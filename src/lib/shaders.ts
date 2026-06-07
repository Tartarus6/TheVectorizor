import mean_shift_cluster_step_shader from '$lib/shaders/mean_shift_cluster_step.wgsl?raw';
import update_density_scores_shader from '$lib/shaders/update_density_scores.wgsl?raw';
import calculate_mean_step_shader from '$lib/shaders/calculate_mean_step.wgsl?raw';
import srgb_to_oklab_shader from '$lib/shaders/srgb_to_oklab.wgsl?raw';
import oklab_to_srgb_shader from '$lib/shaders/oklab_to_srgb.wgsl?raw';
import gaussian_blur_shader from '$lib/shaders/gaussian_blur.wgsl?raw';
import gaussian_gradient_shader from '$lib/shaders/gaussian_gradient.wgsl?raw';
import gradient_max_shader from '$lib/shaders/gradient_maximizing.wgsl?raw';
import edge_tracing_step_shader from '$lib/shaders/edge_tracing_step.wgsl?raw';
import reciprocating_neighbors_shader from '$lib/shaders/reciprocating_neighbors.wgsl?raw';
import edge_visualization_shader from '$lib/shaders/edge_visualization.wgsl?raw';
import face_trace_init_shader from '$lib/shaders/face_trace_init.wgsl?raw';
import face_trace_jump_shader from '$lib/shaders/face_trace_jump.wgsl?raw';
import { faceBuffersToSvg } from '$lib/face_svg';

/*
Idea for turning edges into shapes:

For each pixel with a degree greater than 2 (meaning more than 2 neighbors), say that pixel "points to"
the pixel that is the soonest clockwise pixel (or counterclockwise it shouldnt matter as long as it's
consistent).

͟I͟s͟s͟u͟e͟ ͟1: not sure how to define which direction to start with (easy to tell by looking at it, but not
	sure how to define it)
͟I͟s͟s͟u͟e͟ ͟2: Need to make sure every pixel marked as an edge has at least a degree of 2, otherwise edge
	tracing isn't complete yet, and it won't be possible to turn that edge into part of a closed shape.

For Example:
	╭───────────────────────────────────────────╮
	│Key:                                       │
	│	" ■ " → non-edge pixel (top left shape) │
	│	" ● " → non-edge pixel (bottom shape)   │
	│	" ▲ " → non-edge pixel (top right shape)│
	│	"███" → edge pixel                      │
	╰───────────────────────────────────────────╯

 ╭─1──2──3──4──5──6─╮
A│ ■  ■  ■ ███ ▲  ▲ │
B│███ ■  ■ ███ ▲  ▲ │
C│ ● ███ ■ ███ ▲  ▲ │
D│ ●  ● ████████████│
E│ ●  ●  ●  ●  ●  ● │
 ╰──────────────────╯

We would want to split the pixels above into 3 shapes: the bottom ones, the top left ones, and the top right
ones. In order to make those shapes, we can connect:
- ■: A4 ←→ B4 ←→ C4 ←→ D4 ←→ D3 ←→ C2 ←→ B1
- ●: B1 ←→ C2 ←→ D3 ←→ D4 ←→ D5 ←→ D6
- ▲: D6 ←→ D5 ←→ D4 ←→ C4 ←→ B4 ←→ A4

In the shapes described above, D4 is treated as a "hub" node, so it is included in all 3. I'm not sure how to
make sure that an algorithm to make the shapes will include that central pixel instead of, for example, using
these edges instead (skipping the hub node sometimes):
- ■: A4 ←→ B4 ←→ C4 ←→ D3 ←→ C2 ←→ B1
- ●: B1 ←→ C2 ←→ D3 ←→ D4 ←→ D5 ←→ D6
- ▲: D6 ←→ D5 ←→ C4 ←→ B4 ←→ A4

Notable things for an algorithm:
- The connections described above aren't closed, but that's just because i wanted a short example. We want for
	All shapes to be closed loops. Though, we might want to somehow treat the border of the image as an edge?
	In that case, it would be closed. I think that would make sense.
- The connections descrived above are 2-way, since the outline of a shape isn't directional.
- In the diagram above, D3, C4, D4, and D5 are the high-degree pixels.
- D4 is the "hub" pixel. not sure how to identify that, since D3, D4, and D5 all have a degree of 3, and C4
	even has a degree of 4. So idk how to define a hub node such that it would pick D4.

Maybe could use the theta values of each edge pixel to see which pixels they "point towards", and look for pixels
that are very "pointed at" to find the "hub" pixels? (each pixel would point in 2 directions, since forward and
backward along edge are arbitrary)


Another separate issue is how to deal with staircase pixel patterns

 ╭─1──2──3──4──5─╮
A│███ ▲  ▲  ▲  ▲ │
B│██████ ▲  ▲  ▲ │
C│ ● ██████ ▲  ▲ │
D│ ●  ● ██████ ▲ │
E│ ●  ●  ● ██████│
 ╰───────────────╯

This needs to be identified as a single edge. So like:
A1 ←→ B1 ←→ B2 ←→ C2 ←→ C3 ←→ D3 ←→ D4 ←→ E4 ←→ E5

Pixels do need to be able to connect diagonally (since my edge pixels often have diagonal-only connections). But
sometimes it does end up as a staircase like above. So the example above should be recognised as a single edge,
rather than 2 diagonal edges, or 2 diagonal edges with a staircase on top, or whatever else.

Whatever solution we end up with needs to be efficient, since this project is focused on optimization and
efficiency. So a complex solution like checking all the neighbors of each neighbor to check whether this connection
is needed or not would be too inefficient.




͟U͟p͟d͟a͟t͟e: I have solved the issue of finding the right neighbors. The shader passes now store the connections between
        the edge pixels. So reconstructing the path that the edge tracing steps took is easy. This solves the
        staircase issue.

The issue that remains is tracing the closed faces in the edge pixel graph, and getting the color for each face.

So the idea is to use *Pointer Jumping* to have a shader that combined edges into pairs, then pairs of pairs, etc.,
always picking the "counterclockwise" option.

The trick is that each directed edge corresponds to a closed face.
 ╭1─2─3─4─5─╮
A│  ↙↘  ↙↘  │
B│↓↗  ↓↗  ↖↓│
C│↓↑  ↓↑  ↑↓│
D│↑↘  ↙↑  ↙↑│
E│  ↖↗  ↖↗  │
 ╰──────────╯

The shader will have some data structure like the one shown below:
```wgsl
struct GraphData {
    next_edge : array<u32>,
    face_id   : array<u32>,
};
```
The first pass of the shader will run with one thread per edge connection. It will choose the next "counterclockwise"
connection, and write that as the `next_edge` for the thread's edge connection. It will then take the minimum "ID" of
the 2 edge connections (itsself and the one it just connected to), and write that as the face ID for the thread's edge
pixel.

Then the next pass will again have one thread per edge connection. It will look at the previously chosen `next_edge`
for the thread's edge connection, and then look at the `next_edge` for that edge connection. Then, it will write that
"next next" as the new next for the thread's edge connection, doubling the distance that's been checked. It will also
look at the face ID for the thread's edge connection as well as for the new next, and update the thread's face ID to be
the min of the two.

This will be repeated for some number of iterations until every closed shape has been explored (probably just choose
some arbitrary number of passes that should be enough doublings. something like 5-8, idk). By the end, each edge
connection will correspond to some face ID.

͟I͟s͟s͟u͟e͟ ͟1: Color
This one is pretty easy to solve.
When faces are traced, we need to figure out what color each should be filled in with.

For each edge connection that we look at in the face tracing passes, we can sample the color texture 90 degrees offset
from the direction of the connection at something like a distance of 2 pixels from the connection itsself (in order to
make sure we are getting the internal color, not just edge weirdness). Then each time we do a doubling, we can take the
average of the 2 samples, and write that as the edge's color

That strategy might not work, due to unpredictable comparisons. I think each connection on a face might end up with
slightly different average colors.
So instead, we can just do all of the face tracing, then do another set of passes afterwards. We can split the connectiosn
for each face into pairs, and average them. Then average the pairs, then pairs of pairs, etc. And this should allow us to
get a single average color.

Another idea would be to just use the thing mentioned before, with averaging the colors within the face tracing passes,
then just picking the color of the connection corresponding to the face ID as the color for the shape. If enough passes
happen, all of the connections should have approximately the same color. It should be close enough.


͟I͟s͟s͟u͟e͟ ͟2: Infinitely Large Faces
When certain edges are traced, they will produce an inverted shape with infinite area.
 ╭1─2─3─4─5─6─7─╮
A│              │
B│    ↙→←→←↘    │
C│  ↓↗      ↖↓  │
D│  ↓↑      ↑↓  │
E│  ↑↘      ↙↑  │
F│    ↖→←→←↗    │
G│              │
 ╰──────────────╯

When the graph above is traced, there will be 2 faces created. One face will be the center circle, and the other will
be an open shape that is everything except that circle.

This needs to be dealt with somehow. It might just work to check the area of the resulting directed shape, and if it's
infinite, then throw away the shape?


͟I͟s͟s͟u͟e͟ ͟3: Z-Ordering
With a shader-based face tracing, it will have to be decided which shapes are on top of each other.

This wasn't an issue with the CPU approach, since it iterated through pixels from top left corner to bottom right
corner. This strategy happens to always identify the shapes in the correct order to draw them.

We could do this on the CPU side, using the same method that worked before, but having already traced the faces. As
the CPU algorithm builds the final SVG, we can just have it do so from top left to bottom right. That way it will
(hopefully) correctly order the shapes.
*/

// PERFORMANCE TODOS
// DONE: implement ping-pong textures, stop doing unnecessary texture copies
// DONE: switch from weird uint8array output to just having a canvas context, and having the gpu draw straight to the canvas
// DONE: implement debug canvas view (showing the density scores texture)
// DONE: keep density scores as a buffer, don't turn back into number.
// TODO: automate performance balancing. start at a very low tile size and do some tests, increasing it until it's as big as it can be while meeting max acceptible execution time
// TODO: prevent needing to do texture loads in mean shift cluster step. calculate and store color_dist_squared and image_dist_squared in update_density_scores.wgsl
// TODO: figure out a good value for partial_sum_size
// TODO: (maybe) switch to holding density scores in a texture rather than a general array buffer
// TODO: (maybe) turn update_density_scores into a fragment shader
// TODO: (maybe) turn mean_shift_cluster_step into a fragment shader

// EDGE DETECTION TODOS
// DONE: filter maxima to find only important edges
// DONE: implement Canny double threshold. Above H, maxima are immediately accepted, and abolve L, pixels are taken if connected to valid points. Below L are ignored. (H and L should be calculated based on image)
// DONE: switch to including alpha in gradient math, rather than just Lab
// DONE: better handle pixels on the borders of the image, i think it's currently nearly/completely impossible for them to be marked as an edge. (maybe they should always be edges?)
// DONE: figure out how to turn edges into an actual vector image. How will T-intersections be handled? How will color blocks be identified? How will unclosed edges be handled? etc.
// TODO: add a pass to check whether edge tracing is complete (check if all marked edge pixels have degree of at least 2) (need to do this while avoiding the 100ms waits of cpu-side reads)
// TODO: combine nodes into edges by "Devernay Sub-Pixel Correction" interpolation (quadratic interpolation of the gradient norm between three neighboring positions along the gradient direction)
// TODO: fix junctinons kinda pulling edges in in subpixel offsetting (like the bigger line in a t-junction will get pulled towards the smaller one, like a stitch getting pulled tight)
// TODO: add background removal. primarily for when the background was transparent, so there isn't a mostly transparent square around the svg
// TODO: (size optimizing) recognise straight lines and turn a set of approx. linear points into a line
// TODO: (size optimizing) recognise curves, turn points along approx. continuous curves into a bezier
// TODO: (size optimizing) reduce decimal points saved
// TODO: (size optimizing) remove points that are really close to each other
// TODO: fix Uncaptured WebGPU error: In a dispatch command, indirect:false, caused by: Each current dispatch group size dimension ([73728, 1, 1]) must be less or equal to 65535 (need to split up big images to avoid size limit)

// GENERAL TODOS
// DONE: figure out a name for the stages of the vectorizor (like "cleanup" for the mean shift cluster stuff, and "edge detection" for that, or whatever) and give more descriptive names to functons/files/variables
// DONE: update the canvas view of the output each pass to show incremental work
// DONE: add checks for if device and adapter are defined in each subfunction, to prevent the need to repeat `device!` every time
// DONE: (maybe) remove all or some of the readback buffers. are they needed/used?
// DONE: (question) how exactly do layers work in an svg? what needs to be done to make sure that the stuff on top in the image is drawn on top in the svg?
// TODO: fix clustering messing up outer edges of certain images. it seems like certain images that have white next to a transparent background have their outer edges really mangled. (note for tar: beaSticker_0.png is an example image with this issue)
// TODO: alpha is not respected when drawing directly to canvas context. but it was with the old pixels system. figure out how to make alpha work
// TODO: (maybe) add a mean shift cluster pass at the end that doesn't weight mean by image locality, in order to remove any remaining gradients (prolly not needed though)
// TODO: (maybe) move setup for device, adapter, buffers, etc. into a separate function, just to clean up the main run_shader() function and improve its readability
// TODO: (maybe) make a global const for workgroup sizing (wont sync with shader files, just good to not have multiple possible points of failure)
// TODO: fix super inconsistent naming of edgeData a.k.a. connectionData (should be "connectionData")

// TODO: move this const somewhere better
// TODO: figure out what a good value for this const is
/// each thread in the mean density score passes will be in charge of summing this many elements
const partial_sum_size: number = 8;

const canvas_format: GPUTextureFormat = 'rgba8unorm';

type ImageSize = {
	width: number;
	height: number;
};

type SharedTextures = {
	inputSrgb: GPUTexture;
	oklabPing: GPUTexture;
	oklabPong: GPUTexture;
	gaussianBlurIntermediate: GPUTexture;
	gradientPing: GPUTexture;
	gradientPong: GPUTexture;
	edgePing: GPUTexture;
	edgePong: GPUTexture;
	densityScoresTexture: GPUTexture;
};

type SharedBuffers = {
	densityScores: GPUBuffer;
	inPartialSums: GPUBuffer;
	outPartialSums: GPUBuffer;
	meanDensityScore: GPUBuffer;
	connectionCount: GPUBuffer;
};

type FaceTraceBuffers = {
	edgeDataPing: GPUBuffer;
	edgeDataPong: GPUBuffer;
};

type Pipelines = {
	srgbToOklab: GPURenderPipeline;
	oklabToSrgb: GPURenderPipeline;
	updateDensityScores: GPUComputePipeline;
	meanDensityScore: GPUComputePipeline;
	meanShiftCluster: GPUComputePipeline;
	gaussianBlurH: GPURenderPipeline;
	gaussianBlurV: GPURenderPipeline;
	gaussianGradient: GPURenderPipeline;
	gradientMax: GPUComputePipeline;
	edgeTrace: GPUComputePipeline;
	reciprocatingNeighbors: GPUComputePipeline;
	faceTraceInit: GPUComputePipeline;
	faceTraceJump: GPUComputePipeline;
	edgeVisualization: GPURenderPipeline;
};

export async function run_shader(
	clusterCanvas: GPUCanvasContext,
	edgeCanvas: GPUCanvasContext,
	imageBitMap: ImageBitmap,
	base_bandwidth: number,
	tile_size: number,
	blur_radius: number,
	num_cluster_passes: number,
	num_edge_trace_passes: number
): Promise<[boolean, string]> {
	// -- Setup ---
	console.log('starting mean shift cluster step WGPU');
	const device = await requestDevice();

	if (!device) {
		// TODO: add an actual warning for this on the site, popup or whatever
		alert('need a browser that supports WebGPU');
		return [false, ''];
	}

	const size = getImageSize(imageBitMap);
	const textures = createSharedTextures(device, size);
	const buffers = createSharedBuffers(device, size);
	const pipelines = createPipelines(device);
	const gradientMaxSampler = createGradientMaxSampler(device);

	// TODO: variable texture pointers to make it easier to keep track of which textures are up to date

	// --- Srgb → OkLab ---
	let startTime = performance.now();
	console.log();
	console.log('Srgb -> OkLab:');
	await srgbToOklabPass(
		device,
		pipelines.srgbToOklab,
		imageBitMap,
		textures.inputSrgb,
		textures.oklabPing
	);
	let endTime = performance.now();
	console.log(`execution time: ${(endTime - startTime).toFixed(2)}ms`);

	// --- Blur ---
	startTime = performance.now();
	console.log();
	console.log('Pre-cluster Blur:');
	await gaussianBlurPass(
		device,
		{ gaussianBlurH: pipelines.gaussianBlurH, gaussianBlurV: pipelines.gaussianBlurV },
		textures.gaussianBlurIntermediate,
		blur_radius,
		textures.oklabPing,
		textures.oklabPong
	);
	endTime = performance.now();
	console.log(`execution time: ${(endTime - startTime).toFixed(2)}ms`);

	// --- Mean Shift Cluster Steps ---
	for (let pass_index = 0; pass_index < num_cluster_passes; pass_index++) {
		startTime = performance.now();

		console.log();
		console.log('Pass:', pass_index);

		const clusterInput = (pass_index + 1) % 2 === 1 ? textures.oklabPong : textures.oklabPing;
		const clusterOutput = (pass_index + 1) % 2 === 1 ? textures.oklabPing : textures.oklabPong;

		await densityScoresPass(
			device,
			pipelines.updateDensityScores,
			clusterInput,
			buffers.densityScores,
			textures.densityScoresTexture,
			size,
			base_bandwidth,
			tile_size
		);

		const meanDensityScoreBuffer = await getMeanDensityScore(
			device,
			pipelines.meanDensityScore,
			buffers,
			size
		);
		await meanShiftClusterPass(
			device,
			pipelines.meanShiftCluster,
			meanDensityScoreBuffer,
			clusterInput,
			clusterOutput,
			buffers.densityScores,
			size,
			base_bandwidth,
			tile_size
		);

		endTime = performance.now();
		console.log(`execution time: ${(endTime - startTime).toFixed(2)}ms`);
	}

	// --- Gaussian Gradient ---
	startTime = performance.now();
	console.log();
	console.log('GaussGradient:');
	await gaussianGradientPass(
		device,
		pipelines.gaussianGradient,
		num_cluster_passes % 2 === 0 ? textures.oklabPing : textures.oklabPong,
		textures.gradientPing
	);
	endTime = performance.now();
	console.log(`execution time: ${(endTime - startTime).toFixed(2)}ms`);

	// --- Gradient Maximizing (Edge Seeding) ---
	startTime = performance.now();
	console.log();
	console.log('Gradient Max:');
	await gradientMaxPass(
		device,
		pipelines.gradientMax,
		textures.gradientPing,
		textures.gradientPong,
		textures.edgePing,
		gradientMaxSampler,
		size
	);
	endTime = performance.now();
	console.log(`execution time: ${(endTime - startTime).toFixed(2)}ms`);

	// --- Edge Tracing Steps ---
	// TODO: either switch `final_edge_texture` to not exist (do ping pong like other passes do), or change other passes to use this sort of structure
	let final_edge_texture = textures.edgePing;
	for (
		let edge_trace_pass_index = 0;
		edge_trace_pass_index < num_edge_trace_passes;
		edge_trace_pass_index++
	) {
		startTime = performance.now();
		console.log();
		console.log('Edge Trace Pass:', edge_trace_pass_index);

		const outputTexture = edge_trace_pass_index % 2 === 0 ? textures.edgePong : textures.edgePing;

		await edgeTracePass(
			device,
			pipelines.edgeTrace,
			textures.gradientPong,
			final_edge_texture,
			outputTexture,
			size
		);
		final_edge_texture = outputTexture;

		endTime = performance.now();
		console.log(`execution time: ${(endTime - startTime).toFixed(2)}ms`);
	}

	// --- Reciprocating Neighbors ---
	startTime = performance.now();
	console.log();
	console.log('Reciprocating Neighbors:');
	const reciprocatingNeighborsOutput =
		final_edge_texture === textures.edgePing ? textures.edgePong : textures.edgePing;
	await reciprocatingNeighborsPass(
		device,
		pipelines.reciprocatingNeighbors,
		final_edge_texture,
		reciprocatingNeighborsOutput,
		buffers.connectionCount,
		size
	);
	final_edge_texture = reciprocatingNeighborsOutput;
	endTime = performance.now();
	console.log(`execution time: ${(endTime - startTime).toFixed(2)}ms`);

	// --- Face Tracing Init (directed edges + face ids) ---

	startTime = performance.now();
	console.log();
	console.log('Creating Face Trace Buffers:');
	// making buffers (needed to know how many connections there were in order to know what size to make these buffers)
	const connectionCountNumber = await readU32Buffer(device, buffers.connectionCount);
	const faceBuffers = await createFaceTraceBuffers(device, connectionCountNumber);
	endTime = performance.now();
	console.log(`execution time: ${(endTime - startTime).toFixed(2)}ms`);

	startTime = performance.now();
	console.log();
	console.log('Face Trace Init:');
	await faceTraceInitPass(
		device,
		pipelines.faceTraceInit,
		final_edge_texture,
		textures.inputSrgb,
		faceBuffers.edgeDataPing,
		size
	);
	endTime = performance.now();
	console.log(`execution time: ${(endTime - startTime).toFixed(2)}ms`);

	// --- Face Tracing Pointer Jumping ---
	const faceTracePasses = Math.ceil(Math.log2(Math.max(1, connectionCountNumber)));
	// const faceTracePasses = 500;
	let connectionDataIn = faceBuffers.edgeDataPing;
	let connectionDataOut = faceBuffers.edgeDataPong;
	let finalConnectionData = connectionDataOut;
	for (let passIndex = 0; passIndex < faceTracePasses; passIndex += 1) {
		startTime = performance.now();
		console.log();
		console.log('Face Trace Jump Pass:', passIndex);

		await faceTraceJumpPass(
			device,
			pipelines.faceTraceJump,
			connectionDataIn,
			connectionDataOut,
			connectionCountNumber
		);

		// update final
		finalConnectionData = connectionDataOut;

		// switch in and out for next loop
		const temp = connectionDataIn;
		connectionDataIn = connectionDataOut;
		connectionDataOut = temp;

		endTime = performance.now();
		console.log(`execution time: ${(endTime - startTime).toFixed(2)}ms`);
	}

	// const finalConnectionData = faceBuffers.edgeDataPing;

	// --- Svg Creation ---
	startTime = performance.now();
	console.log();
	console.log('Face SVG:');
	const svg = await faceBuffersToSvg(
		device,
		textures.gradientPong,
		final_edge_texture,
		finalConnectionData,
		size.width,
		size.height,
		connectionCountNumber
	);
	endTime = performance.now();
	console.log(`execution time: ${(endTime - startTime).toFixed(2)}ms`);

	// -- OkLab → Srgb (just for visualization)
	await oklabToSrgbPass(
		device,
		pipelines.oklabToSrgb,
		num_cluster_passes % 2 === 0 ? textures.oklabPing : textures.oklabPong,
		false,
		clusterCanvas
	);

	// -- Edge Visualization
	await edgeVisualizationPass(device, pipelines.edgeVisualization, final_edge_texture, edgeCanvas);

	return [true, svg];
}

async function requestDevice(): Promise<GPUDevice | null> {
	const adapter = await navigator.gpu?.requestAdapter();
	return (await adapter?.requestDevice()) ?? null;
}

function getImageSize(imageBitMap: ImageBitmap): ImageSize {
	return { width: imageBitMap.width, height: imageBitMap.height };
}

function createSharedTextures(device: GPUDevice, size: ImageSize): SharedTextures {
	const inputSrgb = device.createTexture({
		label: 'input srgb texture',
		size: [size.width, size.height],
		format: 'rgba8unorm',
		usage:
			GPUTextureUsage.TEXTURE_BINDING |
			GPUTextureUsage.RENDER_ATTACHMENT |
			GPUTextureUsage.STORAGE_BINDING |
			GPUTextureUsage.COPY_SRC |
			GPUTextureUsage.COPY_DST
	});

	const oklabPing = device.createTexture({
		label: 'input oklab texture',
		size: [size.width, size.height],
		format: 'rgba16float',
		usage:
			GPUTextureUsage.TEXTURE_BINDING |
			GPUTextureUsage.RENDER_ATTACHMENT |
			GPUTextureUsage.STORAGE_BINDING |
			GPUTextureUsage.COPY_SRC |
			GPUTextureUsage.COPY_DST
	});

	const oklabPong = device.createTexture({
		label: 'output oklab texture',
		size: [size.width, size.height],
		format: 'rgba16float',
		usage:
			GPUTextureUsage.TEXTURE_BINDING |
			GPUTextureUsage.RENDER_ATTACHMENT |
			GPUTextureUsage.STORAGE_BINDING |
			GPUTextureUsage.COPY_SRC |
			GPUTextureUsage.COPY_DST
	});

	const gaussianBlurIntermediate = device.createTexture({
		label: 'gaussian blur intermediate texture',
		size: [size.width, size.height],
		format: 'rgba16float',
		usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT
	});

	/*
	gradient texture:
		x → grad_mag        (magnitude of gradient)
		y → theta           (direction of gradient)
		z → subpixel_offset (in the direction of the gradient)
		w → 0               (unused)
	*/
	const gradientPing = device.createTexture({
		label: 'gradient texture ping',
		size: [size.width, size.height],
		format: 'rgba16float',
		usage:
			GPUTextureUsage.TEXTURE_BINDING |
			GPUTextureUsage.STORAGE_BINDING |
			GPUTextureUsage.RENDER_ATTACHMENT |
			GPUTextureUsage.COPY_SRC
	});

	const gradientPong = device.createTexture({
		label: 'gradient texture pong',
		size: [size.width, size.height],
		format: 'rgba16float',
		usage:
			GPUTextureUsage.TEXTURE_BINDING |
			GPUTextureUsage.STORAGE_BINDING |
			GPUTextureUsage.RENDER_ATTACHMENT |
			GPUTextureUsage.COPY_SRC
	});

	/*
	edge textures (rgba16uint):
 		x → edge flag        (whether this pixel is part of an edge)
	    y → packed neighbors (bitmask to say which of the 8 neighbor pixels are connected edge pixels)
	    z → edge_id          (unique edge id, corresponds to the starting index of the pixel's connections)
	    w → 0                (unused)
	*/
	const edgePing = device.createTexture({
		label: 'edge tracing texture ping',
		size: [size.width, size.height],
		format: 'rgba16uint',
		usage:
			GPUTextureUsage.TEXTURE_BINDING |
			GPUTextureUsage.STORAGE_BINDING |
			GPUTextureUsage.COPY_SRC |
			GPUTextureUsage.COPY_DST
	});

	const edgePong = device.createTexture({
		label: 'edge tracing texture pong',
		size: [size.width, size.height],
		format: 'rgba16uint',
		usage:
			GPUTextureUsage.TEXTURE_BINDING |
			GPUTextureUsage.STORAGE_BINDING |
			GPUTextureUsage.COPY_SRC |
			GPUTextureUsage.COPY_DST
	});

	const densityScoresTexture = device.createTexture({
		label: 'density scores texture',
		size: [size.width, size.height],
		format: 'rgba16float',
		usage:
			GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC
	});

	return {
		inputSrgb,
		oklabPing,
		oklabPong,
		gaussianBlurIntermediate,
		gradientPing,
		gradientPong,
		edgePing,
		edgePong,
		densityScoresTexture
	};
}

function createSharedBuffers(device: GPUDevice, size: ImageSize): SharedBuffers {
	const densityScores = device.createBuffer({
		label: 'density scores buffer',
		size: size.width * size.height * 4,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
	});

	const inPartialSums = device.createBuffer({
		label: 'mean density partial sums buffer',
		size: densityScores.size, // same size as density scores buffer
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
	});

	const outPartialSums = device.createBuffer({
		label: 'mean density partial sums buffer',
		size: densityScores.size, // same size as density scores buffer
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
	});

	const meanDensityScore = device.createBuffer({
		label: 'mean density score buffer',
		size: Float32Array.BYTES_PER_ELEMENT,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
	});

	// used to store the number of edge-connections there are
	const connectionCount = device.createBuffer({
		label: 'Connection Count Buffer',
		size: 4, // 4 bytes for a single u32
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
	});

	// TODO: does this overwrite need to be done?
	// initialize counter buffer (in case it wasn't defaulting to 0)
	const initialData = new Uint32Array([0]);
	device.queue.writeBuffer(connectionCount, 0, initialData);

	return {
		densityScores,
		inPartialSums,
		outPartialSums,
		meanDensityScore,
		connectionCount
	};
}

async function createFaceTraceBuffers(
	device: GPUDevice,
	connectionCount: number
): Promise<FaceTraceBuffers> {
	const edgeDataBufferSize = connectionCount * (4 + 4) * Float32Array.BYTES_PER_ELEMENT; // u32 * 4 + vec4f

	const edgeDataPing = device.createBuffer({
		label: 'edge data ping',
		size: edgeDataBufferSize,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
	});

	const edgeDataPong = device.createBuffer({
		label: 'edge data pong',
		size: edgeDataBufferSize,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
	});

	return {
		edgeDataPing,
		edgeDataPong
	};
}

function createGradientMaxSampler(device: GPUDevice): GPUSampler {
	return device.createSampler({
		label: 'gradient max sampler',
		addressModeU: 'clamp-to-edge',
		addressModeV: 'clamp-to-edge',
		magFilter: 'linear',
		minFilter: 'linear',
		mipmapFilter: 'nearest'
	});
}

function createPipelines(device: GPUDevice): Pipelines {
	const srgbToOklabModule = device.createShaderModule({
		label: 'srgb to oklab module',
		code: srgb_to_oklab_shader
	});
	const srgbToOklab = device.createRenderPipeline({
		label: 'srgb to oklab render pipeline',
		layout: 'auto',
		vertex: {
			entryPoint: 'vs_main',
			module: srgbToOklabModule
		},
		fragment: {
			entryPoint: 'fs_main',
			module: srgbToOklabModule,
			targets: [
				{
					format: 'rgba16float'
				}
			]
		}
	});

	const oklabToSrgbModule = device.createShaderModule({
		label: 'oklab to srgb module',
		code: oklab_to_srgb_shader
	});
	const oklabToSrgb = device.createRenderPipeline({
		label: 'oklab to srgb render pipeline',
		layout: 'auto',
		vertex: {
			entryPoint: 'vs_main',
			module: oklabToSrgbModule
		},
		fragment: {
			entryPoint: 'fs_main',
			module: oklabToSrgbModule,
			targets: [
				{
					format: canvas_format
				}
			]
		}
	});

	const updateDensityScores = device.createComputePipeline({
		label: 'update density scores compute pipeline',
		layout: 'auto',
		compute: {
			module: device.createShaderModule({
				label: 'update density scores module',
				code: update_density_scores_shader
			}),
			entryPoint: 'cs_main'
		}
	});

	const meanDensityScore = device.createComputePipeline({
		label: 'mean density score compute pipeline',
		layout: 'auto',
		compute: {
			module: device.createShaderModule({
				label: 'mean density score module',
				code: calculate_mean_step_shader
			}),
			entryPoint: 'cs_main'
		}
	});

	const meanShiftCluster = device.createComputePipeline({
		label: 'mean shift cluster compute pipeline',
		layout: 'auto',
		compute: {
			module: device.createShaderModule({
				label: 'mean shift cluster module',
				code: mean_shift_cluster_step_shader
			}),
			entryPoint: 'cs_main'
		}
	});

	const gaussianBlurModule = device.createShaderModule({
		label: 'gaussian blur module',
		code: gaussian_blur_shader
	});
	const gaussianBlurH = device.createRenderPipeline({
		label: 'gaussian blur horizontal pipeline',
		layout: 'auto',
		vertex: {
			entryPoint: 'vs_main',
			module: gaussianBlurModule
		},
		fragment: {
			entryPoint: 'blur_horizontal',
			module: gaussianBlurModule,
			targets: [
				{
					format: 'rgba16float'
				}
			]
		}
	});
	const gaussianBlurV = device.createRenderPipeline({
		label: 'gaussian blur vertical pipeline',
		layout: 'auto',
		vertex: {
			entryPoint: 'vs_main',
			module: gaussianBlurModule
		},
		fragment: {
			entryPoint: 'blur_vertical',
			module: gaussianBlurModule,
			targets: [
				{
					format: 'rgba16float'
				}
			]
		}
	});

	const gaussianGradientModule = device.createShaderModule({
		label: 'gaussian gradient module',
		code: gaussian_gradient_shader
	});
	const gaussianGradient = device.createRenderPipeline({
		label: 'difference of gaussian pipeline',
		layout: 'auto',
		vertex: {
			entryPoint: 'vs_main',
			module: gaussianGradientModule
		},
		fragment: {
			entryPoint: 'cs_main',
			module: gaussianGradientModule,
			targets: [
				{
					format: 'rgba16float'
				}
			]
		}
	});

	const gradientMaxModule = device.createShaderModule({
		label: 'gradient max module',
		code: gradient_max_shader
	});
	const gradientMax = device.createComputePipeline({
		label: 'gradient max compute pipeline',
		layout: 'auto',
		compute: {
			module: gradientMaxModule,
			entryPoint: 'cs_main'
		}
	});

	const edgeTraceModule = device.createShaderModule({
		label: 'edge tracing module',
		code: edge_tracing_step_shader
	});
	const edgeTrace = device.createComputePipeline({
		label: 'edge tracing compute pipeline',
		layout: 'auto',
		compute: {
			module: edgeTraceModule,
			entryPoint: 'cs_main'
		}
	});

	const reciprocatingNeighborsModule = device.createShaderModule({
		label: 'reciprocating neighbors module',
		code: reciprocating_neighbors_shader
	});
	const reciprocatingNeighbors = device.createComputePipeline({
		label: 'reciprocating neighbors compute pipeline',
		layout: 'auto',
		compute: {
			module: reciprocatingNeighborsModule,
			entryPoint: 'cs_main'
		}
	});

	const faceTraceInitModule = device.createShaderModule({
		label: 'face trace init module',
		code: face_trace_init_shader
	});
	const faceTraceInit = device.createComputePipeline({
		label: 'face trace init compute pipeline',
		layout: 'auto',
		compute: {
			module: faceTraceInitModule,
			entryPoint: 'cs_main'
		}
	});

	const faceTraceJumpModule = device.createShaderModule({
		label: 'face trace jump module',
		code: face_trace_jump_shader
	});
	const faceTraceJump = device.createComputePipeline({
		label: 'face trace jump compute pipeline',
		layout: 'auto',
		compute: {
			module: faceTraceJumpModule,
			entryPoint: 'cs_main'
		}
	});

	const edgeVisualizationModule = device.createShaderModule({
		label: 'edge visualization module',
		code: edge_visualization_shader
	});
	const edgeVisualization = device.createRenderPipeline({
		label: 'edge visualization render pipeline',
		layout: 'auto',
		vertex: {
			entryPoint: 'vs_main',
			module: edgeVisualizationModule
		},
		fragment: {
			entryPoint: 'fs_main',
			module: edgeVisualizationModule,
			targets: [
				{
					format: canvas_format
				}
			]
		}
	});

	return {
		srgbToOklab,
		oklabToSrgb,
		updateDensityScores,
		meanDensityScore,
		meanShiftCluster,
		gaussianBlurH,
		gaussianBlurV,
		gaussianGradient,
		gradientMax,
		edgeTrace,
		reciprocatingNeighbors,
		faceTraceInit,
		faceTraceJump,
		edgeVisualization
	};
}

async function srgbToOklabPass(
	device: GPUDevice,
	pipeline: GPURenderPipeline,
	imageBitMap: ImageBitmap,
	inputSrgbTexture: GPUTexture,
	outputOklabTexture: GPUTexture
): Promise<void> {
	device.queue.copyExternalImageToTexture(
		{ source: imageBitMap },
		{ texture: inputSrgbTexture },
		{ width: imageBitMap.width, height: imageBitMap.height }
	);

	const bindGroup = device.createBindGroup({
		label: 'srgb to oklab bind group',
		layout: pipeline.getBindGroupLayout(0),
		entries: [{ binding: 0, resource: inputSrgbTexture.createView() }]
	});

	const encoder = device.createCommandEncoder({ label: 'srgb to oklab encoder' });
	const pass = encoder.beginRenderPass({
		label: 'srgb to oklab render pass',
		colorAttachments: [
			{
				view: outputOklabTexture.createView(),
				clearValue: [0, 0, 0, 0],
				loadOp: 'clear',
				storeOp: 'store'
			}
		]
	});

	pass.setPipeline(pipeline);
	pass.setBindGroup(0, bindGroup);
	pass.draw(3);
	pass.end();

	device.queue.submit([encoder.finish()]);
}

async function oklabToSrgbPass(
	device: GPUDevice,
	pipeline: GPURenderPipeline,
	texture: GPUTexture,
	showEdgePixels: boolean,
	context: GPUCanvasContext
): Promise<void> {
	const debugUniformsData = new Uint32Array([showEdgePixels ? 1 : 0]);
	const debugUniformsBuffer = device.createBuffer({
		label: 'oklab to srgb debug uniforms buffer',
		size: debugUniformsData.byteLength,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
	});
	device.queue.writeBuffer(debugUniformsBuffer, 0, debugUniformsData);

	const bindGroup = device.createBindGroup({
		label: 'oklab to srgb ping bind group',
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: texture.createView() },
			{ binding: 1, resource: { buffer: debugUniformsBuffer } }
		]
	});

	context.configure({
		device,
		format: canvas_format
	});

	const renderPassDescriptor: GPURenderPassDescriptor = {
		label: 'oklab to srgb render pass',
		colorAttachments: [
			{
				view: context.getCurrentTexture().createView(),
				clearValue: [0, 0, 0, 0],
				loadOp: 'clear',
				storeOp: 'store'
			}
		]
	};

	const encoder = device.createCommandEncoder({ label: 'oklab to srgb encoder' });

	const pass = encoder.beginRenderPass(renderPassDescriptor);

	pass.setPipeline(pipeline);
	pass.setBindGroup(0, bindGroup);
	pass.draw(3);
	pass.end();

	device.queue.submit([encoder.finish()]);
}

async function densityScoresPass(
	device: GPUDevice,
	pipeline: GPUComputePipeline,
	texture: GPUTexture,
	densityScoresBuffer: GPUBuffer,
	densityScoresTexture: GPUTexture,
	size: ImageSize,
	baseBandwidth: number,
	tileSize: number
): Promise<void> {
	/*
	struct Uniforms {
		base_bandwidth: f32,
	}
	*/
	const floatUniformsData = new Float32Array([baseBandwidth]);
	/*
	struct UintUniforms {
		tile_x: u32,    /// the low x value of the current tile (basically the x-offset for this shader pass)
		tile_y: u32,    /// the low y value of the current tile (basically the y-offset for this shader pass)
		tile_size: u32, /// the size of each tile (the range of x and y for this shader pass)
	}
	*/
	const uintUniformsData = new Uint32Array([0, 0, tileSize]);

	const floatUniformsBuffer = device.createBuffer({
		label: 'float uniforms buffer',
		size: floatUniformsData.byteLength,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
	});

	const uintUniformsBuffer = device.createBuffer({
		label: 'uint uniforms buffer',
		size: uintUniformsData.byteLength,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
	});

	const bindGroup = device.createBindGroup({
		label: 'update density scores bind group',
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: floatUniformsBuffer },
			{ binding: 1, resource: uintUniformsBuffer },
			{ binding: 2, resource: texture.createView() },
			{ binding: 3, resource: densityScoresBuffer },
			{ binding: 4, resource: densityScoresTexture.createView() }
		]
	});

	device.queue.writeBuffer(floatUniformsBuffer, 0, floatUniformsData);

	for (let tileX = 0; tileX < size.width; tileX += tileSize) {
		for (let tileY = 0; tileY < size.height; tileY += tileSize) {
			// update the uint uniforms buffer for this pass
			uintUniformsData[0] = tileX;
			uintUniformsData[1] = tileY;
			device.queue.writeBuffer(uintUniformsBuffer, 0, uintUniformsData);

			const encoder = device.createCommandEncoder({
				label: 'update density scores encoder'
			});
			const pass = encoder.beginComputePass({
				label: 'update density scores compute pass'
			});
			pass.setPipeline(pipeline);
			pass.setBindGroup(0, bindGroup);
			pass.dispatchWorkgroups(Math.ceil(tileSize / 16), Math.ceil(tileSize / 16));
			pass.end();

			device.queue.submit([encoder.finish()]);
		}
	}
}

async function getMeanDensityScore(
	device: GPUDevice,
	pipeline: GPUComputePipeline,
	buffers: Pick<
		SharedBuffers,
		'inPartialSums' | 'outPartialSums' | 'meanDensityScore' | 'densityScores'
	>,
	size: ImageSize
): Promise<GPUBuffer> {
	/*
	struct Uniforms {
		partial_sum_size: u32,         /// each thread will be in charge of summing this many elements
		num_remaining_elements: u32,   /// how many elements exist in `in_partial_sums`
	}
	*/
	const uniformsData = new Uint32Array([partial_sum_size, size.width * size.height]);

	const uniformsBuffer = device.createBuffer({
		label: 'mean density uniforms buffer',
		size: uniformsData.byteLength,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
	});

	const bindGroup = device.createBindGroup({
		label: 'mean density bind group',
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: uniformsBuffer },
			{ binding: 1, resource: buffers.inPartialSums },
			{ binding: 2, resource: buffers.outPartialSums }
		]
	});
	device.queue.writeBuffer(uniformsBuffer, 0, uniformsData);

	const initEncoder = device.createCommandEncoder({
		label: 'mean density init encoder'
	});
	initEncoder.copyBufferToBuffer(
		buffers.densityScores,
		0,
		buffers.inPartialSums,
		0,
		buffers.densityScores.size
	);
	device.queue.submit([initEncoder.finish()]);

	let numRemainingElements = size.width * size.height;
	while (numRemainingElements > 1) {
		uniformsData[1] = numRemainingElements;
		device.queue.writeBuffer(uniformsBuffer, 0, uniformsData);

		const nextNumRemainingElements = Math.ceil(numRemainingElements / partial_sum_size);

		const encoder = device.createCommandEncoder({
			label: 'mean density encoder'
		});

		const pass = encoder.beginComputePass({
			label: 'mean density compute pass'
		});
		pass.setPipeline(pipeline);
		pass.setBindGroup(0, bindGroup);
		pass.dispatchWorkgroups(Math.ceil(nextNumRemainingElements / 256));
		pass.end();

		if (nextNumRemainingElements > 1) {
			encoder.copyBufferToBuffer(
				buffers.outPartialSums,
				0,
				buffers.inPartialSums,
				0,
				nextNumRemainingElements * Float32Array.BYTES_PER_ELEMENT
			);
		}

		device.queue.submit([encoder.finish()]);
		numRemainingElements = nextNumRemainingElements;
	}

	const outputEncoder = device.createCommandEncoder({
		label: 'mean density output encoder'
	});
	outputEncoder.copyBufferToBuffer(
		buffers.outPartialSums,
		0,
		buffers.meanDensityScore,
		0,
		Float32Array.BYTES_PER_ELEMENT
	);
	device.queue.submit([outputEncoder.finish()]);

	return buffers.meanDensityScore;
}

async function meanShiftClusterPass(
	device: GPUDevice,
	pipeline: GPUComputePipeline,
	meanDensityScoreBuffer: GPUBuffer,
	textureIn: GPUTexture,
	textureOut: GPUTexture,
	densityScoresBuffer: GPUBuffer,
	size: ImageSize,
	baseBandwidth: number,
	tileSize: number
): Promise<void> {
	/*
	struct FloatUniforms {
		base_bandwidth: f32,
	}
	*/
	const floatUniformsData = new Float32Array([baseBandwidth]);
	/*
	struct UintUniforms {
		tile_x: u32,    /// the low x value of the current tile (basically the x-offset for this shader pass)
		tile_y: u32,    /// the low y value of the current tile (basically the y-offset for this shader pass)
		tile_size: u32, /// the size of each tile (the range of x and y for this shader pass)
	}
	*/
	const uintUniformsData = new Uint32Array([0, 0, tileSize]);

	const floatUniformsBuffer = device.createBuffer({
		label: 'mean shift cluster float uniforms buffer',
		size: floatUniformsData.byteLength,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
	});

	const uintUniformsBuffer = device.createBuffer({
		label: 'mean shift cluster uint uniforms buffer',
		size: uintUniformsData.byteLength,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
	});

	const bindGroup = device.createBindGroup({
		label: 'mean shift cluster ping bind group',
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: floatUniformsBuffer },
			{ binding: 1, resource: meanDensityScoreBuffer },
			{ binding: 2, resource: uintUniformsBuffer },
			{ binding: 3, resource: textureIn.createView() },
			{ binding: 4, resource: densityScoresBuffer },
			{ binding: 5, resource: textureOut.createView() }
		]
	});

	device.queue.writeBuffer(floatUniformsBuffer, 0, floatUniformsData);

	for (let tileX = 0; tileX < size.width; tileX += tileSize) {
		for (let tileY = 0; tileY < size.height; tileY += tileSize) {
			uintUniformsData[0] = tileX;
			uintUniformsData[1] = tileY;
			device.queue.writeBuffer(uintUniformsBuffer, 0, uintUniformsData);

			const encoder = device.createCommandEncoder({
				label: 'mean shift cluster encoder'
			});
			const pass = encoder.beginComputePass({
				label: 'mean shift cluster compute pass'
			});
			pass.setPipeline(pipeline);
			pass.setBindGroup(0, bindGroup);
			pass.dispatchWorkgroups(Math.ceil(tileSize / 16), Math.ceil(tileSize / 16));
			pass.end();

			device.queue.submit([encoder.finish()]);
		}
	}
}

async function gaussianBlurPass(
	device: GPUDevice,
	pipelines: Pick<Pipelines, 'gaussianBlurH' | 'gaussianBlurV'>,
	gaussianBlurIntermediateTexture: GPUTexture,
	radius: number,
	inputTexture: GPUTexture,
	outputTexture: GPUTexture
): Promise<void> {
	const uniformsData = new Uint32Array([radius]);
	const uniformsBuffer = device.createBuffer({
		label: 'gaussian blur uniforms buffer',
		size: uniformsData.byteLength,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
	});
	device.queue.writeBuffer(uniformsBuffer, 0, uniformsData);

	const kernelWeights = compute_gaussian_kernel(radius);
	const kernelBuffer = device.createBuffer({
		label: 'gaussian blur kernel weights buffer',
		size: kernelWeights.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
	});
	device.queue.writeBuffer(kernelBuffer, 0, kernelWeights);

	const hBindGroup = device.createBindGroup({
		label: 'gaussian blur horizontal bind group',
		layout: pipelines.gaussianBlurH.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: uniformsBuffer } },
			{ binding: 1, resource: inputTexture.createView() },
			{ binding: 2, resource: { buffer: kernelBuffer } }
		]
	});

	const vBindGroup = device.createBindGroup({
		label: 'gaussian blur vertical bind group',
		layout: pipelines.gaussianBlurV.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: uniformsBuffer } },
			{ binding: 1, resource: gaussianBlurIntermediateTexture.createView() },
			{ binding: 2, resource: { buffer: kernelBuffer } }
		]
	});

	const hEncoder = device.createCommandEncoder({ label: 'gaussian blur horizontal encoder' });
	const hPass = hEncoder.beginRenderPass({
		label: 'gaussian blur horizontal render pass',
		colorAttachments: [
			{
				view: gaussianBlurIntermediateTexture.createView(),
				clearValue: [0, 0, 0, 0],
				loadOp: 'clear',
				storeOp: 'store'
			}
		]
	});
	hPass.setPipeline(pipelines.gaussianBlurH);
	hPass.setBindGroup(0, hBindGroup);
	hPass.draw(3);
	hPass.end();
	device.queue.submit([hEncoder.finish()]);

	const vEncoder = device.createCommandEncoder({ label: 'gaussian blur vertical encoder' });
	const vPass = vEncoder.beginRenderPass({
		label: 'gaussian blur vertical render pass',
		colorAttachments: [
			{
				view: outputTexture.createView(),
				clearValue: [0, 0, 0, 0],
				loadOp: 'clear',
				storeOp: 'store'
			}
		]
	});
	vPass.setPipeline(pipelines.gaussianBlurV);
	vPass.setBindGroup(0, vBindGroup);
	vPass.draw(3);
	vPass.end();
	device.queue.submit([vEncoder.finish()]);
}

async function gaussianGradientPass(
	device: GPUDevice,
	pipeline: GPURenderPipeline,
	inTexture: GPUTexture,
	outTexture: GPUTexture
): Promise<void> {
	const bindGroup = device.createBindGroup({
		label: 'gaussian gradient bind group',
		layout: pipeline.getBindGroupLayout(0),
		entries: [{ binding: 0, resource: inTexture.createView() }]
	});

	const encoder = device.createCommandEncoder({ label: 'gaussian gradient encoder' });
	const pass = encoder.beginRenderPass({
		label: 'gaussian gradient render pass',
		colorAttachments: [
			{
				view: outTexture.createView(),
				clearValue: [0, 0, 0, 0],
				loadOp: 'clear',
				storeOp: 'store'
			}
		]
	});
	pass.setPipeline(pipeline);
	pass.setBindGroup(0, bindGroup);
	pass.draw(3);
	pass.end();

	device.queue.submit([encoder.finish()]);
}

async function gradientMaxPass(
	device: GPUDevice,
	pipeline: GPUComputePipeline,
	inGradientTexture: GPUTexture,
	outGradientTexture: GPUTexture,
	outEdgeTexture: GPUTexture,
	gradientMaxSampler: GPUSampler,
	size: ImageSize
): Promise<void> {
	const bindGroup = device.createBindGroup({
		label: 'gradient max bind group',
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: inGradientTexture.createView() },
			{ binding: 1, resource: gradientMaxSampler },
			{ binding: 2, resource: outGradientTexture.createView() },
			{ binding: 3, resource: outEdgeTexture.createView() }
		]
	});

	const encoder = device.createCommandEncoder({ label: 'gradient max encoder' });
	const pass = encoder.beginComputePass({
		label: 'gradient max compute pass'
	});
	pass.setPipeline(pipeline);
	pass.setBindGroup(0, bindGroup);
	pass.dispatchWorkgroups(Math.ceil(size.width / 16), Math.ceil(size.height / 16));
	pass.end();

	device.queue.submit([encoder.finish()]);
}

async function edgeTracePass(
	device: GPUDevice,
	pipeline: GPUComputePipeline,
	gradientTexture: GPUTexture,
	inEdgeTexture: GPUTexture,
	outEdgeTexture: GPUTexture,
	size: ImageSize
): Promise<void> {
	const bindGroup = device.createBindGroup({
		label: 'edge tracing bind group',
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: gradientTexture.createView() },
			{ binding: 1, resource: inEdgeTexture.createView() },
			{ binding: 2, resource: outEdgeTexture.createView() }
		]
	});

	const encoder = device.createCommandEncoder({ label: 'edge tracing encoder' });

	encoder.copyTextureToTexture(
		{
			texture: inEdgeTexture
		},
		{
			texture: outEdgeTexture
		},
		{
			width: size.width,
			height: size.height,
			depthOrArrayLayers: 1
		}
	);

	const pass = encoder.beginComputePass({
		label: 'edge tracing compute pass'
	});
	pass.setPipeline(pipeline);
	pass.setBindGroup(0, bindGroup);
	pass.dispatchWorkgroups(Math.ceil(size.width / 16), Math.ceil(size.height / 16));
	pass.end();

	device.queue.submit([encoder.finish()]);
}

async function reciprocatingNeighborsPass(
	device: GPUDevice,
	pipeline: GPUComputePipeline,
	inEdgeTexture: GPUTexture,
	outEdgeTexture: GPUTexture,
	connectionsCountBuffer: GPUBuffer,
	size: ImageSize
): Promise<void> {
	const bindGroup = device.createBindGroup({
		label: 'edge power bind group',
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: inEdgeTexture.createView() },
			{ binding: 1, resource: outEdgeTexture.createView() },
			{ binding: 2, resource: { buffer: connectionsCountBuffer } }
		]
	});

	const encoder = device.createCommandEncoder({ label: 'edge power encoder' });
	const pass = encoder.beginComputePass({
		label: 'edge power compute pass'
	});
	pass.setPipeline(pipeline);
	pass.setBindGroup(0, bindGroup);
	pass.dispatchWorkgroups(Math.ceil(size.width / 16), Math.ceil(size.height / 16));
	pass.end();

	device.queue.submit([encoder.finish()]);
}

async function faceTraceInitPass(
	device: GPUDevice,
	pipeline: GPUComputePipeline,
	edgeTexture: GPUTexture,
	colorTexture: GPUTexture,
	edgeDataOut: GPUBuffer,
	size: ImageSize
): Promise<void> {
	const bindGroup = device.createBindGroup({
		label: 'face trace init bind group',
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: edgeTexture.createView() },
			{ binding: 1, resource: colorTexture.createView() },
			{ binding: 2, resource: { buffer: edgeDataOut } }
		]
	});

	const encoder = device.createCommandEncoder({ label: 'face trace init encoder' });
	const pass = encoder.beginComputePass({ label: 'face trace init compute pass' });
	pass.setPipeline(pipeline);
	pass.setBindGroup(0, bindGroup);
	pass.dispatchWorkgroups(Math.ceil(size.width / 16), Math.ceil(size.height / 16));
	pass.end();

	device.queue.submit([encoder.finish()]);
}

async function faceTraceJumpPass(
	device: GPUDevice,
	pipeline: GPUComputePipeline,
	edgeDataIn: GPUBuffer,
	edgeDataOut: GPUBuffer,
	connectionCount: number
): Promise<void> {
	const params = new Uint32Array([connectionCount]);
	const paramsBuffer = device.createBuffer({
		label: 'face trace jump params',
		size: params.byteLength,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
	});
	device.queue.writeBuffer(paramsBuffer, 0, params);

	const bindGroup = device.createBindGroup({
		label: 'face trace jump bind group',
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: paramsBuffer } },
			{ binding: 1, resource: { buffer: edgeDataIn } },
			{ binding: 2, resource: { buffer: edgeDataOut } }
		]
	});

	const encoder = device.createCommandEncoder({ label: 'face trace jump encoder' });
	const pass = encoder.beginComputePass({ label: 'face trace jump compute pass' });
	pass.setPipeline(pipeline);
	pass.setBindGroup(0, bindGroup);
	pass.dispatchWorkgroups(Math.ceil(connectionCount / 256));
	pass.end();

	device.queue.submit([encoder.finish()]);
}

async function edgeVisualizationPass(
	device: GPUDevice,
	pipeline: GPURenderPipeline,
	edgeTexture: GPUTexture,
	context: GPUCanvasContext
): Promise<void> {
	const bindGroup = device.createBindGroup({
		label: 'edge visualization bind group',
		layout: pipeline.getBindGroupLayout(0),
		entries: [{ binding: 0, resource: edgeTexture.createView() }]
	});

	context.configure({
		device,
		format: canvas_format
	});

	const renderPassDescriptor: GPURenderPassDescriptor = {
		label: 'edge visualization render pass',
		colorAttachments: [
			{
				view: context.getCurrentTexture().createView(),
				clearValue: [0, 0, 0, 0],
				loadOp: 'clear',
				storeOp: 'store'
			}
		]
	};

	const encoder = device.createCommandEncoder({ label: 'edge visualization encoder' });

	const pass = encoder.beginRenderPass(renderPassDescriptor);
	pass.setPipeline(pipeline);
	pass.setBindGroup(0, bindGroup);
	pass.draw(3);
	pass.end();

	device.queue.submit([encoder.finish()]);
}

/// returns whether the colors changed (used to know whether to increase count)
function compute_gaussian_kernel(radius: number): Float32Array {
	const sigma = radius / 3.0;
	const weights = new Float32Array(radius + 1);

	for (let i = 0; i <= radius; i++) {
		weights[i] = Math.exp(-(i * i) / (2 * sigma * sigma));
	}

	let sum = weights[0];
	for (let i = 1; i <= radius; i++) sum += 2 * weights[i];
	for (let i = 0; i <= radius; i++) weights[i] /= sum;

	return weights;
}

async function readU32Buffer(device: GPUDevice, buffer: GPUBuffer): Promise<number> {
	const byteLength = Uint32Array.BYTES_PER_ELEMENT; // TODO: should this just be set to the number 4? or is this fine?
	const readback = device.createBuffer({
		label: 'u32 readback buffer',
		size: byteLength,
		usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
	});

	const encoder = device.createCommandEncoder({ label: 'u32 readback encoder' });
	encoder.copyBufferToBuffer(buffer, 0, readback, 0, byteLength);
	device.queue.submit([encoder.finish()]);
	await readback.mapAsync(GPUMapMode.READ);

	const data = new Uint32Array(readback.getMappedRange().slice(0))[0];
	readback.unmap();
	return data;
}

interface EdgeData {
	nextConnectionIdx: number;
	jumpNextIdx: number;
	faceId: number;
	posIdx: number;
	color: Float32Array; // or [number, number, number, number]
}

async function readEdgeDataBuffer(
	device: GPUDevice,
	buffer: GPUBuffer,
	count: number
): Promise<EdgeData[]> {
	const bytesPerElement = 32; // 2 * u32 (8) + vec4f (16)
	const byteLength = count * bytesPerElement;

	const readback = device.createBuffer({
		label: 'EdgeData readback buffer',
		size: byteLength,
		usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
	});

	const encoder = device.createCommandEncoder();
	encoder.copyBufferToBuffer(buffer, 0, readback, 0, byteLength);
	device.queue.submit([encoder.finish()]);

	await readback.mapAsync(GPUMapMode.READ);
	const mapped = readback.getMappedRange();

	// Use a DataView to handle the mixed types and offsets
	const view = new DataView(mapped);
	const result: EdgeData[] = [];

	for (let i = 0; i < count; i++) {
		const offset = i * bytesPerElement;
		result.push({
			nextConnectionIdx: view.getUint32(offset + 0, true),
			jumpNextIdx: view.getUint32(offset + 4, true),
			faceId: view.getUint32(offset + 8, true),
			posIdx: view.getUint32(offset + 12, true),
			color: new Float32Array([
				view.getFloat32(offset + 16, true),
				view.getFloat32(offset + 20, true),
				view.getFloat32(offset + 24, true),
				view.getFloat32(offset + 28, true)
			])
		});
	}

	readback.unmap();
	readback.destroy();

	console.log(result);

	return result;
}
