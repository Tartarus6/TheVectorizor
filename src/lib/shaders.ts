import mean_shift_cluster_step_shader from '$lib/shaders/mean_shift_cluster_step.wgsl?raw';
import update_density_scores_shader from '$lib/shaders/update_density_scores.wgsl?raw';
import calculate_mean_step_shader from '$lib/shaders/calculate_mean_step.wgsl?raw';
import srgb_to_oklab_shader from '$lib/shaders/srgb_to_oklab.wgsl?raw';
import oklab_to_srgb_shader from '$lib/shaders/oklab_to_srgb.wgsl?raw';
import gaussian_blur_shader from '$lib/shaders/gaussian_blur.wgsl?raw';
import gaussian_gradient_shader from '$lib/shaders/gaussian_gradient.wgsl?raw';
import gradient_max_shader from '$lib/shaders/gradient_maximizing.wgsl?raw';
import edge_tracing_step_shader from '$lib/shaders/edge_tracing_step.wgsl?raw';
import { textureToEdgeSvg } from '$lib/edge_svg';

/*
Idea for turning edges into shapes:

For each pixel with a degree greater than 2 (meaning more than 2 neighbors), say that pixel "points to"
the pixel that is the soonest clockwise pixel (or counterclockwise it shouldnt matter as long as it's
consistent).

*Issue*: not sure how to define which direction to start with (easy to tell by looking at it, but not
sure how to define it)

For Example:

 +-1--2--3--4--5--6-+
A| ·  ·  · [ ] ·  · |
B|[ ] ·  · [ ] ·  · |
C| · [ ] · [ ] ·  · |
D| ·  · [ ][ ][ ][ ]|
E| ·  ·  ·  ·  ·  · |
 +------------------+

We would want to split the pixels above into 3 shapes: the bottom ones, the top left ones, and the top right
ones. In order to make those shapes, we can connect:
	Top Left  - A4 <-> B4 <-> C4 <-> D4 <-> D3 <-> C2 <-> B1
	Bottom    - B1 <-> C2 <-> D3 <-> D4 <-> D5 <-> D6
	Top Right - D6 <-> D5 <-> D4 <-> C4 <-> B4 <-> A4

In the shapes described above, D4 is treated as a "hub" node, so it is included in all 3. I'm not sure how to
make sure that an algorithm to make the shapes will include that central pixel instead of, for example, using
these edges instead (skipping the hub node sometimes).
	Top Left  - A4 <-> B4 <-> C4 <-> D3 <-> C2 <-> B1
	Bottom    - B1 <-> C2 <-> D3 <-> D4 <-> D5 <-> D6
	Top Right - D6 <-> D5 <-> C4 <-> B4 <-> A4

Notable things for an algorithm:
- In the diagram above, D3, C4, D4, and D5 are the high-degree pixels.
- D4 is the "hub" pixel. not sure how to identify that, since D3, D4, and D5 all have a degree of 3, and C4
	even has a degree of 4. So idk how to define a hub node such that it would pick D4.
- The connections described above aren't closed, but that's just because i wanted a short example. We want for
	All shapes to be closed loops.
*/

// TODO: move this const somewhere better
// TODO: figure out what a good value for this const is
/// each thread in the mean density score passes will be in charge of summing this many elements
const partial_sum_size: number = 8;

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
// TODO: add a pass to check whether edge tracing is complete (check if all marked edge pixels have degree of at least 2) (need to do this while avoiding the 100ms waits of cpu-side reads)
// TODO: combine nodes into edges by "Devernay Sub-Pixel Correction" interpolation (quadratic interpolation of the gradient norm between three neighboring positions along the gradient direction)
// TODO: figure out how to turn edges into an actual vector image. How will T-intersections be handled? How will color blocks be identified? How will unclosed edges be handled? etc.

// GENERAL TODOS
// DONE: figure out a name for the stages of the vectorizor (like "cleanup" for the mean shift cluster stuff, and "edge detection" for that, or whatever) and give more descriptive names to functons/files/variables
// TODO: deal with unused alpha in clustering. (need to figure what makes for a good function, and if alpha should be removed or not)
// TODO: handling for transparent pixels: fully transparent pixels should be completely ignored (so the mean density score will have to divide by the number of non-transparent pixels rather than the width * height of the image)
// TODO: add checks for if device and adapter are defined in each subfunction, to prevent the need to repeat `device!` every time
// TODO: update the canvas view of the output each pass to show incremental work
// TODO: alpha is not respected when drawing directly to canvas context. but it was with the old pixels system. figure out how to make alpha work
// TODO: (maybe) add a mean shift cluster pass at the end that doesn't weight mean by image locality, in order to remove any remaining gradients (prolly not needed though)
// TODO: (maybe) move setup for device, adapter, buffers, etc. into a separate function, just to clean up the main run_shader() function and improve its readability
// TODO: (maybe) remove all or some of the readback buffers. are they needed/used?
// TODO: (maybe) make a global const for workgroup sizing (wont sync with shader files, just good to not have multiple possible points of failure)

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
	console.log('starting mean shift cluster step WGPU');
	const adapter = await navigator.gpu?.requestAdapter();
	const device = await adapter?.requestDevice();

	if (!device) {
		// TODO: add an actual warning for this on the site, popup or whatever
		alert('need a browser that supports WebGPU');
		return [false, ''];
	}

	// --- Shared Textures ---
	const input_srgb_texture = device.createTexture({
		label: 'input srgb texture',
		size: [imageBitMap.width, imageBitMap.height],
		format: 'rgba8unorm',
		usage:
			GPUTextureUsage.TEXTURE_BINDING |
			GPUTextureUsage.RENDER_ATTACHMENT |
			GPUTextureUsage.STORAGE_BINDING |
			GPUTextureUsage.COPY_SRC |
			GPUTextureUsage.COPY_DST
	});

	const oklab_texture_ping = device.createTexture({
		label: 'input oklab texture',
		size: [imageBitMap.width, imageBitMap.height],
		format: 'rgba16float',
		usage:
			GPUTextureUsage.TEXTURE_BINDING |
			GPUTextureUsage.RENDER_ATTACHMENT |
			GPUTextureUsage.STORAGE_BINDING |
			GPUTextureUsage.COPY_SRC |
			GPUTextureUsage.COPY_DST
	});

	const oklab_texture_pong = device.createTexture({
		label: 'output oklab texture',
		size: [imageBitMap.width, imageBitMap.height],
		format: 'rgba16float',
		usage:
			GPUTextureUsage.TEXTURE_BINDING |
			GPUTextureUsage.RENDER_ATTACHMENT |
			GPUTextureUsage.STORAGE_BINDING |
			GPUTextureUsage.COPY_SRC |
			GPUTextureUsage.COPY_DST
	});

	const gaussian_blur_intermediate_texture = device.createTexture({
		label: 'gaussian blur intermediate texture',
		size: [imageBitMap.width, imageBitMap.height],
		format: 'rgba16float',
		usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT
	});

	/*
	gradient texture:
		x -> gradient magnitude
		y -> theta              (in the direction of the gradient)
		z -> 0                  (unused)
		w -> 0                  (unused)
	*/
	const gradient_texture = device.createTexture({
		label: 'gaussian gradient output texture',
		size: [imageBitMap.width, imageBitMap.height],
		format: 'rgba16float',
		usage:
			GPUTextureUsage.TEXTURE_BINDING |
			GPUTextureUsage.STORAGE_BINDING |
			GPUTextureUsage.RENDER_ATTACHMENT |
			GPUTextureUsage.COPY_SRC
	});

	/*
	edge textures:
		x -> edge flag       (whether this pixel is part of an edge)
		y -> subpixel_offset (in the direction of the gradient)
		z -> 0               (unused)
		w -> 0               (unused)
	*/
	const edge_texture_ping = device.createTexture({
		label: 'edge tracing texture ping',
		size: [imageBitMap.width, imageBitMap.height],
		format: 'rgba16float',
		usage:
			GPUTextureUsage.TEXTURE_BINDING |
			GPUTextureUsage.STORAGE_BINDING |
			GPUTextureUsage.COPY_SRC |
			GPUTextureUsage.COPY_DST |
			GPUTextureUsage.RENDER_ATTACHMENT
	});

	const edge_texture_pong = device.createTexture({
		label: 'edge tracing texture pong',
		size: [imageBitMap.width, imageBitMap.height],
		format: 'rgba16float',
		usage:
			GPUTextureUsage.TEXTURE_BINDING |
			GPUTextureUsage.STORAGE_BINDING |
			GPUTextureUsage.COPY_SRC |
			GPUTextureUsage.COPY_DST |
			GPUTextureUsage.RENDER_ATTACHMENT
	});

	const gradient_max_sampler = device.createSampler({
		label: 'gradient max sampler',
		addressModeU: 'clamp-to-edge',
		addressModeV: 'clamp-to-edge',
		magFilter: 'linear',
		minFilter: 'linear',
		mipmapFilter: 'nearest'
	});

	// --- Shared Buffers ---
	const density_scores_buffer = device.createBuffer({
		label: 'density scores buffer',
		size: imageBitMap.width * imageBitMap.height * 4,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
	});
	const density_scores_texture = device.createTexture({
		label: 'density scores texture',
		size: [imageBitMap.width, imageBitMap.height],
		format: 'rgba16float',
		usage:
			GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC
	});

	// --- sRGB to OkLab Pass ---
	const srgb_to_oklab_module = device!.createShaderModule({
		label: 'srgb to oklab module',
		code: srgb_to_oklab_shader
	});
	const srgb_to_oklab_pipeline = device!.createRenderPipeline({
		label: 'srgb to oklab render pipeline',
		layout: 'auto',
		vertex: {
			entryPoint: 'vs_main',
			module: srgb_to_oklab_module
		},
		fragment: {
			entryPoint: 'fs_main',
			module: srgb_to_oklab_module,
			targets: [
				{
					format: 'rgba16float'
				}
			]
		}
	});
	async function srgb_to_oklab_pass() {
		device!.queue.copyExternalImageToTexture(
			{ source: imageBitMap },
			{ texture: input_srgb_texture },
			{ width: imageBitMap.width, height: imageBitMap.height }
		);

		const bind_group = device!.createBindGroup({
			label: 'srgb to oklab bind group',
			layout: srgb_to_oklab_pipeline.getBindGroupLayout(0),
			entries: [{ binding: 0, resource: input_srgb_texture.createView() }]
		});

		const encoder = device!.createCommandEncoder({ label: 'srgb to oklab encoder' });
		const pass = encoder.beginRenderPass({
			label: 'srgb to oklab render pass',
			colorAttachments: [
				{
					view: oklab_texture_ping.createView(),
					clearValue: [0, 0, 0, 0],
					loadOp: 'clear',
					storeOp: 'store'
				}
			]
		});

		pass.setPipeline(srgb_to_oklab_pipeline);
		pass.setBindGroup(0, bind_group);
		pass.draw(3);
		pass.end();

		device!.queue.submit([encoder.finish()]);
		// await device!.queue.onSubmittedWorkDone();
	}

	// --- OkLab to sRGB Pass ---
	// const canvas_format = navigator.gpu.getPreferredCanvasFormat();
	const canvas_format: GPUTextureFormat = 'rgba8unorm';

	const oklab_to_srgb_module = device!.createShaderModule({
		label: 'oklab to srgb module',
		code: oklab_to_srgb_shader
	});
	const oklab_to_srgb_pipeline = device!.createRenderPipeline({
		label: 'oklab to srgb render pipeline',
		layout: 'auto',
		vertex: {
			entryPoint: 'vs_main',
			module: oklab_to_srgb_module
		},
		fragment: {
			entryPoint: 'fs_main',
			module: oklab_to_srgb_module,
			targets: [
				{
					format: canvas_format
				}
			]
		}
	});
	async function oklab_to_srgb_pass(
		texture: GPUTexture,
		show_edge_pixels: boolean,
		context: GPUCanvasContext
	) {
		const debug_uniforms_data = new Uint32Array([show_edge_pixels ? 1 : 0]);
		const debug_uniforms_buffer = device!.createBuffer({
			label: 'oklab to srgb debug uniforms buffer',
			size: debug_uniforms_data.byteLength,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});
		device!.queue.writeBuffer(debug_uniforms_buffer, 0, debug_uniforms_data);

		const bind_group = device!.createBindGroup({
			label: 'oklab to srgb ping bind group',
			layout: oklab_to_srgb_pipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: texture.createView() },
				{ binding: 1, resource: { buffer: debug_uniforms_buffer } }
			]
		});

		context.configure({
			device: device!,
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

		const encoder = device!.createCommandEncoder({ label: 'oklab to srgb encoder' });

		const pass = encoder.beginRenderPass(renderPassDescriptor);

		pass.setPipeline(oklab_to_srgb_pipeline);
		pass.setBindGroup(0, bind_group);
		pass.draw(3);
		pass.end();

		device!.queue.submit([encoder.finish()]);
	}

	// --- Density Scores Pass ---
	const update_density_scores_pipeline = device.createComputePipeline({
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

	const in_partial_sums_buffer = device!.createBuffer({
		label: 'mean density partial sums buffer',
		size: density_scores_buffer.size, // same size as density scores buffer
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
	});

	const out_partial_sums_buffer = device!.createBuffer({
		label: 'mean density partial sums buffer',
		size: density_scores_buffer.size, // same size as density scores buffer
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
	});

	const mean_density_score_buffer = device!.createBuffer({
		label: 'mean density score buffer',
		size: Float32Array.BYTES_PER_ELEMENT,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
	});
	async function density_scores_pass(texture: GPUTexture) {
		/*
		struct Uniforms {
			base_bandwidth: f32,
		}
		*/
		const float_uniforms_data = new Float32Array([base_bandwidth]);
		/*
		struct UintUniforms {
			tile_x: u32,    /// the low x value of the current tile (basically the x-offset for this shader pass)
			tile_y: u32,    /// the low y value of the current tile (basically the y-offset for this shader pass)
			tile_size: u32, /// the size of each tile (the range of x and y for this shader pass)
		}
		*/
		const uint_uniforms_data = new Uint32Array([0, 0, tile_size]);

		const float_uniforms_buffer = device!.createBuffer({
			label: 'float uniforms buffer',
			size: float_uniforms_data.byteLength,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});

		const uint_uniforms_buffer = device!.createBuffer({
			label: 'uint uniforms buffer',
			size: uint_uniforms_data.byteLength,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});

		// Setup a bindGroup to tell the shader which
		// buffer to use for the computation
		const bind_group = device!.createBindGroup({
			label: 'update density scores bind group',
			layout: update_density_scores_pipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: float_uniforms_buffer },
				{ binding: 1, resource: uint_uniforms_buffer },
				{ binding: 2, resource: texture.createView() },
				{ binding: 3, resource: density_scores_buffer },
				{ binding: 4, resource: density_scores_texture.createView() }
			]
		});

		device!.queue.writeBuffer(float_uniforms_buffer, 0, float_uniforms_data);

		for (let tile_x = 0; tile_x < imageBitMap.width; tile_x += tile_size) {
			for (let tile_y = 0; tile_y < imageBitMap.height; tile_y += tile_size) {
				// update the uint uniforms buffer for this pass
				uint_uniforms_data[0] = tile_x;
				uint_uniforms_data[1] = tile_y;
				device!.queue.writeBuffer(uint_uniforms_buffer, 0, uint_uniforms_data);

				// Encode commands to do the computation
				const encoder = device!.createCommandEncoder({
					label: 'update density scores encoder'
				});
				const pass = encoder.beginComputePass({
					label: 'update density scores compute pass'
				});
				pass.setPipeline(update_density_scores_pipeline);
				pass.setBindGroup(0, bind_group);
				pass.dispatchWorkgroups(
					Math.ceil(tile_size / 16), // divide by 16 to match shader workgroup size
					Math.ceil(tile_size / 16) // divide by 16 to match shader workgroup size
				);
				pass.end();

				device!.queue.submit([encoder.finish()]);

				// cooperative pacing: prevents long uninterrupted GPU queue bursts
				// await device!.queue.onSubmittedWorkDone();
				// await new Promise((r) => setTimeout(r, 0));
			}
		}
	}

	// --- Mean Density Score Steps Pass ---
	const mean_density_score_pipeline = device.createComputePipeline({
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
	async function get_mean_density_score(): Promise<GPUBuffer> {
		/*
		struct Uniforms {
		    partial_sum_size: u32,         /// each thread will be in charge of summing this many elements
		    num_remaining_elements: u32,   /// how many elements exist in `in_partial_sums`
		}
		*/
		const uniforms_data = new Uint32Array([
			partial_sum_size,
			imageBitMap.width * imageBitMap.height
		]);

		const uniforms_buffer = device!.createBuffer({
			label: 'mean density uniforms buffer',
			size: uniforms_data.byteLength,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});

		// Setup a bindGroup to tell the shader which
		// buffer to use for the computation
		const bind_group = device!.createBindGroup({
			label: 'mean density bind group',
			layout: mean_density_score_pipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: uniforms_buffer },
				{ binding: 1, resource: in_partial_sums_buffer },
				{ binding: 2, resource: out_partial_sums_buffer }
			]
		});
		device!.queue.writeBuffer(uniforms_buffer, 0, uniforms_data);

		// initialize the in_partial_sums_buffer with the density scores
		const init_encoder = device!.createCommandEncoder({
			label: 'mean density init encoder'
		});
		init_encoder.copyBufferToBuffer(
			density_scores_buffer,
			0,
			in_partial_sums_buffer,
			0,
			density_scores_buffer.size
		);
		device!.queue.submit([init_encoder.finish()]);

		// TODO: this should be broken up into multiple passes. its a lot less important than the other shaders, but at larger image sizes this can still cause hitching
		let num_remaining_elements = imageBitMap.width * imageBitMap.height;
		while (num_remaining_elements > 1) {
			// Update the total_elements in the uniforms buffer
			uniforms_data[1] = num_remaining_elements;
			device!.queue.writeBuffer(uniforms_buffer, 0, uniforms_data);

			// update total elements (the pass hasn't happened yet, but i need this value for dispatching. so it's calculated early)
			const next_num_remaining_elements = Math.ceil(num_remaining_elements / partial_sum_size);

			const encoder = device!.createCommandEncoder({
				label: 'mean density encoder'
			});

			const pass = encoder.beginComputePass({
				label: 'mean density compute pass'
			});
			pass.setPipeline(mean_density_score_pipeline);
			pass.setBindGroup(0, bind_group);
			pass.dispatchWorkgroups(
				Math.ceil(next_num_remaining_elements / 256) // divide by 256 to match shader workgroup size
			);
			pass.end();

			if (next_num_remaining_elements > 1) {
				// TODO: does this actually work? does this properly copy to the buffer so that the shader has the new data?
				// set up the output of this pass to be the input to the next pass
				encoder.copyBufferToBuffer(
					out_partial_sums_buffer,
					0,
					in_partial_sums_buffer,
					0,
					next_num_remaining_elements * Float32Array.BYTES_PER_ELEMENT
				);
			}

			device!.queue.submit([encoder.finish()]);
			num_remaining_elements = next_num_remaining_elements;
		}

		// copy reduced result (sum of all density scores) into a dedicated 1-float buffer
		const output_encoder = device!.createCommandEncoder({
			label: 'mean density output encoder'
		});
		output_encoder.copyBufferToBuffer(
			out_partial_sums_buffer,
			0,
			mean_density_score_buffer,
			0,
			Float32Array.BYTES_PER_ELEMENT
		);
		device!.queue.submit([output_encoder.finish()]);

		return mean_density_score_buffer;
	}

	// --- Mean Shift Cluster Pass ---
	const mean_shift_cluster_pipeline = device.createComputePipeline({
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
	async function mean_shift_cluster_pass(
		mean_density_score_buffer: GPUBuffer,
		texture_in: GPUTexture,
		texture_out: GPUTexture
	): Promise<void> {
		/*
		struct FloatUniforms {
			base_bandwidth: f32,
		}
		*/
		const float_uniforms_data = new Float32Array([base_bandwidth]);
		/*
		struct UintUniforms {
			tile_x: u32,    /// the low x value of the current tile (basically the x-offset for this shader pass)
			tile_y: u32,    /// the low y value of the current tile (basically the y-offset for this shader pass)
			tile_size: u32, /// the size of each tile (the range of x and y for this shader pass)
		}
		*/
		const uint_uniforms_data = new Uint32Array([0, 0, tile_size]);

		const float_uniforms_buffer = device!.createBuffer({
			label: 'mean shift cluster float uniforms buffer',
			size: float_uniforms_data.byteLength,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});

		const uint_uniforms_buffer = device!.createBuffer({
			label: 'mean shift cluster uint uniforms buffer',
			size: uint_uniforms_data.byteLength,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});

		// Setup a bindGroup to tell the shader which
		// buffer to use for the computation
		const bind_group = device!.createBindGroup({
			label: 'mean shift cluster ping bind group',
			layout: mean_shift_cluster_pipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: float_uniforms_buffer },
				{ binding: 1, resource: mean_density_score_buffer },
				{ binding: 2, resource: uint_uniforms_buffer },
				{ binding: 3, resource: texture_in.createView() },
				{ binding: 4, resource: density_scores_buffer },
				{ binding: 5, resource: texture_out.createView() }
			]
		});

		// Copy our input data to input buffers
		device!.queue.writeBuffer(float_uniforms_buffer, 0, float_uniforms_data);

		for (let tile_x = 0; tile_x < imageBitMap.width; tile_x += tile_size) {
			for (let tile_y = 0; tile_y < imageBitMap.height; tile_y += tile_size) {
				// update the uint uniforms buffer for this pass
				uint_uniforms_data[0] = tile_x;
				uint_uniforms_data[1] = tile_y;
				device!.queue.writeBuffer(uint_uniforms_buffer, 0, uint_uniforms_data);

				// Encode commands to do the computation
				const encoder = device!.createCommandEncoder({
					label: 'mean shift cluster encoder'
				});
				const pass = encoder.beginComputePass({
					label: 'mean shift cluster compute pass'
				});
				pass.setPipeline(mean_shift_cluster_pipeline);
				pass.setBindGroup(0, bind_group);
				pass.dispatchWorkgroups(
					Math.ceil(tile_size / 16), // divide by 16 to match shader workgroup size
					Math.ceil(tile_size / 16) // divide by 16 to match shader workgroup size
				);
				pass.end();

				device!.queue.submit([encoder.finish()]);

				// cooperative pacing: prevents long uninterrupted GPU queue bursts
				// await device!.queue.onSubmittedWorkDone();
				// await new Promise((r) => setTimeout(r, 0));
			}
		}
	}

	// --- Gaussian Blur Pass ---
	const gaussian_blur_module = device.createShaderModule({
		label: 'gaussian blur module',
		code: gaussian_blur_shader
	});
	const gaussian_blur_h_pipeline = device.createRenderPipeline({
		label: 'gaussian blur horizontal pipeline',
		layout: 'auto',
		vertex: {
			entryPoint: 'vs_main',
			module: gaussian_blur_module
		},
		fragment: {
			entryPoint: 'blur_horizontal',
			module: gaussian_blur_module,
			targets: [
				{
					format: 'rgba16float'
				}
			]
		}
	});
	const gaussian_blur_v_pipeline = device.createRenderPipeline({
		label: 'gaussian blur vertical pipeline',
		layout: 'auto',
		vertex: {
			entryPoint: 'vs_main',
			module: gaussian_blur_module
		},
		fragment: {
			entryPoint: 'blur_vertical',
			module: gaussian_blur_module,
			targets: [
				{
					format: 'rgba16float'
				}
			]
		}
	});

	async function gaussian_blur_pass(
		radius: number,
		input_texture: GPUTexture,
		output_texture: GPUTexture
	): Promise<void> {
		const uniforms_data = new Uint32Array([radius]);
		const uniforms_buffer = device!.createBuffer({
			label: 'gaussian blur uniforms buffer',
			size: uniforms_data.byteLength,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});
		device!.queue.writeBuffer(uniforms_buffer, 0, uniforms_data);

		const kernel_weights = compute_gaussian_kernel(radius);
		const kernel_buffer = device!.createBuffer({
			label: 'gaussian blur kernel weights buffer',
			size: kernel_weights.byteLength,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});
		device!.queue.writeBuffer(kernel_buffer, 0, kernel_weights);

		const h_bind_group = device!.createBindGroup({
			label: 'gaussian blur horizontal bind group',
			layout: gaussian_blur_h_pipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: uniforms_buffer } },
				{ binding: 1, resource: input_texture.createView() },
				{ binding: 2, resource: { buffer: kernel_buffer } }
			]
		});

		const v_bind_group = device!.createBindGroup({
			label: 'gaussian blur vertical bind group',
			layout: gaussian_blur_v_pipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: uniforms_buffer } },
				{ binding: 1, resource: gaussian_blur_intermediate_texture.createView() },
				{ binding: 2, resource: { buffer: kernel_buffer } }
			]
		});

		const h_encoder = device!.createCommandEncoder({ label: 'gaussian blur horizontal encoder' });
		const h_pass = h_encoder.beginRenderPass({
			label: 'gaussian blur horizontal render pass',
			colorAttachments: [
				{
					view: gaussian_blur_intermediate_texture.createView(),
					clearValue: [0, 0, 0, 0],
					loadOp: 'clear',
					storeOp: 'store'
				}
			]
		});
		h_pass.setPipeline(gaussian_blur_h_pipeline);
		h_pass.setBindGroup(0, h_bind_group);
		h_pass.draw(3);
		h_pass.end();
		device!.queue.submit([h_encoder.finish()]);
		// await device!.queue.onSubmittedWorkDone();

		const v_encoder = device!.createCommandEncoder({ label: 'gaussian blur vertical encoder' });
		const v_pass = v_encoder.beginRenderPass({
			label: 'gaussian blur vertical render pass',
			colorAttachments: [
				{
					view: output_texture.createView(),
					clearValue: [0, 0, 0, 0],
					loadOp: 'clear',
					storeOp: 'store'
				}
			]
		});
		v_pass.setPipeline(gaussian_blur_v_pipeline);
		v_pass.setBindGroup(0, v_bind_group);
		v_pass.draw(3);
		v_pass.end();
		device!.queue.submit([v_encoder.finish()]);
		// await device!.queue.onSubmittedWorkDone();
	}

	// --- Gradient Pass ---
	const gaussian_gradient_module = device.createShaderModule({
		label: 'gaussian gradient module',
		code: gaussian_gradient_shader
	});
	const gaussian_grad_pipeline = device.createRenderPipeline({
		label: 'difference of gaussian pipeline',
		layout: 'auto',
		vertex: {
			entryPoint: 'vs_main',
			module: gaussian_gradient_module
		},
		fragment: {
			entryPoint: 'cs_main',
			module: gaussian_gradient_module,
			targets: [
				{
					format: 'rgba16float'
				}
			]
		}
	});

	async function gaussian_gradient_pass(
		in_texture: GPUTexture,
		out_texture: GPUTexture
	): Promise<void> {
		const bind_group = device!.createBindGroup({
			label: 'gaussian gradient bind group',
			layout: gaussian_grad_pipeline.getBindGroupLayout(0),
			entries: [{ binding: 0, resource: in_texture.createView() }]
		});

		const encoder = device!.createCommandEncoder({ label: 'gaussian gradient encoder' });
		const pass = encoder.beginRenderPass({
			label: 'gaussian gradient render pass',
			colorAttachments: [
				{
					view: out_texture.createView(),
					clearValue: [0, 0, 0, 0],
					loadOp: 'clear',
					storeOp: 'store'
				}
			]
		});
		pass.setPipeline(gaussian_grad_pipeline);
		pass.setBindGroup(0, bind_group);
		pass.draw(3);
		pass.end();

		device!.queue.submit([encoder.finish()]);
		// await device!.queue.onSubmittedWorkDone();
	}

	// --- Gradient Maximizing Pass ---
	const gradient_max_module = device.createShaderModule({
		label: 'gradient max module',
		code: gradient_max_shader
	});
	const grad_max_pipeline = device.createRenderPipeline({
		label: 'gradient max pipeline',
		layout: 'auto',
		vertex: {
			entryPoint: 'vs_main',
			module: gradient_max_module
		},
		fragment: {
			entryPoint: 'cs_main',
			module: gradient_max_module,
			targets: [
				{
					format: 'rgba16float'
				}
			]
		}
	});

	async function gradient_max_pass(
		in_gradient_texture: GPUTexture,
		output_texture: GPUTexture
	): Promise<void> {
		const bind_group = device!.createBindGroup({
			label: 'gradient max bind group',
			layout: grad_max_pipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: in_gradient_texture.createView() },
				{ binding: 1, resource: gradient_max_sampler }
			]
		});

		const encoder = device!.createCommandEncoder({ label: 'gradient max encoder' });
		const pass = encoder.beginRenderPass({
			label: 'gradient max render pass',
			colorAttachments: [
				{
					view: output_texture.createView(),
					clearValue: [0, 0, 0, 0],
					loadOp: 'clear',
					storeOp: 'store'
				}
			]
		});
		pass.setPipeline(grad_max_pipeline);
		pass.setBindGroup(0, bind_group);
		pass.draw(3);
		pass.end();

		device!.queue.submit([encoder.finish()]);
		// await device!.queue.onSubmittedWorkDone();
	}

	// --- Edge Trace Pass ---
	const edge_trace_module = device.createShaderModule({
		label: 'edge tracing module',
		code: edge_tracing_step_shader
	});
	const edge_trace_pipeline = device.createComputePipeline({
		label: 'edge tracing compute pipeline',
		layout: 'auto',
		compute: {
			module: edge_trace_module,
			entryPoint: 'cs_main'
		}
	});

	async function edge_trace_pass(
		in_edge_texture: GPUTexture,
		out_edge_texture: GPUTexture
	): Promise<void> {
		const bind_group = device!.createBindGroup({
			label: 'edge tracing bind group',
			layout: edge_trace_pipeline.getBindGroupLayout(0),
			entries: [
				// TODO: might want to use a local gradient_texture instead of global
				{ binding: 0, resource: gradient_texture.createView() },
				{ binding: 1, resource: in_edge_texture.createView() },
				{ binding: 2, resource: out_edge_texture.createView() }
			]
		});

		const encoder = device!.createCommandEncoder({ label: 'edge tracing encoder' });

		encoder.copyTextureToTexture(
			{
				texture: in_edge_texture
			},
			{
				texture: out_edge_texture
			},
			{
				width: imageBitMap.width,
				height: imageBitMap.height,
				depthOrArrayLayers: 1
			}
		);

		const pass = encoder.beginComputePass({
			label: 'edge tracing compute pass'
		});
		pass.setPipeline(edge_trace_pipeline);
		pass.setBindGroup(0, bind_group);
		pass.dispatchWorkgroups(Math.ceil(imageBitMap.width / 16), Math.ceil(imageBitMap.height / 16));
		pass.end();

		device!.queue.submit([encoder.finish()]);
		// await device!.queue.onSubmittedWorkDone();
	}

	let startTime = performance.now();
	console.log();
	console.log('Srgb -> OkLab:');
	await srgb_to_oklab_pass();
	let endTime = performance.now();
	console.log(`execution time: ${(endTime - startTime).toFixed(2)}ms`);

	startTime = performance.now();
	console.log();
	console.log('Pre-cluster Blur:');
	await gaussian_blur_pass(blur_radius, oklab_texture_ping, oklab_texture_pong);
	endTime = performance.now();
	console.log(`execution time: ${(endTime - startTime).toFixed(2)}ms`);

	for (let pass_index = 0; pass_index < num_cluster_passes; pass_index++) {
		startTime = performance.now();

		console.log();
		console.log('Pass:', pass_index);
		await density_scores_pass((pass_index + 1) % 2 == 1 ? oklab_texture_pong : oklab_texture_ping);

		const mean_density_score_buffer = await get_mean_density_score();
		await mean_shift_cluster_pass(
			mean_density_score_buffer,
			(pass_index + 1) % 2 == 1 ? oklab_texture_pong : oklab_texture_ping,
			(pass_index + 1) % 2 == 1 ? oklab_texture_ping : oklab_texture_pong
		);

		endTime = performance.now();
		console.log(`execution time: ${(endTime - startTime).toFixed(2)}ms`);
	}

	startTime = performance.now();
	console.log();
	console.log('GaussGradient:');
	await gaussian_gradient_pass(
		num_cluster_passes % 2 === 0 ? oklab_texture_ping : oklab_texture_pong,
		gradient_texture
	);
	endTime = performance.now();
	console.log(`execution time: ${(endTime - startTime).toFixed(2)}ms`);

	startTime = performance.now();
	console.log();
	console.log('Gradient Max:');
	await gradient_max_pass(gradient_texture, edge_texture_ping);
	endTime = performance.now();
	console.log(`execution time: ${(endTime - startTime).toFixed(2)}ms`);

	// TODO: either switch `final_edge_texture` to not exist (do ping pong like other passes do), or change other passes to use this sort of structure
	let final_edge_texture = edge_texture_ping;
	for (
		let edge_trace_pass_index = 0;
		edge_trace_pass_index < num_edge_trace_passes;
		edge_trace_pass_index++
	) {
		startTime = performance.now();
		console.log();
		console.log('Edge Trace Pass:', edge_trace_pass_index);

		const output_texture = edge_trace_pass_index % 2 === 0 ? edge_texture_pong : edge_texture_ping;

		await edge_trace_pass(final_edge_texture, output_texture);
		final_edge_texture = output_texture;

		endTime = performance.now();
		console.log(`execution time: ${(endTime - startTime).toFixed(2)}ms`);
	}

	// TODO: the svg implementation below is not final. it's just for testing.
	startTime = performance.now();
	console.log();
	console.log('Edge SVG:');
	const svg = await textureToEdgeSvg(
		device,
		gradient_texture,
		final_edge_texture,
		imageBitMap.width,
		imageBitMap.height
	);
	endTime = performance.now();
	console.log(`execution time: ${(endTime - startTime).toFixed(2)}ms`);

	await oklab_to_srgb_pass(
		num_cluster_passes % 2 === 0 ? oklab_texture_ping : oklab_texture_pong,
		false,
		clusterCanvas
	);

	await oklab_to_srgb_pass(final_edge_texture, false, edgeCanvas);

	return [true, svg];
}
