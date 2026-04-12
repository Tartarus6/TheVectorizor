import { get_median, type Oklab } from './utils';
import mean_shift_cluster_step_shader from '$lib/shaders/mean_shift_cluster_step.wgsl?raw';
import update_density_scores_shader from '$lib/shaders/update_density_scores.wgsl?raw';
import mean_density_score_pass_shader from '$lib/shaders/mean_density_score_pass.wgsl?raw';

// TODO: move this const somewhere better
// TODO: figure out what a good value for this const is
/// each thread in the mean density score passes will be in charge of summing this many elements
const partial_sum_size: number = 8;

// TODO: add image locality weighting to cluster. make closer pixels count more toward the final color
// TODO: switch run_shader to actually be multiple passes. it should loop passes until the image isn't changing anymore (or is changing below some threshold)
// TODO: add pass to convert image colors into Oklab, and another to convert it back into rgb
// TODO: keep density scores as a buffer, don't turn back into number[]. this means that we'll need to somehow switch how we calculate the median density score
// TODO: (maybe) move setup for device, adapter, buffers, etc. into a separate function, just to clean up the main run_shader() function and improve its readability
// TODO: (maybe) remove all or some of the readback buffers. are they needed/used?
// TODO: (maybe) make a global const for workgroup sizing (wont sync with shader files, just good to not have multiple possible points of failure)
// TODO: handling for transparent pixels: fully transparent pixels should be completely ignored (so the mean density score will have to divide by the number of non-transparent pixels rather than the width * height of the image)
// TODO: add checks for if device and adapter are defined in each subfunction, to prevent the need to repeat `device!` every time
/// returns whether the colors changed (used to know whether to increase count)
export async function run_shader(
	imageBitMap: ImageBitmap,
	colors: Oklab[],
	base_bandwidth: number,
	cluster_check_radius: number,
	tile_size: number
): Promise<[boolean, Uint8ClampedArray]> {
	console.log('starting mean shift cluster step WGPU');
	const adapter = await navigator.gpu?.requestAdapter();
	const device = await adapter?.requestDevice();

	if (!device) {
		// TODO: add an actual warning for this on the site, popup or whatever
		alert('need a browser that supports WebGPU');
		return [false, new Uint8ClampedArray()];
	}

	// --- Pipelines ---
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

	const mean_density_score_pipeline = device.createComputePipeline({
		label: 'mean density score compute pipeline',
		layout: 'auto',
		compute: {
			module: device.createShaderModule({
				label: 'mean density score module',
				code: mean_density_score_pass_shader
			}),
			entryPoint: 'cs_main'
		}
	});

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

	// --- Shared Textures ---
	const input_color_texture = device.createTexture({
		label: 'input color texture',
		size: [imageBitMap.width, imageBitMap.height],
		format: 'rgba8unorm',
		usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
	});

	const output_colors_texture = device.createTexture({
		label: 'output colors texture',
		size: [imageBitMap.width, imageBitMap.height],
		format: 'rgba8unorm',
		usage: GPUTextureUsage.COPY_SRC | GPUTextureUsage.STORAGE_BINDING
	});

	device.queue.copyExternalImageToTexture(
		{ source: imageBitMap },
		{ texture: input_color_texture },
		{ width: imageBitMap.width, height: imageBitMap.height }
	);

	// --- Shared Buffers ---
	const density_scores_buffer = device.createBuffer({
		label: 'density scores buffer',
		size: imageBitMap.width * imageBitMap.height * 4,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
	});

	// --- Density Scores Pass ---
	async function density_scores_pass() {
		// struct Uniforms {
		// 	  base_bandwidth: f32,
		// }
		const float_uniforms_data = new Float32Array([base_bandwidth, cluster_check_radius]);

		// struct UintUniforms {
		// 	  cluster_check_radius: u32, /// how many a square of double this size, in the texture, around the pixel is the are checked for creating the cluster
		// 	  tile_x: u32,    /// the low x value of the current tile (basically the x-offset for this shader pass)
		// 	  tile_y: u32,    /// the low y value of the current tile (basically the y-offset for this shader pass)
		// 	  tile_size: u32, /// the size of each tile (the range of x and y for this shader pass)
		// }
		const uint_uniforms_data = new Uint32Array([cluster_check_radius, 0, 0, tile_size]);

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
				{ binding: 2, resource: input_color_texture.createView() },
				{ binding: 3, resource: density_scores_buffer }
			]
		});
		device!.queue.writeBuffer(float_uniforms_buffer, 0, float_uniforms_data);

		for (let tile_x = 0; tile_x < imageBitMap.width; tile_x += tile_size) {
			for (let tile_y = 0; tile_y < imageBitMap.height; tile_y += tile_size) {
				// update the uint uniforms buffer for this pass
				uint_uniforms_data[1] = tile_x;
				uint_uniforms_data[2] = tile_y;
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
				await device!.queue.onSubmittedWorkDone();
				await new Promise((r) => setTimeout(r, 0));
			}
		}
	}

	// --- Mean Density Score Passes ---
	async function get_mean_density_score(): Promise<number> {
		// struct Uniforms {
		//     partial_sum_size: u32,         /// each thread will be in charge of summing this many elements
		//     num_remaining_elements: u32,   /// how many elements exist in `in_partial_sums`
		// }
		var uniforms_data = new Uint32Array([partial_sum_size, imageBitMap.width * imageBitMap.height]);

		const uniforms_buffer = device!.createBuffer({
			label: 'median density uniforms buffer',
			size: uniforms_data.byteLength,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
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

		// this buffer is not used by the shader directly. when the shader finishes, its output is copied into this buffer so that it can be better used
		const readback_buffer = device!.createBuffer({
			label: 'median density readback buffer',
			size: density_scores_buffer.size,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
		});

		// Setup a bindGroup to tell the shader which
		// buffer to use for the computation
		const bind_group = device!.createBindGroup({
			label: 'median density bind group',
			layout: mean_density_score_pipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: uniforms_buffer },
				{ binding: 1, resource: in_partial_sums_buffer },
				{ binding: 2, resource: out_partial_sums_buffer }
			]
		});
		device!.queue.writeBuffer(uniforms_buffer, 0, uniforms_data);

		// Encode commands to do the computation
		const encoder = device!.createCommandEncoder({
			label: 'median density encoder'
		});

		// initialize the in_partial_sums_buffer with the density scores
		encoder.copyBufferToBuffer(density_scores_buffer, 0, in_partial_sums_buffer, 0);

		// TODO: this should be broken up into multiple passes. its a lot less important than the other shaders, but at larger image sizes this can still cause hitching
		let num_remaining_elements = imageBitMap.width * imageBitMap.height;
		while (num_remaining_elements > 1) {
			// Update the total_elements in the uniforms buffer
			uniforms_data[1] = num_remaining_elements;
			device!.queue.writeBuffer(uniforms_buffer, 0, uniforms_data);

			// update total elements (the pass hasn't happened yet, but i need this value for dispatching. so it's calculated early)
			num_remaining_elements = Math.ceil(num_remaining_elements / partial_sum_size);

			const pass = encoder.beginComputePass({
				label: 'median density compute pass'
			});
			pass.setPipeline(mean_density_score_pipeline);
			pass.setBindGroup(0, bind_group);
			pass.dispatchWorkgroups(
				Math.ceil(num_remaining_elements / 256) // divide by 256 to match shader workgroup size
			);
			pass.end();

			if (num_remaining_elements > 1) {
				// TODO: does this actually work? does this properly copy to the buffer so that the shader has the new data?
				// set up the output of this pass to be the input to the next pass
				encoder.copyBufferToBuffer(out_partial_sums_buffer, 0, in_partial_sums_buffer, 0);
			}
		}

		encoder.copyBufferToBuffer(out_partial_sums_buffer, 0, readback_buffer, 0);

		// Finish encoding and submit the commands
		const command_buffer = encoder.finish();
		device!.queue.submit([command_buffer]);
		await readback_buffer.mapAsync(GPUMapMode.READ);
		const result = new Float32Array(readback_buffer.getMappedRange().slice());

		const median_density_score = result[0] / (imageBitMap.width * imageBitMap.height);

		readback_buffer.unmap();

		console.log();
		console.log('WGPU Mean Density Score');
		console.log(median_density_score);

		return median_density_score;
	}

	// --- Mean Shift Cluster Pass ---
	async function mean_shift_cluster_pass(median_density_score: number): Promise<Uint8ClampedArray> {
		// struct FloatUniforms {
		// 	  base_bandwidth: f32,
		// 	  median_density_score: f32,
		// }
		const float_uniforms_data = new Float32Array([base_bandwidth, median_density_score]);

		// struct UintUniforms {
		// 	  cluster_check_radius: u32, /// how many a square of double this size, in the texture, around the pixel is the are checked for creating the cluster
		// 	  tile_x: u32,    /// the low x value of the current tile (basically the x-offset for this shader pass)
		// 	  tile_y: u32,    /// the low y value of the current tile (basically the y-offset for this shader pass)
		// 	  tile_size: u32, /// the size of each tile (the range of x and y for this shader pass)
		// }
		const uint_uniforms_data = new Uint32Array([cluster_check_radius, 0, 0, tile_size]);

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
			label: 'mean shift cluster bind group',
			layout: mean_shift_cluster_pipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: float_uniforms_buffer },
				{ binding: 1, resource: uint_uniforms_buffer },
				{ binding: 2, resource: input_color_texture.createView() },
				{ binding: 3, resource: density_scores_buffer },
				{ binding: 4, resource: output_colors_texture.createView() }
			]
		});

		// Copy our input data to input buffers
		device!.queue.writeBuffer(float_uniforms_buffer, 0, float_uniforms_data);

		for (let tile_x = 0; tile_x < imageBitMap.width; tile_x += tile_size) {
			for (let tile_y = 0; tile_y < imageBitMap.height; tile_y += tile_size) {
				// update the uint uniforms buffer for this pass
				uint_uniforms_data[1] = tile_x;
				uint_uniforms_data[2] = tile_y;
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
				await device!.queue.onSubmittedWorkDone();
				await new Promise((r) => setTimeout(r, 0));
			}
		}

		// add padding because reading from a texture to a buffer needs multiples of 256 ??? god I hate shaders
		const bytesPerRow = Math.ceil((imageBitMap.width * 4) / 256) * 256;

		const readback_buffer = device!.createBuffer({
			label: 'color readback buffer',
			size: bytesPerRow * imageBitMap.height,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
		});
		const copy_encoder = device!.createCommandEncoder({ label: 'cluster step copy encoder' });
		copy_encoder.copyTextureToBuffer(
			{ texture: output_colors_texture },
			{ buffer: readback_buffer, bytesPerRow },
			{ width: imageBitMap.width, height: imageBitMap.height }
		);

		device!.queue.submit([copy_encoder.finish()]);

		await readback_buffer.mapAsync(GPUMapMode.READ);
		const raw = new Uint8Array(readback_buffer.getMappedRange().slice());
		readback_buffer.unmap();

		// remove the padding
		const pixels = new Uint8ClampedArray(imageBitMap.width * imageBitMap.height * 4);
		for (let y = 0; y < imageBitMap.height; y++) {
			for (let x = 0; x < imageBitMap.width; x++) {
				const src = y * bytesPerRow + x * 4;
				const dst = y * imageBitMap.width * 4 + x * 4;
				pixels[dst] = raw[src];
				pixels[dst + 1] = raw[src + 1];
				pixels[dst + 2] = raw[src + 2];
				pixels[dst + 3] = raw[src + 3];
			}
		}
		return pixels;
	}

	await density_scores_pass();
	const median_density_score = await get_mean_density_score();
	return [true, await mean_shift_cluster_pass(median_density_score)];
}
