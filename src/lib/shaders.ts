import mean_shift_cluster_step_shader from '$lib/shaders/mean_shift_cluster_step.wgsl?raw';
import update_density_scores_shader from '$lib/shaders/update_density_scores.wgsl?raw';
import mean_density_score_pass_shader from '$lib/shaders/mean_density_score_pass.wgsl?raw';
import srgb_to_oklab_shader from '$lib/shaders/srgb_to_oklab.wgsl?raw';
import oklab_to_srgb_shader from '$lib/shaders/oklab_to_srgb.wgsl?raw';

// TODO: move this const somewhere better
// TODO: figure out what a good value for this const is
/// each thread in the mean density score passes will be in charge of summing this many elements
const partial_sum_size: number = 8;

// PERFORMANCE TODOS
// TODO: implement ping-pong textures, stop doing unnecessary texture copies
// TODO: automate performance balancing. start at a very low tile size and do some tests, increasing it until it's as big as it can be while meeting max acceptible execution time
// TODO: keep density scores as a buffer, don't turn back into number. this means that we'll need to somehow switch how we calculate the median density score
// TODO: figure out a good value for partial_sum_size
// TODO: (maybe) switch to holding density scores in a texture rather than a general array buffer
// TODO: (maybe) turn update_density_scores into a fragment shader
// TODO: (maybe) turn mean_shift_cluster_step into a fragment shader

// GENERAL TODOS
// TODO: figure out a name for the stages of the vectorizor (like "cleanup" for the mean shift cluster stuff, and "edge detection" for that, or whatever) and give more descriptive names to functons/files/variables
// TODO: deal with unused alpha. (need to figure what makes for a good function, and if alpha should be removed or not)
// TODO: handling for transparent pixels: fully transparent pixels should be completely ignored (so the mean density score will have to divide by the number of non-transparent pixels rather than the width * height of the image)
// TODO: add checks for if device and adapter are defined in each subfunction, to prevent the need to repeat `device!` every time
// TODO: (maybe) move setup for device, adapter, buffers, etc. into a separate function, just to clean up the main run_shader() function and improve its readability
// TODO: (maybe) remove all or some of the readback buffers. are they needed/used?
// TODO: (maybe) make a global const for workgroup sizing (wont sync with shader files, just good to not have multiple possible points of failure)
/// returns whether the colors changed (used to know whether to increase count)
export async function run_shader(
	imageBitMap: ImageBitmap,
	base_bandwidth: number,
	cluster_check_radius: number,
	tile_size: number,
	passes: number,
	alpha: number
): Promise<[boolean, Uint8ClampedArray]> {
	console.log('starting mean shift cluster step WGPU');
	console.log(alpha);
	const adapter = await navigator.gpu?.requestAdapter();
	const device = await adapter?.requestDevice();

	if (!device) {
		// TODO: add an actual warning for this on the site, popup or whatever
		alert('need a browser that supports WebGPU');
		return [false, new Uint8ClampedArray()];
	}

	// --- Shared Textures ---
	const input_srgb_texture = device.createTexture({
		label: 'input srgb texture',
		size: [imageBitMap.width, imageBitMap.height],
		format: 'rgba8unorm',
		usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
	});

	const input_oklab_texture = device.createTexture({
		label: 'input oklab texture',
		size: [imageBitMap.width, imageBitMap.height],
		format: 'rgba16float',
		usage:
			GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST
	});

	const output_oklab_texture = device.createTexture({
		label: 'output oklab texture',
		size: [imageBitMap.width, imageBitMap.height],
		format: 'rgba16float',
		usage:
			GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC
	});

	const output_srgb_texture = device.createTexture({
		label: 'output srgb texture',
		size: [imageBitMap.width, imageBitMap.height],
		format: 'rgba8unorm',
		usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC
	});

	// --- Shared Buffers ---
	const density_scores_buffer = device.createBuffer({
		label: 'density scores buffer',
		size: imageBitMap.width * imageBitMap.height * 4,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
	});

	// --- sRGB to OkLab Pass ---
	async function srgb_to_oklab_pass() {
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
					view: input_oklab_texture.createView(),
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
		await device!.queue.onSubmittedWorkDone();
	}

	// --- OkLab to sRGB Pass ---
	async function oklab_to_srgb_pass(): Promise<Uint8ClampedArray> {
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
						format: 'rgba8unorm'
					}
				]
			}
		});

		const bind_group = device!.createBindGroup({
			label: 'oklab to srgb bind group',
			layout: oklab_to_srgb_pipeline.getBindGroupLayout(0),
			entries: [{ binding: 0, resource: output_oklab_texture.createView() }]
		});

		const encoder = device!.createCommandEncoder({ label: 'oklab to srgb encoder' });
		const pass = encoder.beginRenderPass({
			label: 'oklab to srgb render pass',
			colorAttachments: [
				{
					view: output_srgb_texture.createView(),
					clearValue: [0, 0, 0, 0],
					loadOp: 'clear',
					storeOp: 'store'
				}
			]
		});

		pass.setPipeline(oklab_to_srgb_pipeline);
		pass.setBindGroup(0, bind_group);
		pass.draw(3);
		pass.end();

		const bytesPerRow = Math.ceil((imageBitMap.width * 4) / 256) * 256;
		const readback_buffer = device!.createBuffer({
			label: 'srgb readback buffer',
			size: bytesPerRow * imageBitMap.height,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
		});

		encoder.copyTextureToBuffer(
			{ texture: output_srgb_texture },
			{ buffer: readback_buffer, bytesPerRow },
			{ width: imageBitMap.width, height: imageBitMap.height }
		);

		device!.queue.submit([encoder.finish()]);

		await readback_buffer.mapAsync(GPUMapMode.READ);
		const raw = new Uint8Array(readback_buffer.getMappedRange().slice());
		readback_buffer.unmap();

		// remove row padding introduced by WebGPU alignment requirements
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

	// --- Density Scores Pass ---
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
	async function density_scores_pass() {
		// for performance monitoring
		const startTime = performance.now();

		const update_density_scores_pipeline = device!.createComputePipeline({
			label: 'update density scores compute pipeline',
			layout: 'auto',
			compute: {
				module: device!.createShaderModule({
					label: 'update density scores module',
					code: update_density_scores_shader
				}),
				entryPoint: 'cs_main'
			}
		});

		/*
		struct Uniforms {
			base_bandwidth: f32,
		}
		*/
		const float_uniforms_data = new Float32Array([base_bandwidth, cluster_check_radius]);
		/*
		struct UintUniforms {
			cluster_check_radius: u32, /// how many a square of double this size, in the texture, around the pixel is the are checked for creating the cluster
			tile_x: u32,    /// the low x value of the current tile (basically the x-offset for this shader pass)
			tile_y: u32,    /// the low y value of the current tile (basically the y-offset for this shader pass)
			tile_size: u32, /// the size of each tile (the range of x and y for this shader pass)
		}
		*/
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
				{ binding: 2, resource: input_oklab_texture.createView() },
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

		const endTime = performance.now();
		console.log(`density_scores_pass execution time: ${(endTime - startTime).toFixed(2)}ms`);
	}

	// --- Mean Density Score Passes ---
	async function get_mean_density_score(): Promise<number> {
		// for performance monitoring
		const startTime = performance.now();

		const mean_density_score_pipeline = device!.createComputePipeline({
			label: 'mean density score compute pipeline',
			layout: 'auto',
			compute: {
				module: device!.createShaderModule({
					label: 'mean density score module',
					code: mean_density_score_pass_shader
				}),
				entryPoint: 'cs_main'
			}
		});

		/*
		struct Uniforms {
		    partial_sum_size: u32,         /// each thread will be in charge of summing this many elements
		    num_remaining_elements: u32,   /// how many elements exist in `in_partial_sums`
		}
		*/
		var uniforms_data = new Uint32Array([partial_sum_size, imageBitMap.width * imageBitMap.height]);

		const uniforms_buffer = device!.createBuffer({
			label: 'mean density uniforms buffer',
			size: uniforms_data.byteLength,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});

		// this buffer is not used by the shader directly. when the shader finishes, its output is copied into this buffer so that it can be better used
		const readback_buffer = device!.createBuffer({
			label: 'mean density readback buffer',
			size: Float32Array.BYTES_PER_ELEMENT, // size of 1 element
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
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

		// Finish encoding and submit the commands
		const readback_encoder = device!.createCommandEncoder({
			label: 'mean density readback encoder'
		});
		readback_encoder.copyBufferToBuffer(
			out_partial_sums_buffer,
			0,
			readback_buffer,
			0,
			Float32Array.BYTES_PER_ELEMENT
		);
		device!.queue.submit([readback_encoder.finish()]);

		await readback_buffer.mapAsync(GPUMapMode.READ);
		const result = new Float32Array(readback_buffer.getMappedRange().slice());

		const mean_density_score = result[0] / (imageBitMap.width * imageBitMap.height);

		readback_buffer.unmap();

		const endTime = performance.now();
		console.log(`get_mean_density_score execution time: ${(endTime - startTime).toFixed(2)}ms`);

		return mean_density_score;
	}

	// --- Mean Shift Cluster Pass ---
	async function mean_shift_cluster_pass(mean_density_score: number): Promise<void> {
		// for performance monitoring
		const startTime = performance.now();

		const mean_shift_cluster_pipeline = device!.createComputePipeline({
			label: 'mean shift cluster compute pipeline',
			layout: 'auto',
			compute: {
				module: device!.createShaderModule({
					label: 'mean shift cluster module',
					code: mean_shift_cluster_step_shader
				}),
				entryPoint: 'cs_main'
			}
		});
		/*
		struct FloatUniforms {
			base_bandwidth: f32,
			mean_density_score: f32,
    		alpha: f32, /// controls how strongly the density matters
		}
		*/
		const float_uniforms_data = new Float32Array([base_bandwidth, mean_density_score, alpha]);
		/*
		struct UintUniforms {
			cluster_check_radius: u32, /// how many a square of double this size, in the texture, around the pixel is the are checked for creating the cluster
			tile_x: u32,    /// the low x value of the current tile (basically the x-offset for this shader pass)
			tile_y: u32,    /// the low y value of the current tile (basically the y-offset for this shader pass)
			tile_size: u32, /// the size of each tile (the range of x and y for this shader pass)
		}
		*/
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
				{ binding: 2, resource: input_oklab_texture.createView() },
				{ binding: 3, resource: density_scores_buffer },
				{ binding: 4, resource: output_oklab_texture.createView() }
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

		// TODO: switch to a ping-pong pattern to prevent unnecessary memory movement
		// Copy out texture back to in texture for use on next pass
		const texture_copy_encoder = device!.createCommandEncoder({
			label: 'mean shift cluster encoder'
		});
		texture_copy_encoder.copyTextureToTexture(
			{ texture: output_oklab_texture },
			{ texture: input_oklab_texture },
			{ width: imageBitMap.width, height: imageBitMap.height }
		);
		device!.queue.submit([texture_copy_encoder.finish()]);

		const endTime = performance.now();
		console.log(`mean_shift_cluster_pass execution time: ${(endTime - startTime).toFixed(2)}ms`);
	}

	// --- Calling the Code ---
	await srgb_to_oklab_pass();
	for (var i = 0; i < passes; i++) {
		console.log();
		console.log('Pass:', i);
		await density_scores_pass();
		const mean_density_score = await get_mean_density_score();
		await mean_shift_cluster_pass(mean_density_score);
	}
	return [true, await oklab_to_srgb_pass()];
}
