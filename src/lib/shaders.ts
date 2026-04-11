import { get_median, type Oklab } from './utils';
import gpu_test_shader from '$lib/shaders/mean_shift_cluster_step.wgsl?raw';
import update_density_scores_shader from '$lib/shaders/update_density_scores.wgsl?raw';

// TODO: add image locality weighting to cluster. make closer pixels count more toward the final color
// TODO: switch run_shader to actually be multiple passes. it should loop passes until the image isn't changing anymore (or is changing below some threshold)
// TODO: add pass to convert image colors into Oklab, and another to convert it back into rgb
// TODO: keep density scores as a buffer, don't turn back into number[]. this means that we'll need to somehow switch how we calculate the median density score
/// returns whether the colors changed (used to know whether to increase count)
export async function run_shader(
	imageBitMap: ImageBitmap,
	colors: Oklab[],
	base_bandwidth: number,
	cluster_check_radius: number
): Promise<[boolean, Uint8ClampedArray]> {
	// Set up the GPU resources

	console.log('starting mean shift cluster step WGPU');
	const adapter = await navigator.gpu?.requestAdapter();
	const device = await adapter?.requestDevice();

	if (!device) {
		// TODO: add an actual warning for this on the site, popup or whatever
		alert('need a browser that supports WebGPU');
		return [false, new Uint8ClampedArray()];
	}
	// TODO: rename the modules something more descriptive
	const module = device.createShaderModule({
		code: gpu_test_shader
	});

	const module2 = device.createShaderModule({
		code: update_density_scores_shader
	});

	const pipeline1 = device.createComputePipeline({
		label: 'gpu test compute pipeline',
		layout: 'auto',
		compute: {
			module,
			entryPoint: 'cs_main'
		}
	});

	const pipeline2 = device.createComputePipeline({
		label: 'gpu test compute pipeline',
		layout: 'auto',
		compute: {
			module: module2,
			entryPoint: 'cs_main'
		}
	});

	const input_color_texture = device.createTexture({
		label: 'input color texture',
		size: [imageBitMap.width, imageBitMap.height],
		format: 'rgba8unorm',
		usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
	});

	const density_scores_buffer = device.createBuffer({
		label: 'output buffer',
		size: imageBitMap.width * imageBitMap.height * 4,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
	});

	const output_colors_texture = device.createTexture({
		label: 'output colors texture',
		size: [imageBitMap.width, imageBitMap.height],
		format: 'rgba8unorm',
		usage: GPUTextureUsage.COPY_SRC | GPUTextureUsage.STORAGE_BINDING
	});

	device.queue.copyExternalImageToTexture(
		{ source: imageBitMap, flipY: false },
		{ texture: input_color_texture },
		{ width: imageBitMap.width, height: imageBitMap.height }
	);

	// This is where the specific functions starts ----------------------------------

	// mean shift cluster step
	async function gpu_mean_shift_cluster_step(density_scores: number[]): Promise<Uint8ClampedArray> {
		const median_density_score = get_median(density_scores);
		const uniforms_data = new Float32Array([
			base_bandwidth,
			cluster_check_radius,
			median_density_score
		]);

		const uniforms_buffer = device!.createBuffer({
			label: 'uniforms buffer',
			size: uniforms_data.byteLength,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});

		// Setup a bindGroup to tell the shader which
		// buffer to use for the computation
		const bind_group = device!.createBindGroup({
			label: 'gpu test bind group',
			layout: pipeline1.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: uniforms_buffer },
				{ binding: 1, resource: input_color_texture.createView() },
				{ binding: 2, resource: density_scores_buffer },
				{ binding: 3, resource: output_colors_texture.createView() }
			]
		});

		// Copy our input data to input buffers
		device!.queue.writeBuffer(uniforms_buffer, 0, uniforms_data);

		// Encode commands to do the computation
		const encoder = device!.createCommandEncoder({
			label: 'gpu test encoder'
		});
		const pass = encoder.beginComputePass({
			label: 'gpu test compute pass'
		});
		pass.setPipeline(pipeline1);
		pass.setBindGroup(0, bind_group);
		pass.dispatchWorkgroups(
			Math.ceil(imageBitMap.width / 16), // divide by 16 to match shader workgroup size
			Math.ceil(imageBitMap.height / 16) // divide by 16 to match shader workgroup size
		);
		pass.end();

		// Encode a command to copy the results to a mappable buffer.

		// Finish encoding and submit the commands
		const commandBuffer = encoder.finish();
		device!.queue.submit([commandBuffer]);

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

	async function gpu_update_density_scores(base_bandwidth: number): Promise<number[]> {
		const uniforms_data = new Float32Array([base_bandwidth, cluster_check_radius]);

		// this buffer is not used by the shader directly. when the shader finishes, its output is copied into this buffer so that it can be better used
		const readback_buffer = device!.createBuffer({
			label: 'readback buffer',
			size: density_scores_buffer.size,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
		});

		const uniforms_buffer = device!.createBuffer({
			label: 'uniforms buffer',
			size: uniforms_data.byteLength,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});

		// Setup a bindGroup to tell the shader which
		// buffer to use for the computation
		const bind_group = device!.createBindGroup({
			label: 'update density scores bind group',
			layout: pipeline2.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: uniforms_buffer },
				{ binding: 1, resource: input_color_texture.createView() },
				{ binding: 2, resource: density_scores_buffer }
			]
		});
		device!.queue.writeBuffer(uniforms_buffer, 0, uniforms_data);

		// Encode commands to do the computation
		const encoder = device!.createCommandEncoder({
			label: 'update density scores encoder'
		});
		const pass = encoder.beginComputePass({
			label: 'update density scores compute pass'
		});
		pass.setPipeline(pipeline2);
		pass.setBindGroup(0, bind_group);
		pass.dispatchWorkgroups(
			Math.ceil(imageBitMap.width / 16), // divide by 16 to match shader workgroup size
			Math.ceil(imageBitMap.height / 16) // divide by 16 to match shader workgroup size
		);
		pass.end();

		encoder.copyBufferToBuffer(
			density_scores_buffer,
			0,
			readback_buffer,
			0,
			density_scores_buffer.size
		);

		// Finish encoding and submit the commands
		const command_buffer = encoder.finish();
		device!.queue.submit([command_buffer]);
		await readback_buffer.mapAsync(GPUMapMode.READ);
		const result = new Float32Array(readback_buffer.getMappedRange().slice());

		const density_scores: number[] = [];
		for (let i = 0; i < colors.length; i++) {
			const density_score = result[i];
			density_scores.push(density_score);
		}

		readback_buffer.unmap();

		console.log();
		console.log('WGPU Density Scores');
		console.log(density_scores);

		return density_scores;
	}
	// end of the shader BS calling everything now

	const density_scores = await gpu_update_density_scores(base_bandwidth);
	return [true, await gpu_mean_shift_cluster_step(density_scores)];
}
