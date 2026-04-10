import { get_median, type Oklab } from './utils';
import gpu_test_shader from '$lib/shaders/mean_shift_cluster_step.wgsl?raw';
import update_density_scores_shader from '$lib/shaders/update_density_scores.wgsl?raw';

// TODO: rename gpu_test to what it actually does
// TODO: switch base bandwidth to instead use per-color bandwidths
/// returns whether the colors changed (used to know whether to increase count)
export async function gpu_mean_shift_cluster_step(
	colors: Oklab[],
	density_scores: number[],
	base_bandwidth: number
): Promise<[boolean, Oklab[]]> {
	console.log('starting mean shift cluster step WGPU');
	const adapter = await navigator.gpu?.requestAdapter();
	const device = await adapter?.requestDevice();

	if (!device) {
		// TODO: add an actual warning for this on the site, popup or whatever
		alert('need a browser that supports WebGPU');
		return [false, colors];
	}

	const module = device.createShaderModule({
		code: gpu_test_shader
	});

	const pipeline = device.createComputePipeline({
		label: 'gpu test compute pipeline',
		layout: 'auto',
		compute: {
			module,
			entryPoint: 'cs_main'
		}
	});

	const input_colors_data = new Float32Array(colors.length * 4);
	const input_density_scores_data = new Float32Array(colors.length);

	for (let i = 0; i < colors.length; i++) {
		// colors data
		input_colors_data[i * 4 + 0] = colors[i].L;
		input_colors_data[i * 4 + 1] = colors[i].a + 0.5;
		input_colors_data[i * 4 + 2] = colors[i].b + 0.5;
		input_colors_data[i * 4 + 3] = 0;

		// density scores data
		input_density_scores_data[i] = density_scores[i];
	}

	// create a buffer on the GPU to hold our computation
	// input and output
	const input_colors_buffer = device.createBuffer({
		label: 'input colors buffer',
		size: input_colors_data.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
	});

	/// size in bytes of the output buffer
	const density_scores_buffer_size = colors.length * 4;

	// create a buffer on the GPU to get a copy of the results
	const input_density_scores_buffer = device.createBuffer({
		label: 'input density scores buffer',
		size: input_density_scores_data.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
	});

	// create a buffer on the GPU to get a copy of the results
	const output_colors_buffer = device.createBuffer({
		label: 'output buffer',
		size: input_colors_data.byteLength, // same size as input colors buffer
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
	});

	/// this buffer is not used by the shader directly. when the shader finishes, its output is copied into this buffer so that it can be better used
	const readback_buffer = device.createBuffer({
		label: 'readback buffer',
		size: input_colors_data.byteLength,
		usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
	});

	const median_density_score = get_median(density_scores);
	const uniforms_data = new Float32Array([base_bandwidth, median_density_score]);

	const uniforms_buffer = device.createBuffer({
		label: 'uniforms buffer',
		size: uniforms_data.byteLength,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
	});

	// Setup a bindGroup to tell the shader which
	// buffer to use for the computation
	const bind_group = device.createBindGroup({
		label: 'gpu test bind group',
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: uniforms_buffer },
			{ binding: 1, resource: input_colors_buffer },
			{ binding: 2, resource: input_density_scores_buffer },
			{ binding: 3, resource: output_colors_buffer }
		]
	});

	// Copy our input data to input buffers
	device.queue.writeBuffer(uniforms_buffer, 0, uniforms_data);
	device.queue.writeBuffer(input_colors_buffer, 0, input_colors_data);
	device.queue.writeBuffer(input_density_scores_buffer, 0, input_density_scores_data);

	// Encode commands to do the computation
	const encoder = device.createCommandEncoder({
		label: 'gpu test encoder'
	});
	const pass = encoder.beginComputePass({
		label: 'gpu test compute pass'
	});
	pass.setPipeline(pipeline);
	pass.setBindGroup(0, bind_group);
	pass.dispatchWorkgroups(input_colors_data.length);
	pass.end();

	// Encode a command to copy the results to a mappable buffer.
	encoder.copyBufferToBuffer(
		output_colors_buffer,
		0,
		readback_buffer,
		0,
		output_colors_buffer.size
	);

	// Finish encoding and submit the commands
	const commandBuffer = encoder.finish();
	device.queue.submit([commandBuffer]);

	// Read the results
	await readback_buffer.mapAsync(GPUMapMode.READ);
	const result = new Float32Array(readback_buffer.getMappedRange().slice());

	let the_same = true;
	const new_colors: Oklab[] = [];
	for (let i = 0; i < colors.length; i++) {
		const L = result[i * 4 + 0];
		const a = result[i * 4 + 1] - 0.5;
		const b = result[i * 4 + 2] - 0.5;
		new_colors.push({ L, a, b });
		if (L !== colors[i].L || a !== colors[i].a || b !== colors[i].b) {
			the_same = false;
		}
	}

	readback_buffer.unmap();

	if (!the_same) {
		return [true, new_colors];
	}
	return [false, colors];
}

// TODO: keep density scores as a buffer, don't turn back into number[]
export async function gpu_update_density_scores(
	colors: Oklab[],
	base_bandwidth: number
): Promise<number[]> {
	const adapter = await navigator.gpu?.requestAdapter();
	const device = await adapter?.requestDevice();

	if (!device) {
		// TODO: add an actual warning for this on the site, popup or whatever
		alert('need a browser that supports WebGPU');
		return [];
	}

	const module = device.createShaderModule({
		code: update_density_scores_shader
	});

	const pipeline = device.createComputePipeline({
		label: 'update density scores compute pipeline',
		layout: 'auto',
		compute: {
			module,
			entryPoint: 'cs_main'
		}
	});

	const input_colors_data = new Float32Array(colors.length * 4);

	for (let i = 0; i < colors.length; i++) {
		input_colors_data[i * 4 + 0] = colors[i].L;
		input_colors_data[i * 4 + 1] = colors[i].a + 0.5;
		input_colors_data[i * 4 + 2] = colors[i].b + 0.5;
		input_colors_data[i * 4 + 3] = 0;
	}

	// create a buffer on the GPU to hold our computation
	// input and output
	const input_colors_buffer = device.createBuffer({
		label: 'input colors buffer',
		size: input_colors_data.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
	});
	// Copy our input data to that buffer

	/// size in bytes of the output buffer
	const output_buffer_size = colors.length * 4;

	// create a buffer on the GPU to get a copy of the results
	const output_density_scores_buffer = device.createBuffer({
		label: 'output buffer',
		size: output_buffer_size,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
	});

	/// this buffer is not used by the shader directly. when the shader finishes, its output is copied into this buffer so that it can be better used
	const readback_buffer = device.createBuffer({
		label: 'readback buffer',
		size: output_buffer_size,
		usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
	});

	const uniforms_data = new Float32Array([base_bandwidth]);

	const uniforms_buffer = device.createBuffer({
		label: 'uniforms buffer',
		size: uniforms_data.byteLength,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
	});

	// Setup a bindGroup to tell the shader which
	// buffer to use for the computation
	const bind_group = device.createBindGroup({
		label: 'update density scores bind group',
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: uniforms_buffer },
			{ binding: 1, resource: input_colors_buffer },
			{ binding: 2, resource: output_density_scores_buffer }
		]
	});
	device.queue.writeBuffer(uniforms_buffer, 0, uniforms_data);
	device.queue.writeBuffer(input_colors_buffer, 0, input_colors_data);

	// Encode commands to do the computation
	const encoder = device.createCommandEncoder({
		label: 'update density scores encoder'
	});
	const pass = encoder.beginComputePass({
		label: 'update density scores compute pass'
	});
	pass.setPipeline(pipeline);
	pass.setBindGroup(0, bind_group);
	pass.dispatchWorkgroups(input_colors_data.length);
	pass.end();

	// Encode a command to copy the results to a mappable buffer.
	encoder.copyBufferToBuffer(
		output_density_scores_buffer,
		0,
		readback_buffer,
		0,
		output_density_scores_buffer.size
	);

	// Finish encoding and submit the commands
	const command_buffer = encoder.finish();
	device.queue.submit([command_buffer]);

	// Read the results
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
