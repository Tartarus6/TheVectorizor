import type { Oklab } from './utils';

import gpu_test_shader from '$lib/shaders/gpu_test.wgsl?raw';

// TODO: switch base bandwidth to instead use per-color bandwidths
/// returns whether the colors changed (used to know whether to increase count)
export async function gpu_test(
	colors: Oklab[],
	base_bandwidth: number
): Promise<[boolean, Oklab[]]> {
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
		label: 'doubling compute pipeline',
		layout: 'auto',
		compute: {
			module,
			entryPoint: 'cs_main'
		}
	});

	const input = new Float32Array(colors.length * 4);

	for (let i = 0; i < colors.length; i++) {
		input[i * 4 + 0] = colors[i].L;
		input[i * 4 + 1] = colors[i].a + 0.5;
		input[i * 4 + 2] = colors[i].b + 0.5;
		input[i * 4 + 3] = 0;
	}

	// create a buffer on the GPU to hold our computation
	// input and output
	const workBuffer = device.createBuffer({
		label: 'input buffer',
		size: input.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
	});
	// Copy our input data to that buffer

	// create a buffer on the GPU to get a copy of the results
	const outputBuffer = device.createBuffer({
		label: 'output buffer',
		size: input.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
	});

	const readBackBuffer = device.createBuffer({
		label: 'readback buffer',
		size: input.byteLength,
		usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
	});

	const uniformData = new Float32Array([input.length, base_bandwidth]);

	const uniformBuffer = device.createBuffer({
		label: 'uniform buffer',
		size: input.byteLength,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
	});

	// Setup a bindGroup to tell the shader which
	// buffer to use for the computation
	const bindGroup = device.createBindGroup({
		label: 'bindGroup for work buffer',
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: uniformBuffer },
			{ binding: 1, resource: workBuffer },
			{ binding: 2, resource: outputBuffer }
		]
	});
	device.queue.writeBuffer(uniformBuffer, 0, uniformData);
	device.queue.writeBuffer(workBuffer, 0, input);

	// Encode commands to do the computation
	const encoder = device.createCommandEncoder({
		label: 'doubling encoder'
	});
	const pass = encoder.beginComputePass({
		label: 'doubling compute pass'
	});
	pass.setPipeline(pipeline);
	pass.setBindGroup(0, bindGroup);
	pass.dispatchWorkgroups(input.length);
	pass.end();

	// Encode a command to copy the results to a mappable buffer.
	encoder.copyBufferToBuffer(outputBuffer, 0, readBackBuffer, 0, outputBuffer.size);

	// Finish encoding and submit the commands
	const commandBuffer = encoder.finish();
	device.queue.submit([commandBuffer]);

	// Read the results
	await readBackBuffer.mapAsync(GPUMapMode.READ);
	const result = new Float32Array(readBackBuffer.getMappedRange().slice());

	let the_same = true;
	const newColors: Oklab[] = [];
	for (let i = 0; i < colors.length; i++) {
		const L = result[i * 4 + 0];
		const a = result[i * 4 + 1] - 0.5;
		const b = result[i * 4 + 2] - 0.5;
		newColors.push({ L, a, b });
		if (L !== colors[i].L || a !== colors[i].a || b !== colors[i].b) {
			the_same = false;
		}
	}

	readBackBuffer.unmap();

	if (!the_same) {
		return [true, newColors];
	}
	return [false, colors];
} // this is where test gpu ends
