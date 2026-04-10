import type { Oklab } from './utils';

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
		code: /*wgsl*/ `
			struct Uniforms {
                num_colors: f32,
                bandwidth: f32,
            }


            @group(0) @binding(0) var<uniform> uniforms: Uniforms;
            @group(0) @binding(1) var<storage, read> input_colors: array<vec4<f32>>;
            @group(0) @binding(2) var<storage, read_write> output_colors: array<vec4<f32>>;



            @compute @workgroup_size(64)
            fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
                let BANDWIDTH_SQUARED = uniforms.bandwidth * uniforms.bandwidth;
                let idx = id.x;
                if idx >= u32(uniforms.num_colors) { return; }

                let color = input_colors[idx].rgb;

                var cluster_sum = vec3<f32>(0.0);
                var cluster_count = 0u;

                for (var i = 0u; i < u32(uniforms.num_colors); i++) {
                    let other = input_colors[i].rgb;
                    let delta = color - other;
                    let dist = dot(delta, delta);

                    if dist < BANDWIDTH_SQUARED {
                        cluster_sum += other;
                        cluster_count += 1u;

                    }

                }

                let new_color = cluster_sum / f32(cluster_count);
                output_colors[idx] = vec4<f32>(new_color, 1.0);
            }
				`
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
