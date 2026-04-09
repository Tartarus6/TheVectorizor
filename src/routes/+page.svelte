<script lang="ts">
	class RGB {
		r: number;
		g: number;
		b: number;

		constructor(r: number, g: number, b: number) {
			this.r = r;
			this.g = g;
			this.b = b;
		}

		clone() {
			return new RGB(this.r, this.g, this.b);
		}
	}
	let image_canvas: HTMLCanvasElement | undefined = $state();
	let canvas: HTMLCanvasElement | undefined = $state();

	let bandwidth = $state(50);
	let num_points = $state(10);

	let graph_size_rem = $state(32);
	let point_size_rem = $state(2);

	let count = $state(0);

	let colors: RGB[] = $state([]);

	function randomize_colors(n: number) {
		colors = [];
		count = 0;
		for (let i = 0; i < n; i++) {
			let color = new RGB(0, Math.random() * 255, Math.random() * 255);
			colors[i] = color;
		}
	}

	async function gpu_test() {
		const adapter = await navigator.gpu?.requestAdapter();
		const device = await adapter?.requestDevice();
		if (!device) {
			alert('need a browser that supports WebGPU');
			return;
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
			input[i * 4 + 0] = colors[i].r / 255;
			input[i * 4 + 1] = colors[i].g / 255;
			input[i * 4 + 2] = colors[i].b / 255;
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

		const uniformData = new Float32Array([input.length, bandwidth / 255]);

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
		const newColors: RGB[] = [];
		for (let i = 0; i < colors.length; i++) {
			const r = result[i * 4 + 0] * 255;
			const g = result[i * 4 + 1] * 255;
			const b = result[i * 4 + 2] * 255;
			newColors.push(new RGB(r, g, b));
			if (r !== colors[i].r || g !== colors[i].g || b !== colors[i].b) {
				the_same = false;
			}
		}

		readBackBuffer.unmap();

		if (!the_same) {
			count += 1;
			colors = newColors;
		}
	} // this is where test gpu ends

	function mean_shift_cluster_step(b: number) {
		let shifted_colors: RGB[] = [];

		let the_same: boolean = true;

		for (let color of colors) {
			let cluster: RGB[] = [];
			for (let other_color of colors) {
				let delta_r_sq = (color.r - other_color.r) * (color.r - other_color.r);
				let delta_g_sq = (color.g - other_color.g) * (color.g - other_color.g);
				let delta_b_sq = (color.b - other_color.b) * (color.b - other_color.b);
				let dist = Math.sqrt(delta_r_sq + delta_g_sq + delta_b_sq);

				if (dist < bandwidth) {
					cluster.push(other_color);
				}
			}

			let cluster_sum = new RGB(0, 0, 0);
			for (let neighbor_color of cluster) {
				cluster_sum.r += neighbor_color.r;
				cluster_sum.g += neighbor_color.g;
				cluster_sum.b += neighbor_color.b;
			}

			let new_color = new RGB(
				cluster_sum.r / cluster.length,
				cluster_sum.g / cluster.length,
				cluster_sum.b / cluster.length
			);

			if (color.r != new_color.r || color.g != new_color.g || color.b != new_color.b) {
				the_same = false;
			}

			shifted_colors.push(new_color);
		}

		if (!the_same) {
			count += 1;

			colors = shifted_colors;
		}
	}
</script>

<h1 class="text-center">The Vectorizor</h1>

<button
	onmousedown={() => {
		randomize_colors(num_points);
	}}
>
	<div class="bg-green-500">
		<span>randomize colors</span>
	</div>
</button>

<button
	onmousedown={() => {
		mean_shift_cluster_step(bandwidth);
	}}
>
	<div class="bg-yellow-500">
		<span>mean shift cluster step</span>
	</div>
</button>

<button
	onmousedown={() => {
		gpu_test();
	}}
>
	test the gpu
</button>

<span>Number of passes: {count}</span>

<div class="flex flex-col">
	{#each colors as color}
		<div style="background: rgb({color.r}, {color.g}, {color.b})">
			<p>rgb({color.r}, {color.g}, {color.b})</p>
		</div>
	{/each}
</div>

<div
	class="grid grid-cols-1 grid-rows-1 bg-white"
	style="width: {graph_size_rem}rem; height: {graph_size_rem}rem;"
>
	{#each colors as color}
		<div
			style="background: rgb({color.r}, {color.g}, {color.b});
			    width: {point_size_rem}rem; height: {point_size_rem}rem;
			    transform: translate(
					{(color.g / 255) * graph_size_rem - point_size_rem / 2}rem,
					{(color.b / 255) * graph_size_rem - point_size_rem / 2}rem);"
			class="col-start-1 row-start-1 rounded-full"
		></div>
	{/each}
</div>

<div>
	<canvas bind:this={canvas} class="col-start-1 row-start-1 h-100 w-100"></canvas>
	<canvas bind:this={image_canvas} class="col-start-1 row-start-1 h-100 w-100"></canvas>
</div>
