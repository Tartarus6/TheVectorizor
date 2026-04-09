<script lang="ts">
	import { rgbToOklab, oklabToSRGB, type RGB, type Oklab, rgbToHex, get_distance, get_median, clamp } from "$lib/utils";

	let base_bandwidth = $state(0.1);
	let image_canvas: HTMLCanvasElement | undefined = $state();
	let canvas: HTMLCanvasElement | undefined = $state();

	let num_points = $state(10);

	let graph_size_rem = $state(32);
	let point_size_rem = $state(2);

	let count = $state(0);

	let colors: Oklab[] = $state([]);
	let density_scores: number[] = $state([]);

	function randomize_colors(n: number) {
		colors = [];
		count = 0;
		for (let i = 0; i < n; i++) {
			let color: Oklab = { L: 0.5, a: Math.random() - 0.5, b: Math.random() - 0.5};
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
			newColors.push({L, a, b});
			if (L !== colors[i].L || a !== colors[i].a || b !== colors[i].b) {
				the_same = false;
			}
		}

		readBackBuffer.unmap();

		if (!the_same) {
			count += 1;
			colors = newColors;
		}
	} // this is where test gpu ends

	function mean_shift_cluster_step() {
		// initialize density scores if not done
		if (density_scores.length != colors.length) {
			update_density_scores();
		}

		let shifted_colors: Oklab[] = [];

		let the_same: boolean = true;

		let median_density = get_median(density_scores);

		for (let i in colors) {
			let color = colors[i];
			let cluster: Oklab[] = [];

			let bandwidth = get_bandwidth(density_scores[i], median_density);

			// making cluster
			for (let other_color of colors) {
				let dist = get_distance(color, other_color);

				if (dist < bandwidth) {
					cluster.push(other_color);
				}
			}

			// getting cluster average
			let cluster_sum: Oklab = { L: 0, a: 0, b: 0 };
			for (let neighbor_color of cluster) {
				cluster_sum.L += neighbor_color.L;
				cluster_sum.a += neighbor_color.a;
				cluster_sum.b += neighbor_color.b;
			}

			let new_color: Oklab = {
				L: cluster_sum.L / cluster.length,
				a: cluster_sum.a / cluster.length,
				b: cluster_sum.b / cluster.length
			};

			// if color is changed, mark that
			if (color.L != new_color.L || color.a != new_color.a || color.b != new_color.b) {
				the_same = false;
			}

			// push the new color
			shifted_colors.push(new_color);
		}

		// only do the stuff if the colors actually changed
		if (!the_same) {
			count += 1;

			colors = shifted_colors;

			update_density_scores();
		}
	}

	// per-color bandwidth calculation
	function get_bandwidth(density_score: number, median_density: number) {
		const alpha = 0.35; // controls how strongly the density matters
		const epsilon = 0.000001; // prevents divide by zero
		const min_mult = 0.7;
		const max_mult = 1.8;
		
		console.log(density_score);
		console.log(median_density);
		console.log(Math.pow(median_density / density_score, alpha));
		return base_bandwidth * clamp(Math.pow(median_density / (density_score + epsilon), alpha), min_mult, max_mult);
	}

	function get_density_score(color: Oklab) {
		let density_score = 0;
		for (let other_color of colors) {
			if (color == other_color) {
				continue;
			}

			let distance = get_distance(color, other_color);

			density_score += Math.exp(-(distance * distance) / (2 * base_bandwidth * base_bandwidth))
		}

		return density_score;
	}

	function update_density_scores() {
		for (let i in colors) {
			density_scores[i] = get_density_score(colors[i]);
		}
	}

	function add_color_from_graph_click(event: MouseEvent) {
		const graph = event.currentTarget as HTMLButtonElement | null;
		if (!graph) {
			return;
		}

		const rect = graph.getBoundingClientRect();
		const x_ratio = clamp((event.clientX - rect.left) / rect.width, 0, 1);
		const y_ratio = clamp((event.clientY - rect.top) / rect.height, 0, 1);

		const new_color: Oklab = {
			L: 0.5,
			a: x_ratio - 0.5,
			b: y_ratio - 0.5
		};

		colors = [...colors, new_color];
		update_density_scores();
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
	onmousedown={mean_shift_cluster_step}
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
	
<button
	type="button"
	class="grid grid-cols-1 grid-rows-1 bg-white p-0 border-0"
	style="width: {graph_size_rem}rem; height: {graph_size_rem}rem;"
	onmousedown={add_color_from_graph_click}
>
	{#each colors as color}
		<div
			style="background: {rgbToHex(oklabToSRGB(color))};
			    width: {point_size_rem}rem; height: {point_size_rem}rem;
			    transform: translate(
					{(color.a + 0.5) * graph_size_rem - point_size_rem / 2}rem,
					{(color.b + 0.5) * graph_size_rem - point_size_rem / 2}rem);"
			class="col-start-1 row-start-1 rounded-full content-center"
		>
			<p class="w-full text-center">{get_density_score(color).toFixed(1)}</p>
		</div>
	{/each}
</button>

<div class="flex flex-col">
	{#each colors as color}
		<div style="background: {rgbToHex(oklabToSRGB(color))};">
			<span>{rgbToHex(oklabToSRGB(color))}</span>
			<span>OkLab({color.L.toFixed(2)}, {color.a.toFixed(2)}, {color.b.toFixed(2)})</span>
		</div>
	{/each}
</div>

<div>
	<canvas bind:this={canvas} class="col-start-1 row-start-1 h-100 w-100"></canvas>
	<canvas bind:this={image_canvas} class="col-start-1 row-start-1 h-100 w-100"></canvas>
</div>
