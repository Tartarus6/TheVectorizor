<script lang="ts">
	import {
		rgbToOklab,
		oklabToSRGB,
		type RGB,
		type Oklab,
		rgbToHex,
		get_distance,
		get_median,
		clamp
	} from '$lib/utils';
	import { gpu_mean_shift_cluster_step, gpu_update_density_scores } from '$lib/shaders';

	let base_bandwidth = $state(0.1);
	let image_canvas: HTMLCanvasElement | undefined = $state();
	let canvas: HTMLCanvasElement | undefined = $state();

	let num_points = $state(10);

	let graph_size_rem = $state(32);
	let point_size_rem = $state(2);

	let count = $state(0);

	let colors: Oklab[] = $state([]);
	let density_scores: number[] = $state([]);

	async function randomize_colors() {
		colors = [];
		count = 0;
		for (let i = 0; i < num_points; i++) {
			let color: Oklab = { L: 0.5, a: Math.random() - 0.5, b: Math.random() - 0.5 };
			colors[i] = color;
		}
		// await gpu_update_density_scores(colors, base_bandwidth);
	}

	async function mean_shift_cluster_step() {
		console.log('starting mean shift cluster step TS');
		// initialize density scores if not done
		if (density_scores.length != colors.length) {
			// await gpu_update_density_scores(colors, base_bandwidth);
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

			// await gpu_update_density_scores(shifted_colors, base_bandwidth);
			update_density_scores();
		}
	}

	// per-color bandwidth calculation
	function get_bandwidth(density_score: number, median_density: number) {
		const alpha = 0.35; // controls how strongly the density matters
		const epsilon = 0.000001; // prevents divide by zero
		const min_mult = 0.7;
		const max_mult = 1.8;

		return (
			base_bandwidth *
			clamp(Math.pow(median_density / (density_score + epsilon), alpha), min_mult, max_mult)
		);
	}

	function get_density_score(color: Oklab) {
		let density_score = 0;
		for (let other_color of colors) {
			if (color == other_color) {
				continue;
			}

			let distance = get_distance(color, other_color);

			density_score += Math.exp(-(distance * distance) / (2 * base_bandwidth * base_bandwidth));
		}

		return density_score;
	}

	function update_density_scores() {
		density_scores = [];
		for (let i in colors) {
			density_scores[i] = get_density_score(colors[i]);
		}

		console.log();
		console.log('TS Density Scores');
		console.log($state.snapshot(density_scores));
	}

	async function add_color_from_graph_click(event: MouseEvent) {
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
		// await gpu_update_density_scores(colors, base_bandwidth);
		update_density_scores();
	}
</script>

<h1 class="text-center">The Vectorizor</h1>

<div class="flex flex-col">
	<div class="m-2 flex w-64 flex-col bg-slate-500 p-2">
		<span>Num Points: {num_points}</span>
		<input type="range" bind:value={num_points} min="0" max="1000" />
	</div>

	<button onmousedown={randomize_colors} class="m-2 w-fit cursor-pointer bg-green-500 p-2">
		<span>randomize colors</span>
	</button>

	<button onmousedown={mean_shift_cluster_step} class="m-2 w-fit cursor-pointer bg-yellow-500 p-2">
		<span>mean shift cluster step TS</span>
	</button>

	<button
		onmousedown={async () => {
			const [changed, newColors] = await gpu_mean_shift_cluster_step(
				colors,
				density_scores,
				base_bandwidth
			);
			if (changed) {
				colors = newColors;
				count += 1;
				await gpu_update_density_scores(colors, base_bandwidth);
			}
		}}
		class="m-2 w-fit cursor-pointer bg-yellow-500 p-2"
	>
		mean shift cluster step WGPU
	</button>

	<button onmousedown={update_density_scores} class="m-2 w-fit cursor-pointer bg-red-500 p-2">
		<span>update density scores TS</span>
	</button>

	<button
		onmousedown={() => {
			gpu_update_density_scores(colors, base_bandwidth);
		}}
		class="m-2 w-fit cursor-pointer bg-red-500 p-2"
	>
		<span>update density scores WGPU</span>
	</button>

	<span>Number of passes: {count}</span>
</div>

<button
	type="button"
	class="grid grid-cols-1 grid-rows-1 border-0 bg-white p-0"
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
			class="col-start-1 row-start-1 content-center rounded-full"
		>
			<p class="w-full text-center">{get_density_score(color).toFixed(1)}</p>
		</div>
	{/each}
</button>

<div>
	<canvas bind:this={canvas} class="col-start-1 row-start-1 h-100 w-100"></canvas>
	<canvas bind:this={image_canvas} class="col-start-1 row-start-1 h-100 w-100"></canvas>
</div>
