<script lang="ts">
	import { rgbToOklab, oklabToSRGB, type RGB, type Oklab, rgbToHex, get_distance, get_median, clamp } from "$lib/utils";

	let base_bandwidth = $state(0.2);
	let density_probe_radius = $state(0.1);
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
		const min_mult = 1.8;
		const max_mult = 0.7;
		
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

			density_score += Math.exp(-(distance * distance) / (2 * density_probe_radius * density_probe_radius))
		}

		return density_score;
	}

	function update_density_scores() {
		for (let i in colors) {
			density_scores[i] = get_density_score(colors[i]);
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
	onmousedown={mean_shift_cluster_step}
>
	<div class="bg-yellow-500">
		<span>mean shift cluster step</span>
	</div>
</button>

<span>Number of passes: {count}</span>

<div class="flex flex-col">
	{#each colors as color}
		<div style="background: {rgbToHex(oklabToSRGB(color))};">
			<span>{rgbToHex(oklabToSRGB(color))}</span>
			<span>OkLab({color.L.toFixed(2)}, {color.a.toFixed(2)}, {color.b.toFixed(2)})</span>
		</div>
	{/each}
</div>
	
<div
	class="grid grid-cols-1 grid-rows-1 bg-white"
	style="width: {graph_size_rem}rem; height: {graph_size_rem}rem;"
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
</div>
