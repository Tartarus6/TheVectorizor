<script lang="ts">
	import { rgbToOklab, oklabToSRGB, type RGB, type Oklab, rgbToHex } from "$lib/utils";

	let bandwidth = $state(0.2);
	let num_points = $state(10);

	let graph_size_rem = $state(32);
	let point_size_rem = $state(2);

	let count = $state(0);

	let colors: Oklab[] = $state([]);

	function randomize_colors(n: number) {
		colors = [];
		count = 0;
		for (let i = 0; i < n; i++) {
			let color: Oklab = { L: 0.5, a: Math.random() - 0.5, b: Math.random() - 0.5};
			colors[i] = color;
		}
	}

	function mean_shift_cluster_step(b: number) {
		let shifted_colors: Oklab[] = [];

		let the_same: boolean = true;

		for (let color of colors) {
			let cluster: Oklab[] = [];
			for (let other_color of colors) {
				let delta_L_sq = (color.L - other_color.L) * (color.L - other_color.L);
				let delta_a_sq = (color.a - other_color.a) * (color.a - other_color.a);
				let delta_b_sq = (color.b - other_color.b) * (color.b - other_color.b);
				let dist = Math.sqrt(delta_L_sq + delta_a_sq + delta_b_sq);

				if (dist < bandwidth) {
					cluster.push(other_color);
				}
			}

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

			if (color.L != new_color.L || color.a != new_color.a || color.b != new_color.b) {
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
			class="col-start-1 row-start-1 rounded-full"
		></div>
	{/each}
</div>
