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

	function mean_shift_cluster_step(b: number) {
		count += 1;
		let shifted_colors: RGB[] = [];

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

			shifted_colors.push(new_color);
		}

		colors = shifted_colors;
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
