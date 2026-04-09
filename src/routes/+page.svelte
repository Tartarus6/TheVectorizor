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
	}

	let num_points = $state(10);

	let graph_size_rem = 32;
	let point_size_rem = 2;

	let colors: RGB[] = $state([]);

	function randomize_colors(n: number) {
		colors = [];
		for (let i = 0; i < n; i++) {
			let color = new RGB(Math.random() * 255, 0, Math.random() * 255);
			colors[i] = color;
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
					{(color.r / 255) * graph_size_rem - point_size_rem / 2}rem,
					{(color.b / 255) * graph_size_rem - point_size_rem / 2}rem);"
			class="col-start-1 row-start-1 rounded-full"
		></div>
	{/each}
</div>
