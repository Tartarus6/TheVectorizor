<script lang="ts">
	import { run_shader } from '$lib/shaders';

	let svgUrl: string | undefined = $state();

	// TODO: add a thing thatll help show what the mean shift clustering has done by either making an apng or just toggling the visibility of the 2 images, so they can be viewed on top of one another
	// TODO: figure out good ranges for the input variables (like blur radius), and maybe dont hardcode the limits

	let uploadedImageUrl: string | ArrayBuffer | null = $state('google_test.jpg'); // default image to google test
	let image: ImageBitmap | undefined = $state();

	let imageUploaded: HTMLImageElement | undefined = $state();

	let base_bandwidth = $state(0.05);
	/// the width and height of the tiles that the texture is broken into for processing (in order to prevent the system from hanging until jobs are complete)
	let tile_size = $state(512);
	let num_cluster_passes = $state(5);
	let num_edge_trace_passes = $state(1);
	let blur_radius = $state(1);
	let image_canvas: HTMLCanvasElement | undefined = $state();
	let canvas: HTMLCanvasElement | undefined = $state();
	let svg_preview: HTMLImageElement | undefined = $state();
	let canvas_scale = $state(8);

	const onFileSelected = (e: any) => {
		const file = e.target.files[0];
		let reader = new FileReader();
		reader.readAsDataURL(file);
		reader.onload = async (e) => {
			uploadedImageUrl = e.target!.result;

			const res = await fetch(uploadedImageUrl as string);
			image = await createImageBitmap(await res.blob());
		};
		console.log(uploadedImageUrl!);
	};

	function apply_svg_display_scale(target: HTMLImageElement | undefined) {
		if (!image) {
			return;
		}

		if (!target) {
			return;
		}

		target.style.width = `${image.width * canvas_scale}px`;
		target.style.height = `${image.height * canvas_scale}px`;
	}

	function apply_canvas_display_scale(target: HTMLCanvasElement | undefined) {
		if (!image) {
			return;
		}

		if (!target) {
			return;
		}

		target.style.width = `${image.width * canvas_scale}px`;
		target.style.height = `${image.height * canvas_scale}px`;
		target.style.imageRendering = 'pixelated';
	}

	$effect(() => {
		apply_canvas_display_scale(image_canvas);
		apply_svg_display_scale(svg_preview);
	});
</script>

<h1 class="text-center">The Vectorizor</h1>

<div class="flex w-92 flex-col">
	<div class="m-2 flex flex-col bg-slate-500 p-2">
		<span>Canvas Scale: {canvas_scale}</span>
		<input type="range" bind:value={canvas_scale} min={1} max={30} />
	</div>

	<div class="m-2 flex flex-col bg-slate-500 p-2">
		<div class="flex flex-row gap-2">
			<span>Base Bandwidth:</span>
			<input
				type="number"
				bind:value={base_bandwidth}
				min={0}
				max={1}
				step={0.0001}
				class="min-w-20 border-2 border-white"
			/>
		</div>
		<input type="range" bind:value={base_bandwidth} min={0} max={1} step={0.0001} />
	</div>

	<div class="m-2 flex flex-col bg-slate-500 p-2">
		<div class="flex flex-row gap-2">
			<span>Tile Size:</span>
			<input
				type="number"
				bind:value={tile_size}
				min={1}
				max={image ? Math.max(image.width, image.height) : 512}
				step={1}
				class="border-2 border-white"
			/>
		</div>
		<input
			type="range"
			bind:value={tile_size}
			min={1}
			max={image ? Math.max(image.width, image.height) : 512}
			step={1}
		/>
	</div>

	<div class="m-2 flex flex-col bg-slate-500 p-2">
		<div class="flex flex-row gap-2">
			<span>Blur Radius:</span>
			<input
				type="number"
				bind:value={blur_radius}
				min={1}
				max={image ? 50 : 512}
				step={1}
				class="border-2 border-white"
			/>
		</div>
		<input type="range" bind:value={blur_radius} min={1} max={image ? 50 : 512} step={1} />
	</div>

	<div class="m-2 flex flex-col bg-slate-500 p-2">
		<span>Cluster Passes: {num_cluster_passes}</span>
		<input type="range" bind:value={num_cluster_passes} min={1} max={20} />
	</div>

	<div class="m-2 flex flex-col bg-slate-500 p-2">
		<span>Edge Trace Passes: {num_edge_trace_passes}</span>
		<input type="range" bind:value={num_edge_trace_passes} min={0} max={20} />
	</div>

	<button
		onmousedown={async () => {
			const res = await fetch(uploadedImageUrl as string);
			image = await createImageBitmap(await res.blob());

			image_canvas!.width = image.width;
			image_canvas!.height = image.height;
			image_canvas!.getContext('2d')!.drawImage(image, 0, 0);

			const startTime = performance.now();
			const [success, svg, pixels] = await run_shader(
				image,
				base_bandwidth,
				tile_size,
				blur_radius,
				num_cluster_passes,
				num_edge_trace_passes
			);
			const endTime = performance.now();
			console.log(`Shader execution time: ${(endTime - startTime).toFixed(2)}ms`);

			if (success) {
				canvas!.width = image.width;
				canvas!.height = image.height;
				const safePixels = new Uint8ClampedArray(new ArrayBuffer(pixels.length));
				safePixels.set(pixels);
				const ctx = canvas!.getContext('2d')!;

				ctx.putImageData(new ImageData(safePixels, image.width, image.height), 0, 0);
				if (svgUrl) {
					URL.revokeObjectURL(svgUrl);
				}

				svgUrl = URL.createObjectURL(new Blob([svg], { type: 'image/svg+xml' }));
			}
		}}
		class="m-2 w-fit cursor-pointer bg-purple-500 p-2"
	>
		<span>shader pass</span>
	</button>
</div>
<div class="flex w-fit flex-col bg-slate-600 p-2">
	<h3>appload your image here</h3>
	<input type="file" accept="image/*" class="bg-blue-500" onchange={onFileSelected} />
	{#if uploadedImageUrl}
		<img
			src={uploadedImageUrl as string}
			alt="uploaded"
			bind:this={imageUploaded}
			class="max-h-100 max-w-100"
		/>
	{/if}
</div>

<div class="flex w-fit flex-col gap-2 bg-black">
	<canvas bind:this={image_canvas} style="image-rendering: pixelated;"></canvas>
	<canvas bind:this={canvas} style="image-rendering: pixelated;"></canvas>
	{#if svgUrl}
		<img bind:this={svg_preview} src={svgUrl} alt="vector output" class="bg-white" />
	{/if}
</div>
