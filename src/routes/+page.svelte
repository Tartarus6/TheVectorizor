<script lang="ts">
	import { run_shader } from '$lib/shaders';
	import { optimize } from 'svgo/browser';
	import JSZip from 'jszip';

	// TODO: add a thing thatll help show what the mean shift clustering has done by either making an apng or just toggling the visibility of the 2 images, so they can be viewed on top of one another
	// TODO: figure out good ranges for the input variables (like blur radius), and maybe dont hardcode the limits
	// TODO: remember the filename of the input, and name the svg the same
	// TODO: organize complete vs. non-complete jobs so that downloading and uploading makes more sense (uploading a file, running, then uploading more files, then downloading causes the pending ones to be lost)

	type Job = {
		file: File;
		image?: ImageBitmap;
		svgBlob?: Blob;
		status: 'pending' | 'processing' | 'done' | 'error';
	};

	let jobs = $state<Job[]>([]);

	let bitmap: ImageBitmap | undefined = $state();
	let svgUrl: string | undefined = $state(); // just for visualizing

	let base_bandwidth = $state(0.05);
	/// the width and height of the tiles that the texture is broken into for processing (in order to prevent the system from hanging until jobs are complete)
	let tile_size = $state(512);
	let num_cluster_passes = $state(5);
	let num_edge_trace_passes = $state(300);
	let blur_radius = $state(2);
	let image_canvas: HTMLCanvasElement | undefined = $state();
	let blurred_canvas: HTMLCanvasElement | undefined = $state();
	let clustered_canvas: HTMLCanvasElement | undefined = $state();
	let edge_canvas: HTMLCanvasElement | undefined = $state();
	let svg_preview: HTMLImageElement | undefined = $state();
	let canvas_scale = $state(3);

	let working = $state(false);

	async function onFilesSelected(e: Event) {
		const files = Array.from((e.target as HTMLInputElement).files ?? []);

		jobs.push(
			...files.map((file) => ({
				file,
				status: 'pending'
			}))
		);
	}

	function apply_svg_display_scale(target: HTMLImageElement | undefined) {
		if (!bitmap) {
			return;
		}

		if (!target) {
			return;
		}

		target.style.width = `${bitmap.width * canvas_scale}px`;
		target.style.height = `${bitmap.height * canvas_scale}px`;
	}

	function apply_canvas_display_scale(target: HTMLCanvasElement | undefined) {
		if (!bitmap) {
			return;
		}

		if (!target) {
			return;
		}

		target.style.width = `${bitmap.width * canvas_scale}px`;
		target.style.height = `${bitmap.height * canvas_scale}px`;
		target.style.imageRendering = 'pixelated';
	}

	async function on_shader_run() {
		if (!image_canvas || !blurred_canvas || !clustered_canvas || !edge_canvas || working) {
			return;
		}

		working = true;

		for (const job of jobs) {
			// skip jobs that aren't pending
			if (job.status != 'pending') {
				continue;
			}

			job.status = 'processing';

			bitmap = await createImageBitmap(job.file);

			// debug
			image_canvas.width = bitmap.width;
			image_canvas.height = bitmap.height;
			image_canvas.getContext('2d')!.drawImage(bitmap, 0, 0);
			blurred_canvas.width = bitmap.width;
			blurred_canvas.height = bitmap.height;
			clustered_canvas.width = bitmap.width;
			clustered_canvas.height = bitmap.height;
			edge_canvas.width = bitmap.width;
			edge_canvas.height = bitmap.height;
			const blurred_ctx = blurred_canvas.getContext('webgpu');
			const clustered_ctx = clustered_canvas.getContext('webgpu');
			const edge_ctx = edge_canvas.getContext('webgpu');
			if (!blurred_ctx || !clustered_ctx || !edge_ctx) {
				alert('context didnt work');
				break;
			}

			// const [success, svg] = await run_shader
			let startTime = performance.now();
			const [success, svg] = await run_shader(
				blurred_ctx,
				clustered_ctx,
				edge_ctx,
				bitmap,
				base_bandwidth,
				tile_size,
				blur_radius,
				num_cluster_passes,
				num_edge_trace_passes
			);
			let endTime = performance.now();
			console.log(`Shader execution time: ${(endTime - startTime).toFixed(2)}ms`);

			if (success) {
				startTime = performance.now();
				const { data: optimizedSvg } = optimize(svg);
				endTime = performance.now();
				console.log(`Optimize execution time: ${(endTime - startTime).toFixed(2)}ms`);

				job.svgBlob = new Blob([optimizedSvg], { type: 'image/svg+xml' });

				job.status = 'done';

				svgUrl = URL.createObjectURL(job.svgBlob); // just for visualization
			}
		}

		// set working complete
		working = false;
	}

	async function downloadAll() {
		// if there were no jobs, or shaders currently working, skip
		if (jobs.length == 0 || working) return;

		// if there was only one job, then download that file alone, unzipped
		if (jobs.length == 1) {
			let job = jobs[0];
			if (!job.svgBlob) return;

			const name = job.file.name.replace(/\.[^.]+$/, '') + '.svg';
			downloadBlob(job.svgBlob, name);

			// clear jobs list
			jobs = [];

			return;
		}

		const zip = new JSZip();

		let some_done_jobs = false; // store whether there are any jobs that are done, otherwise itll just be an empty zip
		// for each job, get its svg and add it to the zip
		for (const job of jobs) {
			if (!job.svgBlob) continue;

			some_done_jobs = true;

			const name = job.file.name.replace(/\.[^.]+$/, '') + '.svg';

			zip.file(name, job.svgBlob);
		}

		// only download zip if there were some complete jobs
		if (some_done_jobs) {
			const blob = await zip.generateAsync({
				type: 'blob'
			});

			downloadBlob(blob, 'vectorized-images.zip');

			// clear jobs list
			jobs = [];
		}
	}

	function downloadBlob(blob: Blob, filename: string) {
		const url = URL.createObjectURL(blob);

		const a = document.createElement('a');
		a.href = url;
		a.download = filename;
		a.click();

		URL.revokeObjectURL(url);
	}

	$effect(() => {
		apply_canvas_display_scale(image_canvas);
		apply_svg_display_scale(svg_preview);
	});
</script>

<h1 class="text-center">The Vectorizor</h1>

<div class="flex w-128 flex-col gap-2 p-2">
	<div class="flex flex-col bg-slate-500 p-2">
		<span>Canvas Scale: {canvas_scale}</span>
		<input type="range" bind:value={canvas_scale} min={1} max={30} />
	</div>

	<div class="flex flex-col bg-slate-500 p-2">
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

	<div class="flex flex-col bg-slate-500 p-2">
		<div class="flex flex-row gap-2">
			<span>Tile Size:</span>
			<input
				type="number"
				bind:value={tile_size}
				min={1}
				max={bitmap ? Math.max(bitmap.width, bitmap.height) : 512}
				step={1}
				class="border-2 border-white"
			/>
		</div>
		<input
			type="range"
			bind:value={tile_size}
			min={1}
			max={bitmap ? Math.max(bitmap.width, bitmap.height) : 512}
			step={1}
		/>
	</div>

	<div class="flex flex-col bg-slate-500 p-2">
		<div class="flex flex-row gap-2">
			<span>Blur Radius:</span>
			<input
				type="number"
				bind:value={blur_radius}
				min={1}
				max={bitmap ? 50 : 512}
				step={1}
				class="border-2 border-white"
			/>
		</div>
		<input type="range" bind:value={blur_radius} min={1} max={bitmap ? 50 : 512} step={1} />
	</div>

	<div class="flex flex-col bg-slate-500 p-2">
		<span>Cluster Passes: {num_cluster_passes}</span>
		<input type="range" bind:value={num_cluster_passes} min={1} max={20} />
	</div>

	<div class="flex flex-col bg-slate-500 p-2">
		<div class="flex flex-row gap-2">
			<span>Edge Tracing Passes:</span>
			<input
				type="number"
				bind:value={num_edge_trace_passes}
				min={0}
				max={10000}
				step={1}
				class="border-2 border-white"
			/>
		</div>
		<input type="range" bind:value={num_edge_trace_passes} min={0} max={10000} step={1} />
	</div>

	<button
		onmousedown={on_shader_run}
		class="w-fit {working ? 'bg-gray-500' : 'cursor-pointer bg-purple-500'} p-2"
	>
		<span>shader pass</span>
	</button>

	<button onmousedown={downloadAll} class="w-fit cursor-pointer bg-green-500 p-2">
		<span>download svg</span>
	</button>
	<div class="flex w-fit flex-col bg-slate-600 p-2">
		<h3>appload your image here</h3>
		<input class="bg-blue-500" type="file" accept="image/*" multiple onchange={onFilesSelected} />
	</div>
	<div class="flex w-fit flex-col bg-slate-600 p-2">
		<span class="text-2xl">Jobs:</span>
		<hr />
		{#if jobs.length == 0}
			<span>No submitted jobs...</span>
		{/if}
		{#each jobs as job (job.file)}
			<span>
				{job.file.name} - {job.status}
			</span>
		{/each}
	</div>
</div>
<div class="checker flex w-fit flex-col gap-2">
	{#if svgUrl}
		<img bind:this={svg_preview} src={svgUrl} alt="vector output" class="" />
	{/if}
	<canvas bind:this={edge_canvas} style="image-rendering: pixelated;"></canvas>
	<canvas bind:this={clustered_canvas} style="image-rendering: pixelated;"></canvas>
	<canvas bind:this={blurred_canvas} style="image-rendering: pixelated;"></canvas>
	<canvas bind:this={image_canvas} style="image-rendering: pixelated;"></canvas>
</div>

<style>
	canvas,
	img {
		--size: 30px;
		--color-1: #ccc;
		--color-2: #bbb;
		background: conic-gradient(
			var(--color-1) 90deg,
			var(--color-2) 90deg 180deg,
			var(--color-1) 180deg 270deg,
			var(--color-2) 270deg
		);
		background-repeat: repeat;
		background-size: var(--size) var(--size);
		background-position: top left;
	}
</style>
