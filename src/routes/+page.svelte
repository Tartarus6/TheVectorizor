<script lang="ts">
	import { onDestroy, onMount } from 'svelte';
	import { run_shader } from '$lib/shaders';
	import { optimize } from 'svgo/browser';
	import JSZip from 'jszip';
	import InputNumber from '$lib/components/InputNumber.svelte';
	import Button from '$lib/components/Button.svelte';
	import type { Job } from '$lib/types';
	import JobDisplay from '$lib/components/JobDisplay.svelte';
	import VectorizorIcon from '$lib/components/icons/VectorizorIcon.svelte';
	import TitledSection from '$lib/components/TitledSection.svelte';

	// DONE: retry, cancel, and download buttons on each job
	// DONE: add a job result display (maybe show for all jobs, or store for each job and display on click) comparison between input bitmap and output svg (visual difference and file size)
	// DONE: add a warning or error or something to the UI that shows up when there's no WebGPU
	// TODO: add ability to re-vectorize after changing settings or whatever (still needs to be done for main vectorize button)
	// TODO: (accessibility) make buttons work right with keyboard navigation (some still need work)
	// TODO: add ability to halt vectorization

	// DONE: (style) element background colors by doing a diagnoal zigzag line, as it would be on a vector display
	// DONE: (style) make it more obvious when a button is disabled. just turnign it red isnt intuitive enough
	// DONE: (style) apply a tiny round to EVERYTHING to better the circular shape a CRT beam lights up
	// DONE: (style) set global stroke width in layout.css (currently it's 2px), it should also be switched to use rem rather than px

	let webgpu_available: boolean | undefined = $state();

	let jobs = $state<Job[]>([]);
	let working: boolean = $state(false);

	let show_debug: boolean = $state(false);
	let show_settings: boolean = $state(false);

	// Derived state for UI
	let pending_jobs = $derived(jobs.filter((j) => j.status === 'pending'));
	let done_jobs = $derived(jobs.filter((j) => j.status === 'done'));
	let has_pending = $derived(pending_jobs.length > 0);
	let has_done = $derived(done_jobs.length > 0);
	let can_submit = $derived(has_pending && !working);
	let can_download = $derived(has_done && !working);

	let svgUrl: string | undefined = $state(); // just for visualizing

	// variables
	let base_bandwidth: number | undefined = $state();
	let num_cluster_passes: number | undefined = $state();
	let num_edge_trace_passes: number | undefined = $state();

	// canvases
	let image_canvas: HTMLCanvasElement | undefined = $state();
	let clustered_canvas: HTMLCanvasElement | undefined = $state();
	let init_edge_canvas: HTMLCanvasElement | undefined = $state();
	let edge_canvas: HTMLCanvasElement | undefined = $state();

	onMount(() => {
		// evilllll global event listener
		document!.addEventListener('paste', on_image_pasted);

		// check for WebGPU availability
		check_webgpu_availability();

		// remove global event listener
		return () => document.removeEventListener('paste', on_image_pasted);
	});

	onDestroy(() => {
		// Clean up jobs, important to revoke job image URLs
		for (const job of jobs) {
			delete_job(job)
		}
	})

	function check_webgpu_availability() {
		if (navigator.gpu) {
			webgpu_available = true
		}
	}

	function add_files(files: File[]) {
		jobs.push(
			...files.map(
				(file): Job => ({
					file,
					status: 'pending'
				})
			)
		);
	}

	function on_image_pasted(e: ClipboardEvent) {
		const file = Array.from(e.clipboardData?.files ?? [])[0];

		if (!file) return;

		add_files([file]);
	}

	function on_files_selected(e: Event) {
		const files = Array.from((e.target as HTMLInputElement).files ?? []);

		// blocking the svg as it get uploaded
		// const nonvector = files.filter((e) => {
		// 	const v = /svg|ai|esl/.test(e.type);
		// 	if (v) {
		// 		console.error(e.type + ' is of vector image type');
		// 		alert('cannot vectorize vector image type');
		// 	}
		// 	return !v;
		// });
		// addFiles(nonvector);

		add_files(files);
	}

	// TODO: rename this to make it obviousely different than the do_job() function (or whatever it's been renamed to)
	// Helper: process a single job
	async function process_job(job: Job) {
		try {
			job.status = 'processing';

			// make sure variables are defined
			if (!base_bandwidth || !num_cluster_passes || !num_edge_trace_passes) {
				throw new Error("Variables aren't defined");
			}

			// check if file is of vector type
			if (/svg|ai|esl/.test(job.file.type)) {
				throw new Error(job.file.type + ' is of vector image type');
			}
			const bitmap = await createImageBitmap(job.file);

			// Set up canvases for this job
			if (!image_canvas || !clustered_canvas || !init_edge_canvas || !edge_canvas) {
				throw new Error('Canvas elements missing');
			}

			image_canvas.width = bitmap.width;
			image_canvas.height = bitmap.height;
			image_canvas.getContext('2d')!.drawImage(bitmap, 0, 0);

			clustered_canvas.width = bitmap.width;
			clustered_canvas.height = bitmap.height;
			init_edge_canvas.width = bitmap.width;
			init_edge_canvas.height = bitmap.height;
			edge_canvas.width = bitmap.width;
			edge_canvas.height = bitmap.height;

			const clustered_ctx = clustered_canvas.getContext('webgpu');
			const init_edge_ctx = init_edge_canvas.getContext('webgpu');
			const edge_ctx = edge_canvas.getContext('webgpu');
			if (!clustered_ctx || !init_edge_ctx || !edge_ctx) {
				throw new Error('WebGPU context not available');
			}

			let start_time = performance.now();
			const [success, svg] = await run_shader(
				clustered_ctx,
				init_edge_ctx,
				edge_ctx,
				bitmap,
				base_bandwidth,
				num_cluster_passes,
				num_edge_trace_passes
			);
			let end_time = performance.now();
			console.log(`Shader execution time: ${(end_time - start_time).toFixed(2)}ms`);

			if (!success) throw new Error('Shader failed');

			start_time = performance.now();
			const { data: optimizedSvg } = optimize(svg);
			end_time = performance.now();
			console.log(`Optimize execution time: ${(end_time - start_time).toFixed(2)}ms`);

			job.svg_blob = new Blob([optimizedSvg], { type: 'image/svg+xml' });
			job.status = 'done';

			// Update debug preview (clean up old URL)
			if (svgUrl) URL.revokeObjectURL(svgUrl);
			svgUrl = URL.createObjectURL(job.svg_blob);

			// Update stored job svg url
			if (job.svg_url) URL.revokeObjectURL(job.svg_url);
			job.svg_url = URL.createObjectURL(job.svg_blob);

			// Store job bitmap url if it hadn't already been made
			if (!job.image_url) {
				job.image_url = image_canvas.toDataURL()
			}
		} catch (err) {
			console.error(err);
			job.status = 'error';
			const errMessage = err as Error;
			job.error_message = errMessage.message;
		}
	}

	async function on_vectorize() {
		if (!has_pending || working) return;

		working = true;
		// Take a snapshot of only pending jobs at this moment
		const pendingSnapshot = jobs.filter((j) => j.status === 'pending');

		for (const job of pendingSnapshot) {
			await process_job(job);
			// Give UI a chance to update between jobs
			await new Promise((resolve) => setTimeout(resolve, 0));
		}

		working = false;
	}

	async function download_all() {
		if (!has_done || working) return;

		const completed = jobs.filter((j) => j.status === 'done');
		if (completed.length === 0) return;

		// Single job: download as plain SVG
		if (completed.length === 1) {
			const job = completed[0];
			if (!job.svg_blob) return;
			const name = job.file.name.replace(/\.[^.]+$/, '') + '.svg';
			downloadBlob(job.svg_blob, name);
		} else {
			// Multiple jobs: create zip
			const zip = new JSZip();
			for (const job of completed) {
				if (!job.svg_blob) continue;
				const name = job.file.name.replace(/\.[^.]+$/, '') + '.svg';
				zip.file(name, job.svg_blob);
			}
			const blob = await zip.generateAsync({ type: 'blob' });
			downloadBlob(blob, 'vectorized-images.zip');
		}

		// Remove only completed jobs, keep pending/error ones
		done_jobs = jobs.filter((j) => j.status === 'done');
		for (const job of done_jobs) {
			delete_job(job);
		}
	}

	function download_job(job: Job) {
		if (!job.svg_blob) return;
		downloadBlob(job.svg_blob, job.file.name.replace(/\.[^.]+$/, '') + '.svg');
	}

	// Queue a job; it runs on the next "Vectorize" so current settings are used
	async function queue_job(job: Job) {
		if (job.status === 'processing') return;
		if (working) return;

		job.status = 'pending';
		job.svg_blob = undefined;
		job.svg_url = undefined;
		job.image_url = undefined;
		job.error_message = undefined;

		working = true;

		await process_job(job);

		working = false;
	}

	function delete_job(job: Job) {
		// the job is mid-run and can't be cancelled
		if (job.status === 'processing') return;

		// clean up urls
		if (job.image_url) URL.revokeObjectURL(job.image_url);
		if (job.svg_url) URL.revokeObjectURL(job.svg_url);

		jobs = jobs.filter((j) => j.file !== job.file);
	}

	function downloadBlob(blob: Blob, filename: string) {
		const url = URL.createObjectURL(blob);

		const a = document.createElement('a');
		a.href = url;
		a.download = filename;
		a.click();

		URL.revokeObjectURL(url);
	}
</script>

<div class="w-full flex flex-col pb-8 place-items-center p-4">
	<div class="max-w-128 w-full">
		<VectorizorIcon></VectorizorIcon>
	</div>
	<span class="text-center">Vectorize your images entirely localy, with the power of WebGPU!</span>
</div>

<div class="grid grid-cols-1 {show_settings ? 'lg:grid-cols-2 lg:max-w-384' : ''} items-start w-full max-w-192 mx-auto gap-16 pt-2 pb-8 px-8">
	<section class="flex flex-col gap-4">
		<!-- No WebGPU Warning -->
		{#if webgpu_available === false}
			<TitledSection title="WARNING: WebGPU not Available!" class="text-alt">
				<div class="flex flex-col gap-4">
					<span>This website depends entirely on WebGPU to run. WebGPU is still experimental, so some browsers dont have it enabled by default, and it's hard (or impossible) to enable it on some browsers as well. See if you can enable it, before you can use this website. Just reload the page to check. This warning will be gone if WebGPU is available.</span>
				</div>
			</TitledSection>
		{/if}

		<!-- What's this For? -->
		<TitledSection title="What's this For?" class="text-accent-alt">
			<span>This website is a tool to convert bitmap images (JPG, PNG, etc.) into SVGs. This tool is built to handle simple graphics, such as logos, but feel free to try other images.</span>
		</TitledSection>

		<!-- File Chooser -->
		<div class="filled relative flex flex-col items-center gap-2 p-4 text-accent focus-within:text-accent-focus border-vector border-current">
			<div class="p-1 flex flex-col items-center gap-2 bg-background border-vector border-current">
				<span class="font-semibold">Add Images</span>
				<span>Click or drag images here</span>
				<span>or paste anywhere</span>
				<span>Multiple images supported</span>
			</div>

			<input
				type="file"
				accept="image/*"
				multiple
				onchange={on_files_selected}
				class="absolute inset-0 cursor-pointer opacity-0"
			/>
		</div>

		<!-- Buttons -->
		<div class="flex flex-col gap-2 p-1 text-accent-alt border-current border-vector">
			<Button handler={on_vectorize} disabled={!can_submit} label="Vectorize" alt={false}></Button>
			<Button handler={download_all} disabled={!can_download} label={done_jobs.length > 1 ? 'Download ZIP' : 'Download SVG'} alt={false}></Button>
		</div>

		<!-- Jobs -->
		<TitledSection title="Jobs">
			<div class="flex flex-col gap-2">
				{#if jobs.length == 0}
					<div class="text-alt flex flex-col">
						<span>Add images to Vectorize...</span>
						<span></span>
					</div>
				{/if}
				{#each jobs as job (job.file)}
					<JobDisplay job={job} ondownload={download_job} onqueue={queue_job} ondelete={delete_job} working={working}></JobDisplay>
				{/each}
			</div>
		</TitledSection>

		<!-- Fun Fact -->
		<TitledSection title="Fun Fact!" class="text-accent-alt">
			<span>This website makes no use of machine learning in order to Vectorize your images! Instead, it just uses a ton of classical image processing techniques. You can get a peek into how it works by looking at the debug section at the bottom of the settings below.</span>
		</TitledSection>
	</section>

	<section class="flex flex-col gap-4">
		<!-- Show/Hide Settings Button -->
		<Button label={show_settings ? 'Hide Settings' : "Show Settings"} alt={show_settings} disabled={false} handler={() => {show_settings = !show_settings}}></Button>

		<div class="contents" hidden={!show_settings}>
			<!-- Settings -->
			<div class="p-1 border-vector border-current flex flex-col gap-2">
				<InputNumber label="Base Bandwidth" bind:variable={base_bandwidth} min={0} max={1} step={0.0001} default_value={0.05} description="Increasing this value makes color averaging more aggressive. Colors that are farther apart will be grouped together."></InputNumber>
				<InputNumber label="Cluster Passes" bind:variable={num_cluster_passes} min={1} max={20} step={1} default_value={5} description="This is the number of color clustering passes. Increasing this can help if the output colors you are getting aren't accurate enough."></InputNumber>
				<InputNumber label="Edge Tracing Passes" bind:variable={num_edge_trace_passes} min={0} max={10000} step={1} default_value={300} description="This is the number of edge tracing passes. If certain elements in your image aren't ending up in the output, increasing this *might* help."></InputNumber>
			</div>

			<!-- Known Issues -->
			<TitledSection title="Known Issues" class="text-alt">
				<div class="flex flex-col gap-4">
					<span>(Won't Fix) This won't work if you don't have WebGPU enabled</span>
					<span>(Will Fix) Transparency levels don't add up correctly. So sometimes outputs will be less transparent on parts than they should be.</span>
					<span>(Will Fix) Transparent holes do not get drawn (same cause as previous issue). It just draws a transparent shape on top of a filled shape, but does not cut a hole through.</span>
					<span>(Might Fix) Textured images and graphics that use textured lines (like pencil) may struggle.</span>
					<span>(Might Fix) Gradients cannot be properly represented, and can mess up edge detection.</span>
					<span>(Might Fix) Sometimes, edges near the border of the image get all weird.</span>
				</div>
			</TitledSection>

			<!-- Debug -->
			<TitledSection title="Debug" class="text-alt">
				<div class="flex flex-col gap-2">
					<Button label={show_debug ? 'Hide Debug' : 'Show Debug'} disabled={false} handler={() => {show_debug = !show_debug}} alt={show_debug}></Button>
					<div class="contents" hidden={!show_debug}>
						{#if svgUrl}
							<img src={svgUrl} alt="vector output" class="checker" />
						{/if}
						<canvas bind:this={edge_canvas} style="image-rendering: pixelated;" class="checker rounded-none!"></canvas>
						<canvas bind:this={init_edge_canvas} style="image-rendering: pixelated;" class="checker rounded-none!"></canvas>
						<canvas bind:this={clustered_canvas} style="image-rendering: pixelated;" class="checker rounded-none!"></canvas>
						<canvas bind:this={image_canvas} style="image-rendering: pixelated;" class="checker rounded-none!"></canvas>
					</div>
				</div>
			</TitledSection>
		</div>
	</section>
</div>
