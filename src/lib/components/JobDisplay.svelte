<script lang="ts">
	import type { Job } from "$lib/types";
	import BecomesIcon from "./icons/BecomesIcon.svelte";
	import DownloadIcon from "./icons/DownloadIcon.svelte";
	import FullscreenIcon from "./icons/FullscreenIcon.svelte";
	import InfoIcon from "./icons/InfoIcon.svelte";
	import RetryIcon from "./icons/RetryIcon.svelte";
	import RunIcon from "./icons/RunIcon.svelte";
	import XIcon from "./icons/XIcon.svelte";

	interface Props {
		job: Job;
		ondownload: (job: Job) => void;
		onqueue: (job: Job) => void;
		ondelete: (job: Job) => void;
		working: boolean;
	}

	let props: Props = $props();

	let show_error: boolean = $state(false);
	let show_result: boolean = $state(false);
</script>

<div
	class="flex flex-col border-2 border-current {props.job.status == 'done' ? 'border-accent-alt text-accent-alt' : ''}
		{props.job.status == 'processing' ? 'text-accent' : ''}
		{props.job.status == 'pending' ? 'border-dotted text-accent rounded-none!' : ''}
		{props.job.status == 'error' ? 'text-alt' : ''}"
>
	<div class="grid grid-cols-[1fr_auto] grid-flow-col items-center p-2">
		<span class="break-all">
			{props.job.file.name}
		</span>
		<div class="flex flex-row w-fit gap-1">
			{#if props.job.status === 'error'}
				<button class="size-6 text-accent border-2 border-current" title="Info" onclick={() => (show_error = !show_error)}>
					<InfoIcon></InfoIcon>
				</button>
			{:else if props.job.status === 'done'}
				<!-- <button class="size-6 text-accent border-2 border-current" title="Analyze" onclick={() => (show_result = !show_result)}>
					<FullscreenIcon></FullscreenIcon>
				</button> -->
				<button class="size-6 text-accent border-2 border-current" title="View" onclick={() => (show_result = !show_result)}>
					<InfoIcon></InfoIcon>
				</button>
				<button class="size-6 text-accent-alt border-2 border-current" title="Download" onclick={() => props.ondownload(props.job)}>
					<DownloadIcon></DownloadIcon>
				</button>
				<button disabled={props.working} class="size-6 text-accent-alt border-2 border-current" title="Revectorize" onclick={() => props.onqueue(props.job)}>
					<RetryIcon></RetryIcon>
				</button>
			{:else if props.job.status === 'pending'}
				<button disabled={props.working} class="size-6 text-accent-alt border-2 border-current" title="Vectorize" onclick={() => props.onqueue(props.job)}>
					<RunIcon></RunIcon>
				</button>
			{/if}
			<button class="size-6 text-alt border-2 border-current" title="Clear" onclick={() => props.ondelete(props.job)}>
				<XIcon></XIcon>
			</button>
		</div>
	</div>

	{#if props.job.status === 'error'}
		{#if show_error}
			<span class="border-t-2 p-2 rounded-none! border-current">{props.job.error_message}</span>
		{/if}
	{:else if props.job.status === 'done'}
		{#if show_result}
			<div class="border-t-2 p-2 rounded-none! border-current grid grid-cols-[1fr_auto_1fr] gap-2 grid-flow-col">
				<img src={props.job.image_url} alt="original" class="checker rounded-none!" />
				<div class="size-10 self-center text-accent">
					<BecomesIcon></BecomesIcon>
				</div>
				<img src={props.job.svg_url} alt="svg" class="checker rounded-none!" />
			</div>
		{/if}
	{/if}
</div>
