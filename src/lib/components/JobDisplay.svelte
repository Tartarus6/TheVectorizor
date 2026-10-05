<script lang="ts">
	import type { Job } from "$lib/types";
	import AnalyzeModal from "./AnalyzeModal.svelte";
	import BecomesIcon from "./icons/BecomesIcon.svelte";
	import DownloadIcon from "./icons/DownloadIcon.svelte";
	import FullscreenIcon from "./icons/FullscreenIcon.svelte";
	import InfoIcon from "./icons/InfoIcon.svelte";
	import RetryIcon from "./icons/RetryIcon.svelte";
	import RunIcon from "./icons/RunIcon.svelte";
	import XIcon from "./icons/XIcon.svelte";
	import MiniButton from "./MiniButton.svelte";

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
	let show_analyze: boolean = $state(false);
</script>

<div
	class="flex flex-col border-2 border-current {props.job.status == 'done' ? 'border-accent-alt text-accent-alt' : ''}
		{props.job.status == 'processing' ? 'text-accent' : ''}
		{props.job.status == 'pending' ? 'border-dotted text-accent rounded-none!' : ''}
		{props.job.status == 'error' ? 'text-alt' : ''}"
>
	<div class="grid grid-cols-[1fr_auto] grid-flow-col items-center p-1">
		<span class="break-all">
			{props.job.file.name}
		</span>
		<div class="flex flex-row w-fit gap-1">
			{#if props.job.status === 'error'}
				<MiniButton title="Info" class="text-accent" handler={() => (show_error = !show_error)} disabled={false}>
					<InfoIcon></InfoIcon>
				</MiniButton>
			{:else if props.job.status === 'done'}
				<MiniButton title="Analyze" class="text-accent" handler={() => (show_analyze = !show_analyze)} disabled={false}>
					<FullscreenIcon></FullscreenIcon>
				</MiniButton>
				<MiniButton title="View" class="text-accent" handler={() => (show_result = !show_result)} disabled={false}>
					<InfoIcon></InfoIcon>
				</MiniButton>
				<MiniButton title="Download" class="text-accent-alt" handler={() => props.ondownload(props.job)} disabled={false}>
					<DownloadIcon></DownloadIcon>
				</MiniButton>
				<MiniButton title="Revectorize" class="text-accent-alt" handler={() => props.onqueue(props.job)} disabled={props.working}>
					<RetryIcon></RetryIcon>
				</MiniButton>
			{:else if props.job.status === 'pending'}
				<MiniButton title="Vectorize" class="text-accent-alt" handler={() => props.onqueue(props.job)} disabled={props.working}>
					<RunIcon></RunIcon>
				</MiniButton>
			{/if}
			<MiniButton title="Vectorize" class="text-alt" handler={() => props.ondelete(props.job)} disabled={false}>
				<XIcon></XIcon>
			</MiniButton>
		</div>
	</div>

	{#if props.job.status === 'error'}
		{#if show_error}
			<span class="border-t-2 p-1 rounded-none! border-current">{props.job.error_message}</span>
		{/if}
	{:else if props.job.status === 'done'}
		{#if show_result}
			<div class="border-t-2 p-1 rounded-none! border-current grid grid-cols-[1fr_auto_1fr] gap-2 grid-flow-col">
				<img src={props.job.image_url} alt="original" class="checker rounded-none!" />
				<div class="size-10 self-center text-accent">
					<BecomesIcon></BecomesIcon>
				</div>
				<img src={props.job.svg_url} alt="svg" class="checker rounded-none!" />
			</div>
		{/if}
	{/if}
</div>

{#if show_analyze}
	<AnalyzeModal job={props.job} onclose={() => (show_analyze = false)}></AnalyzeModal>
{/if}
