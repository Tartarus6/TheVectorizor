<script lang="ts">
	import type { Job } from "$lib/types";
	import DownloadIcon from "./icons/DownloadIcon.svelte";
	import InfoIcon from "./icons/InfoIcon.svelte";
	import RetryIcon from "./icons/RetryIcon.svelte";
	import XIcon from "./icons/XIcon.svelte";

	interface Props {
		job: Job
		ondownload: (job: Job) => void
		onretry: (job: Job) => void
		ondelete: (job: Job) => void
	}

	let props: Props = $props();

	let show_error: boolean = $state(false);
</script>

<div
	class="flex flex-col gap-2 border-2 {props.job.status == 'done' ? 'border-accent-alt text-accent-alt' : ''}
		{props.job.status == 'processing' ? 'border-accent text-accent' : ''}
		{props.job.status == 'pending' ? 'border-accent border-dashed text-accent' : ''}
		{props.job.status == 'error' ? 'border-alt text-alt' : ''}"
>
	<div class="grid grid-cols-[1fr_auto] grid-flow-col gap-2 items-center">
		<span class="break-all">
			{props.job.file.name}
		</span>
		<div class="flex flex-row w-fit gap-1 m-1">
			{#if props.job.status === 'error'}
				<button class="size-6 text-accent border-2 border-current text-center" onclick={() => (show_error = !show_error)}>
					<InfoIcon></InfoIcon>
				</button>
				<button class="size-6 text-accent border-2 border-current text-center" onclick={() => props.onretry(props.job)}>
					<RetryIcon></RetryIcon>
				</button>
			{:else if props.job.status === 'done'}
				<button class="size-6 text-accent-alt border-2 border-current text-center" onclick={() => props.ondownload(props.job)}>
					<DownloadIcon></DownloadIcon>
				</button>
				<button class="size-6 text-accent-alt border-2 border-current text-center" onclick={() => props.onretry(props.job)}>
					<RetryIcon></RetryIcon>
				</button>
			{/if}
			<button class="size-6 text-alt border-2 border-current text-center" onclick={() => props.ondelete(props.job)}>
				<XIcon></XIcon>
			</button>
		</div>
	</div>

	{#if props.job.status === 'error'}

		{#if show_error}
			<span class="border-t-2 border-alt">{props.job.error_message}</span>
		{/if}
	{/if}
</div>
