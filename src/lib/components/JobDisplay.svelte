<script lang="ts">
	import type { Job } from "$lib/types";

	interface Props {
		job: Job
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
	<div class="flex flex-row gap-2">
		<span>
			{props.job.file.name} - {props.job.status}
		</span>
		{#if props.job.status === 'error'}
			<button class="float-right size-6 cursor-pointer {show_error ? 'rotate-90' : ''}" onclick={() => (show_error = !show_error)}>></button>
		{/if}
	</div>

	{#if props.job.status === 'error'}

		{#if show_error}
			<span class="border-t-2 border-alt">{props.job.eMessage}</span>
		{/if}
	{/if}
</div>
