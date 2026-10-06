<script lang="ts">
	import { onMount } from "svelte";
	import type { Job } from "$lib/types";
	import MiniButton from "./MiniButton.svelte";
	import XIcon from "./icons/XIcon.svelte";
	import BecomesIcon from "./icons/BecomesIcon.svelte";
	import Button from "./Button.svelte";
	import RotateIcon from "./icons/RotateIcon.svelte";
	import { addZoomPan } from "$lib/renderer";
	import { get_size_string } from "$lib/utils";
	import RetryIcon from "./icons/RetryIcon.svelte";

	interface Props {
		job: Job;
		onclose: () => void;
	}

	// TODO: fix the weird zooming behaviour

	let props: Props = $props();

	type Mode = "side" | "toggle";

	let dialog: HTMLDialogElement;
	let swipe_image_container: HTMLElement | undefined = $state();
	let swipe_image: HTMLImageElement | undefined = $state();

	let mode: Mode = $state("side");

	let side_by_side_rotated: boolean = $state(false);  // whether side-by-side comparison is rotated
	let toggle_switch: boolean = $state(false)  // whether to show the bitmap instead of the SVG

	onMount(() => {
		dialog.showModal();

		// TODO: this should be more robust
		if (swipe_image_container && swipe_image) {
			addZoomPan({container: swipe_image_container, image: swipe_image});
		}
	});
</script>

{#snippet original()}
	<img
		src={props.job.image_url}
		alt="original"
		draggable="false"
		class="rounded-none! select-none checker self-center justify-self-center"
		style="image-rendering: pixelated"
	/>
{/snippet}

{#snippet svg(blend: boolean)}
	<img
		src={props.job.svg_url}
		alt="svg"
		draggable="false"
		class="rounded-none! select-none checker self-center justify-self-center"
		style="{blend ? '; mix-blend-mode: difference' : ''}"
	/>
{/snippet}

<dialog
	bind:this={dialog}
	onclose={props.onclose}
	class="fixed inset-0 m-0 h-dvh max-h-none w-screen max-w-none bg-background p-2 text-accent grid grid-cols-1 grid-rows-1"
>
	<!-- NOTE: col-start-1 and row-start-1 are used so that the close button does not affect the positioning of the rest of the elements -->
	<MiniButton handler={props.onclose} disabled={false} title="Close" class="ml-auto mr-0 col-start-1 row-start-1 z-0">
		<XIcon></XIcon>
	</MiniButton>
	<div class="flex flex-col col-start-1 row-start-1 gap-4 h-full {mode === "toggle" ? 'overflow-clip' : ''}">
		<span>Analyzor</span>

		<!-- buttons -->
		<div>
			<div class="flex flex-row gap-2 place-self-center">
				<Button label="Side By Side" alt={false} disabled={mode === "side"} handler={() => {mode = "side"}}></Button>
				<Button label="Toggle" alt={false} disabled={mode === "toggle"} handler={() => {mode = "toggle"}}></Button>

				{#if mode === "side"}
					<MiniButton class={side_by_side_rotated ? 'text-alt' : 'text-accent-alt'} title="Rotate Comparison" handler={() => {side_by_side_rotated = !side_by_side_rotated}} disabled={false}>
						<RotateIcon></RotateIcon>
					</MiniButton>
				{/if}

				{#if mode === "toggle"}
					<MiniButton class={toggle_switch ? 'text-alt' : 'text-accent-alt'} title="Switch Images" handler={() => {toggle_switch = !toggle_switch}} disabled={false}>
						<RetryIcon></RetryIcon>
					</MiniButton>
				{/if}
			</div>
		</div>

		<!-- side by side -->
		{#if mode === "side" && props.job.svg_blob !== undefined}
			<div class="grid gap-2 {side_by_side_rotated ? 'grid-rows-[1fr_auto_1fr]' : 'grid-cols-[1fr_auto_1fr]'}">
				<div class="{!side_by_side_rotated ? 'place-self-end' : 'justify-self-center'} grid grid-rows-[auto_1fr] place-items-center">
					<span>{get_size_string(props.job.file.size)}</span>
					{@render original()}
				</div>
				<div class="size-10 self-center justify-self-center {side_by_side_rotated ? 'rotate-90' : ''}">
					<BecomesIcon></BecomesIcon>
				</div>
				<div class="{!side_by_side_rotated ? 'place-self-start' : 'justify-self-center'} grid grid-rows-[auto_1fr] place-items-center">
					<span>{get_size_string(props.job.svg_blob.size)}</span>
					{@render svg(false)}
				</div>
			</div>
		{/if}

		<!-- swipe -->
		<!-- {#if mode === "swipe"} -->
		<div bind:this={swipe_image_container} hidden={mode !== "toggle"} class="border-10 h-max p-2 flex-1 checker overflow-hidden">
			<img
				bind:this={swipe_image}
				src={toggle_switch ? props.job.image_url : props.job.svg_url}
				alt="original"
				draggable="false"
				class="rounded-none! select-none pointer-events-none max-h-full max-w-full will-change-transform mx-auto my-auto"
				style="image-rendering: pixelated"
			/>
		</div>
		<!-- {/if} -->

	</div>
</dialog>
