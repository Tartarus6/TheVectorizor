<script lang="ts">
	import { onMount } from "svelte";
	import type { Job } from "$lib/types";
	import MiniButton from "./MiniButton.svelte";
	import XIcon from "./icons/XIcon.svelte";
	import BecomesIcon from "./icons/BecomesIcon.svelte";
	import Button from "./Button.svelte";
	import { addZoomPan } from "$lib/renderer";
	import { get_size_string } from "$lib/utils";

	interface Props {
		job: Job;
		onclose: () => void;
	}

	// TODO: fix the weird zooming behaviour
	// TODO: make controlls less ass

	let props: Props = $props();

	type Mode = "side" | "toggle";

	let dialog: HTMLDialogElement;
	let toggle_image_container: HTMLElement | undefined = $state();
	let toggle_image: HTMLImageElement | undefined = $state();

	let mode: Mode = $state("toggle");

	let side_by_side_rotated: boolean = $state(false);  // whether side-by-side comparison is rotated
	let toggle_switch: boolean = $state(false)  // whether to show the bitmap instead of the SVG

	onMount(() => {
		dialog.showModal();

		// TODO: this should be more robust
		if (toggle_image_container && toggle_image) {
			addZoomPan({container: toggle_image_container, image: toggle_image});
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
		<div class="grid grid-cols-2">
			<div class="flex flex-row gap-2">
				<Button label="Side By Side" alt={false} disabled={mode === "side"} handler={() => {mode = "side"}}></Button>
				<Button label="Zoom & Toggle" alt={false} disabled={mode === "toggle"} handler={() => {mode = "toggle"}}></Button>
			</div>
			<div class="flex flex-row gap-2 place-self-end">
				{#if mode === "side"}
					<Button label="Compare Rotate" alt={side_by_side_rotated} disabled={false} handler={() => {side_by_side_rotated = !side_by_side_rotated}}></Button>
					<!-- <MiniButton class={side_by_side_rotated ? 'text-alt' : 'text-accent-alt'} title="Rotate Comparison" handler={() => {side_by_side_rotated = !side_by_side_rotated}} disabled={false}>
						<RotateIcon></RotateIcon>
					</MiniButton> -->
				{/if}

				{#if mode === "toggle"}
					<Button label={toggle_switch ? "Switch to SVG" : "Switch to Original"} alt={toggle_switch} disabled={false} handler={() => {toggle_switch = !toggle_switch}}></Button>
					<!-- <MiniButton class={toggle_switch ? 'text-alt' : 'text-accent-alt'} title="Switch Images" handler={() => {toggle_switch = !toggle_switch}} disabled={false}>
						<RetryIcon></RetryIcon>
					</MiniButton> -->
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

		<!-- toggle -->
		<!-- {#if mode === "toggle"} -->
		<div class="border-vector h-max p-2 flex-1 overflow-hidden" hidden={mode !== "toggle"}>
		<div bind:this={toggle_image_container} hidden={mode !== "toggle"} class="h-full p-2 flex checker overflow-hidden rounded-none!">
			<img
				bind:this={toggle_image}
				src={toggle_switch ? props.job.image_url : props.job.svg_url}
				alt="original"
				draggable="false"
				class="rounded-none! select-none pointer-events-none max-h-full max-w-full will-change-transform mx-auto self-center"
				style="image-rendering: pixelated"
			/>
		</div>
		</div>
		<!-- {/if} -->

	</div>
</dialog>
