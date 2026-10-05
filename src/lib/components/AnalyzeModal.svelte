<script lang="ts">
	import { onMount } from "svelte";
	import type { Job } from "$lib/types";
	import MiniButton from "./MiniButton.svelte";
	import XIcon from "./icons/XIcon.svelte";

	interface Props {
		job: Job;
		onclose: () => void;
	}

	let props: Props = $props();

	type Mode = "side" | "swipe";

	let dialog: HTMLDialogElement;
	let mode: Mode = $state("side");
	let split: number = $state(50); // swipe divider, % of viewport width
	let cursor: { x: number; y: number } | undefined = $state();

	// content size in image pixels; ImageBitmap if we have it, otherwise from the <img> load
	let width: number = $state(0);
	let height: number = $state(0);

	onMount(() => {
		width = props.job.image?.width ?? 0;
		height = props.job.image?.height ?? 0;
		dialog.showModal();
	});

	function layer_style(): string {
		return `width:${width * pz.scale}px;height:${height * pz.scale}px;transform:translate(${pz.x}px,${pz.y}px)`;
	}

	function onkeydown(e: KeyboardEvent) {
		const step = 60;
		switch (e.key) {
			case "+":
			case "=":
				// pz.zoomCenter(1.25);
				break;
			case "-":
				// pz.zoomCenter(1 / 1.25);
				break;
			case "0":
				// pz.fit();
				break;
			case "1":
				// pz.actual();
				break;
			case "ArrowLeft":
				// pz.panBy(step, 0);
				break;
			case "ArrowRight":
				// pz.panBy(-step, 0);
				break;
			case "ArrowUp":
				// pz.panBy(0, step);
				break;
			case "ArrowDown":
				// pz.panBy(0, -step);
				break;
			default:
				return;
		}
		e.preventDefault();
	}

	function onmove(e: PointerEvent) {
		const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
		// const ix = Math.floor((e.clientX - r.left - pz.x) / pz.scale);
		// const iy = Math.floor((e.clientY - r.top - pz.y) / pz.scale);
		// cursor = ix >= 0 && iy >= 0 && ix < width && iy < height ? { x: ix, y: iy } : undefined;
	}

	function drag_divider(e: PointerEvent) {
		e.stopPropagation(); // don't start a pan
		const handle = e.currentTarget as HTMLElement;
		handle.setPointerCapture(e.pointerId);
		const rect = handle.parentElement!.getBoundingClientRect();
		const move = (ev: PointerEvent) => {
			split = Math.min(100, Math.max(0, ((ev.clientX - rect.left) / rect.width) * 100));
		};
		const up = () => {
			handle.removeEventListener("pointermove", move);
			handle.removeEventListener("pointerup", up);
		};
		handle.addEventListener("pointermove", move);
		handle.addEventListener("pointerup", up);
	}
</script>

{#snippet original()}
	<img
		src={props.job.image_url}
		alt="original"
		draggable="false"
		class="absolute top-0 left-0 max-w-none rounded-none! select-none"
		style="{layer_style()}; image-rendering: pixelated}"
		onload={(e) => {
			const img = e.currentTarget as HTMLImageElement;
			if (!width) {
				width = img.naturalWidth;
				height = img.naturalHeight;
			}
		}}
	/>
{/snippet}

{#snippet svg(blend: boolean)}
	<img
		src={props.job.svg_url}
		alt="svg"
		draggable="false"
		class="absolute top-0 left-0 max-w-none rounded-none! select-none"
		style="{layer_style()}{blend ? '; mix-blend-mode: difference' : ''}"
	/>
{/snippet}

<dialog
	bind:this={dialog}
	{onkeydown}
	onclose={props.onclose}
	class="fixed inset-0 m-0 h-dvh max-h-none w-screen max-w-none bg-background p-2 text-accent grid grid-cols-1 grid-rows-1"
>
	<!-- NOTE: col-start-1 and row-start-1 are used so that the close button does not affect the positioning of the rest of the elements -->
	<MiniButton handler={props.onclose} disabled={false} title="Close" class="ml-auto mr-0 col-start-1 row-start-1 z-0">
		<XIcon></XIcon>
	</MiniButton>
	<div class="flex flex-col col-start-1 row-start-1">
		Tada!
	</div>
</dialog>
