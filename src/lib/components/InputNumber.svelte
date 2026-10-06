<script lang="ts">
	import { onMount } from "svelte";
	import RestoreIcon from "$lib/components/icons/RestoreIcon.svelte";
	import MiniButton from "./MiniButton.svelte";

	interface Props {
		label: string;
		variable: number | undefined;
		min: number;
		max: number;
		step: number;
		default_value: number;
		description: string;
	};

	let {label, variable = $bindable(), min, max, step, default_value, description}: Props = $props();

	onMount(() => {
		variable = default_value  // initialize the variable to its default value
	});
</script>

<div class="flex flex-col border-2 border-accent">
	<div class="grid grid-cols-[auto_1fr] grid-rows-1 gap-2 items-center p-2">
		<span>{label}:</span>
		<div class="p-1 grid grid-cols-[1fr_auto] w-full border-2 text-accent-alt border-accent-alt focus-within:border-accent-alt-focus">
			<input
				type="number"
				bind:value={variable}
				min={min}
				max={max}
				step={step}
				class="h-fit w-full outline-none"
			/>
			<MiniButton handler={() => {variable = default_value}} title="Reset" disabled={false}>
				<RestoreIcon></RestoreIcon>
			</MiniButton>
		</div>

	</div>
	<input type="range" bind:value={variable} min={min} max={max} step={step} />
	<div class="w-full py-1 px-2 border-t-2 border-current">
		<span>{description}</span>
	</div>
</div>

<style>
	/* --- Number Input --- */
	input[type='number'] {
		appearance: textfield; /* hides spinners in Firefox */
	}
	input[type='number']::-webkit-outer-spin-button,
	input[type='number']::-webkit-inner-spin-button {
		appearance: none;
		margin: 0;
	}

	/* --- Range Input --- */
	input[type='range'] {
		color: var(--color-accent-alt);

		--border: var(--border-width) solid currentColor;

		--track-height: 0.2rem;

		--thumb-width: 0.75rem;
		--thumb-height: 1.75rem;


		appearance: none;
		background: transparent;
		width: 100%;
		margin: 0.5rem 0;
		cursor: pointer;
	}
	input[type='range']:focus {
		outline: none;
		color: var(--color-accent-focus);
	}

	input[type='range']::-webkit-slider-runnable-track {
		height: calc(var(--track-height) / 2);
		background: currentColor;
	}
	input[type='range']::-moz-range-track {
		height: calc(var(--track-height) / 2);
		background: currentColor;
		border-radius: var(--border-width);
	}

	input[type='range']::-webkit-slider-thumb {
		appearance: none;
		width: var(--thumb-width);
		height: var(--thumb-height);
		background: var(--color-background);
		border: var(--border);
		border-radius: var(--border-width);
		/* center the thumb on the track (webkit doesn't do this automatically) */
		margin-top: calc((var(--track-height) - var(--thumb-height)) / 2 - 2px);

		background: repeating-linear-gradient(
			45deg,
			var(--color-background),
			var(--color-background) 4px,
			currentColor 4px,
			currentColor 6px
		);
	}
	input[type='range']::-moz-range-thumb {
		width: var(--thumb-width);
		height: var(--thumb-height);
		background: var(--color-background);
		border: var(--border);
		border-radius: var(--border-width);
		box-sizing: border-box;

		background: repeating-linear-gradient(
			45deg,
			var(--color-background),
			var(--color-background) 4px,
			currentColor 4px,
			currentColor 6px
		);
	}
</style>
