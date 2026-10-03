<script lang="ts">
	import { onMount } from "svelte";
	import RestoreIcon from "./RestoreIcon.svelte";

	interface Props {
		label: string;
		variable: number | undefined;
		min: number;
		max: number;
		step: number;
		default_value: number;
	};

	let {label, variable = $bindable(), min, max, step, default_value}: Props = $props();

	onMount(() => {
		variable = default_value  // initialize the variable to its default value
	});
</script>

<div class="flex flex-col border-2 border-accent p-2">
	<div class="grid grid-cols-[auto_1fr] grid-rows-1 gap-2">
		<span>{label}:</span>
		<div class="grid grid-cols-[1fr_auto] w-full border-2 text-accent-alt border-accent-alt focus-within:border-accent-alt-focus">
			<input
				type="number"
				bind:value={variable}
				min={min}
				max={max}
				step={step}
				class="h-fit w-full outline-0 pl-1.5"
			/>
			<button class="size-5 self-center cursor-pointer text-accent-alt focus-within:text-alt-focus" onmousedown={() => {variable = default_value}}>
				<RestoreIcon></RestoreIcon>
			</button>
		</div>

	</div>
	<input type="range" bind:value={variable} min={min} max={max} step={step} />
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
		--border-color: var(--color-accent-alt);
		--border: var(--border-width) solid var(--border-color);

		--track-height: 0.2rem;
		--track-color: var(--color-accent-alt);

		--thumb-width: 0.75rem;
		--thumb-height: 1.75rem;
		--thumb-color: var(--color-background);


		appearance: none;
		background: transparent;
		width: 100%;
		margin: 0.5rem 0;
		cursor: pointer;
	}
	input[type='range']:focus {
		outline: none;
		--border-color: var(--color-accent-focus);
	}

	input[type='range']::-webkit-slider-runnable-track {
		height: var(--track-height);
		background: var(--track-color);
		/*border: var(--border);*/
	}
	input[type='range']::-moz-range-track {
		height: var(--track-height);
		background: var(--track-color);
		/*border: var(--border);*/
	}

	input[type='range']::-webkit-slider-thumb {
		appearance: none;
		width: var(--thumb-width);
		height: var(--thumb-height);
		background: var(--thumb-color);
		border: var(--border);
		border-radius: 0;
		/* center the thumb on the track (webkit doesn't do this automatically) */
		margin-top: calc((var(--track-height) - var(--thumb-height)) / 2 - 2px);
	}
	input[type='range']::-moz-range-thumb {
		width: var(--thumb-width);
		height: var(--thumb-height);
		background: var(--thumb-color);
		border: var(--border);
		border-radius: 0;
		box-sizing: border-box;
	}
</style>
