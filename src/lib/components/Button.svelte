<script lang="ts">
	interface Props {
		label: string;
		onmousedownHandler: () => void;
		disabled: boolean;
	};

	let props: Props = $props()
</script>

<button
	onmousedown={props.onmousedownHandler}
	disabled={props.disabled}
	class="btn"
>
	<span class="bg-background p-1 btn-text">
		{props.label}
	</span>
</button>

<style>
	.btn {
		--btn-color: var(--color-accent-alt);
		border: var(--border-width) solid var(--btn-color);

		color: var(--color-accent-alt);

		background: repeating-linear-gradient(
			45deg,
			transparent,
			transparent 4px,
			var(--btn-color) 4px,
			var(--btn-color) 6px
		);
	}

	.btn:focus-within {
		--btn-color: var(--color-accent-alt-focus);
	}

	.btn:disabled {
		/* change style to show button is disabled */
		--btn-color: var(--color-alt);
		color: var(--color-alt);
	}

	.btn:not(:disabled) {
		/* change style to show button is enabled */
		cursor: pointer;
	}

	.btn-text {
		border: var(--border-width) solid var(--btn-color);
	}
</style>
