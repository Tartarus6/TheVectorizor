<script lang="ts">
	interface Props {
		label: string;
		onmousedown_handler: () => void;
		disabled: boolean;
		alt: boolean;
	};

	let props: Props = $props()
</script>

<button
	onmousedown={props.onmousedown_handler}
	disabled={props.disabled}
	class="filled border-2 border-current {props.alt ? 'text-alt focus-within:text-alt-focus' : 'text-accent-alt focus-within:text-accent-alt-focus'} disabled:opacity-40 p-2"
>
	<span class="bg-background p-1 border-2 border-current">
		{props.label}
	</span>
</button>
