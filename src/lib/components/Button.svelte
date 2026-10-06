<script lang="ts">
	import { button_keyboard_handler } from "$lib/handlers";

	interface Props {
		label: string;
		handler: () => void;
		disabled: boolean;
		alt: boolean;
	};

	let props: Props = $props()
</script>

<button
	onmousedown={props.handler} onkeydown={(e) => {button_keyboard_handler(e, props.handler)}}
	disabled={props.disabled}
	class="filled border-2 border-current {props.alt ? 'text-alt focus-within:text-alt-focus' : 'text-accent-alt focus-within:text-accent-alt-focus'} p-2"
>
	<span class="bg-background p-1 border-2 border-current">
		{props.label}
	</span>
</button>
