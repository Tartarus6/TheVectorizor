<script lang="ts">
	import { button_keyboard_handler } from "$lib/handlers";
	import type { Snippet } from "svelte";

	interface Props {
		handler: () => void;
		disabled: boolean;
		title: string;
		class?: string;
		children: Snippet;
	};

	let props: Props = $props()
</script>

<button disabled={props.disabled} class="{props.class} size-6 border-2 border-current" title={props.title} onkeydown={(e) => button_keyboard_handler(e, props.handler)} onmousedown={props.handler}>
	{@render props.children?.()}
</button>
