<script lang="ts">
	import type { Snippet } from "svelte";

	interface Props {
		title: string;
		class?: string;
		children: Snippet;
	}

	let props: Props = $props();
</script>

<div class="flex flex-col border-vector border-current {props.class}">
	<span class="self-center p-1">{props.title}</span>
	<div class="p-2 border-t-vector rounded-none! border-current">
		{@render props.children()}
	</div>
</div>
