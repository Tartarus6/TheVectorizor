export function button_keyboard_handler(event: KeyboardEvent, handler: () => void): void {
	console.log(event.key)
	if (event.key === "Enter" || event.key === "Space" || event.key === " ") {
		handler()
	}
}
