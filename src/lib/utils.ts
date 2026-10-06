export function get_size_string(size_bytes: number): string {
	const sizes = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];

	// get index of size string
	// safely return index 0 in case input is 0 (can't take log(0))
	const i = size_bytes === 0 ? 0 : parseInt(Math.floor(Math.log(size_bytes) / Math.log(1024)).toString());

	return Math.round(size_bytes / Math.pow(1024, i)) + ' ' + sizes[i];
}
