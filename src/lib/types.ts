export type Job = {
	file: File;

	image?: ImageBitmap;
	svg_blob?: Blob;

	image_url?: string;
	svg_url?: string;

	status: 'pending' | 'processing' | 'done' | 'error';
	error_message?: string;
};
