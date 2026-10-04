export type Job = {
	file: File;
	image?: ImageBitmap;
	svg_blob?: Blob;
	status: 'pending' | 'processing' | 'done' | 'error';
	error_message?: string;
};
