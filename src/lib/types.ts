export type Job = {
	file: File;
	image?: ImageBitmap;
	svgBlob?: Blob;
	status: 'pending' | 'processing' | 'done' | 'error';
	eMessage?: string;
};
