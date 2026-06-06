/*
 ? This code is mostly LLM-written, and is just for testing. it will not be the final thing.
 ? The final code will probably be written for WASM so that it can be a lot faster.
 * What this file does is take in specially formatted textures, and turn them into an SVG.
*/

type EdgePoint = {
	x: number;
	y: number;
};

type FacePath = {
	points: EdgePoint[];
	color: string;
	minPixelIndex: number;
};

const INVALID_EDGE = 0xffffffff;
const EDGE_DIRS = 8;

export async function faceBuffersToSvg(
	device: GPUDevice,
	gradTexture: GPUTexture,
	nextEdgeBuffer: GPUBuffer,
	faceIdBuffer: GPUBuffer,
	edgeColorBuffer: GPUBuffer,
	width: number,
	height: number
): Promise<string> {
	const edgeCount = width * height * EDGE_DIRS;

	const [gradData, nextEdges, faceIds, edgeColors] = await Promise.all([
		readRgba16FloatTexture(device, gradTexture, width, height),
		readUint32Buffer(device, nextEdgeBuffer, edgeCount),
		readUint32Buffer(device, faceIdBuffer, edgeCount),
		readFloat32Buffer(device, edgeColorBuffer, edgeCount * 4)
	]);

	const subpixelPoints: EdgePoint[] = new Array(width * height);
	for (let index = 0; index < width * height; index += 1) {
		const base = index * 4;
		const theta = gradData[base + 1];
		const offset = gradData[base + 2];
		subpixelPoints[index] = {
			x: (index % width) + 0.5 + Math.cos(theta) * offset,
			y: Math.floor(index / width) + 0.5 + Math.sin(theta) * offset
		};
	}

	type FaceAccumulator = {
		startEdge: number;
		minPixelIndex: number;
		colorSum: [number, number, number, number];
		count: number;
	};

	const faces = new Map<number, FaceAccumulator>();

	for (let edgeIndex = 0; edgeIndex < edgeCount; edgeIndex += 1) {
		const faceId = faceIds[edgeIndex];
		if (faceId === INVALID_EDGE) {
			continue;
		}

		if (nextEdges[edgeIndex] === INVALID_EDGE) {
			continue;
		}

		let entry = faces.get(faceId);
		if (!entry) {
			entry = {
				startEdge: edgeIndex,
				minPixelIndex: Math.floor(edgeIndex / EDGE_DIRS),
				colorSum: [0, 0, 0, 0],
				count: 0
			};
			faces.set(faceId, entry);
		}

		const pixelIndex = Math.floor(edgeIndex / EDGE_DIRS);
		if (pixelIndex < entry.minPixelIndex) {
			entry.minPixelIndex = pixelIndex;
		}

		const colorBase = edgeIndex * 4;
		entry.colorSum[0] += edgeColors[colorBase];
		entry.colorSum[1] += edgeColors[colorBase + 1];
		entry.colorSum[2] += edgeColors[colorBase + 2];
		entry.colorSum[3] += edgeColors[colorBase + 3];
		entry.count += 1;
	}

	const facePaths: FacePath[] = [];
	for (const [, entry] of faces) {
		const { startEdge } = entry;
		const points: EdgePoint[] = [];
		const visited = new Set<number>();
		let currentEdge = startEdge;
		let closed = false;

		for (let step = 0; step <= edgeCount; step += 1) {
			if (currentEdge === INVALID_EDGE || visited.has(currentEdge)) {
				break;
			}

			visited.add(currentEdge);
			const pixelIndex = Math.floor(currentEdge / EDGE_DIRS);
			points.push(subpixelPoints[pixelIndex]);

			const nextEdge = nextEdges[currentEdge];
			if (nextEdge === startEdge) {
				closed = true;
				break;
			}

			currentEdge = nextEdge;
		}

		if (!closed || points.length < 3) {
			continue;
		}

		const area = polygonSignedArea(points);
		if (area <= 0) {
			continue;
		}

		const color = averageColor(entry.colorSum, entry.count);
		facePaths.push({ points, color, minPixelIndex: entry.minPixelIndex });
	}

	facePaths.sort((a, b) => a.minPixelIndex - b.minPixelIndex);

	const strokeWidth = Math.max(1 / Math.max(width, height), 0.75);
	const pathElements = facePaths
		.map((path) => {
			const commands = path.points.map((point, index) => {
				const command = index === 0 ? 'M' : 'L';
				return `${command} ${point.x.toFixed(3)} ${point.y.toFixed(3)}`;
			});

			const d = `${commands.join(' ')} Z`;
			return `<path fill="${path.color}" d="${d}" />`;
		})
		.join('');

	return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" fill="none" stroke="none" stroke-width="${strokeWidth}" stroke-linecap="round" stroke-linejoin="round" shape-rendering="geometricPrecision">${pathElements}</svg>`;
}

function averageColor(colorSum: [number, number, number, number], count: number): string {
	if (count === 0) {
		return '#000000';
	}

	const r = clamp01(colorSum[0] / count);
	const g = clamp01(colorSum[1] / count);
	const b = clamp01(colorSum[2] / count);
	const a = clamp01(colorSum[3] / count);

	if (a < 0.999) {
		const r8 = Math.round(r * 255);
		const g8 = Math.round(g * 255);
		const b8 = Math.round(b * 255);
		return `rgba(${r8}, ${g8}, ${b8}, ${a.toFixed(3)})`;
	}

	return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}

function toHex(value: number): string {
	const byte = Math.round(clamp01(value) * 255);
	return byte.toString(16).padStart(2, '0');
}

function clamp01(value: number): number {
	return Math.min(1, Math.max(0, value));
}

function polygonSignedArea(points: EdgePoint[]): number {
	let sum = 0;
	for (let i = 0; i < points.length; i += 1) {
		const current = points[i];
		const next = points[(i + 1) % points.length];
		sum += current.x * next.y - next.x * current.y;
	}

	return sum * 0.5;
}

async function readRgba16FloatTexture(
	device: GPUDevice,
	texture: GPUTexture,
	width: number,
	height: number
): Promise<Float32Array> {
	const bytesPerPixel = 8;
	const bytesPerRow = Math.ceil((width * bytesPerPixel) / 256) * 256;
	const readbackBuffer = device.createBuffer({
		label: 'face svg grad readback buffer',
		size: bytesPerRow * height,
		usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
	});

	const encoder = device.createCommandEncoder({ label: 'face svg grad readback encoder' });
	encoder.copyTextureToBuffer(
		{ texture },
		{ buffer: readbackBuffer, bytesPerRow },
		{ width, height }
	);

	device.queue.submit([encoder.finish()]);
	await readbackBuffer.mapAsync(GPUMapMode.READ);

	const mapped = new Uint8Array(readbackBuffer.getMappedRange().slice());
	readbackBuffer.unmap();

	const data = new Float32Array(width * height * 4);
	const view = new DataView(mapped.buffer, mapped.byteOffset, mapped.byteLength);

	for (let y = 0; y < height; y += 1) {
		const rowOffset = y * bytesPerRow;
		for (let x = 0; x < width; x += 1) {
			const sourceOffset = rowOffset + x * bytesPerPixel;
			const targetOffset = (y * width + x) * 4;
			data[targetOffset] = decodeFloat16(view.getUint16(sourceOffset, true));
			data[targetOffset + 1] = decodeFloat16(view.getUint16(sourceOffset + 2, true));
			data[targetOffset + 2] = decodeFloat16(view.getUint16(sourceOffset + 4, true));
			data[targetOffset + 3] = decodeFloat16(view.getUint16(sourceOffset + 6, true));
		}
	}

	return data;
}

async function readUint32Buffer(
	device: GPUDevice,
	buffer: GPUBuffer,
	length: number
): Promise<Uint32Array> {
	const byteLength = length * Uint32Array.BYTES_PER_ELEMENT;
	const readback = device.createBuffer({
		label: 'face svg u32 readback buffer',
		size: byteLength,
		usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
	});

	const encoder = device.createCommandEncoder({ label: 'face svg u32 readback encoder' });
	encoder.copyBufferToBuffer(buffer, 0, readback, 0, byteLength);
	device.queue.submit([encoder.finish()]);
	await readback.mapAsync(GPUMapMode.READ);

	const data = new Uint32Array(readback.getMappedRange().slice(0));
	readback.unmap();
	return data;
}

async function readFloat32Buffer(
	device: GPUDevice,
	buffer: GPUBuffer,
	length: number
): Promise<Float32Array> {
	const byteLength = length * Float32Array.BYTES_PER_ELEMENT;
	const readback = device.createBuffer({
		label: 'face svg f32 readback buffer',
		size: byteLength,
		usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
	});

	const encoder = device.createCommandEncoder({ label: 'face svg f32 readback encoder' });
	encoder.copyBufferToBuffer(buffer, 0, readback, 0, byteLength);
	device.queue.submit([encoder.finish()]);
	await readback.mapAsync(GPUMapMode.READ);

	const data = new Float32Array(readback.getMappedRange().slice(0));
	readback.unmap();
	return data;
}

function decodeFloat16(bits: number): number {
	const sign = bits & 0x8000 ? -1 : 1;
	const exponent = (bits >> 10) & 0x1f;
	const fraction = bits & 0x03ff;

	if (exponent === 0) {
		if (fraction === 0) {
			return sign === 1 ? 0 : -0;
		}

		return sign * Math.pow(2, -14) * (fraction / 1024);
	}

	if (exponent === 0x1f) {
		return fraction === 0 ? sign * Infinity : Number.NaN;
	}

	return sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
}
