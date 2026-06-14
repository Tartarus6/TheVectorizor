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

const INVALID_CONNECTION = 0xffffffff;

export async function faceBuffersToSvg(
	device: GPUDevice,
	gradTexture: GPUTexture,
	edgeTexture: GPUTexture,
	edgeDataBuffer: GPUBuffer,
	width: number,
	height: number,
	connectionCount: number
): Promise<string> {
	const connectionsData = await readEdgeDataBuffer(device, edgeDataBuffer, connectionCount);

	const [gradTexData, edgeTexData] = await Promise.all([
		readRgba16FloatTexture(device, gradTexture, width, height),
		readRgba16UintTexture(device, edgeTexture, width, height)
	]);

	const subpixelPoints: EdgePoint[] = new Array(width * height);
	const connectionsDataIdx: number[] = new Array(width * height);
	for (let index = 0; index < width * height; index += 1) {
		const x = index % width;
		const y = Math.floor(index / width);
		const base = index * 4;
		const theta = gradTexData[base + 1];
		const offset = gradTexData[base + 2];
		const idx = edgeTexData[base + 2];

		let subpixel_x;
		let subpixel_y;
		if (x == 0) {
			subpixel_x = 0;
		} else if (x == width - 1) {
			subpixel_x = width;
		} else {
			subpixel_x = x + 0.5 + Math.cos(theta) * offset;
		}

		if (y == 0) {
			subpixel_y = 0;
		} else if (y == height - 1) {
			subpixel_y = height;
		} else {
			subpixel_y = y + 0.5 + Math.sin(theta) * offset;
		}

		subpixelPoints[index] = {
			x: subpixel_x,
			y: subpixel_y
		};
		connectionsDataIdx[index] = idx;
	}

	type FaceAccumulator = {
		startEdge: number;
		minPixelIdx: number;
		color: [number, number, number, number];
	};

	const faces = new Map<number, FaceAccumulator>();

	for (let connectionIndex = 0; connectionIndex < connectionCount; connectionIndex += 1) {
		const connection = connectionsData[connectionIndex];
		if (connection.faceId === INVALID_CONNECTION) {
			continue;
		}

		if (connection.nextConnectionIdx === INVALID_CONNECTION) {
			continue;
		}

		let entry = faces.get(connection.faceId);
		if (!entry) {
			entry = {
				startEdge: connectionIndex,
				minPixelIdx: connection.posIdx,
				color: [0, 0, 0, 0]
			};
			faces.set(connection.faceId, entry);
		}

		const pixelIndex = connection.posIdx;
		if (pixelIndex < entry.minPixelIdx) {
			entry.minPixelIdx = pixelIndex;
		}

		if (connectionIndex == connection.faceId) {
			entry.color[0] = connection.color[0];
			entry.color[1] = connection.color[1];
			entry.color[2] = connection.color[2];
			entry.color[3] = connection.color[3];
		}

		// entry.colorSum[0] += connection.color[0];
		// entry.colorSum[1] += connection.color[1];
		// entry.colorSum[2] += connection.color[2];
		// entry.colorSum[3] += connection.color[3];
		// entry.count += 1;
	}

	const facePaths: FacePath[] = [];
	for (const [, entry] of faces) {
		const { startEdge: startConnection } = entry;

		const points: EdgePoint[] = [];

		// edge -> point index in current contour
		const visitedEdgeToPointIndex = new Map<number, number>();

		// edges belonging to discarded loops
		const ignoredEdges = new Set<number>();

		// path history
		const pathEdges: number[] = [];

		let currentConnectionIdx = startConnection;
		let closed = false;

		for (let step = 0; step <= connectionCount; step += 1) {
			if (currentConnectionIdx === INVALID_CONNECTION) {
				break;
			}

			// hit the starting edge again => proper closure
			if (currentConnectionIdx === startConnection && pathEdges.length > 0) {
				closed = true;
				break;
			}

			// somehow walked back into a loop that was discarded
			if (ignoredEdges.has(currentConnectionIdx)) {
				break;
			}

			const existingIndex = visitedEdgeToPointIndex.get(currentConnectionIdx);

			// false loop detected
			if (existingIndex !== undefined) {
				const loopEdges = pathEdges.slice(existingIndex);

				for (const edge of loopEdges) {
					ignoredEdges.add(edge);
					visitedEdgeToPointIndex.delete(edge);
				}

				pathEdges.length = existingIndex;
				points.length = existingIndex;

				currentConnectionIdx = connectionsData[currentConnectionIdx].nextConnectionIdx;

				continue;
			}

			const connection = connectionsData[currentConnectionIdx];

			visitedEdgeToPointIndex.set(currentConnectionIdx, pathEdges.length);

			pathEdges.push(currentConnectionIdx);

			points.push(subpixelPoints[connection.posIdx]);

			currentConnectionIdx = connection.nextConnectionIdx;
		}

		if (!closed || points.length < 3) {
			continue;
		}

		// remove negative area shapes as well as small ones (small ones can be created from small loops in neighbor connections)
		const area = polygonSignedArea(points);
		if (area <= 1) {
			continue;
		}

		const color = averageColor(entry.color);
		facePaths.push({ points, color, minPixelIndex: entry.minPixelIdx });
	}

	facePaths.sort((a, b) => a.minPixelIndex - b.minPixelIndex);

	const strokeWidth = Math.max(1 / Math.max(width, height), 0.75);
	const pathElements = facePaths
		.map((path) => {
			const commands = path.points.map((point, index) => {
				const command = index === 0 ? 'M' : 'L';
				return `${command} ${point.x.toFixed(1)} ${point.y.toFixed(1)}`;
			});

			const d = `${commands.join(' ')} Z`;
			return `<path fill="${path.color}" stroke="${path.color}" stroke-width="0.5px" d="${d}" />`;
		})
		.join('');

	return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" fill="none" stroke="none" stroke-width="${strokeWidth}" stroke-linecap="round" stroke-linejoin="round" shape-rendering="geometricPrecision">${pathElements}</svg>`;
}

function averageColor(colorSum: [number, number, number, number]): string {
	const r = clamp01(colorSum[0]);
	const g = clamp01(colorSum[1]);
	const b = clamp01(colorSum[2]);
	const a = clamp01(colorSum[3]);

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

async function readRgba16UintTexture(
	device: GPUDevice,
	texture: GPUTexture,
	width: number,
	height: number
): Promise<Uint32Array> {
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

	const data = new Uint32Array(width * height * 4);
	const view = new DataView(mapped.buffer, mapped.byteOffset, mapped.byteLength);

	for (let y = 0; y < height; y += 1) {
		const rowOffset = y * bytesPerRow;
		for (let x = 0; x < width; x += 1) {
			const sourceOffset = rowOffset + x * bytesPerPixel;
			const targetOffset = (y * width + x) * 4;
			data[targetOffset] = view.getUint16(sourceOffset, true);
			data[targetOffset + 1] = view.getUint16(sourceOffset + 2, true);
			data[targetOffset + 2] = view.getUint16(sourceOffset + 4, true);
			data[targetOffset + 3] = view.getUint16(sourceOffset + 6, true);
		}
	}

	return data;
}

interface EdgeData {
	nextConnectionIdx: number;
	jumpNextIdx: number;
	faceId: number;
	posIdx: number;
	color: Float32Array; // or [number, number, number, number]
}

async function readEdgeDataBuffer(
	device: GPUDevice,
	buffer: GPUBuffer,
	count: number
): Promise<EdgeData[]> {
	const bytesPerElement = 32; // 2 * u32 (8) + vec4f (16)
	const byteLength = count * bytesPerElement;

	const readback = device.createBuffer({
		label: 'EdgeData readback buffer',
		size: byteLength,
		usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
	});

	const encoder = device.createCommandEncoder();
	encoder.copyBufferToBuffer(buffer, 0, readback, 0, byteLength);
	device.queue.submit([encoder.finish()]);

	await readback.mapAsync(GPUMapMode.READ);
	const mapped = readback.getMappedRange();

	// Use a DataView to handle the mixed types and offsets
	const view = new DataView(mapped);
	const result: EdgeData[] = [];

	for (let i = 0; i < count; i++) {
		const offset = i * bytesPerElement;
		result.push({
			nextConnectionIdx: view.getUint32(offset + 0, true),
			jumpNextIdx: view.getUint32(offset + 4, true),
			faceId: view.getUint32(offset + 8, true),
			posIdx: view.getUint32(offset + 12, true),
			color: new Float32Array([
				view.getFloat32(offset + 16, true),
				view.getFloat32(offset + 20, true),
				view.getFloat32(offset + 24, true),
				view.getFloat32(offset + 28, true)
			])
		});
	}

	readback.unmap();
	readback.destroy();

	return result;
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
