/*
 ? This code is mostly LLM-written, and is just for testing. it will not be the final thing.
 ? The final code will probably be written for WASM so that it can be a lot faster.
 * What this file does is take in a specially formatted texture, and turn it into an SVG.
 * Format:
    r -> edge_flag
    g -> grad_mag
    b -> theta
    a -> subpixel_offset
*/

type EdgePoint = {
	x: number;
	y: number;
};

type EdgePath = {
	points: EdgePoint[];
	closed: boolean;
};

const neighborOffsets = [
	// diagonals
	[1, 1],
	[-1, 1],
	[-1 - 1],
	[1, -1],
	// cardinals
	[1, 0],
	[0, 1],
	[-1, 0],
	[0, -1]
];

export async function textureToEdgeSvg(
	device: GPUDevice,
	grad_tex: GPUTexture,
	edge_tex: GPUTexture,
	width: number,
	height: number
): Promise<string> {
	const grad_data = await readRgba16FloatTexture(device, grad_tex, width, height);
	const edge_data = await readRgba16FloatTexture(device, edge_tex, width, height);
	const paths = edgeTextureToPaths(grad_data, edge_data, width, height);
	return pathsToSvg(paths, width, height);
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
		label: 'edge svg readback buffer',
		size: bytesPerRow * height,
		usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
	});

	const encoder = device.createCommandEncoder({ label: 'edge svg readback encoder' });
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

function edgeTextureToPaths(
	grad_data: Float32Array,
	edge_data: Float32Array,
	width: number,
	height: number
): EdgePath[] {
	const pixelCount = width * height;
	const edgeMask = new Uint8Array(pixelCount);
	const thetaValues = new Float32Array(pixelCount);
	const degreeValues = new Uint8Array(pixelCount);
	const subpixelOffsetValues = new Float32Array(pixelCount);

	for (let index = 0; index < pixelCount; index += 1) {
		const base = index * 4;
		const edgeFlag = edge_data[base];
		// const magnitude = grad_data[base];

		if (edgeFlag > 0.5) {
			edgeMask[index] = 1;
			thetaValues[index] = grad_data[base + 1];
			subpixelOffsetValues[index] = edge_data[base + 1];
		}
	}

	for (let index = 0; index < pixelCount; index += 1) {
		if (edgeMask[index] === 0) {
			continue;
		}

		degreeValues[index] = getNeighborIndices(index, width, height, edgeMask).length;
	}

	const visitedEdges = new Set<string>();
	const paths: EdgePath[] = [];

	const traceIfNeeded = (startIndex: number, neighborIndices: number[]) => {
		for (const neighborIndex of neighborIndices) {
			if (visitedEdges.has(edgeKey(startIndex, neighborIndex))) {
				continue;
			}

			const path = tracePath(
				startIndex,
				neighborIndex,
				width,
				height,
				edgeMask,
				thetaValues,
				subpixelOffsetValues,
				visitedEdges
			);
			if (path.points.length >= 2) {
				paths.push(path);
			}
		}
	};

	for (let index = 0; index < pixelCount; index += 1) {
		// if (edgeMask[index] === 0) {
		if (edgeMask[index] === 0 || degreeValues[index] === 2) {
			continue;
		}

		traceIfNeeded(index, getNeighborIndices(index, width, height, edgeMask));
	}

	for (let index = 0; index < pixelCount; index += 1) {
		if (edgeMask[index] === 0) {
			continue;
		}

		traceIfNeeded(index, getNeighborIndices(index, width, height, edgeMask));
	}

	return paths;
}

function pathsToSvg(paths: EdgePath[], width: number, height: number): string {
	const strokeWidth = Math.max(1 / Math.max(width, height), 0.75);
	const pathElements = paths
		.map((path) => {
			if (path.points.length < 2) {
				return '';
			}

			const commands = path.points.map((point, index) => {
				const command = index === 0 ? 'M' : 'L';
				return `${command} ${point.x.toFixed(3)} ${point.y.toFixed(3)}`;
			});

			const d = `${commands.join(' ')}${path.closed ? ' Z' : ''}`;
			// random stroke color
			return `<path stroke="#${Math.floor(Math.random() * 16777215).toString(16)}" d="${d}" />`;
			// return `<path stroke="black" d="${d}" />`;
		})
		.filter(Boolean)
		.join('');

	return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" fill="none" stroke="black" stroke-width="${strokeWidth}" stroke-linecap="round" stroke-linejoin="round" shape-rendering="geometricPrecision">${pathElements}</svg>`;
}

function tracePath(
	startIndex: number,
	nextIndex: number,
	width: number,
	height: number,
	edgeMask: Uint8Array,
	thetaValues: Float32Array,
	subpixelOffsetValues: Float32Array,
	visitedEdges: Set<string>
): EdgePath {
	const points: EdgePoint[] = [
		indexToPoint(startIndex, width, thetaValues, subpixelOffsetValues),
		indexToPoint(nextIndex, width, thetaValues, subpixelOffsetValues)
	];
	visitedEdges.add(edgeKey(startIndex, nextIndex));

	let previousIndex: number | null = startIndex;
	let currentIndex = nextIndex;
	let closed = false;

	while (true) {
		const neighbors = getNeighborIndices(currentIndex, width, height, edgeMask).filter(
			(candidateIndex) =>
				candidateIndex !== previousIndex && !visitedEdges.has(edgeKey(currentIndex, candidateIndex))
		);

		if (neighbors.length === 0) {
			break;
		}

		const chosenIndex = chooseNextIndex(currentIndex, previousIndex, neighbors, thetaValues, width);

		visitedEdges.add(edgeKey(currentIndex, chosenIndex));
		points.push(indexToPoint(chosenIndex, width, thetaValues, subpixelOffsetValues));

		previousIndex = currentIndex;
		currentIndex = chosenIndex;

		if (currentIndex === startIndex) {
			closed = true;
			break;
		}
	}

	return {
		points,
		closed
	};
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

function edgeKey(a: number, b: number): string {
	return a < b ? `${a}:${b}` : `${b}:${a}`;
}

function indexToPoint(
	index: number,
	width: number,
	thetaValues: Float32Array,
	subpixelOffsetValues: Float32Array
): EdgePoint {
	return {
		x: (index % width) + 0.5 + Math.cos(thetaValues[index]) * subpixelOffsetValues[index],
		y: Math.floor(index / width) + 0.5 + Math.sin(thetaValues[index]) * subpixelOffsetValues[index]
	};
}

function vectorLength(x: number, y: number): number {
	return Math.hypot(x, y);
}

function normalizeVector(x: number, y: number): [number, number] {
	const length = vectorLength(x, y);
	if (length === 0) {
		return [0, 0];
	}

	return [x / length, y / length];
}

function dot(aX: number, aY: number, bX: number, bY: number): number {
	return aX * bX + aY * bY;
}

function getNeighborIndices(
	index: number,
	width: number,
	height: number,
	edgeMask: Uint8Array
): number[] {
	const x = index % width;
	const y = Math.floor(index / width);
	const neighbors: number[] = [];

	// cardinal neighbors
	for (const [offsetX, offsetY] of neighborOffsets) {
		const nextX = x + offsetX;
		const nextY = y + offsetY;

		if (nextX < 0 || nextX >= width || nextY < 0 || nextY >= height) {
			continue;
		}

		const nextIndex = nextY * width + nextX;
		if (edgeMask[nextIndex] !== 0) {
			neighbors.push(nextIndex);
		}
	}

	return neighbors;
}

function chooseNextIndex(
	currentIndex: number,
	previousIndex: number | null,
	candidates: number[],
	thetaValues: Float32Array,
	width: number
): number {
	const currentX = currentIndex % width;
	const currentY = Math.floor(currentIndex / width);
	let previousDirX = 0;
	let previousDirY = 0;

	if (previousIndex !== null) {
		const previousX = previousIndex % width;
		const previousY = Math.floor(previousIndex / width);
		[previousDirX, previousDirY] = normalizeVector(currentX - previousX, currentY - previousY);
	}

	const theta = thetaValues[currentIndex] || 0;
	const tangentX = Math.cos(theta + Math.PI / 2);
	const tangentY = Math.sin(theta + Math.PI / 2);

	let bestCandidate = candidates[0];
	let bestScore = Number.NEGATIVE_INFINITY;

	for (const candidate of candidates) {
		const candidateX = candidate % width;
		const candidateY = Math.floor(candidate / width);
		const [dirX, dirY] = normalizeVector(candidateX - currentX, candidateY - currentY);

		let score = Math.abs(dot(dirX, dirY, tangentX, tangentY));
		if (previousIndex !== null) {
			score += dot(previousDirX, previousDirY, dirX, dirY) * 2;
		}

		if (score > bestScore) {
			bestScore = score;
			bestCandidate = candidate;
		}
	}

	return bestCandidate;
}
