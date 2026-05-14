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

type DirectedEdge = {
	from: number;
	to: number;
};

const neighborOffsets = [
	// diagonals
	[1, 1],
	[-1, 1],
	[-1, -1],
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

	const subpixelPoints: EdgePoint[] = new Array(pixelCount);
	for (let index = 0; index < pixelCount; index += 1) {
		subpixelPoints[index] = indexToPoint(index, width, thetaValues, subpixelOffsetValues);
	}

	// Keep only the 2-core of the edge graph so dangling/open chains are removed
	// before any closed-face tracing.
	const cycleEdgeMask = cullOpenEdgePixels(width, height, edgeMask);

	const adjacency = buildSortedAdjacency(width, height, cycleEdgeMask, subpixelPoints);
	return traceClosedFaces(adjacency, subpixelPoints);
}

function pathsToSvg(paths: EdgePath[], width: number, height: number): string {
	const strokeWidth = Math.max(1 / Math.max(width, height), 0.75);
	const pathElements = paths
		.map((path) => {
			// By this point, paths should already be filtered for validity
			// but keep a sanity check just in case
			if (!path.closed || path.points.length < 3) {
				return '';
			}

			const commands = path.points.map((point, index) => {
				const command = index === 0 ? 'M' : 'L';
				return `${command} ${point.x.toFixed(3)} ${point.y.toFixed(3)}`;
			});

			const d = `${commands.join(' ')}${path.closed ? ' Z' : ''}`;
			return `<path fill="#${Math.floor(Math.random() * 16777215).toString(16)}" d="${d}" />`;
		})
		.filter(Boolean)
		.join('');

	return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" fill="none" stroke="black" stroke-width="${strokeWidth}" stroke-linecap="round" stroke-linejoin="round" shape-rendering="geometricPrecision">${pathElements}</svg>`;
}

function buildSortedAdjacency(
	width: number,
	height: number,
	edgeMask: Uint8Array,
	points: EdgePoint[]
): number[][] {
	const pixelCount = width * height;
	const adjacency: number[][] = new Array(pixelCount);

	for (let index = 0; index < pixelCount; index += 1) {
		if (edgeMask[index] === 0) {
			adjacency[index] = [];
			continue;
		}

		const neighbors = getNeighborIndices(index, width, height, edgeMask);
		const center = points[index];
		neighbors.sort((a, b) => {
			const pointA = points[a];
			const pointB = points[b];
			const angleA = Math.atan2(pointA.y - center.y, pointA.x - center.x);
			const angleB = Math.atan2(pointB.y - center.y, pointB.x - center.x);
			return angleA - angleB;
		});

		adjacency[index] = neighbors;
	}

	return adjacency;
}

function cullOpenEdgePixels(width: number, height: number, edgeMask: Uint8Array): Uint8Array {
	const pixelCount = width * height;
	const keptMask = edgeMask.slice();
	const degree = new Uint16Array(pixelCount);
	const neighborLists: number[][] = new Array(pixelCount);
	const queue: number[] = [];

	for (let index = 0; index < pixelCount; index += 1) {
		if (keptMask[index] === 0) {
			neighborLists[index] = [];
			continue;
		}

		const neighbors = getNeighborIndices(index, width, height, keptMask);
		neighborLists[index] = neighbors;
		degree[index] = neighbors.length;

		if (degree[index] < 2) {
			queue.push(index);
		}
	}

	while (queue.length > 0) {
		const current = queue.pop();
		if (current === undefined) {
			continue;
		}

		if (keptMask[current] === 0 || degree[current] >= 2) {
			continue;
		}

		keptMask[current] = 0;

		for (const neighbor of neighborLists[current]) {
			if (keptMask[neighbor] === 0) {
				continue;
			}

			if (degree[neighbor] > 0) {
				degree[neighbor] -= 1;
			}

			if (degree[neighbor] < 2) {
				queue.push(neighbor);
			}
		}
	}

	return keptMask;
}

function traceClosedFaces(adjacency: number[][], points: EdgePoint[]): EdgePath[] {
	const visitedHalfEdges = new Set<string>();
	const paths: EdgePath[] = [];
	const totalHalfEdges = adjacency.reduce((count, neighbors) => count + neighbors.length, 0);
	const MIN_CYCLE_POINTS = 5; // Require at least 5 points to filter staircase artifacts
	const MIN_AREA = 1.0; // Minimum area to avoid tiny noise loops

	for (let startFrom = 0; startFrom < adjacency.length; startFrom += 1) {
		const neighbors = adjacency[startFrom];
		if (neighbors.length === 0) {
			continue;
		}

		for (const startTo of neighbors) {
			const startKey = directedEdgeKey(startFrom, startTo);
			if (visitedHalfEdges.has(startKey)) {
				continue;
			}

			const cycle = traceSingleFace(
				{ from: startFrom, to: startTo },
				adjacency,
				visitedHalfEdges,
				totalHalfEdges
			);

			if (cycle === null) {
				continue;
			}

			// Remove duplicate end point if present
			if (cycle[0] === cycle[cycle.length - 1]) {
				cycle.pop();
			}

			// Require minimum cycle length to filter staircase artifacts
			if (cycle.length < MIN_CYCLE_POINTS) {
				continue;
			}

			const polygonPoints = cycle.map((index) => points[index]);
			const signedArea = polygonSignedArea(polygonPoints);

			// In image coordinates (y-down), clockwise loops have positive area.
			// Keeping only positive area removes the outer/exterior face cycles.
			// Also require minimum area to filter tiny noise from staircases.
			if (signedArea < MIN_AREA) {
				continue;
			}

			paths.push({
				points: polygonPoints,
				closed: true
			});
		}
	}

	return paths;
}

function traceSingleFace(
	startEdge: DirectedEdge,
	adjacency: number[][],
	visitedHalfEdges: Set<string>,
	maxSteps: number
): number[] | null {
	const localHalfEdges = new Set<string>();
	const cycleIndices: number[] = [startEdge.from];

	let currentFrom = startEdge.from;
	let currentTo = startEdge.to;

	for (let step = 0; step <= maxSteps; step += 1) {
		const currentKey = directedEdgeKey(currentFrom, currentTo);
		if (localHalfEdges.has(currentKey)) {
			return null;
		}

		localHalfEdges.add(currentKey);
		visitedHalfEdges.add(currentKey);
		cycleIndices.push(currentTo);

		const neighbors = adjacency[currentTo];
		if (neighbors.length < 2) {
			return null;
		}

		const incomingIndex = neighbors.indexOf(currentFrom);
		if (incomingIndex === -1) {
			return null;
		}

		// Right-hand rule: choose the first clockwise edge from the reverse incoming edge.
		const nextNeighborIndex = (incomingIndex - 1 + neighbors.length) % neighbors.length;
		const nextTo = neighbors[nextNeighborIndex];

		currentFrom = currentTo;
		currentTo = nextTo;

		if (currentFrom === startEdge.from && currentTo === startEdge.to) {
			return cycleIndices;
		}
	}

	return null;
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

function directedEdgeKey(from: number, to: number): string {
	return `${from}:${to}`;
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

function polygonSignedArea(points: EdgePoint[]): number {
	let sum = 0;
	for (let i = 0; i < points.length; i += 1) {
		const current = points[i];
		const next = points[(i + 1) % points.length];
		sum += current.x * next.y - next.x * current.y;
	}

	return sum * 0.5;
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
