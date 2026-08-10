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
	edges: number[];
	color: string;
	minPixelIndex: number;
};

const INVALID_CONNECTION = 0xffffffff;

// Maximum distance (in pixels) any fitted Bézier is allowed to deviate from the
// detected sub-pixel polyline. This bounds fidelity only — gap-safety is
// guaranteed structurally by fitting each shared edge once (see buildFacePath).
export const FIT_ERROR = 0.3;

// Vertices whose turn angle exceeds this are treated as hard corners: the chain
// is split there so the corner is never rounded. Computed from geometry alone,
// so both faces sharing the vertex always agree.
const CORNER_ANGLE_RAD = (75 * Math.PI) / 180;

// 8-connected direction table. MUST match the DIRS array used in the WGSL
// shaders (face_trace_init / reciprocating_neighbors): index -> (dx, dy).
// Order: E, NE, N, NW, W, SW, S, SE.
const DIRS: ReadonlyArray<readonly [number, number]> = [
	[1, 0], // 0: E
	[1, -1], // 1: NE
	[0, -1], // 2: N
	[-1, -1], // 3: NW
	[-1, 0], // 4: W
	[-1, 1], // 5: SW
	[0, 1], // 6: S
	[1, 1] // 7: SE
];

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
		// edge_id is split across the z (low 16 bits) and w (high 16 bits) channels
		const idx = edgeTexData[base + 2] | (edgeTexData[base + 3] << 16);

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
		facePaths.push({ points, edges: pathEdges.slice(), color, minPixelIndex: entry.minPixelIdx });
	}

	facePaths.sort((a, b) => a.minPixelIndex - b.minPixelIndex);

	// Shared context for turning per-face edge loops into Bézier paths. The fit
	// cache is keyed by a chain's canonical pixel-index sequence, so the two
	// faces bordering a shared edge fit it exactly once and reuse the identical
	// curve (one of them reversed). This makes seams gap-free by construction.
	const svgCtx: SvgBuildContext = {
		connectionsData,
		edgeTexData,
		subpixelPoints,
		width,
		fitCache: new Map<string, CubicSegment[]>()
	};

	const strokeWidth = Math.max(1 / Math.max(width, height), 0.75);
	const pathElements = facePaths
		.map((path) => {
			const d = buildFacePathData(path.edges, svgCtx);
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

// ============================================================================
// Shared-edge Bézier path construction
//
// A face boundary is a closed loop of directed half-edges. Two faces that touch
// share a run of half-edges (each directed edge on one side has a twin on the
// other). To render seams without gaps, every such shared run ("chain") must be
// fit to a curve exactly once, with both faces emitting the identical geometry
// (one of them reversed — reversing a cubic Bézier is exact). We achieve this
// by keying a fit cache on the chain's canonical (direction-independent) pixel
// sequence, so both faces resolve to the same cache entry.
//
// Chains are bounded by junction vertices (pixel degree != 2) and by hard
// corners (turn angle > CORNER_ANGLE_RAD). Both boundary conditions are derived
// from geometry/topology alone, so the two faces sharing a chain always split
// it identically.
// ============================================================================

type Vec = EdgePoint;
type CubicSegment = [Vec, Vec, Vec, Vec]; // P0, P1, P2, P3 (absolute coordinates)

type SvgBuildContext = {
	connectionsData: EdgeData[];
	edgeTexData: Uint32Array;
	subpixelPoints: EdgePoint[];
	width: number;
	fitCache: Map<string, CubicSegment[]>;
};

function buildFacePathData(edges: number[], ctx: SvgBuildContext): string {
	const n = edges.length;
	if (n < 2) {
		return polylineFallback(edges, ctx);
	}

	// Vertex i is the source pixel of directed edge i; edge i runs from vertex i
	// to vertex i+1 (mod n). Validate that reconstructed topology is consistent;
	// if anything is off, fall back to a plain polyline for this face.
	const vertexPixel: number[] = new Array(n);
	for (let i = 0; i < n; i += 1) {
		vertexPixel[i] = ctx.connectionsData[edges[i]].posIdx;
	}
	for (let i = 0; i < n; i += 1) {
		const dst = dstPixelOf(ctx, edges[i]);
		if (dst < 0 || dst !== vertexPixel[(i + 1) % n]) {
			return polylineFallback(edges, ctx);
		}
	}

	const pts: Vec[] = vertexPixel.map((px) => ctx.subpixelPoints[px]);

	// Determine chain-break vertices: junctions or hard corners.
	const isBreak: boolean[] = new Array(n);
	for (let i = 0; i < n; i += 1) {
		const junction = degreeOf(ctx, vertexPixel[i]) !== 2;
		const corner = turnAngle(pts[(i - 1 + n) % n], pts[i], pts[(i + 1) % n]) > CORNER_ANGLE_RAD;
		isBreak[i] = junction || corner;
	}

	const breaks: number[] = [];
	for (let i = 0; i < n; i += 1) {
		if (isBreak[i]) breaks.push(i);
	}

	const orderedSegs: CubicSegment[] = [];

	if (breaks.length >= 2) {
		// Multiple open chains, each spanning one break vertex to the next.
		for (let k = 0; k < breaks.length; k += 1) {
			const startI = breaks[k];
			const endI = breaks[(k + 1) % breaks.length];
			const chainPix: number[] = [];
			let i = startI;
			for (;;) {
				chainPix.push(vertexPixel[i]);
				if (i === endI) break;
				i = (i + 1) % n;
			}
			appendOpenChain(chainPix, ctx, orderedSegs);
		}
	} else {
		// No junction/corner: the whole loop is one closed chain. Pin the seam at
		// the sole break if there is one, otherwise at the lowest pixel index so
		// both faces choose the same seam.
		const seam = breaks.length === 1 ? breaks[0] : indexOfMinPixel(vertexPixel);
		const chainPix: number[] = [];
		let i = seam;
		do {
			chainPix.push(vertexPixel[i]);
			i = (i + 1) % n;
		} while (i !== seam);
		appendClosedChain(chainPix, ctx, orderedSegs);
	}

	if (orderedSegs.length === 0) {
		return polylineFallback(edges, ctx);
	}

	// Assemble the path. Consecutive chains share endpoint vertices exactly, so
	// the concatenation is C0-continuous with no gaps; Z closes the final seam.
	let d = `M ${fmtVec(orderedSegs[0][0])}`;
	for (const seg of orderedSegs) {
		d += ` C ${fmtVec(seg[1])} ${fmtVec(seg[2])} ${fmtVec(seg[3])}`;
	}
	d += ' Z';
	return d;
}

// Fit an open chain (face-order pixel list) and append its segments in face
// traversal order, reusing the cached canonical fit if present.
function appendOpenChain(chainPix: number[], ctx: SvgBuildContext, out: CubicSegment[]): void {
	const { canonicalPix, reversed } = canonicalizeOpen(chainPix);
	const key = canonicalPix.join(',');
	let segs = ctx.fitCache.get(key);
	if (!segs) {
		const cpts = canonicalPix.map((px) => ctx.subpixelPoints[px]);
		const left = normalizeV(subV(cpts[1], cpts[0]));
		const right = normalizeV(subV(cpts[cpts.length - 2], cpts[cpts.length - 1]));
		segs = fitPolyline(cpts, left, right);
		ctx.fitCache.set(key, segs);
	}
	const emitted = reversed ? reverseCubicSegments(segs) : segs;
	for (const seg of emitted) out.push(seg);
}

// Fit a closed chain (single loop, seam already chosen as chainPix[0]).
function appendClosedChain(chainPix: number[], ctx: SvgBuildContext, out: CubicSegment[]): void {
	const { canonicalPix, reversed } = canonicalizeClosed(chainPix);
	const key = 'C:' + canonicalPix.join(',');
	let segs = ctx.fitCache.get(key);
	if (!segs) {
		const cpts = canonicalPix.map((px) => ctx.subpixelPoints[px]);
		cpts.push(cpts[0]); // close the loop for fitting
		// Smooth tangent across the seam for C1 continuity there.
		const seamTan = normalizeV(subV(cpts[1], cpts[cpts.length - 2]));
		segs = fitPolyline(cpts, seamTan, negateV(seamTan));
		ctx.fitCache.set(key, segs);
	}
	const emitted = reversed ? reverseCubicSegments(segs) : segs;
	for (const seg of emitted) out.push(seg);
}

// Canonicalize an open chain: pick the lexicographically-smaller of the
// sequence and its reverse. `reversed` is true when the caller's (face) order
// differs from canonical, meaning the cached fit must be reversed before use.
export function canonicalizeOpen(chainPix: number[]): {
	canonicalPix: number[];
	reversed: boolean;
} {
	const rev = chainPix.slice().reverse();
	const useForward = lexLessOrEqual(chainPix, rev);
	return { canonicalPix: useForward ? chainPix : rev, reversed: !useForward };
}

// Canonicalize a closed chain already rotated to its seam vertex. Both
// directions start at the same seam; choose the lexicographically-smaller.
export function canonicalizeClosed(chainPix: number[]): {
	canonicalPix: number[];
	reversed: boolean;
} {
	// Reverse direction while keeping the seam vertex first.
	const rev = [chainPix[0], ...chainPix.slice(1).reverse()];
	const useForward = lexLessOrEqual(chainPix, rev);
	return { canonicalPix: useForward ? chainPix : rev, reversed: !useForward };
}

function lexLessOrEqual(a: number[], b: number[]): boolean {
	const len = Math.min(a.length, b.length);
	for (let i = 0; i < len; i += 1) {
		if (a[i] < b[i]) return true;
		if (a[i] > b[i]) return false;
	}
	return a.length <= b.length;
}

function indexOfMinPixel(pixels: number[]): number {
	let best = 0;
	for (let i = 1; i < pixels.length; i += 1) {
		if (pixels[i] < pixels[best]) best = i;
	}
	return best;
}

// Reverse a cubic Bézier sequence so it traces the same curve backwards: reverse
// the segment order and swap (P0,P1,P2,P3) -> (P3,P2,P1,P0) within each. Exact.
export function reverseCubicSegments(segs: CubicSegment[]): CubicSegment[] {
	const out: CubicSegment[] = [];
	for (let i = segs.length - 1; i >= 0; i -= 1) {
		const s = segs[i];
		out.push([s[3], s[2], s[1], s[0]]);
	}
	return out;
}

function polylineFallback(edges: number[], ctx: SvgBuildContext): string {
	const commands = edges.map((conn, index) => {
		const p = ctx.subpixelPoints[ctx.connectionsData[conn].posIdx];
		const command = index === 0 ? 'M' : 'L';
		return `${command} ${p.x.toFixed(2)} ${p.y.toFixed(2)}`;
	});
	return `${commands.join(' ')} Z`;
}

// --- topology helpers (reconstruct the directed half-edge graph) ------------

function popcount(v: number): number {
	v = v - ((v >> 1) & 0x55555555);
	v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
	return (((v + (v >> 4)) & 0x0f0f0f0f) * 0x01010101) >> 24;
}

function packedOf(ctx: SvgBuildContext, pixelIdx: number): number {
	return ctx.edgeTexData[pixelIdx * 4 + 1];
}

function baseOf(ctx: SvgBuildContext, pixelIdx: number): number {
	const b = pixelIdx * 4;
	return ctx.edgeTexData[b + 2] | (ctx.edgeTexData[b + 3] << 16);
}

function degreeOf(ctx: SvgBuildContext, pixelIdx: number): number {
	return popcount(packedOf(ctx, pixelIdx));
}

// Recover the 8-connected direction of a directed edge from its sparse index.
// The sparse index is base(pixel) + (number of set neighbor bits below its
// direction), matching get_sparse_index in face_trace_init.wgsl. Inverting that
// means finding the direction of the (localOffset)-th set bit.
function dirOfConnection(ctx: SvgBuildContext, conn: number): number {
	const src = ctx.connectionsData[conn].posIdx;
	const packed = packedOf(ctx, src);
	const localOffset = conn - baseOf(ctx, src);
	let count = 0;
	for (let d = 0; d < 8; d += 1) {
		if (packed & (1 << d)) {
			if (count === localOffset) return d;
			count += 1;
		}
	}
	return -1;
}

function dstPixelOf(ctx: SvgBuildContext, conn: number): number {
	const dir = dirOfConnection(ctx, conn);
	if (dir < 0) return -1;
	const src = ctx.connectionsData[conn].posIdx;
	const x = src % ctx.width;
	const y = Math.floor(src / ctx.width);
	const [dx, dy] = DIRS[dir];
	return x + dx + ctx.width * (y + dy);
}

// --- vector helpers ---------------------------------------------------------

function subV(a: Vec, b: Vec): Vec {
	return { x: a.x - b.x, y: a.y - b.y };
}

function addV(a: Vec, b: Vec): Vec {
	return { x: a.x + b.x, y: a.y + b.y };
}

function scaleV(a: Vec, s: number): Vec {
	return { x: a.x * s, y: a.y * s };
}

function negateV(a: Vec): Vec {
	return { x: -a.x, y: -a.y };
}

function dotV(a: Vec, b: Vec): number {
	return a.x * b.x + a.y * b.y;
}

function distanceV(a: Vec, b: Vec): number {
	return Math.hypot(a.x - b.x, a.y - b.y);
}

function normalizeV(a: Vec): Vec {
	const len = Math.hypot(a.x, a.y);
	if (len < 1e-12) return { x: 0, y: 0 };
	return { x: a.x / len, y: a.y / len };
}

// Turn angle at b between segments a->b and b->c (0 = straight, PI = reversal).
function turnAngle(a: Vec, b: Vec, c: Vec): number {
	const v1 = normalizeV(subV(b, a));
	const v2 = normalizeV(subV(c, b));
	const d = Math.max(-1, Math.min(1, dotV(v1, v2)));
	return Math.acos(d);
}

function fmtVec(p: Vec): string {
	return `${p.x.toFixed(2)} ${p.y.toFixed(2)}`;
}

// --- Schneider cubic Bézier fitting -----------------------------------------
// Port of Philip J. Schneider, "An Algorithm for Automatically Fitting
// Digitized Curves" (Graphics Gems, 1990). Distances are true (un-squared)
// pixels so FIT_ERROR reads directly as a pixel tolerance.

export function fitPolyline(points: Vec[], leftTangent: Vec, rightTangent: Vec): CubicSegment[] {
	const out: CubicSegment[] = [];
	if (points.length < 2) return out;
	fitCubicRec(points, 0, points.length - 1, leftTangent, rightTangent, out);
	return out;
}

function fitCubicRec(
	pts: Vec[],
	first: number,
	last: number,
	tHat1: Vec,
	tHat2: Vec,
	out: CubicSegment[]
): void {
	const nPts = last - first + 1;

	if (nPts === 2) {
		const dist = distanceV(pts[first], pts[last]) / 3;
		const p0 = pts[first];
		const p3 = pts[last];
		out.push([p0, addV(p0, scaleV(tHat1, dist)), addV(p3, scaleV(tHat2, dist)), p3]);
		return;
	}

	let u = chordLengthParameterize(pts, first, last);
	let bez = generateBezier(pts, first, last, u, tHat1, tHat2);
	const initial = computeMaxError(pts, first, last, bez, u);
	const maxError = initial.maxError;
	let splitPoint = initial.splitPoint;

	if (maxError < FIT_ERROR) {
		out.push(bez);
		return;
	}

	// If we're close, try Newton-Raphson reparameterization before splitting.
	if (maxError < FIT_ERROR * 4) {
		for (let i = 0; i < 4; i += 1) {
			const uPrime = reparameterize(pts, first, last, u, bez);
			bez = generateBezier(pts, first, last, uPrime, tHat1, tHat2);
			const r = computeMaxError(pts, first, last, bez, uPrime);
			u = uPrime;
			if (r.maxError < FIT_ERROR) {
				out.push(bez);
				return;
			}
			splitPoint = r.splitPoint;
		}
	}

	if (splitPoint <= first) splitPoint = first + 1;
	if (splitPoint >= last) splitPoint = last - 1;

	const tHatCenter = computeCenterTangent(pts, splitPoint);
	fitCubicRec(pts, first, splitPoint, tHat1, tHatCenter, out);
	fitCubicRec(pts, splitPoint, last, negateV(tHatCenter), tHat2, out);
}

function computeCenterTangent(pts: Vec[], center: number): Vec {
	const v1 = subV(pts[center - 1], pts[center]);
	const v2 = subV(pts[center], pts[center + 1]);
	return normalizeV({ x: (v1.x + v2.x) / 2, y: (v1.y + v2.y) / 2 });
}

function chordLengthParameterize(pts: Vec[], first: number, last: number): number[] {
	const u: number[] = [0];
	for (let i = first + 1; i <= last; i += 1) {
		u.push(u[i - 1 - first] + distanceV(pts[i], pts[i - 1]));
	}
	const total = u[u.length - 1];
	if (total > 0) {
		for (let i = 0; i < u.length; i += 1) u[i] /= total;
	}
	return u;
}

// Least-squares fit of the two interior control points, endpoints and tangent
// directions fixed.
function generateBezier(
	pts: Vec[],
	first: number,
	last: number,
	u: number[],
	tHat1: Vec,
	tHat2: Vec
): CubicSegment {
	const nPts = last - first + 1;
	const p0 = pts[first];
	const p3 = pts[last];

	let c00 = 0;
	let c01 = 0;
	let c11 = 0;
	let x0 = 0;
	let x1 = 0;

	for (let i = 0; i < nPts; i += 1) {
		const ui = u[i];
		const b0 = bernstein0(ui);
		const b1 = bernstein1(ui);
		const b2 = bernstein2(ui);
		const b3 = bernstein3(ui);

		const a0 = scaleV(tHat1, b1);
		const a1 = scaleV(tHat2, b2);

		c00 += dotV(a0, a0);
		c01 += dotV(a0, a1);
		c11 += dotV(a1, a1);

		const tmp = subV(pts[first + i], addV(scaleV(p0, b0 + b1), scaleV(p3, b2 + b3)));
		x0 += dotV(a0, tmp);
		x1 += dotV(a1, tmp);
	}

	const detC = c00 * c11 - c01 * c01;
	let alphaL = 0;
	let alphaR = 0;
	if (Math.abs(detC) > 1e-12) {
		alphaL = (x0 * c11 - x1 * c01) / detC;
		alphaR = (c00 * x1 - c01 * x0) / detC;
	}

	const segLength = distanceV(p0, p3);
	const epsilon = 1e-6 * segLength;
	if (alphaL < epsilon || alphaR < epsilon) {
		// Fall back to Wu/Barsky heuristic: place handles 1/3 along the chord.
		const d = segLength / 3;
		return [p0, addV(p0, scaleV(tHat1, d)), addV(p3, scaleV(tHat2, d)), p3];
	}

	return [p0, addV(p0, scaleV(tHat1, alphaL)), addV(p3, scaleV(tHat2, alphaR)), p3];
}

function computeMaxError(
	pts: Vec[],
	first: number,
	last: number,
	bez: CubicSegment,
	u: number[]
): { maxError: number; splitPoint: number } {
	let maxError = 0;
	let splitPoint = (first + last) >> 1;
	for (let i = first + 1; i < last; i += 1) {
		// Honest error: distance to the *closest* point on the curve, not the
		// point at the chord-length parameter. The chord-length parameter can be
		// far from the projection, which would under-report the deviation and
		// accept an over-tolerance fit. FIT_ERROR then reads as a true per-vertex
		// bound; between vertices (~1px apart on real edge data) the continuous
		// deviation stays sub-pixel.
		const dist = closestDistance(bez, pts[i], u[i - first]);
		if (dist >= maxError) {
			maxError = dist;
			splitPoint = i;
		}
	}
	return { maxError, splitPoint };
}

// Distance from a point to the curve, found by refining the parameter with a
// few Newton steps of the closest-point condition (Q(t)-P)·Q'(t) = 0.
function closestDistance(bez: CubicSegment, point: Vec, uGuess: number): number {
	let t = Math.max(0, Math.min(1, uGuess));
	for (let iter = 0; iter < 6; iter += 1) {
		const next = Math.max(0, Math.min(1, newtonRaphson(bez, point, t)));
		if (Math.abs(next - t) < 1e-6) {
			t = next;
			break;
		}
		t = next;
	}
	return distanceV(bezierEval(bez, t), point);
}

// Newton-Raphson: refine each parameter u[i] toward the closest point on bez.
function reparameterize(
	pts: Vec[],
	first: number,
	last: number,
	u: number[],
	bez: CubicSegment
): number[] {
	const out: number[] = [];
	for (let i = first; i <= last; i += 1) {
		out.push(newtonRaphson(bez, pts[i], u[i - first]));
	}
	return out;
}

function newtonRaphson(bez: CubicSegment, point: Vec, u: number): number {
	const q = bezierEval(bez, u);

	// First and second derivative control points.
	const q1: Vec[] = [
		scaleV(subV(bez[1], bez[0]), 3),
		scaleV(subV(bez[2], bez[1]), 3),
		scaleV(subV(bez[3], bez[2]), 3)
	];
	const q2: Vec[] = [scaleV(subV(q1[1], q1[0]), 2), scaleV(subV(q1[2], q1[1]), 2)];

	const q1u = bezierEvalGeneric(q1, u);
	const q2u = bezierEvalGeneric(q2, u);

	const diff = subV(q, point);
	const numerator = dotV(diff, q1u);
	const denominator = dotV(q1u, q1u) + dotV(diff, q2u);

	if (Math.abs(denominator) < 1e-12) return u;
	const next = u - numerator / denominator;
	if (!Number.isFinite(next)) return u;
	return next;
}

export function bezierEval(bez: CubicSegment, t: number): Vec {
	return bezierEvalGeneric(bez as unknown as Vec[], t);
}

// de Casteljau evaluation for a Bézier of any degree.
function bezierEvalGeneric(ctrl: Vec[], t: number): Vec {
	const tmp = ctrl.map((p) => ({ x: p.x, y: p.y }));
	for (let k = 1; k < tmp.length; k += 1) {
		for (let i = 0; i < tmp.length - k; i += 1) {
			tmp[i].x = (1 - t) * tmp[i].x + t * tmp[i + 1].x;
			tmp[i].y = (1 - t) * tmp[i].y + t * tmp[i + 1].y;
		}
	}
	return tmp[0];
}

function bernstein0(u: number): number {
	const mu = 1 - u;
	return mu * mu * mu;
}
function bernstein1(u: number): number {
	const mu = 1 - u;
	return 3 * u * mu * mu;
}
function bernstein2(u: number): number {
	const mu = 1 - u;
	return 3 * u * u * mu;
}
function bernstein3(u: number): number {
	return u * u * u;
}
