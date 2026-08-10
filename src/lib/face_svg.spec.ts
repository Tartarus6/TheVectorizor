import { describe, it, expect } from 'vitest';
import {
	fitPolyline,
	reverseCubicSegments,
	canonicalizeOpen,
	bezierEval,
	FIT_ERROR
} from './face_svg';

type Vec = { x: number; y: number };

function normalize(v: Vec): Vec {
	const len = Math.hypot(v.x, v.y);
	return len < 1e-12 ? { x: 0, y: 0 } : { x: v.x / len, y: v.y / len };
}
function sub(a: Vec, b: Vec): Vec {
	return { x: a.x - b.x, y: a.y - b.y };
}

// Minimum distance from a point to the fitted curve. Sampled finely (segments
// can be long, so coarse sampling would overestimate the true closest distance).
function distToCurve(segs: [Vec, Vec, Vec, Vec][], p: Vec): number {
	let best = Infinity;
	for (const seg of segs) {
		for (let i = 0; i <= 500; i += 1) {
			const q = bezierEval(seg, i / 500);
			const d = Math.hypot(q.x - p.x, q.y - p.y);
			if (d < best) best = d;
		}
	}
	return best;
}

function fitWithEndpointTangents(pts: Vec[]) {
	const left = normalize(sub(pts[1], pts[0]));
	const right = normalize(sub(pts[pts.length - 2], pts[pts.length - 1]));
	return fitPolyline(pts, left, right);
}

describe('Schneider fit honors FIT_ERROR', () => {
	it('fits a straight line with near-zero error', () => {
		const pts: Vec[] = [];
		for (let i = 0; i <= 20; i += 1) pts.push({ x: i * 0.9, y: 3 + i * 0.4 });
		const segs = fitWithEndpointTangents(pts);
		for (const p of pts) expect(distToCurve(segs, p)).toBeLessThan(FIT_ERROR);
	});

	it('fits a circular arc within tolerance at every sample', () => {
		const pts: Vec[] = [];
		const R = 50;
		for (let i = 0; i <= 60; i += 1) {
			const a = (Math.PI / 2) * (i / 60); // quarter circle
			pts.push({ x: R * Math.cos(a), y: R * Math.sin(a) });
		}
		const segs = fitWithEndpointTangents(pts);
		// Small numerical slack beyond the tolerance threshold.
		for (const p of pts) expect(distToCurve(segs, p)).toBeLessThan(FIT_ERROR + 1e-3);
	});

	it('fits an S-curve within tolerance', () => {
		const pts: Vec[] = [];
		for (let i = 0; i <= 80; i += 1) {
			const x = i * 0.5;
			pts.push({ x, y: 20 * Math.sin(x / 8) });
		}
		const segs = fitWithEndpointTangents(pts);
		for (const p of pts) expect(distToCurve(segs, p)).toBeLessThan(FIT_ERROR + 1e-3);
	});
});

describe('reverseCubicSegments is an exact inverse', () => {
	it('double reversal returns the original control points', () => {
		const pts: Vec[] = [];
		for (let i = 0; i <= 30; i += 1) pts.push({ x: i, y: 10 * Math.sin(i / 5) });
		const segs = fitWithEndpointTangents(pts);
		const back = reverseCubicSegments(reverseCubicSegments(segs));
		expect(back).toEqual(segs);
	});

	it('reversed curve traces the same geometry (rev(t) == fwd(1-t))', () => {
		const seg: [Vec, Vec, Vec, Vec] = [
			{ x: 0, y: 0 },
			{ x: 1, y: 4 },
			{ x: 5, y: 4 },
			{ x: 6, y: 0 }
		];
		const rev = reverseCubicSegments([seg])[0];
		for (let i = 0; i <= 10; i += 1) {
			const t = i / 10;
			const a = bezierEval(seg, t);
			const b = bezierEval(rev, 1 - t);
			expect(Math.hypot(a.x - b.x, a.y - b.y)).toBeLessThan(1e-9);
		}
	});
});

describe('shared-edge gap-safety: twin faces resolve to identical geometry', () => {
	it('a pixel sequence and its reverse canonicalize to the same key', () => {
		const forward = [5, 6, 17, 28, 40];
		const backward = forward.slice().reverse();
		const a = canonicalizeOpen(forward);
		const b = canonicalizeOpen(backward);
		expect(a.canonicalPix).toEqual(b.canonicalPix);
		// Exactly one of the two faces must reverse the cached fit.
		expect(a.reversed).not.toEqual(b.reversed);
	});

	it('both faces emit the same curve for a shared edge', () => {
		// A shared chain seen forward by face A and reversed by face B. Both fit
		// the canonical order once, then whichever is "reversed" flips the result.
		const pts: Vec[] = [];
		for (let i = 0; i <= 25; i += 1) pts.push({ x: i, y: 8 * Math.sin(i / 6) });

		const canonical = fitWithEndpointTangents(pts);

		const emittedA = canonical; // face traversing in canonical order
		const emittedB = reverseCubicSegments(canonical); // twin traversing reversed

		// The two emitted curves must be the same geometry (one is the exact
		// reverse of the other), so no gap can exist between the faces.
		const reReversed = reverseCubicSegments(emittedB);
		expect(reReversed).toEqual(emittedA);

		// Endpoints coincide exactly.
		expect(emittedA[0][0]).toEqual(emittedB[emittedB.length - 1][3]);
		expect(emittedA[emittedA.length - 1][3]).toEqual(emittedB[0][0]);
	});
});
