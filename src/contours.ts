// ─────────────────────────────────────────────────────────────
// Contours — a complete render of the object as iso-lines
//
// Slice the whole mesh with a stack of evenly spaced parallel planes
// (perpendicular to a chosen axis) and draw each intersection curve as a
// Gaussian tube. The result is a topographic / contour-line rendering of the
// entire object.
//
// For each plane at value L we march every triangle: a triangle whose vertices
// straddle L is crossed by exactly one segment, found by linearly interpolating
// the two crossing edges. Each segment becomes one short splat, and across the
// dense mesh the segments abut into continuous contour lines.
// ─────────────────────────────────────────────────────────────

import { Gaussian3D } from './gaussian-generator.ts';
import { jitterColor, lerpColor, pointsToGaussians } from './curves.ts';
import { Mesh } from './obj-loader.ts';
import { Vec3 } from './math.ts';
import { createRng } from './surface-walk.ts';

export type ContourAxis = 'x' | 'y' | 'z';

export interface ContourStyle {
  axis: ContourAxis;
  levels: number;      // number of slicing planes
  radius: number;
  overlap: number;
  scaleMul: number;
  // Colour & opacity gradients across the level stack (A = low, B = high) + jitter.
  opacity: number;     // opacity A
  opacityB: number;    // opacity B
  colorA: Vec3;
  colorB: Vec3;
  hueJitter: number;
  brightJitter: number;
}

const clamp01 = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x);

/** Intersection segment of triangle (i0,i1,i2) with the plane axis = L, or null
 *  if the triangle isn't crossed. */
function triangleIso(V: Float32Array, ax: number, i0: number, i1: number, i2: number, L: number): Vec3[] | null {
  const idx = [i0, i1, i2];
  const f = [V[i0 * 3 + ax] - L, V[i1 * 3 + ax] - L, V[i2 * 3 + ax] - L];
  const pts: Vec3[] = [];
  const edges = [[0, 1], [1, 2], [2, 0]];
  for (const [a, b] of edges) {
    const fa = f[a], fb = f[b];
    if ((fa < 0) !== (fb < 0)) {
      const t = fa / (fa - fb); // where the edge crosses L
      const ia = idx[a] * 3, ib = idx[b] * 3;
      pts.push([
        V[ia] + (V[ib] - V[ia]) * t,
        V[ia + 1] + (V[ib + 1] - V[ia + 1]) * t,
        V[ia + 2] + (V[ib + 2] - V[ia + 2]) * t,
      ]);
    }
  }
  return pts.length === 2 ? pts : null;
}

export function contoursToGaussians(mesh: Mesh, style: ContourStyle): Gaussian3D[] {
  const ax = style.axis === 'x' ? 0 : style.axis === 'y' ? 1 : 2;
  const V = mesh.vertices;
  const F = mesh.faces;

  // Range of the field (the chosen coordinate) over the mesh.
  let lo = Infinity, hi = -Infinity;
  for (let i = ax; i < V.length; i += 3) {
    const v = V[i];
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  const range = hi - lo;
  if (range < 1e-9) return [];

  const levels = Math.max(1, Math.round(style.levels));
  const rng = createRng(0xc07024);

  const out: Gaussian3D[] = [];
  for (let l = 0; l < levels; l++) {
    const t = (l + 0.5) / levels;           // 0..1, centred so planes avoid the extremes
    const L = lo + range * t;
    // Each ring: a mix of A→B across the stack, then a touch of jitter.
    const color = jitterColor(lerpColor(style.colorA, style.colorB, t), style.hueJitter, style.brightJitter, rng);
    const opacity = clamp01(style.opacity + (style.opacityB - style.opacity) * t);
    for (let fi = 0; fi < F.length; fi += 3) {
      const seg = triangleIso(V, ax, F[fi], F[fi + 1], F[fi + 2], L);
      if (seg) {
        out.push(...pointsToGaussians(seg, style.radius, style.overlap, style.scaleMul, opacity, color, color));
      }
    }
  }
  return out;
}
