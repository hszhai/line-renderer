// ─────────────────────────────────────────────────────────────
// Flow Field — a noise-field flow over the ENTIRE model
//
// Like Contours, this is a complete render of the object — but instead of
// slicing with planes, it seeds flow strands all over the surface and lets each
// follow a shared smooth noise field (projected onto the local tangent plane so
// strokes hug the form). The result is a coherent flow/hatching that swirls
// across the whole bunny.
//
// Start points are drawn once from a seed RNG (so coverage stays put while you
// tweak length / smoothing / scale), then each strand walks the surface steered
// by the field and is smoothed into a flowing curve.
// ─────────────────────────────────────────────────────────────

import {
  jitterColor, lerpColor, pointsToGaussians, smoothPolyline,
} from './curves.ts';
import { Gaussian3D } from './gaussian-generator.ts';
import { Mesh } from './obj-loader.ts';
import { Vec3 } from './math.ts';
import { createRng, SteerFn, surfaceWalkPoints } from './surface-walk.ts';
import { noiseFieldDir, projectToTangent } from './walk-cluster.ts';

export interface FlowFieldStyle {
  density: number;     // number of flow strands seeded over the whole surface
  steps: number;       // length of each strand (vertices traversed)
  wander: number;      // jitter off the field direction
  smoothing: number;   // Chaikin iterations
  noiseScale: number;  // field frequency
  variant: number;     // field phase / start-point seed (change for a new field)
  radius: number;
  overlap: number;
  scaleMul: number;
  profile: (t: number) => number;
  // Colour & opacity gradients (mapped to vertical position over the model).
  opacity: number;     // opacity A (bottom)
  opacityB: number;    // opacity B (top)
  colorA: Vec3;
  colorB: Vec3;
  hueJitter: number;
  brightJitter: number;
}

const clamp01 = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x);

function normalAt(normals: Float32Array, i: number): Vec3 {
  const k = i * 3;
  return [normals[k], normals[k + 1], normals[k + 2]];
}

export function flowFieldToGaussians(
  mesh: Mesh,
  adjacency: number[][],
  normals: Float32Array,
  style: FlowFieldStyle
): Gaussian3D[] {
  const numVerts = mesh.vertices.length / 3;
  if (numVerts === 0) return [];

  const count = Math.max(1, Math.round(style.density));
  const phase = style.variant;

  // Draw start vertices once from a dedicated RNG so coverage is stable while
  // other params change. A separate RNG drives the walks + colour jitter.
  const seedRng = createRng((Math.round(phase * 1000) ^ 0x9e3779b1) >>> 0);
  const starts: number[] = [];
  for (let i = 0; i < count; i++) starts.push(Math.floor(seedRng() * numVerts));

  // Sort start vertices by height so the colour/opacity gradient reads as a
  // clean spatial gradient (bottom = A, top = B) across the whole model.
  let yLo = Infinity, yHi = -Infinity;
  for (let i = 1; i < mesh.vertices.length; i += 3) {
    const y = mesh.vertices[i];
    if (y < yLo) yLo = y;
    if (y > yHi) yHi = y;
  }
  const yRange = yHi - yLo || 1;

  const walkRng = createRng((Math.round(phase * 1000) ^ 0x85ebca77) >>> 0);

  // Steer along the shared noise field, kept in the local tangent plane.
  const steer: SteerFn = (pos, _heading, idx) =>
    projectToTangent(noiseFieldDir(pos, phase, style.noiseScale), normalAt(normals, idx));

  const profile = style.profile;
  const out: Gaussian3D[] = [];

  for (const start of starts) {
    let points = surfaceWalkPoints(mesh, adjacency, start, style.steps, style.wander, walkRng, steer);
    if (style.smoothing > 0) points = smoothPolyline(points, Math.round(style.smoothing));

    // Gradient by the strand's height over the model, then per-strand jitter.
    const g = (mesh.vertices[start * 3 + 1] - yLo) / yRange;
    const color = jitterColor(lerpColor(style.colorA, style.colorB, g), style.hueJitter, style.brightJitter, walkRng);
    const opacity = clamp01(style.opacity + (style.opacityB - style.opacity) * g);

    out.push(...pointsToGaussians(points, style.radius, style.overlap, style.scaleMul, opacity, color, color, profile));
  }
  return out;
}
