// ─────────────────────────────────────────────────────────────
// Spline through points
//
// A Kochanek–Bartels (TCB) spline that passes THROUGH every control point, with
// three shape knobs:
//   tension    — how tightly the curve hugs the points (1 = straight segments,
//                0 = round Catmull-Rom, negative = looser/overshooting).
//   bias       — pulls each tangent toward the incoming/outgoing segment.
//   continuity — sharpens (−) or rounds (+) the corners at each point.
// It reuses our Hermite evaluator, and the sampled polyline drops straight into
// pointsToGaussians so it can be enveloped by the width profile.
// ─────────────────────────────────────────────────────────────

import { hermitePoint } from './curves.ts';
import { Vec3 } from './math.ts';

export interface SplineParams {
  tension: number;          // typically -1..1
  bias: number;             // -1..1
  continuity: number;       // -1..1
  samplesPerSegment: number;
}

/** Sample a TCB spline that interpolates `points`. Returns a dense polyline. */
export function splineThroughPoints(points: Vec3[], params: SplineParams): Vec3[] {
  const n = points.length;
  if (n < 2) return points.map((p) => [...p] as Vec3);

  const T = params.tension, B = params.bias, C = params.continuity;
  const seg = Math.max(1, Math.round(params.samplesPerSegment));
  const at = (i: number): Vec3 => points[Math.max(0, Math.min(n - 1, i))];

  // KB tangent weights (endpoints reuse the clamped neighbour, giving a natural
  // one-sided tangent there).
  const a = ((1 - T) * (1 + B) * (1 + C)) / 2;
  const b = ((1 - T) * (1 - B) * (1 - C)) / 2;
  const c = ((1 - T) * (1 + B) * (1 - C)) / 2;
  const d = ((1 - T) * (1 - B) * (1 + C)) / 2;

  const out: Vec3[] = [];
  for (let i = 0; i < n - 1; i++) {
    const pPrev = at(i - 1), p0 = at(i), p1 = at(i + 1), pNext = at(i + 2);
    // Outgoing tangent at p0 and incoming tangent at p1.
    const m0: Vec3 = [
      a * (p0[0] - pPrev[0]) + b * (p1[0] - p0[0]),
      a * (p0[1] - pPrev[1]) + b * (p1[1] - p0[1]),
      a * (p0[2] - pPrev[2]) + b * (p1[2] - p0[2]),
    ];
    const m1: Vec3 = [
      c * (p1[0] - p0[0]) + d * (pNext[0] - p1[0]),
      c * (p1[1] - p0[1]) + d * (pNext[1] - p1[1]),
      c * (p1[2] - p0[2]) + d * (pNext[2] - p1[2]),
    ];
    for (let s = 0; s < seg; s++) {
      out.push(hermitePoint(s / seg, p0, m0, p1, m1));
    }
  }
  out.push([...points[n - 1]] as Vec3);
  return out;
}

/** Generate `count` control points (≥2) from a fixed start to a fixed end, with
 *  the interior points randomly offset off the start→end line by `spread`
 *  (fraction of the span). Deterministic given `rng`. */
export function randomPointsBetween(
  start: Vec3,
  end: Vec3,
  count: number,
  spread: number,
  rng: () => number = Math.random
): Vec3[] {
  const total = Math.max(2, Math.round(count));
  const span = Math.hypot(end[0] - start[0], end[1] - start[1], end[2] - start[2]) || 1;
  const amp = spread * span;
  const pts: Vec3[] = [[...start] as Vec3];
  for (let i = 1; i < total - 1; i++) {
    const t = i / (total - 1);
    pts.push([
      start[0] + (end[0] - start[0]) * t + (rng() * 2 - 1) * amp,
      start[1] + (end[1] - start[1]) * t + (rng() * 2 - 1) * amp,
      start[2] + (end[2] - start[2]) * t + (rng() * 2 - 1) * amp,
    ]);
  }
  pts.push([...end] as Vec3);
  return pts;
}
