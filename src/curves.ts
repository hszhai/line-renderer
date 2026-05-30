import { Gaussian3D } from './gaussian-generator.ts';
import { Mesh } from './obj-loader.ts';
import { pickRandomVertex } from './mesh-utils.ts';
import { mat3ToQuat, rotateVector, Vec3, v3cross, v3dot, v3length, v3normalize, v3scale, v3sub, WORLD_SCALE } from './math.ts';

// ─────────────────────────────────────────────────────────────
// Hermite Spline Math
// ─────────────────────────────────────────────────────────────

/** Evaluate a cubic Hermite spline at parameter t ∈ [0,1]. */
export function hermitePoint(t: number, p0: Vec3, t0: Vec3, p1: Vec3, t1: Vec3): Vec3 {
  const t2 = t * t;
  const t3 = t2 * t;
  const h00 = 2 * t3 - 3 * t2 + 1;
  const h10 = t3 - 2 * t2 + t;
  const h01 = -2 * t3 + 3 * t2;
  const h11 = t3 - t2;
  return [
    h00 * p0[0] + h10 * t0[0] + h01 * p1[0] + h11 * t1[0],
    h00 * p0[1] + h10 * t0[1] + h01 * p1[1] + h11 * t1[1],
    h00 * p0[2] + h10 * t0[2] + h01 * p1[2] + h11 * t1[2],
  ];
}

/** Evaluate the derivative (tangent) of a Hermite spline at t. */
export function hermiteTangent(t: number, p0: Vec3, t0: Vec3, p1: Vec3, t1: Vec3): Vec3 {
  const t2 = t * t;
  const dh00 = 6 * t2 - 6 * t;
  const dh10 = 3 * t2 - 4 * t + 1;
  const dh01 = -6 * t2 + 6 * t;
  const dh11 = 3 * t2 - 2 * t;
  return [
    dh00 * p0[0] + dh10 * t0[0] + dh01 * p1[0] + dh11 * t1[0],
    dh00 * p0[1] + dh10 * t0[1] + dh01 * p1[1] + dh11 * t1[1],
    dh00 * p0[2] + dh10 * t0[2] + dh01 * p1[2] + dh11 * t1[2],
  ];
}

// ─────────────────────────────────────────────────────────────
// Curve → Gaussians
// ─────────────────────────────────────────────────────────────

export interface CurveParams {
  startPoint: Vec3;
  startTangent: Vec3;
  endPoint: Vec3;
  endTangent: Vec3;
  samples: number;
  radius: number;      // perpendicular tube radius (XY sigma, before WORLD_SCALE)
  overlap: number;     // along-curve coverage: 1.0 = splat spans its segment, >1 overlaps neighbors
  scaleMul: number;    // global multiplier applied to every axis
  opacity: number;     // per-splat alpha
  startColor: Vec3;
  endColor: Vec3;
}

/** Persistent seed so curves can be regenerated with new param values. */
export interface CurveSeed {
  startIdx: number;
  endIdx: number;
  hueStart: number;
  hueEnd: number;
}

/** Compute a Parallel Transport Frame along a sampled curve.
 *  Returns T (tangents), N (normals), B (binormals) for each segment.
 */
function computeParallelTransportFrames(points: Vec3[]): { T: Vec3[]; N: Vec3[]; B: Vec3[] } {
  const segCount = points.length - 1;
  const T: Vec3[] = [];
  const N: Vec3[] = [];
  const B: Vec3[] = [];

  for (let i = 0; i < segCount; i++) {
    T.push(v3normalize(v3sub(points[i + 1], points[i])));
  }

  // Initial normal: pick an arbitrary vector not parallel to T[0]
  let arb: Vec3 = [0, 1, 0];
  if (Math.abs(v3dot(T[0], arb)) > 0.99) arb = [1, 0, 0];
  N.push(v3normalize(v3cross(T[0], arb)));
  B.push(v3normalize(v3cross(T[0], N[0])));

  // Transport the frame along the curve
  for (let i = 1; i < segCount; i++) {
    const t0 = T[i - 1];
    const t1 = T[i];
    const axis = v3cross(t0, t1);
    const axisLen = v3length(axis);

    if (axisLen < 1e-8) {
      // Tangent didn't change — keep same frame
      N.push(N[i - 1]);
      B.push(B[i - 1]);
    } else {
      const cosAngle = Math.max(-1, Math.min(1, v3dot(t0, t1)));
      const angle = Math.acos(cosAngle);
      const nNew = rotateVector(N[i - 1], axis, angle);
      N.push(v3normalize(nNew));
      B.push(v3normalize(v3cross(t1, N[i])));
    }
  }

  return { T, N, B };
}

/** Build Gaussians from a Hermite curve using Parallel Transport Frames.
 *  Each Gaussian is a flat disc perpendicular to the tangent,
 *  with its local X axis aligned to the curve's normal (pointing
 *  toward the center of curvature) and Y aligned to the binormal.
 */
export function curveToGaussians(params: CurveParams): Gaussian3D[] {
  const { startPoint, startTangent, endPoint, endTangent, samples, radius, overlap, scaleMul, opacity, startColor, endColor } = params;

  const points: Vec3[] = [];
  for (let i = 0; i <= samples; i++) {
    points.push(hermitePoint(i / samples, startPoint, startTangent, endPoint, endTangent));
  }

  return pointsToGaussians(points, radius, overlap, scaleMul, opacity, startColor, endColor);
}

/** Lay a tube of Gaussians along an arbitrary point polyline. Each segment
 *  becomes one splat: thin (`radius`) on the curve's normal/binormal and
 *  stretched along the tangent, oriented by a Parallel Transport Frame so the
 *  tube doesn't twist. Shared by Hermite curves and surface walks. */
export function pointsToGaussians(
  points: Vec3[],
  radius: number,
  overlap: number,
  scaleMul: number,
  opacity: number,
  startColor: Vec3,
  endColor: Vec3
): Gaussian3D[] {
  const gaussians: Gaussian3D[] = [];
  if (points.length < 2) return gaussians;

  const frames = computeParallelTransportFrames(points);
  // Perpendicular sigma (tube radius) is constant along the curve.
  const sx = radius * WORLD_SCALE * scaleMul;   // scale in N direction (curve normal)
  const sy = radius * WORLD_SCALE * scaleMul;   // scale in B direction (binormal)

  for (let i = 0; i < points.length - 1; i++) {
    const a = points[i];
    const b = points[i + 1];

    const segLen = v3length(v3sub(b, a));
    if (segLen < 1e-8) continue;

    // Along-curve sigma stretches the splat to span its own segment, so a
    // single splat covers the gap between samples. `overlap` blends neighbors
    // (1.0 ≈ meet at ~0.6 alpha each). The +sx term gives a rounded cap.
    const sz = (segLen * 0.5) * overlap + sx;

    const mid: Vec3 = [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5];

    // Build rotation matrix: X→N, Y→B, Z→T (column-major)
    const rotMat = new Float32Array([
      frames.N[i][0], frames.N[i][1], frames.N[i][2],  // col 0
      frames.B[i][0], frames.B[i][1], frames.B[i][2],  // col 1
      frames.T[i][0], frames.T[i][1], frames.T[i][2],  // col 2
    ]);
    const rot = mat3ToQuat(rotMat);

    const t = (i + 0.5) / (points.length - 1);
    const color: Vec3 = [
      startColor[0] + (endColor[0] - startColor[0]) * t,
      startColor[1] + (endColor[1] - startColor[1]) * t,
      startColor[2] + (endColor[2] - startColor[2]) * t,
    ];

    gaussians.push({
      position: mid,
      scale: [sx, sy, sz],
      rotation: [rot[0], rot[1], rot[2], rot[3]],
      color,
      opacity,
    });
  }

  return gaussians;
}

// ─────────────────────────────────────────────────────────────
// Generative helpers
// ─────────────────────────────────────────────────────────────

/** Convert HSV → RGB. All inputs in [0,1]. */
export function hsvToRgb(h: number, s: number, v: number): Vec3 {
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s);
  const q = v * (1 - f * s);
  const t = v * (1 - (1 - f) * s);
  switch (i % 6) {
    case 0: return [v, t, p];
    case 1: return [q, v, p];
    case 2: return [p, v, t];
    case 3: return [p, q, v];
    case 4: return [t, p, v];
    case 5: return [v, p, q];
  }
  return [v, v, v];
}

/** Generate a random curve seed + Gaussians.
 *  The end-point normal is NEGATED so the curve approaches the surface
 *  from the outside (both endpoints grow outward). */
export function generateRandomCurveOnMesh(
  mesh: Mesh,
  vertexNormals: Float32Array,
  radius = 6,
  overlap = 1.0,
  scaleMul = 1.0,
  opacity = 1.0,
  samples = 40,
  tangentMultiplier = 0.6
): { gaussians: Gaussian3D[]; seed: CurveSeed } {
  const start = pickRandomVertex(mesh, vertexNormals);
  let end = pickRandomVertex(mesh, vertexNormals);
  let attempts = 0;
  while (end.index === start.index && attempts++ < 10) {
    end = pickRandomVertex(mesh, vertexNormals);
  }

  const dist = v3length(v3sub(end.position, start.position));
  const scale = Math.max(0.05, dist * tangentMultiplier);

  // Start grows outward; end approaches from outside → negate its normal
  const startTangent = v3scale(start.normal, scale);
  const endTangent = v3scale(end.normal, -scale);

  const hueStart = Math.random();
  const hueEnd = (hueStart + 0.25 + Math.random() * 0.3) % 1.0;

  const params: CurveParams = {
    startPoint: start.position,
    startTangent,
    endPoint: end.position,
    endTangent,
    samples,
    radius,
    overlap,
    scaleMul,
    opacity,
    startColor: hsvToRgb(hueStart, 0.85, 1.0),
    endColor: hsvToRgb(hueEnd, 0.85, 1.0),
  };

  return {
    gaussians: curveToGaussians(params),
    seed: { startIdx: start.index, endIdx: end.index, hueStart, hueEnd },
  };
}

/** Rebuild a CurveParams from a seed + current global settings. */
export function paramsFromSeed(
  seed: CurveSeed,
  mesh: Mesh,
  vertexNormals: Float32Array,
  radius: number,
  overlap: number,
  scaleMul: number,
  opacity: number,
  samples: number,
  tangentMultiplier: number
): CurveParams {
  const s = seed.startIdx * 3;
  const e = seed.endIdx * 3;
  const startPos: Vec3 = [mesh.vertices[s], mesh.vertices[s + 1], mesh.vertices[s + 2]];
  const endPos: Vec3 = [mesh.vertices[e], mesh.vertices[e + 1], mesh.vertices[e + 2]];
  const startNorm: Vec3 = [vertexNormals[s], vertexNormals[s + 1], vertexNormals[s + 2]];
  const endNorm: Vec3 = [vertexNormals[e], vertexNormals[e + 1], vertexNormals[e + 2]];

  const dist = v3length(v3sub(endPos, startPos));
  const scale = Math.max(0.05, dist * tangentMultiplier);

  return {
    startPoint: startPos,
    startTangent: v3scale(startNorm, scale),
    endPoint: endPos,
    endTangent: v3scale(endNorm, -scale),
    samples,
    radius,
    overlap,
    scaleMul,
    opacity,
    startColor: hsvToRgb(seed.hueStart, 0.85, 1.0),
    endColor: hsvToRgb(seed.hueEnd, 0.85, 1.0),
  };
}
