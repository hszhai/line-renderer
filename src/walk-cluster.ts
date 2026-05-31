// ─────────────────────────────────────────────────────────────
// Cluster — the unified modeling primitive
//
// A cluster is a bundle of strands grown from a local patch of the mesh. Each
// strand is either:
//   'walk'  → a surface walk (vertex-to-vertex), steered by a strategy:
//               'direction' (all toward one shared world direction),
//               'noise'     (follow a shared smooth flow field),
//               'random'    (each on its own).
//   'curve' → a Hermite curve bursting from the patch out to a random vertex.
//
// A "single" strand is just a cluster with count = 1.
//
// Style (colour, splat sizes, and per-strand variation in width / opacity /
// length / colour) is applied LIVE from the UI, so editing sliders re-styles
// every existing cluster. Only the structural identity (strand type, steering
// strategy, center, RNG) is stored in the seed so a cluster replays identically.
// ─────────────────────────────────────────────────────────────

import { CurveParams, curveToGaussians, hsvToRgb, pointsToGaussians, rgbToHsv, smoothPolyline } from './curves.ts';
import { Gaussian3D } from './gaussian-generator.ts';
import { Mesh } from './obj-loader.ts';
import { Vec3, v3length, v3normalize, v3scale, v3sub } from './math.ts';
import { createRng, gatherNearbyVertices, SteerFn, surfaceWalkPoints, vertexPos } from './surface-walk.ts';

export type StrandType = 'walk' | 'curve';
export type ClusterStrategy = 'direction' | 'noise' | 'curvature' | 'random';

/** Structural identity of a cluster — everything needed to replay it exactly.
 *  Style lives separately (see ClusterStyle) so it can be edited live. */
export interface ClusterSeed {
  strandType: StrandType;
  strategy: ClusterStrategy;
  centerIdx: number;
  direction: Vec3;   // shared flow direction for the 'direction' strategy
  noiseSeed: number; // phase offset for the 'noise' field
  rngSeed: number;
}

/** Global, live styling shared by all clusters. */
export interface ClusterStyle {
  // Cluster shape
  count: number;        // strands per cluster
  spread: number;       // start-patch radius (vertex rings)
  // Walk strands
  steps: number;
  wander: number;
  noiseScale: number;
  smoothing: number;   // Chaikin iterations: 0 = raw faceted path, higher = smoother
  // Curve strands
  samples: number;
  tangentMult: number;
  // Splat / Gaussian
  radius: number;
  overlap: number;
  scaleMul: number;
  opacity: number;
  // Colour (base + jitter)
  baseColor: Vec3;
  hueJitter: number;
  brightJitter: number;
  // Per-strand variation (0..1 fractions)
  widthVar: number;
  opacityVar: number;
  lengthVar: number;
}

const clamp01 = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x);

/** A smooth pseudo flow field: a unit direction that varies continuously over
 *  space so neighbouring strands swirl together. `scale` sets the turn rate. */
function noiseFieldDir(pos: Vec3, seed: number, scale: number): Vec3 {
  const x = pos[0] * scale, y = pos[1] * scale, z = pos[2] * scale;
  const dx = Math.sin(y + seed) + Math.cos(z * 1.3 + seed * 1.7);
  const dy = Math.sin(z + seed * 2.1) + Math.cos(x * 1.3 + seed * 0.7);
  const dz = Math.sin(x + seed * 1.4) + Math.cos(y * 1.3 + seed * 2.3);
  return v3normalize([dx, dy, dz]);
}

/** Project a world direction onto the tangent plane at normal `n`, so a steered
 *  strand flows ALONG the surface instead of trying to leave it. */
function projectToTangent(d: Vec3, n: Vec3): Vec3 {
  const dot = d[0] * n[0] + d[1] * n[1] + d[2] * n[2];
  return v3normalize([d[0] - dot * n[0], d[1] - dot * n[1], d[2] - dot * n[2]]);
}

function normalAt(normals: Float32Array, i: number): Vec3 {
  const k = i * 3;
  return [normals[k], normals[k + 1], normals[k + 2]];
}

function makeSteer(
  seed: ClusterSeed,
  normals: Float32Array,
  principalDirs: Float32Array,
  noiseScale: number
): SteerFn | undefined {
  switch (seed.strategy) {
    case 'direction':
      // Global direction, but kept in the local tangent plane → flows over form.
      return (_pos, _heading, idx) => projectToTangent(seed.direction, normalAt(normals, idx));
    case 'noise':
      return (pos, _heading, idx) =>
        projectToTangent(noiseFieldDir(pos, seed.noiseSeed, noiseScale), normalAt(normals, idx));
    case 'curvature':
      // Follow the principal-curvature field; flip 180° to stay aligned with the
      // heading (principal directions are sign-ambiguous line fields).
      return (_pos, heading, idx) => {
        const k = idx * 3;
        const d: Vec3 = [principalDirs[k], principalDirs[k + 1], principalDirs[k + 2]];
        const dot = d[0] * heading[0] + d[1] * heading[1] + d[2] * heading[2];
        return dot < 0 ? [-d[0], -d[1], -d[2]] : d;
      };
    default:
      return undefined; // 'random' → heading-based, jitter via wander
  }
}

/** Jitter a base colour in HSV: hue by ±hueJitter, value by ±brightJitter. */
function jitterColor(base: Vec3, hueJitter: number, brightJitter: number, rng: () => number): Vec3 {
  const [h, s, v] = rgbToHsv(base);
  const nh = (h + (rng() * 2 - 1) * hueJitter * 0.5 + 1) % 1;
  const nv = clamp01(v * (1 + (rng() * 2 - 1) * brightJitter));
  return hsvToRgb(nh, s, nv);
}

function vertexNormal(normals: Float32Array, idx: number): Vec3 {
  const i = idx * 3;
  return [normals[i], normals[i + 1], normals[i + 2]];
}

/** Rebuild a cluster's Gaussians from its seed + current global style. */
export function clusterSeedToGaussians(
  seed: ClusterSeed,
  mesh: Mesh,
  adjacency: number[][],
  vertexNormals: Float32Array,
  principalDirs: Float32Array,
  style: ClusterStyle
): Gaussian3D[] {
  const numVerts = mesh.vertices.length / 3;
  const rng = createRng(seed.rngSeed);
  const pool = gatherNearbyVertices(adjacency, seed.centerIdx, Math.max(1, Math.round(style.spread)));
  const steer = makeSteer(seed, vertexNormals, principalDirs, style.noiseScale);

  const out: Gaussian3D[] = [];
  const n = Math.max(1, Math.round(style.count));
  for (let i = 0; i < n; i++) {
    // Per-strand jittered style.
    const radius = Math.max(0.05, style.radius * (1 + (rng() * 2 - 1) * style.widthVar));
    const opacity = clamp01(style.opacity * (1 + (rng() * 2 - 1) * style.opacityVar));
    const lengthMul = Math.max(0.2, 1 + (rng() * 2 - 1) * style.lengthVar);
    const color = jitterColor(style.baseColor, style.hueJitter, style.brightJitter, rng);
    const startIdx = pool[Math.floor(rng() * pool.length)];

    if (seed.strandType === 'walk') {
      const steps = Math.max(2, Math.round(style.steps * lengthMul));
      let points = surfaceWalkPoints(mesh, adjacency, startIdx, steps, style.wander, rng, steer);
      if (style.smoothing > 0) points = smoothPolyline(points, Math.round(style.smoothing));
      out.push(...pointsToGaussians(points, radius, style.overlap, style.scaleMul, opacity, color, color));
    } else {
      // Curve strand: burst from the patch out to a random vertex.
      const endIdx = Math.floor(rng() * numVerts);
      const start = vertexPos(mesh, startIdx);
      const end = vertexPos(mesh, endIdx);
      const dist = v3length(v3sub(end, start));
      const tScale = Math.max(0.05, dist * style.tangentMult);
      const samples = Math.max(2, Math.round(style.samples * lengthMul));
      const params: CurveParams = {
        startPoint: start,
        startTangent: v3scale(vertexNormal(vertexNormals, startIdx), tScale),
        endPoint: end,
        endTangent: v3scale(vertexNormal(vertexNormals, endIdx), -tScale),
        samples,
        radius,
        overlap: style.overlap,
        scaleMul: style.scaleMul,
        opacity,
        startColor: color,
        endColor: color,
      };
      out.push(...curveToGaussians(params));
    }
  }
  return out;
}

/** Create a new cluster seed (structural identity only; styling is live). */
export function createClusterSeed(
  mesh: Mesh,
  strandType: StrandType,
  strategy: ClusterStrategy
): ClusterSeed {
  const numVerts = mesh.vertices.length / 3;
  return {
    strandType,
    strategy,
    centerIdx: Math.floor(Math.random() * numVerts),
    direction: v3normalize([Math.random() * 2 - 1, Math.random() * 2 - 1, Math.random() * 2 - 1]),
    noiseSeed: Math.random() * 1000,
    rngSeed: (Math.random() * 0xffffffff) >>> 0,
  };
}
