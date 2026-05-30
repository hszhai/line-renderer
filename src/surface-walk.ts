// ─────────────────────────────────────────────────────────────
// Surface Walk
//
// Start at a vertex on the mesh and "walk" across the surface vertex-to-vertex
// for N steps. At each vertex we look at the neighbour vertices (excluding the
// one we just came from) and choose the next hop. A `wander` knob blends:
//   0 → always continue as straight as possible (smallest turn): a geodesic-
//       like strand that shoots across the surface.
//   1 → pick a neighbour uniformly at random: a meandering squiggle.
// Every vertex lies exactly on the mesh, so the resulting polyline hugs it.
// ─────────────────────────────────────────────────────────────

import { Gaussian3D } from './gaussian-generator.ts';
import { hsvToRgb, pointsToGaussians } from './curves.ts';
import { Mesh } from './obj-loader.ts';
import { Vec3, v3dot, v3normalize, v3sub } from './math.ts';

/** Persistent seed so a walk can be regenerated with new params. The path is
 *  fully determined by the start vertex + RNG seed; `steps` and `wander` are
 *  applied live at regeneration time (like a curve's `samples`). */
export interface WalkSeed {
  startIdx: number;
  rngSeed: number;
  hueStart: number;
  hueEnd: number;
}

/** Build a vertex → unique-neighbour adjacency list from the mesh triangles.
 *  Computed once per mesh and reused for every walk. */
export function buildVertexAdjacency(mesh: Mesh): number[][] {
  const numVerts = mesh.vertices.length / 3;
  const sets: Set<number>[] = Array.from({ length: numVerts }, () => new Set<number>());
  const f = mesh.faces;
  for (let i = 0; i < f.length; i += 3) {
    const a = f[i], b = f[i + 1], c = f[i + 2];
    sets[a].add(b); sets[a].add(c);
    sets[b].add(a); sets[b].add(c);
    sets[c].add(a); sets[c].add(b);
  }
  return sets.map((s) => Array.from(s));
}

/** Small deterministic PRNG so a stored seed always replays the same walk. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function vertexPos(mesh: Mesh, idx: number): Vec3 {
  const i = idx * 3;
  return [mesh.vertices[i], mesh.vertices[i + 1], mesh.vertices[i + 2]];
}

interface Candidate { idx: number; dir: Vec3; straightness: number; }

/** Choose the next vertex. `wander` ∈ [0,1] blends from straightest (argmax
 *  alignment with the current heading) to uniformly random. */
function chooseNext(cands: Candidate[], wander: number, rng: () => number): Candidate {
  if (cands.length === 1) return cands[0];

  if (wander < 1e-3) {
    let best = cands[0];
    for (const c of cands) if (c.straightness > best.straightness) best = c;
    return best;
  }

  // Weighted random: weight = alignment^exponent. High exponent (low wander)
  // sharply favours the straightest option; exponent 0 (wander=1) is uniform.
  const exponent = (1 - wander) * 10;
  let total = 0;
  const weights = cands.map((c) => {
    const w = Math.pow(Math.max(1e-4, (c.straightness + 1) * 0.5), exponent);
    total += w;
    return w;
  });
  let r = rng() * total;
  for (let i = 0; i < cands.length; i++) {
    r -= weights[i];
    if (r <= 0) return cands[i];
  }
  return cands[cands.length - 1];
}

/** Produce the polyline of vertex positions for a surface walk. */
export function surfaceWalkPoints(
  mesh: Mesh,
  adjacency: number[][],
  startIdx: number,
  steps: number,
  wander: number,
  rng: () => number
): Vec3[] {
  const points: Vec3[] = [vertexPos(mesh, startIdx)];

  const startNbrs = adjacency[startIdx];
  if (!startNbrs || startNbrs.length === 0) return points;

  // Seed the heading with a random neighbour of the start vertex.
  let current = startIdx;
  let next = startNbrs[Math.floor(rng() * startNbrs.length)];
  let heading = v3normalize(v3sub(vertexPos(mesh, next), vertexPos(mesh, current)));

  for (let s = 0; s < steps; s++) {
    const prev = current;
    current = next;
    points.push(vertexPos(mesh, current));

    const here = vertexPos(mesh, current);
    const cands: Candidate[] = [];
    for (const c of adjacency[current]) {
      if (c === prev) continue; // don't immediately backtrack
      const dir = v3normalize(v3sub(vertexPos(mesh, c), here));
      cands.push({ idx: c, dir, straightness: v3dot(dir, heading) });
    }
    if (cands.length === 0) break; // dead end (only neighbour was prev)

    const chosen = chooseNext(cands, wander, rng);
    next = chosen.idx;
    heading = chosen.dir; // continue relative to the step we actually took
  }

  return points;
}

/** Rebuild a walk's Gaussians from its seed + current global splat params. */
export function walkSeedToGaussians(
  seed: WalkSeed,
  mesh: Mesh,
  adjacency: number[][],
  radius: number,
  overlap: number,
  scaleMul: number,
  opacity: number,
  steps: number,
  wander: number
): Gaussian3D[] {
  const rng = mulberry32(seed.rngSeed);
  const points = surfaceWalkPoints(mesh, adjacency, seed.startIdx, steps, wander, rng);
  return pointsToGaussians(
    points, radius, overlap, scaleMul, opacity,
    hsvToRgb(seed.hueStart, 0.85, 1.0),
    hsvToRgb(seed.hueEnd, 0.85, 1.0)
  );
}

/** Start a new random surface walk and return its Gaussians + replayable seed. */
export function generateSurfaceWalk(
  mesh: Mesh,
  adjacency: number[][],
  radius: number,
  overlap: number,
  scaleMul: number,
  opacity: number,
  steps: number,
  wander: number
): { gaussians: Gaussian3D[]; seed: WalkSeed } {
  const numVerts = mesh.vertices.length / 3;
  const hueStart = Math.random();
  const seed: WalkSeed = {
    startIdx: Math.floor(Math.random() * numVerts),
    rngSeed: (Math.random() * 0xffffffff) >>> 0,
    hueStart,
    hueEnd: (hueStart + 0.25 + Math.random() * 0.3) % 1.0,
  };
  return {
    gaussians: walkSeedToGaussians(seed, mesh, adjacency, radius, overlap, scaleMul, opacity, steps, wander),
    seed,
  };
}
