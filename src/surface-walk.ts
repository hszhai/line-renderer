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

import { Mesh } from './obj-loader.ts';
import { Vec3, v3dot, v3length, v3normalize, v3sub } from './math.ts';

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
export function createRng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function vertexPos(mesh: Mesh, idx: number): Vec3 {
  const i = idx * 3;
  return [mesh.vertices[i], mesh.vertices[i + 1], mesh.vertices[i + 2]];
}

/** Collect vertices within `rings` graph hops of `center` (breadth-first).
 *  Used to scatter a cluster's strand start points over a local patch. */
export function gatherNearbyVertices(adjacency: number[][], center: number, rings: number): number[] {
  const visited = new Set<number>([center]);
  let frontier = [center];
  for (let r = 0; r < rings; r++) {
    const next: number[] = [];
    for (const v of frontier) {
      for (const n of adjacency[v]) {
        if (!visited.has(n)) {
          visited.add(n);
          next.push(n);
        }
      }
    }
    if (next.length === 0) break;
    frontier = next;
  }
  return Array.from(visited);
}

/** A steering function: given the current position, heading, and the current
 *  vertex index, returns the world-space direction the walk "wants" to go.
 *  Candidate hops are scored by alignment with this. Omit to walk straightest
 *  (continue along the heading). */
export type SteerFn = (pos: Vec3, heading: Vec3, vertexIdx: number) => Vec3;

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
  rng: () => number,
  steer?: SteerFn
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
    const here = vertexPos(mesh, current);
    points.push(here);

    // The reference direction candidates are scored against: a steering target
    // (global direction / noise field) if supplied, else the current heading.
    let ref = heading;
    if (steer) {
      const want = v3normalize(steer(here, heading, current));
      if (v3length(want) > 1e-6) ref = want;
    }

    const cands: Candidate[] = [];
    for (const c of adjacency[current]) {
      if (c === prev) continue; // don't immediately backtrack
      const dir = v3normalize(v3sub(vertexPos(mesh, c), here));
      cands.push({ idx: c, dir, straightness: v3dot(dir, ref) });
    }
    if (cands.length === 0) break; // dead end (only neighbour was prev)

    const chosen = chooseNext(cands, wander, rng);
    next = chosen.idx;
    heading = chosen.dir; // continue relative to the step we actually took
  }

  return points;
}
