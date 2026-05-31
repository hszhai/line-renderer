import { Mesh } from './obj-loader.ts';
import { eigenDecompose2D, Vec3, v3cross, v3length, v3normalize } from './math.ts';

/**
 * Compute per-vertex normals by averaging adjacent face normals.
 * Returns a Float32Array of the same length as mesh.vertices (3 floats per vertex).
 */
export function computeVertexNormals(mesh: Mesh): Float32Array {
  const { vertices, faces } = mesh;
  const numVerts = vertices.length / 3;
  const vNormals = new Float32Array(vertices.length);
  const vCounts = new Uint32Array(numVerts);

  for (let f = 0; f < faces.length; f += 3) {
    const i0 = faces[f] * 3;
    const i1 = faces[f + 1] * 3;
    const i2 = faces[f + 2] * 3;

    const ax = vertices[i1] - vertices[i0];
    const ay = vertices[i1 + 1] - vertices[i0 + 1];
    const az = vertices[i1 + 2] - vertices[i0 + 2];
    const bx = vertices[i2] - vertices[i0];
    const by = vertices[i2 + 1] - vertices[i0 + 1];
    const bz = vertices[i2 + 2] - vertices[i0 + 2];

    const nx = ay * bz - az * by;
    const ny = az * bx - ax * bz;
    const nz = ax * by - ay * bx;

    vNormals[i0] += nx; vNormals[i0 + 1] += ny; vNormals[i0 + 2] += nz;
    vNormals[i1] += nx; vNormals[i1 + 1] += ny; vNormals[i1 + 2] += nz;
    vNormals[i2] += nx; vNormals[i2 + 1] += ny; vNormals[i2 + 2] += nz;

    vCounts[faces[f]]++;
    vCounts[faces[f + 1]]++;
    vCounts[faces[f + 2]]++;
  }

  for (let i = 0; i < numVerts; i++) {
    const idx = i * 3;
    const count = vCounts[i];
    if (count === 0) continue;

    const nx = vNormals[idx] / count;
    const ny = vNormals[idx + 1] / count;
    const nz = vNormals[idx + 2] / count;
    const len = Math.sqrt(nx * nx + ny * ny + nz * nz);

    if (len > 1e-8) {
      vNormals[idx] = nx / len;
      vNormals[idx + 1] = ny / len;
      vNormals[idx + 2] = nz / len;
    }
  }

  return vNormals;
}

/** Vertex-cluster decimation: snap all vertices into a uniform grid, merge each
 *  occupied cell to its average position, and drop faces that collapse. Simple
 *  and fast (one pass), good enough for an artistic tool. `gridRes` = cells
 *  along the longest axis. */
function clusterDecimate(mesh: Mesh, gridRes: number): Mesh {
  const v = mesh.vertices;
  let mnx = Infinity, mny = Infinity, mnz = Infinity;
  let mxx = -Infinity, mxy = -Infinity, mxz = -Infinity;
  for (let i = 0; i < v.length; i += 3) {
    mnx = Math.min(mnx, v[i]); mxx = Math.max(mxx, v[i]);
    mny = Math.min(mny, v[i + 1]); mxy = Math.max(mxy, v[i + 1]);
    mnz = Math.min(mnz, v[i + 2]); mxz = Math.max(mxz, v[i + 2]);
  }
  const cell = (Math.max(mxx - mnx, mxy - mny, mxz - mnz) || 1) / gridRes;
  const numVerts = v.length / 3;
  const cellToNew = new Map<number, number>();
  const remap = new Int32Array(numVerts);
  const sums: number[] = [];
  const counts: number[] = [];

  for (let vi = 0; vi < numVerts; vi++) {
    const i = vi * 3;
    const ix = Math.min(gridRes - 1, Math.floor((v[i] - mnx) / cell));
    const iy = Math.min(gridRes - 1, Math.floor((v[i + 1] - mny) / cell));
    const iz = Math.min(gridRes - 1, Math.floor((v[i + 2] - mnz) / cell));
    const key = ix + iy * gridRes + iz * gridRes * gridRes;
    let idx = cellToNew.get(key);
    if (idx === undefined) {
      idx = cellToNew.size;
      cellToNew.set(key, idx);
      sums.push(0, 0, 0);
      counts.push(0);
    }
    sums[idx * 3] += v[i]; sums[idx * 3 + 1] += v[i + 1]; sums[idx * 3 + 2] += v[i + 2];
    counts[idx]++;
    remap[vi] = idx;
  }

  const newVerts = new Float32Array(cellToNew.size * 3);
  for (let idx = 0; idx < counts.length; idx++) {
    const c = counts[idx];
    newVerts[idx * 3] = sums[idx * 3] / c;
    newVerts[idx * 3 + 1] = sums[idx * 3 + 1] / c;
    newVerts[idx * 3 + 2] = sums[idx * 3 + 2] / c;
  }

  const f = mesh.faces;
  const newFaces: number[] = [];
  for (let i = 0; i < f.length; i += 3) {
    const a = remap[f[i]], b = remap[f[i + 1]], c = remap[f[i + 2]];
    if (a !== b && b !== c && a !== c) newFaces.push(a, b, c);
  }
  return { vertices: newVerts, faces: new Uint32Array(newFaces) };
}

/** Decimate a mesh down to roughly `maxVerts` vertices (no-op if already under).
 *  Sweeps the cluster grid coarser until the result fits. */
export function decimateMesh(mesh: Mesh, maxVerts: number): Mesh {
  if (mesh.vertices.length / 3 <= maxVerts) return mesh;
  let g = 128;
  let out = clusterDecimate(mesh, g);
  while (out.vertices.length / 3 > maxVerts && g > 8) {
    g = Math.max(8, Math.floor(g * 0.8));
    out = clusterDecimate(mesh, g);
  }
  return out;
}

/** Solve a 3×3 system M·x = r (M given as the 6 unique symmetric entries
 *  [m00,m01,m02,m11,m12,m22]). Returns null if near-singular. */
function solveSym3(M: number[], r: number[]): [number, number, number] | null {
  const a = M[0], b = M[1], c = M[2], e = M[3], f = M[4], i = M[5];
  // Cofactors of the symmetric matrix [[a,b,c],[b,e,f],[c,f,i]]
  const A = e * i - f * f;
  const B = -(b * i - f * c);
  const C = b * f - e * c;
  const det = a * A + b * B + c * C;
  if (Math.abs(det) < 1e-14) return null;
  const inv = 1 / det;
  const E = a * i - c * c;
  const F = -(a * f - b * c);
  const I = a * e - b * b;
  return [
    (A * r[0] + B * r[1] + C * r[2]) * inv,
    (B * r[0] + E * r[1] + F * r[2]) * inv,
    (C * r[0] + F * r[1] + I * r[2]) * inv,
  ];
}

/**
 * Estimate, per vertex, the principal direction of MINIMUM curvature — the
 * "flattest" tangent direction, the one shape-revealing hatching follows (it
 * runs along elongated forms rather than around them). Returns 3 floats per
 * vertex (a unit tangent). Fits the second fundamental form II over the 1-ring
 * by least squares, then takes the eigenvector of smallest |curvature|.
 */
export function computePrincipalDirections(
  mesh: Mesh,
  normals: Float32Array,
  adjacency: number[][]
): Float32Array {
  const { vertices } = mesh;
  const numVerts = vertices.length / 3;
  const dirs = new Float32Array(numVerts * 3);

  for (let v = 0; v < numVerts; v++) {
    const vi = v * 3;
    const n: Vec3 = [normals[vi], normals[vi + 1], normals[vi + 2]];

    // Build an orthonormal tangent basis (t1, t2) ⟂ n.
    let t1 = v3cross(n, [0, 1, 0]);
    if (v3length(t1) < 1e-4) t1 = v3cross(n, [1, 0, 0]);
    t1 = v3normalize(t1);
    const t2 = v3normalize(v3cross(n, t1));

    const px = vertices[vi], py = vertices[vi + 1], pz = vertices[vi + 2];

    // Least-squares fit of κ·len² = a·u² + 2b·uv + c·v² over neighbours.
    const M = [0, 0, 0, 0, 0, 0];
    const r = [0, 0, 0];
    let count = 0;
    for (const j of adjacency[v]) {
      const ji = j * 3;
      const ex = vertices[ji] - px, ey = vertices[ji + 1] - py, ez = vertices[ji + 2] - pz;
      const u = ex * t1[0] + ey * t1[1] + ez * t1[2];
      const w = ex * t2[0] + ey * t2[1] + ez * t2[2];
      const h = ex * n[0] + ey * n[1] + ez * n[2]; // out-of-plane component
      const len2 = u * u + w * w;
      if (len2 < 1e-12) continue;
      const kappaLen2 = (2 * h / (len2 + h * h)) * len2; // normal curvature × len²
      const f0 = u * u, f1 = 2 * u * w, f2 = w * w;
      M[0] += f0 * f0; M[1] += f0 * f1; M[2] += f0 * f2;
      M[3] += f1 * f1; M[4] += f1 * f2; M[5] += f2 * f2;
      r[0] += f0 * kappaLen2; r[1] += f1 * kappaLen2; r[2] += f2 * kappaLen2;
      count++;
    }

    let dir: Vec3 = t1; // fallback for degenerate fits
    if (count >= 3) {
      const sol = solveSym3(M, r);
      if (sol) {
        const eig = eigenDecompose2D(sol[0], sol[1], sol[2]);
        // Min |curvature| eigenvector = flattest direction → follow it.
        const useV2 = Math.abs(eig.lambda2) <= Math.abs(eig.lambda1);
        const ex = useV2 ? eig.v2x : eig.v1x;
        const ey = useV2 ? eig.v2y : eig.v1y;
        dir = v3normalize([
          t1[0] * ex + t2[0] * ey,
          t1[1] * ex + t2[1] * ey,
          t1[2] * ex + t2[2] * ey,
        ]);
      }
    }
    dirs[vi] = dir[0]; dirs[vi + 1] = dir[1]; dirs[vi + 2] = dir[2];
  }

  return dirs;
}
