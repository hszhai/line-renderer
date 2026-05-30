import { Mesh } from './obj-loader.ts';
import { Vec3 } from './math.ts';

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

/**
 * Get a random vertex index and its position + normal from the mesh.
 */
export function pickRandomVertex(mesh: Mesh, normals: Float32Array): { index: number; position: Vec3; normal: Vec3 } {
  const numVerts = mesh.vertices.length / 3;
  const idx = Math.floor(Math.random() * numVerts);
  const i = idx * 3;
  return {
    index: idx,
    position: [mesh.vertices[i], mesh.vertices[i + 1], mesh.vertices[i + 2]],
    normal: [normals[i], normals[i + 1], normals[i + 2]],
  };
}
