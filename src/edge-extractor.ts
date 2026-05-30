import { Mesh } from './obj-loader.ts';
import { Vec3, v3cross, v3dot, v3length, v3normalize, v3sub } from './math.ts';

export interface Edge {
  v0: number;
  v1: number;
}

export function extractAllEdges(mesh: Mesh): Edge[] {
  const edgeSet = new Set<string>();
  const edges: Edge[] = [];
  const faces = mesh.faces;

  for (let i = 0; i < faces.length; i += 3) {
    const a = faces[i];
    const b = faces[i + 1];
    const c = faces[i + 2];

    const addEdge = (v0: number, v1: number) => {
      const i0 = Math.min(v0, v1);
      const i1 = Math.max(v0, v1);
      const key = `${i0}_${i1}`;
      if (!edgeSet.has(key)) {
        edgeSet.add(key);
        edges.push({ v0: i0, v1: i1 });
      }
    };

    addEdge(a, b);
    addEdge(b, c);
    addEdge(c, a);
  }

  return edges;
}

export function computeFaceNormals(mesh: Mesh): Float32Array {
  const { vertices, faces } = mesh;
  const normals = new Float32Array((faces.length / 3) * 3);

  for (let f = 0, j = 0; f < faces.length; f += 3, j += 3) {
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

    const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
    if (len > 1e-8) {
      normals[j] = nx / len;
      normals[j + 1] = ny / len;
      normals[j + 2] = nz / len;
    }
  }

  return normals;
}

export function extractSilhouetteEdges(mesh: Mesh, cameraPos: Vec3): Edge[] {
  const { vertices, faces } = mesh;
  const edgeFaces = new Map<string, number[]>();

  // Build edge -> face adjacency
  for (let f = 0; f < faces.length / 3; f++) {
    const a = faces[f * 3];
    const b = faces[f * 3 + 1];
    const c = faces[f * 3 + 2];

    const add = (v0: number, v1: number) => {
      const i0 = Math.min(v0, v1);
      const i1 = Math.max(v0, v1);
      const key = `${i0}_${i1}`;
      if (!edgeFaces.has(key)) edgeFaces.set(key, []);
      edgeFaces.get(key)!.push(f);
    };

    add(a, b);
    add(b, c);
    add(c, a);
  }

  const faceNormals = computeFaceNormals(mesh);
  const silEdges: Edge[] = [];

  for (const [key, fidxs] of edgeFaces) {
    if (fidxs.length === 1) {
      // Boundary edge - always a silhouette
      const [v0, v1] = key.split('_').map(Number);
      silEdges.push({ v0, v1 });
    } else if (fidxs.length === 2) {
      const f0 = fidxs[0];
      const f1 = fidxs[1];

      const n0x = faceNormals[f0 * 3];
      const n0y = faceNormals[f0 * 3 + 1];
      const n0z = faceNormals[f0 * 3 + 2];
      const n1x = faceNormals[f1 * 3];
      const n1y = faceNormals[f1 * 3 + 1];
      const n1z = faceNormals[f1 * 3 + 2];

      // Compute view direction from camera to edge midpoint
      const [v0, v1] = key.split('_').map(Number);
      const midX = (vertices[v0 * 3] + vertices[v1 * 3]) * 0.5;
      const midY = (vertices[v0 * 3 + 1] + vertices[v1 * 3 + 1]) * 0.5;
      const midZ = (vertices[v0 * 3 + 2] + vertices[v1 * 3 + 2]) * 0.5;
      const vdX = midX - cameraPos[0];
      const vdY = midY - cameraPos[1];
      const vdZ = midZ - cameraPos[2];
      const vLen = Math.sqrt(vdX * vdX + vdY * vdY + vdZ * vdZ);
      if (vLen < 1e-8) continue;
      const vdx = vdX / vLen;
      const vdy = vdY / vLen;
      const vdz = vdZ / vLen;

      const d0 = n0x * vdx + n0y * vdy + n0z * vdz;
      const d1 = n1x * vdx + n1y * vdy + n1z * vdz;

      // Silhouette: one face faces toward camera, the other faces away
      // Note: In view space, camera looks down -Z, so facing camera means dot < 0
      // But here viewDir points from camera to object, so facing camera means dot > 0
      // Actually, let's just check sign difference
      if ((d0 >= 0) !== (d1 >= 0)) {
        silEdges.push({ v0, v1 });
      }
    }
  }

  return silEdges;
}

export function getEdgeMidpoint(mesh: Mesh, edge: Edge): Vec3 {
  const i0 = edge.v0 * 3;
  const i1 = edge.v1 * 3;
  return [
    (mesh.vertices[i0] + mesh.vertices[i1]) * 0.5,
    (mesh.vertices[i0 + 1] + mesh.vertices[i1 + 1]) * 0.5,
    (mesh.vertices[i0 + 2] + mesh.vertices[i1 + 2]) * 0.5,
  ];
}

export function getEdgeDirection(mesh: Mesh, edge: Edge): Vec3 {
  const i0 = edge.v0 * 3;
  const i1 = edge.v1 * 3;
  return v3normalize([
    mesh.vertices[i1] - mesh.vertices[i0],
    mesh.vertices[i1 + 1] - mesh.vertices[i0 + 1],
    mesh.vertices[i1 + 2] - mesh.vertices[i0 + 2],
  ]);
}
