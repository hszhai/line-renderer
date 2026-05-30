import { Edge, getEdgeDirection, getEdgeMidpoint } from './edge-extractor.ts';
import { Mesh } from './obj-loader.ts';
import { quatFromVectors, Vec3, vec3, v3dot, v3length, v3normalize, v3sub } from './math.ts';

export interface Gaussian3D {
  position: Vec3;
  scale: Vec3;
  rotation: [number, number, number, number]; // quaternion
  color: Vec3;
  opacity: number;
}

export type ColorMode = 'white' | 'cyan' | 'orange' | 'rainbow';

function getColor(mode: ColorMode, direction: Vec3): Vec3 {
  switch (mode) {
    case 'cyan':
      return [0.3, 0.9, 1.0];
    case 'orange':
      return [1.0, 0.6, 0.2];
    case 'rainbow': {
      // Map direction to color using a simple hash
      const dx = Math.abs(direction[0]);
      const dy = Math.abs(direction[1]);
      const dz = Math.abs(direction[2]);
      const sum = dx + dy + dz;
      if (sum < 1e-8) return [1, 1, 1];
      return [dx / sum, dy / sum, dz / sum];
    }
    case 'white':
    default:
      return [1.0, 1.0, 1.0];
  }
}

export function edgesToGaussians(
  mesh: Mesh,
  edges: Edge[],
  lineWidth: number,
  colorMode: ColorMode
): Gaussian3D[] {
  const gaussians: Gaussian3D[] = [];

  for (const edge of edges) {
    const pos = getEdgeMidpoint(mesh, edge);
    const dir = getEdgeDirection(mesh, edge);

    const i0 = edge.v0 * 3;
    const i1 = edge.v1 * 3;
    const dx = mesh.vertices[i1] - mesh.vertices[i0];
    const dy = mesh.vertices[i1 + 1] - mesh.vertices[i0 + 1];
    const dz = mesh.vertices[i1 + 2] - mesh.vertices[i0 + 2];
    const len = Math.sqrt(dx * dx + dy * dy + dz * dz);

    // Rotation: align local Z axis with edge direction
    const rot = quatFromVectors([0, 0, 1], dir);

    // Convert pixel-like thickness to world-space scale.
    // The bunny is ~0.15 units tall; at distance ~0.8 with 45° FOV,
    // 1 px ≈ 0.00035 world units. We use 0.0003 as a practical factor.
    const worldThickness = lineWidth * 0.0003;
    const scale: Vec3 = [worldThickness, worldThickness, len * 0.5 + worldThickness];

    gaussians.push({
      position: pos,
      scale,
      rotation: [rot[0], rot[1], rot[2], rot[3]],
      color: getColor(colorMode, dir),
      opacity: 1.0,
    });
  }

  return gaussians;
}
