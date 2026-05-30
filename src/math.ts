// Minimal 3D math utilities for the Gaussian Splatting renderer

export type Vec3 = [number, number, number];
export type Vec4 = [number, number, number, number];
export type Mat4 = Float32Array; // 16 elements, column-major

export function vec3(x = 0, y = 0, z = 0): Vec3 {
  return [x, y, z];
}

export function v3add(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

export function v3sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

export function v3scale(a: Vec3, s: number): Vec3 {
  return [a[0] * s, a[1] * s, a[2] * s];
}

export function v3dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function v3cross(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

export function v3length(a: Vec3): number {
  return Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}

export function v3normalize(a: Vec3): Vec3 {
  const len = v3length(a);
  if (len < 1e-8) return [0, 0, 0];
  return [a[0] / len, a[1] / len, a[2] / len];
}

export function mat4Identity(): Mat4 {
  return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
}

export function mat4Perspective(fovY: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1.0 / Math.tan(fovY / 2);
  const nf = 1 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, 2 * far * near * nf, 0,
  ]);
}

export function mat4LookAt(eye: Vec3, target: Vec3, up: Vec3): Mat4 {
  const zAxis = v3normalize(v3sub(eye, target));
  const xAxis = v3normalize(v3cross(up, zAxis));
  const yAxis = v3cross(zAxis, xAxis);

  return new Float32Array([
    xAxis[0], yAxis[0], zAxis[0], 0,
    xAxis[1], yAxis[1], zAxis[1], 0,
    xAxis[2], yAxis[2], zAxis[2], 0,
    -v3dot(xAxis, eye), -v3dot(yAxis, eye), -v3dot(zAxis, eye), 1,
  ]);
}

export function mat4Multiply(a: Mat4, b: Mat4): Mat4 {
  const out = new Float32Array(16);
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) {
        sum += a[k * 4 + i] * b[j * 4 + k];
      }
      out[j * 4 + i] = sum;
    }
  }
  return out;
}

export function mat4TransformPoint(m: Mat4, p: Vec3): Vec3 {
  const x = p[0], y = p[1], z = p[2];
  const w = m[3] * x + m[7] * y + m[11] * z + m[15];
  if (Math.abs(w) < 1e-8) return [0, 0, 0];
  return [
    (m[0] * x + m[4] * y + m[8] * z + m[12]) / w,
    (m[1] * x + m[5] * y + m[9] * z + m[13]) / w,
    (m[2] * x + m[6] * y + m[10] * z + m[14]) / w,
  ];
}

export function mat4TransformDirection(m: Mat4, d: Vec3): Vec3 {
  const x = d[0], y = d[1], z = d[2];
  return [
    m[0] * x + m[4] * y + m[8] * z,
    m[1] * x + m[5] * y + m[9] * z,
    m[2] * x + m[6] * y + m[10] * z,
  ];
}

export function quatFromVectors(from: Vec3, to: Vec3): Vec4 {
  const a = v3normalize(from);
  const b = v3normalize(to);
  const dot = v3dot(a, b);

  if (dot > 0.999999) {
    return [0, 0, 0, 1];
  }

  if (dot < -0.999999) {
    // 180-degree rotation around a perpendicular axis
    let perp: Vec3 = [1, 0, 0];
    if (Math.abs(a[1]) < Math.abs(a[0])) perp = [0, 1, 0];
    const axis = v3normalize(v3cross(a, perp));
    return [axis[0], axis[1], axis[2], 0];
  }

  const c = v3cross(a, b);
  const w = 1 + dot;
  const len = Math.sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2] + w * w);
  return [c[0] / len, c[1] / len, c[2] / len, w / len];
}

export function quatToMat3(q: Vec4): Float32Array {
  const x = q[0], y = q[1], z = q[2], w = q[3];
  const xx = x * x, yy = y * y, zz = z * z;
  const xy = x * y, xz = x * z, yz = y * z;
  const wx = w * x, wy = w * y, wz = w * z;

  return new Float32Array([
    1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
    2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
    2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
  ]);
}

// Compute eigenvalues and eigenvectors of a 2x2 symmetric matrix [[a, b], [b, c]]
export interface Eigen2D {
  lambda1: number;
  lambda2: number;
  v1x: number;
  v1y: number;
  v2x: number;
  v2y: number;
}

export function eigenDecompose2D(a: number, b: number, c: number): Eigen2D {
  const trace = a + c;
  const det = a * c - b * b;
  const discriminant = Math.sqrt(Math.max(0, (a - c) * (a - c) + 4 * b * b));
  const lambda1 = (trace + discriminant) * 0.5;
  const lambda2 = (trace - discriminant) * 0.5;

  let v1x: number, v1y: number;
  if (Math.abs(b) < 1e-8) {
    v1x = 1; v1y = 0;
  } else {
    v1x = lambda1 - c;
    v1y = b;
    const len1 = Math.sqrt(v1x * v1x + v1y * v1y);
    v1x /= len1; v1y /= len1;
  }

  // Second eigenvector is perpendicular to first
  const v2x = -v1y;
  const v2y = v1x;

  return { lambda1, lambda2, v1x, v1y, v2x, v2y };
}

// Invert a 2x2 symmetric matrix [[a, b], [b, c]]
export function invertSym2D(a: number, b: number, c: number): { x: number; y: number; z: number } {
  const det = a * c - b * b;
  if (Math.abs(det) < 1e-8) return { x: 1, y: 0, z: 1 };
  return { x: c / det, y: -b / det, z: a / det };
}

/** Rotate a vector v around an axis by an angle (Rodrigues' formula). */
export function rotateVector(v: Vec3, axis: Vec3, angle: number): Vec3 {
  const k = v3normalize(axis);
  const cosA = Math.cos(angle);
  const sinA = Math.sin(angle);
  const dot = v3dot(k, v);
  const term1 = v3scale(v, cosA);
  const term2 = v3scale(v3cross(k, v), sinA);
  const term3 = v3scale(k, dot * (1 - cosA));
  return v3add(v3add(term1, term2), term3);
}

/** Convert a column-major 3×3 rotation matrix to a quaternion [x, y, z, w]. */
export function mat3ToQuat(m: Float32Array): Vec4 {
  const trace = m[0] + m[4] + m[8];
  let x: number, y: number, z: number, w: number;

  if (trace > 0) {
    const s = 0.5 / Math.sqrt(trace + 1.0);
    w = 0.25 / s;
    x = (m[5] - m[7]) * s;
    y = (m[6] - m[2]) * s;
    z = (m[1] - m[3]) * s;
  } else if (m[0] > m[4] && m[0] > m[8]) {
    const s = 2.0 * Math.sqrt(1.0 + m[0] - m[4] - m[8]);
    w = (m[5] - m[7]) / s;
    x = 0.25 * s;
    y = (m[3] + m[1]) / s;
    z = (m[6] + m[2]) / s;
  } else if (m[4] > m[8]) {
    const s = 2.0 * Math.sqrt(1.0 + m[4] - m[0] - m[8]);
    w = (m[6] - m[2]) / s;
    x = (m[3] + m[1]) / s;
    y = 0.25 * s;
    z = (m[7] + m[5]) / s;
  } else {
    const s = 2.0 * Math.sqrt(1.0 + m[8] - m[0] - m[4]);
    w = (m[1] - m[3]) / s;
    x = (m[6] + m[2]) / s;
    y = (m[7] + m[5]) / s;
    z = 0.25 * s;
  }

  const len = Math.sqrt(x * x + y * y + z * z + w * w);
  return [x / len, y / len, z / len, w / len];
}
