import { Gaussian3D } from './gaussian-generator.ts';
import { eigenDecompose2D, invertSym2D, Mat4, mat4TransformPoint, quatToMat3 } from './math.ts';

const VERT_SRC = `#version 300 es
precision highp float;

layout(location = 0) in vec4 a_position;
layout(location = 1) in vec2 a_uv;
layout(location = 2) in vec3 a_color;
layout(location = 3) in float a_opacity;
layout(location = 4) in vec3 a_conic;

out vec2 v_uv;
out vec3 v_color;
out float v_opacity;
out vec3 v_conic;

void main() {
  gl_Position = a_position;
  v_uv = a_uv;
  v_color = a_color;
  v_opacity = a_opacity;
  v_conic = a_conic;
}
`;

const FRAG_SRC = `#version 300 es
precision highp float;

in vec2 v_uv;
in vec3 v_color;
in float v_opacity;
in vec3 v_conic;

out vec4 fragColor;

void main() {
  float power = -0.5 * (v_conic.x * v_uv.x * v_uv.x +
                         2.0 * v_conic.y * v_uv.x * v_uv.y +
                         v_conic.z * v_uv.y * v_uv.y);
  float alpha = exp(power) * v_opacity;

  if (alpha < 0.003) discard;

  fragColor = vec4(v_color, alpha);
}
`;

interface ProjectedGaussian {
  depth: number;
  // 4 vertices, each with:
  // position (vec4), uv (vec2), color (vec3), opacity (float), conic (vec3)
  // = 13 floats per vertex
  data: Float32Array;
}

export class GaussianSplatRenderer {
  private gl: WebGL2RenderingContext;
  private program: WebGLProgram;
  private vao: WebGLVertexArrayObject;
  private vertexBuffer: WebGLBuffer;
  private indexBuffer: WebGLBuffer;

  private gaussians: Gaussian3D[] = [];
  private numGaussians = 0;
  private maxGaussians = 0;

  // Pre-allocated buffers to avoid GC
  private vertexData!: Float32Array;
  private indexData!: Uint32Array;
  private projected: ProjectedGaussian[] = [];

  viewportWidth = 1;
  viewportHeight = 1;

  constructor(gl: WebGL2RenderingContext) {
    this.gl = gl;
    this.program = this.createProgram(VERT_SRC, FRAG_SRC);
    this.vao = gl.createVertexArray()!;
    this.vertexBuffer = gl.createBuffer()!;
    this.indexBuffer = gl.createBuffer()!;
    this.setupVAO();

    // Enable blending for alpha compositing (back-to-front over blending)
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
  }

  setGaussians(gaussians: Gaussian3D[]) {
    this.gaussians = gaussians;
    this.numGaussians = gaussians.length;

    if (this.numGaussians > this.maxGaussians) {
      this.maxGaussians = this.numGaussians;
      this.allocateBuffers(this.maxGaussians);
    }
  }

  private allocateBuffers(count: number) {
    const vertsPerQuad = 4;
    const floatsPerVertex = 13; // pos(4) + uv(2) + color(3) + opacity(1) + conic(3)
    this.vertexData = new Float32Array(count * vertsPerQuad * floatsPerVertex);
    this.indexData = new Uint32Array(count * 6); // 2 triangles per quad

    // Build static index buffer
    for (let i = 0; i < count; i++) {
      const base = i * 4;
      const idx = i * 6;
      this.indexData[idx + 0] = base + 0;
      this.indexData[idx + 1] = base + 1;
      this.indexData[idx + 2] = base + 2;
      this.indexData[idx + 3] = base + 0;
      this.indexData[idx + 4] = base + 2;
      this.indexData[idx + 5] = base + 3;
    }

    const gl = this.gl;
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, this.indexData, gl.STATIC_DRAW);

    this.projected = new Array(count);
    for (let i = 0; i < count; i++) {
      this.projected[i] = { depth: 0, data: new Float32Array(vertsPerQuad * floatsPerVertex) };
    }
  }

  private setupVAO() {
    const gl = this.gl;
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indexBuffer);

    const F = Float32Array.BYTES_PER_ELEMENT;
    const stride = 13 * F;

    // a_position (location 0) - vec4
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 4, gl.FLOAT, false, stride, 0);

    // a_uv (location 1)
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 2, gl.FLOAT, false, stride, 4 * F);

    // a_color (location 2)
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 3, gl.FLOAT, false, stride, 6 * F);

    // a_opacity (location 3)
    gl.enableVertexAttribArray(3);
    gl.vertexAttribPointer(3, 1, gl.FLOAT, false, stride, 9 * F);

    // a_conic (location 4)
    gl.enableVertexAttribArray(4);
    gl.vertexAttribPointer(4, 3, gl.FLOAT, false, stride, 10 * F);

    gl.bindVertexArray(null);
  }

  private createProgram(vs: string, fs: string): WebGLProgram {
    const gl = this.gl;
    const vert = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vert, vs);
    gl.compileShader(vert);
    if (!gl.getShaderParameter(vert, gl.COMPILE_STATUS)) {
      throw new Error('Vertex shader error: ' + gl.getShaderInfoLog(vert));
    }

    const frag = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(frag, fs);
    gl.compileShader(frag);
    if (!gl.getShaderParameter(frag, gl.COMPILE_STATUS)) {
      throw new Error('Fragment shader error: ' + gl.getShaderInfoLog(frag));
    }

    const prog = gl.createProgram()!;
    gl.attachShader(prog, vert);
    gl.attachShader(prog, frag);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error('Program link error: ' + gl.getProgramInfoLog(prog));
    }

    gl.deleteShader(vert);
    gl.deleteShader(frag);
    return prog;
  }

  render(viewMatrix: Mat4, projMatrix: Mat4) {
    if (this.numGaussians === 0) return;

    const gl = this.gl;
    const w = this.viewportWidth;
    const h = this.viewportHeight;

    // Focal lengths in pixels, derived from the actual projection matrix so
    // they track the real fov AND aspect ratio. projMatrix[0] = f/aspect and
    // projMatrix[5] = f, so these come out equal (square pixels) regardless of
    // window shape. (Hardcoding w/(2·tan(fov/2)) for X was aspect-wrong: it
    // stretched every splat horizontally by the window's aspect ratio.)
    const focalX = projMatrix[0] * w * 0.5;
    const focalY = projMatrix[5] * h * 0.5;

    // View rotation W (upper-left 3x3 of the column-major view matrix), as rows.
    // EWA splatting needs the covariance in CAMERA space, so we rotate each
    // Gaussian's world rotation by W before projecting: Σ_2D = J·W·Σ·Wᵀ·Jᵀ.
    const w00 = viewMatrix[0], w01 = viewMatrix[4], w02 = viewMatrix[8];
    const w10 = viewMatrix[1], w11 = viewMatrix[5], w12 = viewMatrix[9];
    const w20 = viewMatrix[2], w21 = viewMatrix[6], w22 = viewMatrix[10];

    // Project all Gaussians to 2D on the CPU
    for (let i = 0; i < this.numGaussians; i++) {
      const g = this.gaussians[i];
      const pg = this.projected[i];

      // Transform center to view space
      const camPos = mat4TransformPoint(viewMatrix, g.position);
      const z = camPos[2];

      if (z > -0.2) {
        pg.depth = Infinity; // behind camera or too close, skip
        continue;
      }

      pg.depth = -z;

      // Build 3D covariance in CAMERA space: Σ = (R_cam·S)(R_cam·S)^T,
      // where R_cam = W · R_world rotates the splat into the camera frame.
      const Rw = quatToMat3(g.rotation); // world-space rotation (row-major)
      const sx = g.scale[0];
      const sy = g.scale[1];
      const sz = g.scale[2];

      // R_cam = W * Rw  (camera-space rotation)
      const c0 = w00 * Rw[0] + w01 * Rw[3] + w02 * Rw[6];
      const c1 = w00 * Rw[1] + w01 * Rw[4] + w02 * Rw[7];
      const c2 = w00 * Rw[2] + w01 * Rw[5] + w02 * Rw[8];
      const c3 = w10 * Rw[0] + w11 * Rw[3] + w12 * Rw[6];
      const c4 = w10 * Rw[1] + w11 * Rw[4] + w12 * Rw[7];
      const c5 = w10 * Rw[2] + w11 * Rw[5] + w12 * Rw[8];
      const c6 = w20 * Rw[0] + w21 * Rw[3] + w22 * Rw[6];
      const c7 = w20 * Rw[1] + w21 * Rw[4] + w22 * Rw[7];
      const c8 = w20 * Rw[2] + w21 * Rw[5] + w22 * Rw[8];

      // RS = R_cam * diag(sx, sy, sz)
      const rs0 = c0 * sx, rs1 = c1 * sy, rs2 = c2 * sz;
      const rs3 = c3 * sx, rs4 = c4 * sy, rs5 = c5 * sz;
      const rs6 = c6 * sx, rs7 = c7 * sy, rs8 = c8 * sz;

      // Sigma = RS * (RS)^T
      const sig00 = rs0 * rs0 + rs1 * rs1 + rs2 * rs2;
      const sig01 = rs0 * rs3 + rs1 * rs4 + rs2 * rs5;
      const sig02 = rs0 * rs6 + rs1 * rs7 + rs2 * rs8;
      const sig11 = rs3 * rs3 + rs4 * rs4 + rs5 * rs5;
      const sig12 = rs3 * rs6 + rs4 * rs7 + rs5 * rs8;
      const sig22 = rs6 * rs6 + rs7 * rs7 + rs8 * rs8;

      // Projection Jacobian J (approximate for perspective)
      const z2 = z * z;
      const j00 = focalX / z;
      const j02 = -focalX * camPos[0] / z2;
      const j11 = focalY / z;
      const j12 = -focalY * camPos[1] / z2;

      // T = J * Sigma * J^T (only upper-left 2x2 matters)
      let cov2d_xx = j00 * j00 * sig00 + 2 * j00 * j02 * sig02 + j02 * j02 * sig22;
      let cov2d_xy = j00 * j11 * sig01 + j00 * j12 * sig02 + j02 * j11 * sig12 + j02 * j12 * sig22;
      let cov2d_yy = j11 * j11 * sig11 + 2 * j11 * j12 * sig12 + j12 * j12 * sig22;

      // Add epsilon for numerical stability (keeps very thin lines renderable)
      cov2d_xx += 0.15;
      cov2d_yy += 0.15;

      // Compute conic (inverse of 2D covariance)
      const conic = invertSym2D(cov2d_xx, cov2d_xy, cov2d_yy);

      // Eigen-decomposition for bounding quad
      const eig = eigenDecompose2D(cov2d_xx, cov2d_xy, cov2d_yy);
      const r1 = 3.0 * Math.sqrt(Math.max(0, eig.lambda1));
      const r2 = 3.0 * Math.sqrt(Math.max(0, eig.lambda2));

      // Project center to clip space
      const clipW = projMatrix[3] * camPos[0] + projMatrix[7] * camPos[1] + projMatrix[11] * camPos[2] + projMatrix[15];
      if (Math.abs(clipW) < 1e-6) {
        pg.depth = Infinity;
        continue;
      }
      const clipX = projMatrix[0] * camPos[0] + projMatrix[4] * camPos[1] + projMatrix[8] * camPos[2] + projMatrix[12];
      const clipY = projMatrix[1] * camPos[0] + projMatrix[5] * camPos[1] + projMatrix[9] * camPos[2] + projMatrix[13];
      const clipZ = projMatrix[2] * camPos[0] + projMatrix[6] * camPos[1] + projMatrix[10] * camPos[2] + projMatrix[14];

      const ndcX = clipX / clipW;
      const ndcY = clipY / clipW;

      // Build 4 quad corners
      const corners = [
        [-1, -1],
        [-1, 1],
        [1, 1],
        [1, -1],
      ];

      const data = pg.data;
      let off = 0;

      for (let c = 0; c < 4; c++) {
        const cx = corners[c][0];
        const cy = corners[c][1];

        // Pixel-space offset = corner * (eigenvector1 * r1 + eigenvector2 * r2)
        const pxOff = cx * eig.v1x * r1 + cy * eig.v2x * r2;
        const pyOff = cx * eig.v1y * r1 + cy * eig.v2y * r2;

        // Convert pixel offset to NDC offset
        const ndcOffX = pxOff / (w * 0.5);
        const ndcOffY = pyOff / (h * 0.5);

        const vNdcX = ndcX + ndcOffX;
        const vNdcY = ndcY + ndcOffY;

        // Store clip-space position
        data[off + 0] = vNdcX * clipW;
        data[off + 1] = vNdcY * clipW;
        data[off + 2] = clipZ;
        data[off + 3] = clipW;

        // UV in pixel space
        data[off + 4] = pxOff;
        data[off + 5] = pyOff;

        // Color
        data[off + 6] = g.color[0];
        data[off + 7] = g.color[1];
        data[off + 8] = g.color[2];

        // Opacity
        data[off + 9] = g.opacity;

        // Conic
        data[off + 10] = conic.x;
        data[off + 11] = conic.y;
        data[off + 12] = conic.z;

        off += 13;
      }
    }

    // Sort by depth (back to front for alpha blending)
    this.projected.sort((a, b) => b.depth - a.depth);

    // Build interleaved vertex buffer in sorted order
    let vOff = 0;
    for (let i = 0; i < this.numGaussians; i++) {
      const pg = this.projected[i];
      if (pg.depth === Infinity) continue;
      const src = pg.data;
      for (let j = 0; j < 52; j++) { // 4 verts * 13 floats
        this.vertexData[vOff + j] = src[j];
      }
      vOff += 52;
    }

    const visibleCount = vOff / 52;
    if (visibleCount === 0) return;

    // Upload vertex data
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, this.vertexData.subarray(0, vOff), gl.DYNAMIC_DRAW);

    // Render
    gl.viewport(0, 0, w, h);
    gl.clearColor(0.04, 0.04, 0.04, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    gl.useProgram(this.program);
    gl.bindVertexArray(this.vao);
    gl.drawElements(gl.TRIANGLES, visibleCount * 6, gl.UNSIGNED_INT, 0);
    gl.bindVertexArray(null);
  }

  dispose() {
    const gl = this.gl;
    gl.deleteProgram(this.program);
    gl.deleteVertexArray(this.vao);
    gl.deleteBuffer(this.vertexBuffer);
    gl.deleteBuffer(this.indexBuffer);
  }
}
