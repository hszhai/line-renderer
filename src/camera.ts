import { Mat4, mat4Identity, mat4LookAt, mat4Multiply, mat4Perspective, Vec3, v3add, v3scale, vec3 } from './math.ts';

export class OrbitCamera {
  position: Vec3 = [0, 0, 5];
  target: Vec3 = [0, 0, 0];
  up: Vec3 = [0, 1, 0];

  distance = 5;
  azimuth = 0; // radians, around Y axis
  elevation = 0.3; // radians, from horizontal

  fov = Math.PI / 4;
  near = 0.001;
  far = 100;

  viewMatrix: Mat4 = mat4Identity();
  projMatrix: Mat4 = mat4Identity();
  vpMatrix: Mat4 = mat4Identity();

  private canvas: HTMLCanvasElement;
  private isDragging = false;
  private lastMouseX = 0;
  private lastMouseY = 0;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.setupEvents();
    this.updateMatrices();
  }

  private setupEvents() {
    const canvas = this.canvas;

    canvas.addEventListener('mousedown', (e) => {
      this.isDragging = true;
      this.lastMouseX = e.clientX;
      this.lastMouseY = e.clientY;
    });

    window.addEventListener('mousemove', (e) => {
      if (!this.isDragging) return;
      const dx = e.clientX - this.lastMouseX;
      const dy = e.clientY - this.lastMouseY;
      this.lastMouseX = e.clientX;
      this.lastMouseY = e.clientY;

      this.azimuth -= dx * 0.008;
      this.elevation += dy * 0.008;
      this.elevation = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, this.elevation));

      this.updatePosition();
    });

    window.addEventListener('mouseup', () => {
      this.isDragging = false;
    });

    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.distance *= 1 + e.deltaY * 0.001;
      this.distance = Math.max(0.03, Math.min(50, this.distance));
      this.updatePosition();
    }, { passive: false });
  }

  updatePosition() {
    const cosEl = Math.cos(this.elevation);
    const sinEl = Math.sin(this.elevation);
    const cosAz = Math.cos(this.azimuth);
    const sinAz = Math.sin(this.azimuth);

    this.position = [
      this.target[0] + this.distance * cosEl * sinAz,
      this.target[1] + this.distance * sinEl,
      this.target[2] + this.distance * cosEl * cosAz,
    ];

    this.updateMatrices();
  }

  updateMatrices() {
    this.viewMatrix = mat4LookAt(this.position, this.target, this.up);
    const aspect = this.canvas.clientWidth / this.canvas.clientHeight;
    this.projMatrix = mat4Perspective(this.fov, aspect, this.near, this.far);
    this.vpMatrix = mat4Multiply(this.projMatrix, this.viewMatrix);
  }

  resize() {
    this.updateMatrices();
  }
}
