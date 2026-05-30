// =============================================================================
// 3DGS Sandbox — the absolute fundamentals
//
// Goal: understand what a single 3D Gaussian "splat" is, and how a handful of
// them combine. We place THREE points in space, then connect them with THREE
// splats so the result reads as a triangle.
//
// A 3D Gaussian splat is just an oriented, scaled blob defined by:
//   - position : where its center sits in the world           (Vec3)
//   - scale    : its size along its own local x / y / z axes   (Vec3, "sigmas")
//   - rotation : how those local axes are oriented in the world (quaternion)
//   - color    : RGB                                            (Vec3, 0..1)
//   - opacity  : how solid it is                                (0..1)
//
// The renderer (src/renderer.ts) takes that 3D blob, builds its 3x3 covariance
// matrix  Σ = R · S · Sᵀ · Rᵀ , projects Σ down to a 2D ellipse on screen, and
// shades each pixel with a Gaussian falloff. That projection math is THE core
// idea of 3D Gaussian Splatting — everything else is just feeding it blobs.
//
// To make an edge look like a tube/line, we use an anisotropic scale: thin on
// two axes (the tube radius) and long on the third (half the edge length), then
// rotate so that long axis points down the edge.
// =============================================================================

import { OrbitCamera } from './camera.ts';
import { GaussianSplatRenderer } from './renderer.ts';
import { Gaussian3D } from './gaussian-generator.ts';
import {
  quatFromVectors,
  v3length,
  v3scale,
  v3sub,
  v3normalize,
  Vec3,
} from './math.ts';

// --- 1. Three points in space -----------------------------------------------
// A simple upright triangle centered on the origin. Tweak these to see the
// splats follow.
const A: Vec3 = [0.0, 1.0, 0.0]; // top
const B: Vec3 = [-1.0, -0.7, 0.0]; // bottom-left
const C: Vec3 = [1.0, -0.7, 0.0]; // bottom-right

// Colors for the three edges so they're easy to tell apart...
const EDGE_COLORS: Vec3[] = [
  [1.0, 0.35, 0.35], // A→B  red
  [0.35, 1.0, 0.45], // B→C  green
  [0.4, 0.55, 1.0], // C→A  blue
];

// ...and a distinct color for each of the three original points.
const POINT_COLORS: Vec3[] = [
  [1.0, 0.85, 0.2], // A  yellow
  [1.0, 0.35, 0.9], // B  magenta
  [0.3, 0.95, 0.95], // C  cyan
];

// Live, slider-driven splat parameters. Everything the UI tweaks lives here so
// the scene can be rebuilt from a single source of truth.
const params = {
  edgeMode: 'single' as 'single' | 'beads', // one stretched splat, or many
  edgeSigma: 0.5, // SINGLE mode: σ along edge, as a fraction of half-length
  splatsPerEdge: 14, // BEADS mode: how many small Gaussians make up each edge
  tubeRadius: 0.04, // σ x/y of each edge splat (its thinness)
  pointSize: 0.072, // radius of the round vertex-marker splats
  opacity: 1.0, // alpha applied to every splat
  globalScale: 1.0, // multiplies ALL scales (zoom the blobs in/out)
};

// --- 2a. ONE stretched splat per edge ---------------------------------------
// A single anisotropic Gaussian DOES link the two points: thin on x/y, long on
// z (aligned to the edge). The catch is the tails. With σ_z = half-length the
// endpoints sit at 1σ (~61% bright) — clearly linked — but the renderer draws
// every splat out to 3σ, so a glow runs ~2× the half-length PAST each point.
// Shrink `edgeSigma` to cut the overshoot, and watch the ends fade instead:
// that brightness-vs-overshoot tradeoff is exactly why scenes stack many.
function edgeSplatSingle(p0: Vec3, p1: Vec3, radius: number, color: Vec3): Gaussian3D {
  const mid: Vec3 = v3scale([p0[0] + p1[0], p0[1] + p1[1], p0[2] + p1[2]], 0.5);
  const delta = v3sub(p1, p0);
  const length = v3length(delta);
  const dir = v3normalize(delta);
  const rot = quatFromVectors([0, 0, 1], dir);

  const g = params.globalScale;
  const sigmaZ = length * 0.5 * params.edgeSigma * g; // σ along the edge
  return {
    position: mid,
    scale: [radius * g, radius * g, sigmaZ],
    rotation: [rot[0], rot[1], rot[2], rot[3]],
    color,
    opacity: params.opacity,
  };
}

// --- 2b. MANY small splats per edge -----------------------------------------
// Lay down a row of small "beads" evenly from p0 to p1 (endpoints included),
// each elongated just enough to overlap its neighbour into a smooth tube. This
// gives uniform brightness AND clean ends — at the cost of N splats per edge.
// Crank `splatsPerEdge` low to see discrete blobs, high for a clean line.
function edgeSplats(p0: Vec3, p1: Vec3, radius: number, color: Vec3): Gaussian3D[] {
  const n = Math.max(2, Math.round(params.splatsPerEdge));
  const g = params.globalScale;

  // Edge direction + length, and the rotation that aligns each bead's long
  // (local Z) axis with the edge.
  const delta = v3sub(p1, p0);
  const length = v3length(delta);
  const dir = v3normalize(delta);
  const rot = quatFromVectors([0, 0, 1], dir);

  // Spacing between consecutive beads. Give each bead σ_z ≈ 0.6× the spacing so
  // neighbours overlap and the tail of one fills the gap to the next.
  const spacing = length / (n - 1);
  const sigmaZ = spacing * 0.6 * g;
  const sigmaXY = radius * g;

  const out: Gaussian3D[] = [];
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1); // 0..1 along the edge, hits both endpoints
    const pos: Vec3 = [
      p0[0] + delta[0] * t,
      p0[1] + delta[1] * t,
      p0[2] + delta[2] * t,
    ];
    out.push({
      position: pos,
      scale: [sigmaXY, sigmaXY, sigmaZ],
      rotation: [rot[0], rot[1], rot[2], rot[3]],
      color,
      opacity: params.opacity,
    });
  }
  return out;
}

// --- 3. A tiny round splat to mark each original point ----------------------
// Isotropic (equal scale on all axes) => a round dot, so you can literally see
// the three points the triangle is built from.
function pointSplat(p: Vec3, radius: number, color: Vec3): Gaussian3D {
  const r = radius * params.globalScale;
  return {
    position: p,
    scale: [r, r, r],
    rotation: [0, 0, 0, 1], // identity rotation (a sphere needs none)
    color,
    opacity: params.opacity,
  };
}

function buildScene(showPoints: boolean): Gaussian3D[] {
  // THREE edges. In 'single' mode each is one stretched splat; in 'beads' mode
  // each is a row of small splats.
  const edges: [Vec3, Vec3, Vec3][] = [
    [A, B, EDGE_COLORS[0]],
    [B, C, EDGE_COLORS[1]],
    [C, A, EDGE_COLORS[2]],
  ];
  const splats: Gaussian3D[] = [];
  for (const [p0, p1, color] of edges) {
    if (params.edgeMode === 'single') {
      splats.push(edgeSplatSingle(p0, p1, params.tubeRadius, color));
    } else {
      splats.push(...edgeSplats(p0, p1, params.tubeRadius, color));
    }
  }

  if (showPoints) {
    // THREE point markers at the original vertices, each its own color.
    splats.push(pointSplat(A, params.pointSize, POINT_COLORS[0]));
    splats.push(pointSplat(B, params.pointSize, POINT_COLORS[1]));
    splats.push(pointSplat(C, params.pointSize, POINT_COLORS[2]));
  }

  return splats;
}

// --- 4. Boot the WebGL renderer + orbit camera ------------------------------
const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const gl = canvas.getContext('webgl2', { antialias: false, premultipliedAlpha: false });
if (!gl) throw new Error('WebGL2 not supported in this browser.');

const camera = new OrbitCamera(canvas);
camera.distance = 4.5;
camera.elevation = 0.15;
camera.updatePosition();

const renderer = new GaussianSplatRenderer(gl);

const showPointsCheckbox = document.getElementById('show-points') as HTMLInputElement;
function refresh() {
  renderer.setGaussians(buildScene(showPointsCheckbox.checked));
}
showPointsCheckbox.addEventListener('change', refresh);

// --- Wire each slider to a field on `params`, then rebuild the scene. --------
// `format` just controls how the live value is printed next to the slider.
// Only the numeric fields of `params` can be driven by a range slider.
type NumericParam = {
  [K in keyof typeof params]: typeof params[K] extends number ? K : never;
}[keyof typeof params];

function bindSlider(
  id: NumericParam,
  format: (v: number) => string
) {
  const slider = document.getElementById(id) as HTMLInputElement;
  const label = document.getElementById(id + '-val') as HTMLElement;
  const apply = () => {
    params[id] = parseFloat(slider.value);
    label.textContent = format(params[id]);
    refresh();
  };
  slider.addEventListener('input', apply);
  apply(); // initialise label + params from the slider's default
}

bindSlider('edgeSigma', (v) => v.toFixed(2) + ' × ½-len');
bindSlider('splatsPerEdge', (v) => String(Math.round(v)));
bindSlider('tubeRadius', (v) => v.toFixed(3));
bindSlider('pointSize', (v) => v.toFixed(3));
bindSlider('opacity', (v) => v.toFixed(2));
bindSlider('globalScale', (v) => v.toFixed(1) + '×');

// Edge mode select: switch between one stretched splat and many beads, and
// show only the slider that matters for the active mode.
const edgeModeSelect = document.getElementById('edgeMode') as HTMLSelectElement;
const singleRow = document.getElementById('edgeSigma-row') as HTMLElement;
const beadsRow = document.getElementById('splatsPerEdge-row') as HTMLElement;
function applyEdgeMode() {
  params.edgeMode = edgeModeSelect.value as typeof params.edgeMode;
  const isSingle = params.edgeMode === 'single';
  singleRow.style.display = isSingle ? '' : 'none';
  beadsRow.style.display = isSingle ? 'none' : '';
  refresh();
}
edgeModeSelect.addEventListener('change', applyEdgeMode);
applyEdgeMode();

refresh();

function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.floor(canvas.clientWidth * dpr);
  const h = Math.floor(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  renderer.viewportWidth = canvas.width;
  renderer.viewportHeight = canvas.height;
  camera.resize();
}
window.addEventListener('resize', resize);
resize();

function frame() {
  resize();
  renderer.render(camera.viewMatrix, camera.projMatrix);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
