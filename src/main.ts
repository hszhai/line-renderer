import { OrbitCamera } from './camera.ts';
import { extractAllEdges, extractSilhouetteEdges } from './edge-extractor.ts';
import { edgesToGaussians, ColorMode, Gaussian3D } from './gaussian-generator.ts';
import { loadOBJ, Mesh } from './obj-loader.ts';
import { loadPLY } from './ply-loader.ts';
import { computePrincipalDirections, computeVertexNormals, decimateMesh } from './mesh-utils.ts';
import { GaussianSplatRenderer } from './renderer.ts';
import { buildVertexAdjacency } from './surface-walk.ts';
import {
  ClusterSeed, ClusterStrategy, ClusterStyle, StrandType,
  clusterSeedToGaussians, createClusterSeed,
} from './walk-cluster.ts';
import { ContourAxis, contoursToGaussians } from './contours.ts';
import { flowFieldToGaussians } from './flow-field.ts';
import { ProfileEditor } from './profile-editor.ts';

type RenderMode = 'wireframe' | 'silhouette' | 'modeling' | 'contours' | 'flow';

// Canonical size every model is normalized to, so the splat sizing, camera, and
// noise scales stay calibrated regardless of the source model's units.
const MODEL_SIZE = 0.2;
// High-res meshes are decimated to ~this many vertices on load, so the surface
// walks step meaningfully and the reference/contours stay performant.
const MAX_VERTS = 30000;

function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

/** Center a mesh at the origin and scale it so its largest bbox extent = size. */
function normalizeMesh(m: Mesh, size: number) {
  const v = m.vertices;
  let mnx = Infinity, mny = Infinity, mnz = Infinity;
  let mxx = -Infinity, mxy = -Infinity, mxz = -Infinity;
  for (let i = 0; i < v.length; i += 3) {
    mnx = Math.min(mnx, v[i]); mxx = Math.max(mxx, v[i]);
    mny = Math.min(mny, v[i + 1]); mxy = Math.max(mxy, v[i + 1]);
    mnz = Math.min(mnz, v[i + 2]); mxz = Math.max(mxz, v[i + 2]);
  }
  const cx = (mnx + mxx) / 2, cy = (mny + mxy) / 2, cz = (mnz + mxz) / 2;
  const s = size / (Math.max(mxx - mnx, mxy - mny, mxz - mnz) || 1);
  for (let i = 0; i < v.length; i += 3) {
    v[i] = (v[i] - cx) * s;
    v[i + 1] = (v[i + 1] - cy) * s;
    v[i + 2] = (v[i + 2] - cz) * s;
  }
}

async function main() {
  const canvas = document.getElementById('canvas') as HTMLCanvasElement;
  if (!canvas) throw new Error('Canvas not found');

  function resizeCanvas() {
    const dpr = Math.min(window.devicePixelRatio, 2);
    canvas.width = canvas.clientWidth * dpr;
    canvas.height = canvas.clientHeight * dpr;
  }
  resizeCanvas();
  window.addEventListener('resize', () => {
    resizeCanvas();
    camera.resize();
  });

  const gl = canvas.getContext('webgl2', { alpha: false, antialias: false });
  if (!gl) {
    alert('WebGL2 is not supported in your browser');
    throw new Error('WebGL2 not supported');
  }

  // Mesh + derived data — (re)populated by loadModel() so the model can switch.
  let mesh: Mesh;
  let allEdges: ReturnType<typeof extractAllEdges>;
  let vertexNormals: Float32Array;
  let vertexAdjacency: number[][];
  let principalDirs: Float32Array;

  // ─── UI refs ───
  const modeSelect = document.getElementById('mode') as HTMLSelectElement;
  const meshControls = document.getElementById('mesh-controls') as HTMLDivElement;
  const modelingControls = document.getElementById('modeling-controls') as HTMLDivElement;
  const contourControls = document.getElementById('contour-controls') as HTMLDivElement;
  const flowControls = document.getElementById('flow-controls') as HTMLDivElement;
  const splatControls = document.getElementById('splat-controls') as HTMLDivElement;
  const walkParams = document.getElementById('walk-params') as HTMLDivElement;
  const curveParams = document.getElementById('curve-params') as HTMLDivElement;
  const meshInfo = document.getElementById('mesh-info') as HTMLSpanElement;
  const curveInfo = document.getElementById('curve-info') as HTMLSpanElement;
  const edgeCountDisplay = document.getElementById('edge-count') as HTMLSpanElement;
  const clusterCountDisplay = document.getElementById('curve-count') as HTMLSpanElement;
  const splatCountDisplay = document.getElementById('seg-count') as HTMLSpanElement;

  const thicknessInput = document.getElementById('thickness') as HTMLInputElement;
  const colorSelect = document.getElementById('color') as HTMLSelectElement;
  const modelSelect = document.getElementById('model-select') as HTMLSelectElement;
  const bgColorInput = document.getElementById('bg-color') as HTMLInputElement;
  const colorAInput = document.getElementById('c-a') as HTMLInputElement;
  const colorBInput = document.getElementById('c-b') as HTMLInputElement;
  const colorStrip = document.getElementById('color-strip') as HTMLDivElement;
  const profileCanvas = document.getElementById('profile-canvas') as HTMLCanvasElement;
  const clTypeSelect = document.getElementById('cl-type') as HTMLSelectElement;
  const clStrategySelect = document.getElementById('cl-strategy') as HTMLSelectElement;
  const ctAxisSelect = document.getElementById('ct-axis') as HTMLSelectElement;
  const showReferenceCheck = document.getElementById('show-reference') as HTMLInputElement;
  const genClusterBtn = document.getElementById('gen-cluster') as HTMLButtonElement;
  const clearBtn = document.getElementById('clear-curves') as HTMLButtonElement;

  // ─── State ───
  let currentMode: RenderMode = modeSelect.value as RenderMode;
  let lineThickness = parseFloat(thicknessInput.value);
  let lineColor = colorSelect.value as ColorMode;
  let strandType = clTypeSelect.value as StrandType;
  let strategy = clStrategySelect.value as ClusterStrategy;
  let showReference = showReferenceCheck.checked;

  // Contour params
  let contourAxis = ctAxisSelect.value as ContourAxis;
  let contourLevels = 24;

  // Flow field params
  let ffDensity = 200;
  let ffSteps = 40;
  let ffNoiseScale = 35;
  let ffWander = 0.08;
  let ffSmoothing = 2;
  let ffVariant = 0;

  // Interactive width-profile editor; sampled per-segment along each stroke.
  const profileEditor = new ProfileEditor(profileCanvas, () => regenerate());

  // Global, live styling shared by every cluster.
  const style: ClusterStyle = {
    count: 12, spread: 6,
    steps: 80, wander: 0.1, noiseScale: 40, smoothing: 2,
    samples: 24, tangentMult: 0.6,
    radius: 4, overlap: 1, scaleMul: 1,
    profile: (t) => profileEditor.sample(t),
    colorA: hexToRgb(colorAInput.value),
    colorB: hexToRgb(colorBInput.value),
    hueJitter: 0.1, brightJitter: 0.2,
    opacity: 1, opacityB: 1,
    widthVar: 0.3, opacityVar: 0.2, lengthVar: 0.3,
  };

  let clusterSeeds: ClusterSeed[] = [];
  let meshRefGaussians: Gaussian3D[] = [];

  // Camera & renderer (camera framing is set by loadModel once the mesh loads).
  const camera = new OrbitCamera(canvas);

  const renderer = new GaussianSplatRenderer(gl);
  renderer.backgroundColor = hexToRgb(bgColorInput.value);

  function updateColorStrip() {
    colorStrip.style.background = `linear-gradient(90deg, ${colorAInput.value}, ${colorBInput.value})`;
  }
  updateColorStrip();

  // Load (or switch) the model: normalize it, recompute derived data, reframe.
  async function loadModel(file: string) {
    const url = '/models/' + file;
    mesh = file.toLowerCase().endsWith('.ply') ? await loadPLY(url) : await loadOBJ(url);
    normalizeMesh(mesh, MODEL_SIZE);
    const before = mesh.vertices.length / 3;
    mesh = decimateMesh(mesh, MAX_VERTS);
    const after = mesh.vertices.length / 3;
    if (after < before) console.log(`Decimated ${file}: ${before} → ${after} verts`);
    allEdges = extractAllEdges(mesh);
    vertexNormals = computeVertexNormals(mesh);
    vertexAdjacency = buildVertexAdjacency(mesh);
    principalDirs = computePrincipalDirections(mesh, vertexNormals, vertexAdjacency);
    edgeCountDisplay.textContent = String(allEdges.length);

    clusterSeeds = [];               // seeds index vertices — invalid for a new mesh
    camera.target = [0, 0, 0];
    camera.distance = MODEL_SIZE * 1.9;
    camera.near = MODEL_SIZE * 0.002;
    camera.far = MODEL_SIZE * 20;
    camera.updatePosition();
    regenerate();
  }

  function buildMeshReferenceGaussians() {
    meshRefGaussians = edgesToGaussians(mesh, allEdges, 1.5, 'white').map((g) => ({
      ...g,
      opacity: 0.12,
    }));
  }

  function regenerate() {
    if (currentMode === 'wireframe') {
      renderer.setGaussians(edgesToGaussians(mesh, allEdges, lineThickness, lineColor));
    } else if (currentMode === 'silhouette') {
      const silEdges = extractSilhouetteEdges(mesh, camera.position);
      renderer.setGaussians(edgesToGaussians(mesh, silEdges, lineThickness, lineColor));
    } else if (currentMode === 'contours') {
      const gaussians = contoursToGaussians(mesh, {
        axis: contourAxis,
        levels: contourLevels,
        radius: style.radius,
        overlap: style.overlap,
        scaleMul: style.scaleMul,
        opacity: style.opacity,
        opacityB: style.opacityB,
        colorA: style.colorA,
        colorB: style.colorB,
        hueJitter: style.hueJitter,
        brightJitter: style.brightJitter,
      });
      renderer.setGaussians(gaussians);
      clusterCountDisplay.textContent = String(contourLevels);
      splatCountDisplay.textContent = String(gaussians.length);
    } else if (currentMode === 'flow') {
      const gaussians = flowFieldToGaussians(mesh, vertexAdjacency, vertexNormals, {
        density: ffDensity,
        steps: ffSteps,
        wander: ffWander,
        smoothing: ffSmoothing,
        noiseScale: ffNoiseScale,
        variant: ffVariant,
        radius: style.radius,
        overlap: style.overlap,
        scaleMul: style.scaleMul,
        profile: style.profile,
        opacity: style.opacity,
        opacityB: style.opacityB,
        colorA: style.colorA,
        colorB: style.colorB,
        hueJitter: style.hueJitter,
        brightJitter: style.brightJitter,
      });
      renderer.setGaussians(gaussians);
      clusterCountDisplay.textContent = String(Math.round(ffDensity));
      splatCountDisplay.textContent = String(gaussians.length);
    } else {
      let all: Gaussian3D[] = [];
      for (const seed of clusterSeeds) {
        all.push(...clusterSeedToGaussians(seed, mesh, vertexAdjacency, vertexNormals, principalDirs, style));
      }
      const splatCount = all.length;
      if (showReference) {
        buildMeshReferenceGaussians();
        all = all.concat(meshRefGaussians);
      }
      renderer.setGaussians(all);
      clusterCountDisplay.textContent = String(clusterSeeds.length);
      splatCountDisplay.textContent = String(splatCount);
    }
  }

  function updateUIVisibility() {
    const modeling = currentMode === 'modeling';
    const contours = currentMode === 'contours';
    const flow = currentMode === 'flow';
    const meshMode = currentMode === 'wireframe' || currentMode === 'silhouette';
    const splat = modeling || contours || flow;
    meshControls.style.display = meshMode ? 'block' : 'none';
    modelingControls.style.display = modeling ? 'block' : 'none';
    contourControls.style.display = contours ? 'block' : 'none';
    flowControls.style.display = flow ? 'block' : 'none';
    splatControls.style.display = splat ? 'block' : 'none';
    meshInfo.style.display = meshMode ? 'inline' : 'none';
    curveInfo.style.display = meshMode ? 'none' : 'inline';
    // The profile canvas needs a real size once its panel is visible.
    if (splat) profileEditor.resize();
  }

  function updateStrandParamVisibility() {
    walkParams.style.display = strandType === 'walk' ? 'block' : 'none';
    curveParams.style.display = strandType === 'curve' ? 'block' : 'none';
  }

  // ─── Generic binders ───
  function bindRange(id: string, set: (v: number) => void, fmt: (v: number) => string) {
    const el = document.getElementById(id) as HTMLInputElement;
    const val = document.getElementById(id + '-val');
    const run = () => {
      const v = parseFloat(el.value);
      set(v);
      if (val) val.textContent = fmt(v);
    };
    el.addEventListener('input', () => { run(); regenerate(); });
    run(); // initialise label + state without regenerating
  }

  // Cluster shape
  bindRange('cl-count', (v) => (style.count = v), (v) => String(Math.round(v)));
  bindRange('cl-spread', (v) => (style.spread = v), (v) => String(Math.round(v)));
  bindRange('cl-length-var', (v) => (style.lengthVar = v), (v) => v.toFixed(2));
  // Walk strands
  bindRange('cl-steps', (v) => (style.steps = v), (v) => String(Math.round(v)));
  bindRange('cl-wander', (v) => (style.wander = v), (v) => v.toFixed(2));
  bindRange('cl-noise', (v) => (style.noiseScale = v), (v) => v.toFixed(0));
  bindRange('cl-smooth', (v) => (style.smoothing = v), (v) => String(Math.round(v)));
  // Curve strands
  bindRange('cl-samples', (v) => (style.samples = v), (v) => String(Math.round(v)));
  bindRange('cl-tangent', (v) => (style.tangentMult = v), (v) => v.toFixed(1) + '×');
  // Gaussian / splat
  bindRange('g-radius', (v) => (style.radius = v), (v) => v.toFixed(1));
  bindRange('g-overlap', (v) => (style.overlap = v), (v) => v.toFixed(2) + '×');
  bindRange('g-scale', (v) => (style.scaleMul = v), (v) => v.toFixed(1) + '×');
  bindRange('g-opacity', (v) => (style.opacity = v), (v) => v.toFixed(2));
  bindRange('g-opacity-b', (v) => (style.opacityB = v), (v) => v.toFixed(2));
  bindRange('g-width-var', (v) => (style.widthVar = v), (v) => v.toFixed(2));
  bindRange('g-opacity-var', (v) => (style.opacityVar = v), (v) => v.toFixed(2));
  // Colour jitter
  bindRange('c-hue-jitter', (v) => (style.hueJitter = v), (v) => v.toFixed(2));
  bindRange('c-bright-jitter', (v) => (style.brightJitter = v), (v) => v.toFixed(2));
  // Contours
  bindRange('ct-levels', (v) => (contourLevels = v), (v) => String(Math.round(v)));
  // Flow field
  bindRange('ff-density', (v) => (ffDensity = v), (v) => String(Math.round(v)));
  bindRange('ff-steps', (v) => (ffSteps = v), (v) => String(Math.round(v)));
  bindRange('ff-noise', (v) => (ffNoiseScale = v), (v) => v.toFixed(0));
  bindRange('ff-wander', (v) => (ffWander = v), (v) => v.toFixed(2));
  bindRange('ff-smooth', (v) => (ffSmoothing = v), (v) => String(Math.round(v)));
  bindRange('ff-variant', (v) => (ffVariant = v), (v) => String(Math.round(v)));
  // Mesh line thickness
  bindRange('thickness', (v) => (lineThickness = v), (v) => v.toFixed(1) + ' px');

  // ─── Selects, colours, checkboxes ───
  modeSelect.addEventListener('change', () => {
    currentMode = modeSelect.value as RenderMode;
    updateUIVisibility();
    regenerate();
  });

  colorSelect.addEventListener('change', () => {
    lineColor = colorSelect.value as ColorMode;
    regenerate();
  });

  bgColorInput.addEventListener('input', () => {
    renderer.backgroundColor = hexToRgb(bgColorInput.value);
  });

  colorAInput.addEventListener('input', () => {
    style.colorA = hexToRgb(colorAInput.value);
    updateColorStrip();
    regenerate();
  });
  colorBInput.addEventListener('input', () => {
    style.colorB = hexToRgb(colorBInput.value);
    updateColorStrip();
    regenerate();
  });

  // Width-profile preset buttons load shapes into the editor.
  document.querySelectorAll<HTMLButtonElement>('[data-preset]').forEach((btn) => {
    btn.addEventListener('click', () => profileEditor.loadPreset(btn.dataset.preset!));
  });

  modelSelect.addEventListener('change', () => { void loadModel(modelSelect.value); });

  // Strand type / strategy are stored per-cluster at creation, so changing them
  // only affects the NEXT cluster — no regenerate of existing clusters.
  clTypeSelect.addEventListener('change', () => {
    strandType = clTypeSelect.value as StrandType;
    updateStrandParamVisibility();
  });
  clStrategySelect.addEventListener('change', () => {
    strategy = clStrategySelect.value as ClusterStrategy;
  });

  ctAxisSelect.addEventListener('change', () => {
    contourAxis = ctAxisSelect.value as ContourAxis;
    regenerate();
  });

  showReferenceCheck.addEventListener('change', () => {
    showReference = showReferenceCheck.checked;
    regenerate();
  });

  genClusterBtn.addEventListener('click', () => {
    clusterSeeds.push(createClusterSeed(mesh, strandType, strategy));
    regenerate();
  });

  clearBtn.addEventListener('click', () => {
    clusterSeeds = [];
    regenerate();
  });

  // Collapsible panels
  document.querySelectorAll('.panel-header').forEach((hdr) => {
    hdr.addEventListener('click', () => hdr.parentElement!.classList.toggle('collapsed'));
  });

  // Initial UI state + load the selected model (which triggers the first render).
  updateUIVisibility();
  updateStrandParamVisibility();
  await loadModel(modelSelect.value);

  // Silhouette recompute on camera move
  let lastCamPos = [...camera.position] as [number, number, number];
  let framesSinceSilUpdate = 0;

  function updateSilhouetteIfNeeded() {
    if (currentMode !== 'silhouette') return;
    framesSinceSilUpdate++;
    if (framesSinceSilUpdate < 3) return;

    const dx = camera.position[0] - lastCamPos[0];
    const dy = camera.position[1] - lastCamPos[1];
    const dz = camera.position[2] - lastCamPos[2];
    if (Math.sqrt(dx * dx + dy * dy + dz * dz) > 0.005) {
      lastCamPos = [...camera.position] as [number, number, number];
      regenerate();
      framesSinceSilUpdate = 0;
    }
  }

  // Animation loop
  function frame() {
    renderer.viewportWidth = canvas.width;
    renderer.viewportHeight = canvas.height;
    updateSilhouetteIfNeeded();
    renderer.render(camera.viewMatrix, camera.projMatrix);
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}

main().catch((err) => {
  console.error('Failed to initialize:', err);
});
