import { OrbitCamera } from './camera.ts';
import { extractAllEdges, extractSilhouetteEdges } from './edge-extractor.ts';
import { edgesToGaussians, ColorMode, Gaussian3D } from './gaussian-generator.ts';
import { loadOBJ } from './obj-loader.ts';
import { computePrincipalDirections, computeVertexNormals } from './mesh-utils.ts';
import { GaussianSplatRenderer } from './renderer.ts';
import { buildVertexAdjacency } from './surface-walk.ts';
import {
  ClusterSeed, ClusterStrategy, ClusterStyle, StrandType,
  clusterSeedToGaussians, createClusterSeed,
} from './walk-cluster.ts';
import { ContourAxis, contoursToGaussians } from './contours.ts';

type RenderMode = 'wireframe' | 'silhouette' | 'modeling' | 'contours';

function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
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

  // Load mesh & precompute data
  console.log('Loading Stanford Bunny...');
  const mesh = await loadOBJ('/models/stanford_bunny.obj');
  console.log(`Loaded: ${mesh.vertices.length / 3} vertices, ${mesh.faces.length / 3} faces`);

  const allEdges = extractAllEdges(mesh);
  const vertexNormals = computeVertexNormals(mesh);
  const vertexAdjacency = buildVertexAdjacency(mesh);
  const principalDirs = computePrincipalDirections(mesh, vertexNormals, vertexAdjacency);

  // ─── UI refs ───
  const modeSelect = document.getElementById('mode') as HTMLSelectElement;
  const meshControls = document.getElementById('mesh-controls') as HTMLDivElement;
  const modelingControls = document.getElementById('modeling-controls') as HTMLDivElement;
  const contourControls = document.getElementById('contour-controls') as HTMLDivElement;
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
  const bgColorInput = document.getElementById('bg-color') as HTMLInputElement;
  const baseColorInput = document.getElementById('c-base') as HTMLInputElement;
  const clTypeSelect = document.getElementById('cl-type') as HTMLSelectElement;
  const clStrategySelect = document.getElementById('cl-strategy') as HTMLSelectElement;
  const ctAxisSelect = document.getElementById('ct-axis') as HTMLSelectElement;
  const showReferenceCheck = document.getElementById('show-reference') as HTMLInputElement;
  const genClusterBtn = document.getElementById('gen-cluster') as HTMLButtonElement;
  const clearBtn = document.getElementById('clear-curves') as HTMLButtonElement;

  edgeCountDisplay.textContent = String(allEdges.length);

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
  let contourHueRange = 0;

  // Global, live styling shared by every cluster.
  const style: ClusterStyle = {
    count: 12, spread: 6,
    steps: 80, wander: 0.1, noiseScale: 40, smoothing: 2,
    samples: 24, tangentMult: 0.6,
    radius: 4, overlap: 1, scaleMul: 1, opacity: 1,
    baseColor: hexToRgb(baseColorInput.value),
    hueJitter: 0.1, brightJitter: 0.2,
    widthVar: 0.3, opacityVar: 0.2, lengthVar: 0.3,
  };

  let clusterSeeds: ClusterSeed[] = [];
  let meshRefGaussians: Gaussian3D[] = [];

  // Camera & renderer
  const camera = new OrbitCamera(canvas);
  camera.distance = 0.8;
  camera.target = [0, 0.1, 0];
  camera.updatePosition();

  const renderer = new GaussianSplatRenderer(gl);
  renderer.backgroundColor = hexToRgb(bgColorInput.value);

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
        baseColor: style.baseColor,
        hueRange: contourHueRange,
      });
      renderer.setGaussians(gaussians);
      clusterCountDisplay.textContent = String(contourLevels);
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
    const mesh = currentMode === 'wireframe' || currentMode === 'silhouette';
    meshControls.style.display = mesh ? 'block' : 'none';
    modelingControls.style.display = modeling ? 'block' : 'none';
    contourControls.style.display = contours ? 'block' : 'none';
    splatControls.style.display = modeling || contours ? 'block' : 'none';
    meshInfo.style.display = mesh ? 'inline' : 'none';
    curveInfo.style.display = mesh ? 'none' : 'inline';
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
  bindRange('g-width-var', (v) => (style.widthVar = v), (v) => v.toFixed(2));
  bindRange('g-opacity-var', (v) => (style.opacityVar = v), (v) => v.toFixed(2));
  // Colour jitter
  bindRange('c-hue-jitter', (v) => (style.hueJitter = v), (v) => v.toFixed(2));
  bindRange('c-bright-jitter', (v) => (style.brightJitter = v), (v) => v.toFixed(2));
  // Contours
  bindRange('ct-levels', (v) => (contourLevels = v), (v) => String(Math.round(v)));
  bindRange('ct-huerange', (v) => (contourHueRange = v), (v) => v.toFixed(2));
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

  baseColorInput.addEventListener('input', () => {
    style.baseColor = hexToRgb(baseColorInput.value);
    regenerate();
  });

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

  // Initial render
  updateUIVisibility();
  updateStrandParamVisibility();
  regenerate();

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
