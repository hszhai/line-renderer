import { OrbitCamera } from './camera.ts';
import { CurveSeed, curveToGaussians, generateRandomCurveOnMesh, paramsFromSeed } from './curves.ts';
import { extractAllEdges, extractSilhouetteEdges } from './edge-extractor.ts';
import { edgesToGaussians, ColorMode, Gaussian3D } from './gaussian-generator.ts';
import { loadOBJ, Mesh } from './obj-loader.ts';
import { computeVertexNormals } from './mesh-utils.ts';
import { GaussianSplatRenderer } from './renderer.ts';
import { buildVertexAdjacency, generateSurfaceWalk, walkSeedToGaussians, WalkSeed } from './surface-walk.ts';

type RenderMode = 'wireframe' | 'silhouette' | 'curves';

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
  console.log(`Unique edges: ${allEdges.length}`);

  // ─── UI refs ───
  const modeSelect = document.getElementById('mode') as HTMLSelectElement;
  const meshControls = document.getElementById('mesh-controls') as HTMLDivElement;
  const curveControls = document.getElementById('curve-controls') as HTMLDivElement;
  const meshInfo = document.getElementById('mesh-info') as HTMLSpanElement;
  const curveInfo = document.getElementById('curve-info') as HTMLSpanElement;

  const thicknessInput = document.getElementById('thickness') as HTMLInputElement;
  const thicknessVal = document.getElementById('thickness-val') as HTMLDivElement;
  const colorSelect = document.getElementById('color') as HTMLSelectElement;
  const edgeCountDisplay = document.getElementById('edge-count') as HTMLSpanElement;

  const curveRadiusInput = document.getElementById('curve-radius') as HTMLInputElement;
  const curveRadiusVal = document.getElementById('curve-radius-val') as HTMLDivElement;
  const curveOverlapInput = document.getElementById('curve-overlap') as HTMLInputElement;
  const curveOverlapVal = document.getElementById('curve-overlap-val') as HTMLDivElement;
  const curveScaleMulInput = document.getElementById('curve-scalemul') as HTMLInputElement;
  const curveScaleMulVal = document.getElementById('curve-scalemul-val') as HTMLDivElement;
  const curveOpacityInput = document.getElementById('curve-opacity') as HTMLInputElement;
  const curveOpacityVal = document.getElementById('curve-opacity-val') as HTMLDivElement;
  const curveTmultInput = document.getElementById('curve-tmult') as HTMLInputElement;
  const curveTmultVal = document.getElementById('curve-tmult-val') as HTMLDivElement;
  const curveSamplesInput = document.getElementById('curve-samples') as HTMLInputElement;
  const curveSamplesVal = document.getElementById('curve-samples-val') as HTMLDivElement;
  const showReferenceCheck = document.getElementById('show-reference') as HTMLInputElement;
  const walkStepsInput = document.getElementById('walk-steps') as HTMLInputElement;
  const walkStepsVal = document.getElementById('walk-steps-val') as HTMLDivElement;
  const walkWanderInput = document.getElementById('walk-wander') as HTMLInputElement;
  const walkWanderVal = document.getElementById('walk-wander-val') as HTMLDivElement;
  const genWalkBtn = document.getElementById('gen-walk') as HTMLButtonElement;
  const genCurveBtn = document.getElementById('gen-curve') as HTMLButtonElement;
  const clearCurvesBtn = document.getElementById('clear-curves') as HTMLButtonElement;
  const curveCountDisplay = document.getElementById('curve-count') as HTMLSpanElement;
  const segCountDisplay = document.getElementById('seg-count') as HTMLSpanElement;

  edgeCountDisplay.textContent = String(allEdges.length);

  // ─── State ───
  let currentMode: RenderMode = modeSelect.value as RenderMode;
  let currentThickness = parseFloat(thicknessInput.value);
  thicknessVal.textContent = currentThickness.toFixed(1) + ' px';
  let currentColor = colorSelect.value as ColorMode;

  let curveRadius = parseFloat(curveRadiusInput.value);
  curveRadiusVal.textContent = curveRadius.toFixed(1);
  let curveOverlap = parseFloat(curveOverlapInput.value);
  curveOverlapVal.textContent = curveOverlap.toFixed(2) + '×';
  let curveScaleMul = parseFloat(curveScaleMulInput.value);
  curveScaleMulVal.textContent = curveScaleMul.toFixed(1) + '×';
  let curveOpacity = parseFloat(curveOpacityInput.value);
  curveOpacityVal.textContent = curveOpacity.toFixed(2);
  let curveTangentMultiplier = parseFloat(curveTmultInput.value);
  curveTmultVal.textContent = curveTangentMultiplier.toFixed(1) + '×';
  let curveSamples = parseInt(curveSamplesInput.value);
  curveSamplesVal.textContent = String(curveSamples);
  let showReference = showReferenceCheck.checked;

  let walkSteps = parseInt(walkStepsInput.value);
  walkStepsVal.textContent = String(walkSteps);
  let walkWander = parseFloat(walkWanderInput.value);
  walkWanderVal.textContent = walkWander.toFixed(2);

  let curveSeeds: CurveSeed[] = [];
  let walkSeeds: WalkSeed[] = [];
  let curveGaussians: Gaussian3D[] = [];
  let meshRefGaussians: Gaussian3D[] = [];

  // Camera & renderer
  const camera = new OrbitCamera(canvas);
  camera.distance = 0.8;
  camera.target = [0, 0.1, 0];
  camera.updatePosition();

  const renderer = new GaussianSplatRenderer(gl);

  function updateUIVisibility() {
    if (currentMode === 'curves') {
      meshControls.style.display = 'none';
      curveControls.style.display = 'block';
      meshInfo.style.display = 'none';
      curveInfo.style.display = 'inline';
    } else {
      meshControls.style.display = 'block';
      curveControls.style.display = 'none';
      meshInfo.style.display = 'inline';
      curveInfo.style.display = 'none';
    }
  }

  function buildMeshReferenceGaussians() {
    meshRefGaussians = edgesToGaussians(mesh, allEdges, 1.5, 'white').map(g => ({
      ...g,
      opacity: 0.12,
    }));
  }

  function regenerateAllCurves() {
    curveGaussians = [];
    for (const seed of curveSeeds) {
      const params = paramsFromSeed(
        seed, mesh, vertexNormals,
        curveRadius, curveOverlap, curveScaleMul, curveOpacity, curveSamples, curveTangentMultiplier
      );
      curveGaussians.push(...curveToGaussians(params));
    }
    for (const seed of walkSeeds) {
      curveGaussians.push(...walkSeedToGaussians(
        seed, mesh, vertexAdjacency,
        curveRadius, curveOverlap, curveScaleMul, curveOpacity, walkSteps, walkWander
      ));
    }
    regenerateGaussians();
  }

  function regenerateGaussians() {
    if (currentMode === 'wireframe') {
      const gaussians = edgesToGaussians(mesh, allEdges, currentThickness, currentColor);
      renderer.setGaussians(gaussians);
    } else if (currentMode === 'silhouette') {
      const silEdges = extractSilhouetteEdges(mesh, camera.position);
      const gaussians = edgesToGaussians(mesh, silEdges, currentThickness, currentColor);
      renderer.setGaussians(gaussians);
    } else if (currentMode === 'curves') {
      let all: Gaussian3D[] = [...curveGaussians];
      if (showReference) {
        buildMeshReferenceGaussians();
        all = all.concat(meshRefGaussians);
      }
      renderer.setGaussians(all);
      curveCountDisplay.textContent = String(curveSeeds.length + walkSeeds.length);
      segCountDisplay.textContent = String(curveGaussians.length);
    }
  }

  function generateCurve() {
    const result = generateRandomCurveOnMesh(
      mesh, vertexNormals,
      curveRadius, curveOverlap, curveScaleMul, curveOpacity, curveSamples, curveTangentMultiplier
    );
    curveSeeds.push(result.seed);
    curveGaussians.push(...result.gaussians);
    regenerateGaussians();
  }

  function generateWalk() {
    const result = generateSurfaceWalk(
      mesh, vertexAdjacency,
      curveRadius, curveOverlap, curveScaleMul, curveOpacity, walkSteps, walkWander
    );
    walkSeeds.push(result.seed);
    regenerateAllCurves();
  }

  function clearCurves() {
    curveSeeds = [];
    walkSeeds = [];
    curveGaussians = [];
    regenerateGaussians();
  }

  // Initial render
  regenerateGaussians();
  updateUIVisibility();

  // ─── Events ───
  modeSelect.addEventListener('change', () => {
    currentMode = modeSelect.value as RenderMode;
    updateUIVisibility();
    regenerateGaussians();
  });

  thicknessInput.addEventListener('input', () => {
    currentThickness = parseFloat(thicknessInput.value);
    thicknessVal.textContent = currentThickness.toFixed(1) + ' px';
    regenerateGaussians();
  });

  colorSelect.addEventListener('change', () => {
    currentColor = colorSelect.value as ColorMode;
    regenerateGaussians();
  });

  // Curve param events — regenerate all existing curves with new values
  curveRadiusInput.addEventListener('input', () => {
    curveRadius = parseFloat(curveRadiusInput.value);
    curveRadiusVal.textContent = curveRadius.toFixed(1);
    if (currentMode === 'curves') regenerateAllCurves();
  });

  curveOverlapInput.addEventListener('input', () => {
    curveOverlap = parseFloat(curveOverlapInput.value);
    curveOverlapVal.textContent = curveOverlap.toFixed(2) + '×';
    if (currentMode === 'curves') regenerateAllCurves();
  });

  curveScaleMulInput.addEventListener('input', () => {
    curveScaleMul = parseFloat(curveScaleMulInput.value);
    curveScaleMulVal.textContent = curveScaleMul.toFixed(1) + '×';
    if (currentMode === 'curves') regenerateAllCurves();
  });

  curveOpacityInput.addEventListener('input', () => {
    curveOpacity = parseFloat(curveOpacityInput.value);
    curveOpacityVal.textContent = curveOpacity.toFixed(2);
    if (currentMode === 'curves') regenerateAllCurves();
  });

  curveTmultInput.addEventListener('input', () => {
    curveTangentMultiplier = parseFloat(curveTmultInput.value);
    curveTmultVal.textContent = curveTangentMultiplier.toFixed(1) + '×';
    if (currentMode === 'curves') regenerateAllCurves();
  });

  curveSamplesInput.addEventListener('input', () => {
    curveSamples = parseInt(curveSamplesInput.value);
    curveSamplesVal.textContent = String(curveSamples);
    if (currentMode === 'curves') regenerateAllCurves();
  });

  showReferenceCheck.addEventListener('change', () => {
    showReference = showReferenceCheck.checked;
    regenerateGaussians();
  });

  walkStepsInput.addEventListener('input', () => {
    walkSteps = parseInt(walkStepsInput.value);
    walkStepsVal.textContent = String(walkSteps);
    if (currentMode === 'curves') regenerateAllCurves();
  });

  walkWanderInput.addEventListener('input', () => {
    walkWander = parseFloat(walkWanderInput.value);
    walkWanderVal.textContent = walkWander.toFixed(2);
    if (currentMode === 'curves') regenerateAllCurves();
  });

  genWalkBtn.addEventListener('click', generateWalk);
  genCurveBtn.addEventListener('click', generateCurve);
  clearCurvesBtn.addEventListener('click', clearCurves);

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
    if (Math.sqrt(dx*dx + dy*dy + dz*dz) > 0.005) {
      lastCamPos = [...camera.position] as [number, number, number];
      regenerateGaussians();
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
