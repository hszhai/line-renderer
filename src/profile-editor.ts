// ─────────────────────────────────────────────────────────────
// Profile Editor
//
// A small interactive curve editor for the stroke WIDTH profile. The x-axis is
// position along the stroke (t: 0 = start → 1 = end); the y-axis is the width
// multiplier (0 → 1× the Tube Radius). Drag control points to shape the stroke,
// click empty space to add a point, double-click a middle point to remove it.
// `sample(t)` returns the piecewise-linear multiplier used by the renderer, so
// you can author calligraphic / tapered / bulging strokes.
// ─────────────────────────────────────────────────────────────

export interface ProfilePoint { t: number; w: number; }

const MIN_W = 0.02;
const clamp01 = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x);

export const PROFILE_PRESETS: Record<string, ProfilePoint[]> = {
  uniform: [{ t: 0, w: 1 }, { t: 1, w: 1 }],
  taper: [{ t: 0, w: 0.08 }, { t: 0.5, w: 1 }, { t: 1, w: 0.08 }],
  grow: [{ t: 0, w: 0.1 }, { t: 1, w: 1 }],
  shrink: [{ t: 0, w: 1 }, { t: 1, w: 0.1 }],
  // A calligraphic stroke: thin entry, swell, thin waist, swell, thin exit.
  calligraphic: [{ t: 0, w: 0.05 }, { t: 0.22, w: 1 }, { t: 0.5, w: 0.45 }, { t: 0.78, w: 1 }, { t: 1, w: 0.05 }],
};

export class ProfileEditor {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private points: ProfilePoint[];
  private onChange: () => void;
  private dragIdx = -1;
  private dpr = Math.min(window.devicePixelRatio || 1, 2);
  private pendingChange = false;

  constructor(canvas: HTMLCanvasElement, onChange: () => void) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d')!;
    this.onChange = onChange;
    this.points = PROFILE_PRESETS.uniform.map((p) => ({ ...p }));
    this.bindEvents();
    this.resize();
  }

  /** Re-measure the backing store (call when the panel becomes visible). */
  resize() {
    const w = this.canvas.clientWidth || 240;
    const h = this.canvas.clientHeight || 90;
    this.canvas.width = Math.round(w * this.dpr);
    this.canvas.height = Math.round(h * this.dpr);
    this.render();
  }

  loadPreset(name: string) {
    const preset = PROFILE_PRESETS[name] || PROFILE_PRESETS.uniform;
    this.points = preset.map((p) => ({ ...p }));
    this.render();
    this.onChange();
  }

  /** Piecewise-linear width multiplier at position t ∈ [0,1]. */
  sample(t: number): number {
    const pts = this.points;
    if (pts.length === 0) return 1;
    const x = clamp01(t);
    if (x <= pts[0].t) return Math.max(MIN_W, pts[0].w);
    const last = pts.length - 1;
    if (x >= pts[last].t) return Math.max(MIN_W, pts[last].w);
    for (let i = 0; i < last; i++) {
      const a = pts[i], b = pts[i + 1];
      if (x >= a.t && x <= b.t) {
        const u = (x - a.t) / Math.max(1e-6, b.t - a.t);
        return Math.max(MIN_W, a.w + (b.w - a.w) * u);
      }
    }
    return 1;
  }

  private cssSize(): [number, number] {
    return [this.canvas.clientWidth || 240, this.canvas.clientHeight || 90];
  }

  private toPx(p: ProfilePoint): [number, number] {
    const [w, h] = this.cssSize();
    return [p.t * w, (1 - p.w) * h];
  }

  private fromMouse(e: MouseEvent): ProfilePoint {
    const r = this.canvas.getBoundingClientRect();
    return { t: clamp01((e.clientX - r.left) / r.width), w: clamp01(1 - (e.clientY - r.top) / r.height) };
  }

  private nearestPoint(e: MouseEvent): number {
    const r = this.canvas.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    let best = -1, bestD = 12;
    for (let i = 0; i < this.points.length; i++) {
      const [cx, cy] = this.toPx(this.points[i]);
      const d = Math.hypot(cx - mx, cy - my);
      if (d < bestD) { bestD = d; best = i; }
    }
    return best;
  }

  private scheduleChange() {
    if (this.pendingChange) return;
    this.pendingChange = true;
    requestAnimationFrame(() => { this.pendingChange = false; this.onChange(); });
  }

  private bindEvents() {
    this.canvas.addEventListener('mousedown', (e) => {
      e.preventDefault();
      const idx = this.nearestPoint(e);
      if (idx >= 0) {
        this.dragIdx = idx;
      } else {
        const p = this.fromMouse(e);
        this.points.push(p);
        this.points.sort((a, b) => a.t - b.t);
        this.dragIdx = this.points.indexOf(p);
        this.scheduleChange();
      }
      this.render();
    });

    window.addEventListener('mousemove', (e) => {
      if (this.dragIdx < 0) return;
      const m = this.fromMouse(e);
      const last = this.points.length - 1;
      const p = this.points[this.dragIdx];
      p.w = m.w;
      if (this.dragIdx !== 0 && this.dragIdx !== last) {
        const lo = this.points[this.dragIdx - 1].t + 0.001;
        const hi = this.points[this.dragIdx + 1].t - 0.001;
        p.t = Math.min(hi, Math.max(lo, m.t));
      }
      this.render();
      this.scheduleChange();
    });

    window.addEventListener('mouseup', () => {
      if (this.dragIdx >= 0) { this.dragIdx = -1; this.onChange(); }
    });

    this.canvas.addEventListener('dblclick', (e) => {
      const idx = this.nearestPoint(e);
      const last = this.points.length - 1;
      if (idx > 0 && idx < last) {
        this.points.splice(idx, 1);
        this.render();
        this.onChange();
      }
    });
  }

  render() {
    const ctx = this.ctx;
    const [W, H] = this.cssSize();
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    ctx.fillStyle = '#161616';
    ctx.fillRect(0, 0, W, H);

    // Grid
    ctx.strokeStyle = 'rgba(255,255,255,0.07)';
    ctx.lineWidth = 1;
    for (let i = 1; i < 4; i++) {
      const x = (W * i) / 4;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    }
    const ymid = H / 2;
    ctx.beginPath(); ctx.moveTo(0, ymid); ctx.lineTo(W, ymid); ctx.stroke();

    // Curve + filled area under it
    ctx.beginPath();
    ctx.moveTo(this.toPx(this.points[0])[0], H);
    for (const p of this.points) {
      const [x, y] = this.toPx(p);
      ctx.lineTo(x, y);
    }
    ctx.lineTo(this.toPx(this.points[this.points.length - 1])[0], H);
    ctx.closePath();
    ctx.fillStyle = 'rgba(79,195,247,0.18)';
    ctx.fill();

    ctx.beginPath();
    this.points.forEach((p, i) => {
      const [x, y] = this.toPx(p);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = '#4fc3f7';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Control points
    for (const p of this.points) {
      const [x, y] = this.toPx(p);
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fillStyle = '#fff';
      ctx.fill();
    }
  }
}
