#!/usr/bin/env node
// ─────────────────────────────────────────────────────────────
// spline-cli — build a Kochanek–Bartels (TCB) spline through points.
//
// Mirrors src/spline.ts so it runs standalone with plain Node (no deps/build).
//
// Examples:
//   node scripts/spline-cli.mjs --points '[[0,0,0],[1,2,0],[3,1,1]]' --tension 0.3
//   node scripts/spline-cli.mjs --random --count 6 --start 0,0,0 --end 3,0,0 --seed 7
//   node scripts/spline-cli.mjs --file pts.json --samples 40 --out curve.json
//
// Output: JSON { control: [[x,y,z]...], curve: [[x,y,z]...] }  (or --format csv).
// ─────────────────────────────────────────────────────────────

import { readFileSync, writeFileSync } from 'node:fs';

function parseArgs(argv) {
  const a = {};
  for (let i = 0; i < argv.length; i++) {
    const t = argv[i];
    if (t.startsWith('--')) {
      const key = t.slice(2);
      const next = argv[i + 1];
      if (next === undefined || next.startsWith('--')) { a[key] = true; }
      else { a[key] = next; i++; }
    }
  }
  return a;
}

const vec = (s) => s.split(',').map(Number);
const num = (v, d) => (v === undefined ? d : Number(v));

function mulberry32(seed) {
  let x = seed >>> 0;
  return () => {
    x = (x + 0x6d2b79f5) | 0;
    let t = Math.imul(x ^ (x >>> 15), 1 | x);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function hermitePoint(t, p0, m0, p1, m1) {
  const t2 = t * t, t3 = t2 * t;
  const h00 = 2 * t3 - 3 * t2 + 1;
  const h10 = t3 - 2 * t2 + t;
  const h01 = -2 * t3 + 3 * t2;
  const h11 = t3 - t2;
  return [
    h00 * p0[0] + h10 * m0[0] + h01 * p1[0] + h11 * m1[0],
    h00 * p0[1] + h10 * m0[1] + h01 * p1[1] + h11 * m1[1],
    h00 * p0[2] + h10 * m0[2] + h01 * p1[2] + h11 * m1[2],
  ];
}

function splineThroughPoints(points, { tension: T, bias: B, continuity: C, samplesPerSegment }) {
  const n = points.length;
  if (n < 2) return points.map((p) => [...p]);
  const seg = Math.max(1, Math.round(samplesPerSegment));
  const at = (i) => points[Math.max(0, Math.min(n - 1, i))];
  const a = ((1 - T) * (1 + B) * (1 + C)) / 2;
  const b = ((1 - T) * (1 - B) * (1 - C)) / 2;
  const c = ((1 - T) * (1 + B) * (1 - C)) / 2;
  const d = ((1 - T) * (1 - B) * (1 + C)) / 2;
  const out = [];
  for (let i = 0; i < n - 1; i++) {
    const pPrev = at(i - 1), p0 = at(i), p1 = at(i + 1), pNext = at(i + 2);
    const m0 = [
      a * (p0[0] - pPrev[0]) + b * (p1[0] - p0[0]),
      a * (p0[1] - pPrev[1]) + b * (p1[1] - p0[1]),
      a * (p0[2] - pPrev[2]) + b * (p1[2] - p0[2]),
    ];
    const m1 = [
      c * (p1[0] - p0[0]) + d * (pNext[0] - p1[0]),
      c * (p1[1] - p0[1]) + d * (pNext[1] - p1[1]),
      c * (p1[2] - p0[2]) + d * (pNext[2] - p1[2]),
    ];
    for (let s = 0; s < seg; s++) out.push(hermitePoint(s / seg, p0, m0, p1, m1));
  }
  out.push([...points[n - 1]]);
  return out;
}

function randomPointsBetween(start, end, count, spread, rng) {
  const total = Math.max(2, Math.round(count));
  const span = Math.hypot(end[0] - start[0], end[1] - start[1], end[2] - start[2]) || 1;
  const amp = spread * span;
  const pts = [[...start]];
  for (let i = 1; i < total - 1; i++) {
    const t = i / (total - 1);
    pts.push([
      start[0] + (end[0] - start[0]) * t + (rng() * 2 - 1) * amp,
      start[1] + (end[1] - start[1]) * t + (rng() * 2 - 1) * amp,
      start[2] + (end[2] - start[2]) * t + (rng() * 2 - 1) * amp,
    ]);
  }
  pts.push([...end]);
  return pts;
}

const USAGE = `spline-cli — TCB spline through points

Input (choose one):
  --points '[[x,y,z],...]'   control points as JSON
  --file PATH                JSON array of [x,y,z] (use - for stdin)
  --random                   generate points; with:
       --count N             number of points (default 6)
       --start x,y,z         default 0,0,0
       --end x,y,z           default 1,0,0
       --spread S            interior scatter, fraction of span (default 0.3)
       --seed K              RNG seed (default 1)

Shape:
  --tension T  --bias B  --continuity C   each -1..1 (default 0)
  --samples N                              samples per segment (default 24)

Output:
  --format points|csv        default points (JSON)
  --out PATH                  write to file (default stdout)`;

const args = parseArgs(process.argv.slice(2));
if (args.help || args.h || Object.keys(args).length === 0) {
  console.log(USAGE);
  process.exit(0);
}

let control;
if (args.points) {
  control = JSON.parse(args.points);
} else if (args.file) {
  const text = args.file === '-' ? readFileSync(0, 'utf8') : readFileSync(args.file, 'utf8');
  control = JSON.parse(text);
} else if (args.random) {
  const rng = mulberry32(num(args.seed, 1));
  control = randomPointsBetween(
    args.start ? vec(args.start) : [0, 0, 0],
    args.end ? vec(args.end) : [1, 0, 0],
    num(args.count, 6),
    num(args.spread, 0.3),
    rng,
  );
} else {
  console.error('No input. Pass --points, --file, or --random. (--help for usage)');
  process.exit(1);
}

const curve = splineThroughPoints(control, {
  tension: num(args.tension, 0),
  bias: num(args.bias, 0),
  continuity: num(args.continuity, 0),
  samplesPerSegment: num(args.samples, 24),
});

let output;
if (args.format === 'csv') {
  output = curve.map((p) => p.join(',')).join('\n') + '\n';
} else {
  output = JSON.stringify({ control, curve }, null, 2);
}

if (args.out) {
  writeFileSync(args.out, output);
  console.error(`Wrote ${curve.length} points → ${args.out}`);
} else {
  process.stdout.write(output + '\n');
}
