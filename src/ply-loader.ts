// ─────────────────────────────────────────────────────────────
// PLY loader → our Mesh abstraction ({ vertices, faces }).
//
// Handles ASCII and binary (little/big-endian) PLY. Connectivity may be stored
// either as a `face` element (polygons → fan-triangulated) or as a `tristrips`
// element (triangle strips with -1 restart markers → expanded to triangles).
// Only x/y/z vertex properties are kept; any other properties are read past so
// the binary cursor stays aligned. Once we hand back { vertices, faces } the
// rest of the pipeline is identical to the OBJ path — no algorithm changes.
// ─────────────────────────────────────────────────────────────

import { Mesh } from './obj-loader.ts';

interface PlyProp { name: string; isList: boolean; type: string; countType?: string; }
interface PlyElement { name: string; count: number; props: PlyProp[]; }

const SIZES: Record<string, number> = {
  char: 1, int8: 1, uchar: 1, uint8: 1,
  short: 2, int16: 2, ushort: 2, uint16: 2,
  int: 4, int32: 4, uint: 4, uint32: 4, float: 4, float32: 4,
  double: 8, float64: 8,
};
const isFloatType = (t: string) => t === 'float' || t === 'float32' || t === 'double' || t === 'float64';

export async function loadPLY(url: string): Promise<Mesh> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to load ${url}: ${res.status}`);
  return parsePLY(await res.arrayBuffer());
}

export function parsePLY(buffer: ArrayBuffer): Mesh {
  const bytes = new Uint8Array(buffer);
  // The header is ASCII; latin1 keeps byte index == char index so the body
  // offset we compute from the text is also the byte offset.
  const headerStr = new TextDecoder('latin1').decode(bytes.subarray(0, Math.min(bytes.length, 1 << 16)));
  const endPos = headerStr.indexOf('end_header');
  if (endPos < 0) throw new Error('PLY: missing end_header');
  const bodyOffset = headerStr.indexOf('\n', endPos) + 1;

  let little = true;
  let ascii = false;
  const elements: PlyElement[] = [];
  for (const raw of headerStr.slice(0, endPos).split(/\r?\n/)) {
    const p = raw.trim().split(/\s+/);
    if (p[0] === 'format') {
      ascii = p[1] === 'ascii';
      little = p[1] !== 'binary_big_endian';
    } else if (p[0] === 'element') {
      elements.push({ name: p[1], count: parseInt(p[2], 10), props: [] });
    } else if (p[0] === 'property' && elements.length) {
      const el = elements[elements.length - 1];
      if (p[1] === 'list') el.props.push({ name: p[4], isList: true, countType: p[2], type: p[3] });
      else el.props.push({ name: p[2], isList: false, type: p[1] });
    }
  }

  // One reader for both encodings; advances a cursor and returns numbers.
  const dv = new DataView(buffer);
  let off = bodyOffset;
  let tokens: string[] = [];
  let ti = 0;
  if (ascii) tokens = new TextDecoder('latin1').decode(bytes).slice(bodyOffset).split(/\s+/).filter(Boolean);

  function readVal(type: string): number {
    if (ascii) {
      const t = tokens[ti++];
      return isFloatType(type) ? parseFloat(t) : parseInt(t, 10);
    }
    let v: number;
    switch (type) {
      case 'char': case 'int8': v = dv.getInt8(off); break;
      case 'uchar': case 'uint8': v = dv.getUint8(off); break;
      case 'short': case 'int16': v = dv.getInt16(off, little); break;
      case 'ushort': case 'uint16': v = dv.getUint16(off, little); break;
      case 'int': case 'int32': v = dv.getInt32(off, little); break;
      case 'uint': case 'uint32': v = dv.getUint32(off, little); break;
      case 'double': case 'float64': v = dv.getFloat64(off, little); break;
      default: v = dv.getFloat32(off, little); break;
    }
    off += SIZES[type] ?? 4;
    return v;
  }

  let vertices: Float32Array | null = null;
  const faces: number[] = [];

  for (const el of elements) {
    if (el.name === 'vertex') {
      vertices = new Float32Array(el.count * 3);
      const xi = el.props.findIndex((p) => p.name === 'x');
      const yi = el.props.findIndex((p) => p.name === 'y');
      const zi = el.props.findIndex((p) => p.name === 'z');
      for (let i = 0; i < el.count; i++) {
        for (let pj = 0; pj < el.props.length; pj++) {
          const pr = el.props[pj];
          if (pr.isList) {
            const c = readVal(pr.countType!);
            for (let k = 0; k < c; k++) readVal(pr.type);
          } else {
            const val = readVal(pr.type);
            if (pj === xi) vertices[i * 3] = val;
            else if (pj === yi) vertices[i * 3 + 1] = val;
            else if (pj === zi) vertices[i * 3 + 2] = val;
          }
        }
      }
    } else if (el.name === 'tristrips') {
      // Triangle strips: -1 restarts a strip; winding alternates each triangle.
      const strip: number[] = [];
      const flush = () => {
        for (let k = 0; k + 2 < strip.length; k++) {
          const a = strip[k], b = strip[k + 1], c = strip[k + 2];
          if (a === b || b === c || a === c) continue; // degenerate (often used as a joiner)
          if (k & 1) faces.push(b, a, c); else faces.push(a, b, c);
        }
        strip.length = 0;
      };
      for (let i = 0; i < el.count; i++) {
        for (const pr of el.props) {
          if (pr.isList) {
            const c = readVal(pr.countType!);
            for (let k = 0; k < c; k++) {
              const idx = readVal(pr.type);
              if (idx < 0) flush(); else strip.push(idx);
            }
          } else readVal(pr.type);
        }
      }
      flush();
    } else if (el.name === 'face') {
      for (let i = 0; i < el.count; i++) {
        for (const pr of el.props) {
          if (pr.isList) {
            const c = readVal(pr.countType!);
            const poly: number[] = [];
            for (let k = 0; k < c; k++) poly.push(readVal(pr.type));
            for (let k = 1; k + 1 < poly.length; k++) faces.push(poly[0], poly[k], poly[k + 1]); // fan
          } else readVal(pr.type);
        }
      }
    } else {
      // Unknown element: read past it so the cursor stays aligned.
      for (let i = 0; i < el.count; i++) {
        for (const pr of el.props) {
          if (pr.isList) {
            const c = readVal(pr.countType!);
            for (let k = 0; k < c; k++) readVal(pr.type);
          } else readVal(pr.type);
        }
      }
    }
  }

  if (!vertices) throw new Error('PLY: no vertex element');
  return { vertices, faces: new Uint32Array(faces) };
}
