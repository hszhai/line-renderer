export interface Mesh {
  vertices: Float32Array;
  faces: Uint32Array;
}

export async function loadOBJ(url: string): Promise<Mesh> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to load ${url}: ${response.status}`);
  const text = await response.text();
  return parseOBJ(text);
}

export function parseOBJ(text: string): Mesh {
  const verts: number[] = [];
  const faces: number[] = [];

  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;

    const parts = trimmed.split(/\s+/);
    if (parts[0] === 'v') {
      verts.push(
        parseFloat(parts[1]),
        parseFloat(parts[2]),
        parseFloat(parts[3])
      );
    } else if (parts[0] === 'f') {
      // Handle f v/vt/vn and f v//vn formats
      for (let i = 1; i <= 3; i++) {
        const idxStr = parts[i].split('/')[0];
        const idx = parseInt(idxStr, 10) - 1; // OBJ is 1-indexed
        faces.push(idx);
      }
    }
  }

  return {
    vertices: new Float32Array(verts),
    faces: new Uint32Array(faces),
  };
}
