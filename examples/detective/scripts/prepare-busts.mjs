/**
 * Turn the raw TRELLIS busts (assets-src/busts/<id>_bust.glb, ~15–30k triangles,
 * 1 MB PNG texture) into game assets (public/busts/<id>.glb): one mesh
 * simplified with meshoptimizer (BUST_TRIS, default 4k) with UV seams locked, smooth
 * normals, and the texture re-encoded as JPEG (<id>_tex.jpg, made beforehand).
 *
 * The LOD baker (@mnemopay/augengine-lod) isn't used here on purpose: it
 * flat-shades and drops textures, which suits the city kit but not faces,
 * and the busts are only ever seen up close in one room, so they need no LODs.
 *
 *   node scripts/prepare-busts.mjs
 */
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { MeshoptSimplifier } from "meshoptimizer";

const here = dirname(fileURLToPath(import.meta.url));
const SRC = join(here, "..", "assets-src", "busts");
const OUT = join(here, "..", "public", "busts");
const IDS = ["mara", "desmond", "ines", "teodor"];
const TARGET_TRIS = Number(process.env.BUST_TRIS ?? 4000);

function readGlb(buf) {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const jsonLen = dv.getUint32(12, true);
  const json = JSON.parse(new TextDecoder().decode(buf.subarray(20, 20 + jsonLen)));
  const binOff = 20 + jsonLen;
  const bin = buf.subarray(binOff + 8, binOff + 8 + dv.getUint32(binOff, true));
  const acc = (i) => {
    const a = json.accessors[i];
    const bv = json.bufferViews[a.bufferView];
    const n = { SCALAR: 1, VEC2: 2, VEC3: 3 }[a.type] * a.count;
    const start = bin.byteOffset + (bv.byteOffset ?? 0) + (a.byteOffset ?? 0);
    const Ctor = { 5126: Float32Array, 5125: Uint32Array, 5123: Uint16Array }[a.componentType];
    return new Ctor(bin.buffer.slice(start, start + n * Ctor.BYTES_PER_ELEMENT));
  };
  const p = json.meshes[0].primitives[0];
  return { pos: acc(p.attributes.POSITION), uv: acc(p.attributes.TEXCOORD_0), idx: Uint32Array.from(acc(p.indices)) };
}

function smoothNormals(pos, idx) {
  const n = new Float32Array(pos.length);
  for (let t = 0; t < idx.length; t += 3) {
    const [a, b, c] = [idx[t] * 3, idx[t + 1] * 3, idx[t + 2] * 3];
    const e1 = [pos[b] - pos[a], pos[b + 1] - pos[a + 1], pos[b + 2] - pos[a + 2]];
    const e2 = [pos[c] - pos[a], pos[c + 1] - pos[a + 1], pos[c + 2] - pos[a + 2]];
    const f = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
    for (const v of [a, b, c]) for (let k = 0; k < 3; k++) n[v + k] += f[k];
  }
  for (let i = 0; i < n.length; i += 3) {
    const l = Math.hypot(n[i], n[i + 1], n[i + 2]) || 1;
    n[i] /= l; n[i + 1] /= l; n[i + 2] /= l;
  }
  return n;
}

/** Keep only vertices the simplified index buffer still uses. */
function compact(pos, uv, idx) {
  const remap = new Int32Array(pos.length / 3).fill(-1);
  const P = [], U = [];
  const out = new Uint32Array(idx.length);
  for (let i = 0; i < idx.length; i++) {
    const v = idx[i];
    if (remap[v] < 0) {
      remap[v] = P.length / 3;
      P.push(pos[v * 3], pos[v * 3 + 1], pos[v * 3 + 2]);
      U.push(uv[v * 2], uv[v * 2 + 1]);
    }
    out[i] = remap[v];
  }
  return { pos: new Float32Array(P), uv: new Float32Array(U), idx: out };
}

function writeGlb({ pos, nrm, uv, idx }, jpeg, name) {
  const parts = [];
  let len = 0;
  const views = [];
  const add = (bytes, target) => {
    const pad = (4 - (len % 4)) % 4;
    if (pad) { parts.push(new Uint8Array(pad)); len += pad; }
    views.push({ buffer: 0, byteOffset: len, byteLength: bytes.byteLength, ...(target ? { target } : {}) });
    parts.push(bytes);
    len += bytes.byteLength;
    return views.length - 1;
  };
  const u8 = (a) => new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < pos.length; i += 3) for (let k = 0; k < 3; k++) { min[k] = Math.min(min[k], pos[i + k]); max[k] = Math.max(max[k], pos[i + k]); }
  const vc = pos.length / 3;
  const json = {
    asset: { version: "2.0", generator: "augengine detective prepare-busts" },
    scene: 0, scenes: [{ nodes: [0] }], nodes: [{ mesh: 0, name }],
    meshes: [{ name, primitives: [{ attributes: { POSITION: 0, NORMAL: 1, TEXCOORD_0: 2 }, indices: 3, material: 0 }] }],
    materials: [{ name: `${name}-paint`, pbrMetallicRoughness: { baseColorTexture: { index: 0 }, metallicFactor: 0, roughnessFactor: 0.85 } }],
    textures: [{ source: 0, sampler: 0 }],
    samplers: [{ magFilter: 9729, minFilter: 9987, wrapS: 33071, wrapT: 33071 }],
    images: [{ bufferView: 4, mimeType: "image/jpeg" }],
    accessors: [
      { bufferView: add(u8(pos), 34962), componentType: 5126, count: vc, type: "VEC3", min, max },
      { bufferView: add(u8(nrm), 34962), componentType: 5126, count: vc, type: "VEC3" },
      { bufferView: add(u8(uv), 34962), componentType: 5126, count: vc, type: "VEC2" },
      { bufferView: add(u8(idx), 34963), componentType: 5125, count: idx.length, type: "SCALAR" },
    ],
    bufferViews: views,
  };
  add(jpeg);
  json.buffers = [{ byteLength: len }];
  let jsonBytes = new TextEncoder().encode(JSON.stringify(json));
  const jpad = (4 - (jsonBytes.length % 4)) % 4;
  jsonBytes = Uint8Array.from([...jsonBytes, ...new Array(jpad).fill(0x20)]);
  const binPad = (4 - (len % 4)) % 4;
  const total = 12 + 8 + jsonBytes.length + 8 + len + binPad;
  const out = new Uint8Array(total);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, 0x46546c67, true); dv.setUint32(4, 2, true); dv.setUint32(8, total, true);
  dv.setUint32(12, jsonBytes.length, true); dv.setUint32(16, 0x4e4f534a, true); out.set(jsonBytes, 20);
  let o = 20 + jsonBytes.length;
  dv.setUint32(o, len + binPad, true); dv.setUint32(o + 4, 0x004e4942, true); o += 8;
  for (const p of parts) { out.set(p, o); o += p.byteLength; }
  return out;
}

await MeshoptSimplifier.ready;
mkdirSync(OUT, { recursive: true });
for (const id of IDS) {
  const raw = readGlb(new Uint8Array(readFileSync(join(SRC, `${id}_bust.glb`))));
  const tris = raw.idx.length / 3;
  const target = Math.min(raw.idx.length, TARGET_TRIS * 3);
  // UVs are vertex attributes here, so seams are separate vertices and stay intact.
  // UV seams are split vertices, i.e. open borders; locking them stops seam edges collapsing into holes.
  const [simple, err] = MeshoptSimplifier.simplify(raw.idx, raw.pos, 3, target, 0.02, ["LockBorder"]);
  const m = compact(raw.pos, raw.uv, simple);
  const glb = writeGlb({ ...m, nrm: smoothNormals(m.pos, m.idx) }, new Uint8Array(readFileSync(join(SRC, `${id}_tex.jpg`))), id);
  writeFileSync(join(OUT, `${id}.glb`), glb);
  console.log(`${id}: ${tris} -> ${m.idx.length / 3} triangles (error ${err.toFixed(4)}), ${(glb.byteLength / 1024).toFixed(0)} KB`);
}
