import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { ARKIT_52 } from "./humanoid.js";
import {
  FACE_RIG_INDEX,
  FACE_RIG_SHAPES,
  buildFaceRig,
  detectFaceLandmarks,
  faceMaps,
  faceRigWeights,
  findHeadBox,
  OrthoCaster,
  remapFaceRigWeights,
  visemeFrameToFaceRig,
  type FaceSourceMesh,
  type FaceTextureSampler,
} from "./face-rig.js";

const here = dirname(fileURLToPath(import.meta.url));

// ── a synthetic generated character: box body, neck, sculpted head ─────────
// Head centre and features, in metres (character faces +Z, its left is +X).
const HEAD = { y: 1.62, rx: 0.08, ry: 0.11, rz: 0.09 };
const EYE_THETA = 1.3;
const EYE_DPHI = 0.42;
const MOUTH_THETA = 1.95;

function sculpt(theta: number, phi: number): [number, number, number] {
  const nx = Math.sin(theta) * Math.cos(phi), ny = Math.cos(theta), nz = Math.sin(theta) * Math.sin(phi);
  const dPhi = Math.atan2(Math.sin(phi - Math.PI / 2), Math.cos(phi - Math.PI / 2));
  let z = nz * HEAD.rz;
  const nose = Math.exp(-(dPhi ** 2) / 0.12 ** 2) * Math.exp(-((theta - 1.62) ** 2) / 0.2 ** 2);
  z += nose * 0.03;
  const chin = Math.exp(-((theta - 2.3) ** 2) / 0.12 ** 2) * Math.exp(-(dPhi ** 2) / 0.25 ** 2);
  z += chin * 0.01;
  return [nx * HEAD.rx, HEAD.y + ny * HEAD.ry, z];
}

/** Skin, with dark eyes and a dark red mouth painted where the sculpt puts them. */
const sampler: FaceTextureSampler = (u, v) => {
  const phi = u * Math.PI * 2, theta = v * Math.PI;
  const dPhi = Math.atan2(Math.sin(phi - Math.PI / 2), Math.cos(phi - Math.PI / 2));
  const eye = Math.min((theta - EYE_THETA) ** 2 / 0.06 ** 2 + (Math.abs(dPhi) - EYE_DPHI) ** 2 / 0.1 ** 2, 9);
  if (eye < 1) return [0.08, 0.06, 0.05];
  if (Math.abs(theta - MOUTH_THETA) < 0.035 && Math.abs(dPhi) < 0.26) return [0.45, 0.12, 0.13];
  return [0.82, 0.62, 0.5];
};

function character(): FaceSourceMesh {
  const pos: number[] = [], uv: number[] = [], idx: number[] = [];
  const box = (c: [number, number, number], h: [number, number, number]) => {
    const b = pos.length / 3;
    for (const [x, y, z] of [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]) {
      pos.push(c[0] + x! * h[0], c[1] + y! * h[1], c[2] + z! * h[2]); uv.push(0.99, 0.99);
    }
    for (const f of [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2], [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]]) idx.push(b + f[0]!, b + f[1]!, b + f[2]!);
  };
  box([0, 1.2, 0], [0.2, 0.25, 0.1]);   // torso
  box([0.1, 0.48, 0], [0.06, 0.47, 0.06]); // legs
  box([-0.1, 0.48, 0], [0.06, 0.47, 0.06]);
  // Neck: a column of rings from the shoulders into the head.
  const nb = pos.length / 3, RING = 16;
  for (let r = 0; r <= 6; r++) {
    for (let k = 0; k < RING; k++) {
      const a = (k / RING) * Math.PI * 2;
      pos.push(Math.cos(a) * 0.045, 1.45 + r * 0.03, Math.sin(a) * 0.045); uv.push(0.99, 0.99);
    }
  }
  for (let r = 0; r < 6; r++) for (let k = 0; k < RING; k++) {
    const a = nb + r * RING + k, b = nb + r * RING + ((k + 1) % RING), c = a + RING, d = b + RING;
    idx.push(a, c, b, b, c, d);
  }
  const LAT = 48, LON = 64;
  const hb = pos.length / 3;
  for (let y = 0; y <= LAT; y++) {
    const theta = (y / LAT) * Math.PI;
    for (let x = 0; x <= LON; x++) {
      const phi = (x / LON) * Math.PI * 2;
      pos.push(...sculpt(theta, phi));
      uv.push(x / LON, y / LAT);
    }
  }
  for (let y = 0; y < LAT; y++) for (let x = 0; x < LON; x++) {
    const i0 = hb + y * (LON + 1) + x, i1 = i0 + 1, i2 = i0 + LON + 2, i3 = i0 + LON + 1;
    idx.push(i0, i1, i2, i0, i2, i3);
  }
  return { positions: new Float32Array(pos), uvs: new Float32Array(uv), indices: new Uint32Array(idx) };
}

const eyeY = HEAD.y + Math.cos(EYE_THETA) * HEAD.ry;
const mouthY = HEAD.y + Math.cos(MOUTH_THETA) * HEAD.ry;

describe("face rig naming", () => {
  it("has the 52 ARKit names, each once", () => {
    expect(FACE_RIG_SHAPES).toHaveLength(52);
    expect(new Set(FACE_RIG_SHAPES).size).toBe(52);
    expect([...FACE_RIG_SHAPES].sort()).toEqual([...ARKIT_52].sort());
  });

  // The audio2face package lives next to the engine in the monorepo but is not in the
  // public mirror, so this cross-package check skips where it is absent.
  const A2F_SHAPES = join(here, "../../../augengine-audio2face/src/arkit-shapes.ts");
  it.skipIf(!existsSync(A2F_SHAPES))("uses the audio2face channel order the live lip-sync emits", () => {
    const src = readFileSync(A2F_SHAPES, "utf8");
    const block = src.slice(src.indexOf("ARKIT_SHAPES = ["), src.indexOf("] as const"));
    const names = [...block.matchAll(/"([A-Za-z]+)"/g)].map((m) => m[1]);
    expect(names).toEqual([...FACE_RIG_SHAPES]);
  });

  it("builds and re-orders weights by name", () => {
    const w = faceRigWeights({ jawOpen: 0.8, eyeBlinkLeft: 2, notAShape: 1 });
    expect(w[FACE_RIG_INDEX.jawOpen]).toBeCloseTo(0.8);
    expect(w[FACE_RIG_INDEX.eyeBlinkLeft]).toBe(1);
    const alpha = remapFaceRigWeights(w, ["Face.jawOpen", ...ARKIT_52.filter((n) => n !== "jawOpen"), "extra"]);
    expect(alpha[0]).toBeCloseTo(0.8);
    expect(alpha[1 + ARKIT_52.filter((n) => n !== "jawOpen").indexOf("eyeBlinkLeft")]).toBe(1);
    expect(alpha[alpha.length - 1]).toBe(0);
  });

  it("maps visemes and procedural channels onto named shapes", () => {
    const w = visemeFrameToFaceRig({ visemes: { aa: 1 }, procedural: { blinkR: 0.6 } });
    expect(w[FACE_RIG_INDEX.jawOpen]).toBeCloseTo(0.72);
    expect(w[FACE_RIG_INDEX.eyeBlinkRight]).toBeCloseTo(0.6);
    expect(w[FACE_RIG_INDEX.eyeBlinkLeft]).toBe(0);
  });
});

describe("face detection", () => {
  const mesh = character();
  const box = findHeadBox(mesh);

  it("finds the head above the neck", () => {
    expect(box.maxY).toBeCloseTo(HEAD.y + HEAD.ry, 2);
    expect(box.minY).toBeGreaterThan(1.45);
    expect(box.minY).toBeLessThan(HEAD.y - 0.06);
    expect(box.maxX - box.minX).toBeCloseTo(2 * HEAD.rx, 1);
  });

  it("reads the eyes and mouth off the texture and the nose off the depth", () => {
    const caster = new OrthoCaster(mesh, { minX: -0.2, maxX: 0.2, minY: 1.3, maxY: 1.9 });
    const L = detectFaceLandmarks(faceMaps(mesh, box, caster, sampler), box);
    expect(L.found).toMatchObject({ eyes: true, mouth: true, nose: true });
    expect(Math.abs(L.eyeY - eyeY)).toBeLessThan(0.012);
    expect(Math.abs(L.mouthY - mouthY)).toBeLessThan(0.01);
    expect(Math.abs(L.eyeX - Math.cos(Math.PI / 2 - EYE_DPHI) * HEAD.rx)).toBeLessThan(0.012);
    expect(L.chinY).toBeLessThan(L.mouthY);
    expect(L.noseY).toBeGreaterThan(L.mouthY);
    expect(L.noseY).toBeLessThan(L.eyeY);
  });

  it("falls back to proportions without a colour source", () => {
    const caster = new OrthoCaster(mesh, { minX: -0.2, maxX: 0.2, minY: 1.3, maxY: 1.9 });
    const L = detectFaceLandmarks(faceMaps(mesh, box, caster), box);
    expect(L.found.eyes).toBe(false);
    expect(L.found.mouth).toBe(false);
    expect(L.chinY < L.mouthY && L.mouthY < L.noseY && L.noseY < L.eyeY && L.eyeY < L.browY).toBe(true);
  });
});

describe("face patch deformers", () => {
  const mesh = character();
  const rig = buildFaceRig(mesh, { sampler })!;
  const n = rig.positions.length / 3;
  const P = rig.positions;
  const delta = (shape: (typeof FACE_RIG_SHAPES)[number], v: number) => {
    const d = rig.morphPositions[FACE_RIG_INDEX[shape]]!;
    return [d[v * 3]!, d[v * 3 + 1]!, d[v * 3 + 2]!] as const;
  };
  const mag = (shape: (typeof FACE_RIG_SHAPES)[number], v: number) => Math.hypot(...delta(shape, v));

  it("builds geometry with 52 targets and finite values", () => {
    expect(rig).not.toBeNull();
    expect(rig.morphPositions).toHaveLength(52);
    expect(rig.indices.length % 3).toBe(0);
    for (const t of rig.morphPositions) expect(t.length).toBe(n * 3);
    for (const a of [rig.positions, rig.normals, rig.uvs!, ...rig.morphPositions]) expect(a.every(Number.isFinite)).toBe(true);
    expect(rig.stats.triangles).toBeGreaterThan(1500);
  });

  it("splits the lips: rest copies coincide, jawOpen pulls only the lower one down", () => {
    let pairs = 0;
    for (let a = 0; a < n; a++) {
      for (let b = a + 1; b < Math.min(n, a + 3); b++) {
        if (P[a * 3] !== P[b * 3] || P[a * 3 + 1] !== P[b * 3 + 1] || P[a * 3 + 2] !== P[b * 3 + 2]) continue;
        if (Math.abs(P[a * 3 + 1]! - rig.landmarks.mouthY) > 0.002) continue;
        const da = delta("jawOpen", a)[1], db = delta("jawOpen", b)[1];
        expect(Math.min(da, db)).toBeLessThan(-0.003);
        expect(Math.max(da, db)).toBe(0);
        pairs++;
      }
    }
    expect(pairs).toBeGreaterThan(3);
  });

  it("never moves the patch border (no tearing from the head)", () => {
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (let v = 0; v < n; v++) { minX = Math.min(minX, P[v * 3]!); maxX = Math.max(maxX, P[v * 3]!); minY = Math.min(minY, P[v * 3 + 1]!); maxY = Math.max(maxY, P[v * 3 + 1]!); }
    let border = 0;
    for (let v = 0; v < n; v++) {
      const x = P[v * 3]!, y = P[v * 3 + 1]!;
      const edge = Math.abs(x - minX) < 1e-3 || Math.abs(x - maxX) < 1e-3 || Math.abs(y - minY) < 1e-3;
      if (!edge) continue;
      border++;
      for (const s of FACE_RIG_SHAPES) expect(mag(s, v)).toBe(0);
    }
    expect(border).toBeGreaterThan(20);
  });

  it("closes the character's own eye from above and below", () => {
    let moved = 0;
    for (let v = 0; v < n; v++) {
      const d = delta("eyeBlinkLeft", v);
      if (Math.hypot(...d) === 0) continue;
      moved++;
      expect(P[v * 3]!).toBeGreaterThan(0);
      expect(Number.isFinite(d[1])).toBe(true);
    }
    expect(moved).toBeGreaterThan(10);
    const maxDrop = Math.min(...Array.from({ length: n }, (_, v) => delta("eyeBlinkLeft", v)[1]));
    expect(maxDrop).toBeLessThan(-0.005);
    expect(Math.max(...Array.from({ length: n }, (_, v) => delta("eyeBlinkLeft", v)[1]))).toBeGreaterThan(0);
  });

  it("mirrors left and right shapes", () => {
    for (const [l, r] of [["mouthSmileLeft", "mouthSmileRight"], ["browOuterUpLeft", "browOuterUpRight"], ["eyeBlinkLeft", "eyeBlinkRight"]] as const) {
      let sumL = 0, sumR = 0, dxL = 0, dxR = 0;
      for (let v = 0; v < n; v++) { sumL += mag(l, v); sumR += mag(r, v); dxL += delta(l, v)[0] * Math.sign(P[v * 3]!); dxR += delta(r, v)[0] * Math.sign(P[v * 3]!); }
      expect(sumL).toBeGreaterThan(0);
      expect(Math.abs(sumL - sumR) / sumL).toBeLessThan(0.15);
      expect(Math.sign(dxL)).toBe(Math.sign(dxR) || Math.sign(dxL));
    }
  });

  it("removes only front-of-face source triangles and hides them behind the patch", () => {
    expect(rig.stats.covered).toBeGreaterThan(50);
    const I = mesh.indices, S = mesh.positions;
    for (let t = 0; t < rig.coveredTriangles.length; t++) {
      if (!rig.coveredTriangles[t]) continue;
      for (let k = 0; k < 3; k++) expect(S[I[t * 3 + k]! * 3 + 2]!).toBeGreaterThan(0.02);
    }
  });

  it("has a cavity behind the closed lips that follows the jaw", () => {
    const L = rig.landmarks;
    let lipZ = -Infinity;
    for (let v = 0; v < n; v++) if (Math.abs(P[v * 3]!) < 0.004 && Math.abs(P[v * 3 + 1]! - L.mouthY) < 0.004) lipZ = Math.max(lipZ, P[v * 3 + 2]!);
    let behind = 0;
    for (let v = 0; v < n; v++) {
      const x = P[v * 3]!, y = P[v * 3 + 1]!, z = P[v * 3 + 2]!;
      if (Math.abs(x) < L.mouthHalfW && y < L.mouthY - 0.02 * rig.scale && y > L.mouthY - 0.1 * rig.scale && z < lipZ - 0.03 * rig.scale) {
        if (delta("jawOpen", v)[1] < -0.002) behind++;
      }
    }
    expect(behind).toBeGreaterThan(5);
  });
});
