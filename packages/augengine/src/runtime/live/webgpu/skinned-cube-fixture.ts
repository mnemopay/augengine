/**
 * In-memory skinned-cube GLB fixture — Track 2 integration test asset.
 *
 * Generates a minimal valid glTF 2.0 GLB describing:
 *   - 8-vertex unit cube (positions + normals + JOINTS_0 + WEIGHTS_0)
 *   - 2-joint skeleton (root at origin, child at +Y)
 *   - 1 rotation channel/sampler animating the child joint 90° about Y
 *     over 1 second (linear)
 *
 * Used by `skinned-cube.test.ts` (vitest) and reusable by Track-2 e2e
 * harnesses that need a known-fingerprint skinned asset without checking
 * a binary into the repo.
 *
 * Keep this builder pure-functional + dep-free so it stays vendor-able
 * into other workspaces (the same fixture concept will land in
 * `examples/augengine-toolkit` once Track 2 is merged).
 */

const GLB_MAGIC = 0x46546c67;
const GLB_VERSION = 2;
const CHUNK_JSON = 0x4e4f534a;
const CHUNK_BIN = 0x004e4942;

interface BufferRegion {
  byteOffset: number;
  byteLength: number;
}

interface BuildResult {
  /** The complete .glb byte stream, suitable for parseGlb/loadGlb. */
  bytes: Uint8Array;
  /** Stable metadata callers can assert against. */
  meta: {
    jointCount: number;
    vertexCount: number;
    indexCount: number;
    clipName: string;
    clipDuration: number;
  };
}

/**
 * Build the skinned-cube GLB. Layout choice: vertices are split into a
 * top half and a bottom half — top-half vertices weight 100% to joint 1
 * (the child); bottom-half vertices weight 100% to joint 0 (the root).
 * That makes the child-joint rotation visibly bend the top of the cube
 * without affecting the bottom, which is the simplest possible visual
 * check for "skin works."
 */
export function buildSkinnedCubeGlb(): BuildResult {
  // 8 cube corners — bottom 4 (y=-0.5) then top 4 (y=+0.5).
  // Layout: [-x,-z], [+x,-z], [+x,+z], [-x,+z] per layer.
  const positions = new Float32Array([
    -0.5, -0.5, -0.5,
     0.5, -0.5, -0.5,
     0.5, -0.5,  0.5,
    -0.5, -0.5,  0.5,
    -0.5,  0.5, -0.5,
     0.5,  0.5, -0.5,
     0.5,  0.5,  0.5,
    -0.5,  0.5,  0.5,
  ]);
  // Outward-pointing diagonal normals — cheap and acceptable for a
  // fixture (real assets carry per-face normals; we don't care for a
  // skinning test).
  const normals = new Float32Array(8 * 3);
  for (let i = 0; i < 8; i++) {
    const x = positions[i * 3]!;
    const y = positions[i * 3 + 1]!;
    const z = positions[i * 3 + 2]!;
    const len = Math.hypot(x, y, z) || 1;
    normals[i * 3] = x / len;
    normals[i * 3 + 1] = y / len;
    normals[i * 3 + 2] = z / len;
  }
  // 12 triangles, CCW from outside.
  const indices = new Uint16Array([
    // bottom (-Y)
    0, 2, 1,  0, 3, 2,
    // top (+Y)
    4, 5, 6,  4, 6, 7,
    // -Z
    0, 1, 5,  0, 5, 4,
    // +Z
    3, 7, 6,  3, 6, 2,
    // -X
    0, 4, 7,  0, 7, 3,
    // +X
    1, 2, 6,  1, 6, 5,
  ]);

  // JOINTS_0 (u16x4) + WEIGHTS_0 (float32x4). 4 entries per vertex.
  const joints = new Uint16Array(8 * 4);
  const weights = new Float32Array(8 * 4);
  for (let i = 0; i < 8; i++) {
    const topHalf = i >= 4; // 0..3 bottom, 4..7 top
    joints[i * 4] = topHalf ? 1 : 0;
    weights[i * 4] = 1;
    // joints[1..3] / weights[1..3] left at 0 — single-bone influence.
  }

  // Skin joints: 2 nodes. Inverse bind matrices = identity (both joints
  // sit at the world origin in bind pose — the root joint and the child
  // are co-located which keeps the math trivial).
  //
  // NB: we keep both joints unparented at the origin in bind pose so
  // inverseBindMatrices = identity and the test fixture stays
  // dependency-free of TRS composition.
  const ibm = new Float32Array(2 * 16);
  ibm[0] = ibm[5] = ibm[10] = ibm[15] = 1;
  ibm[16 + 0] = ibm[16 + 5] = ibm[16 + 10] = ibm[16 + 15] = 1;

  // Animation: child joint (node index 2, see node table below) rotates
  // 90° about Y over 1 second. Keyframes at t=0 (identity) and t=1.
  const animTimes = new Float32Array([0, 1]);
  const animValues = new Float32Array([
    0, 0, 0, 1,
    0, 0.7071068, 0, 0.7071068,
  ]);

  // ── Assemble the BIN buffer ─────────────────────────────────────────
  // Order: positions, normals, indices, joints, weights, ibm, animTimes,
  // animValues. Each section 4-byte aligned.
  const sections: { name: string; bytes: Uint8Array }[] = [
    { name: "pos", bytes: byteCopy(positions) },
    { name: "nrm", bytes: byteCopy(normals) },
    { name: "idx", bytes: byteCopy(indices) },
    { name: "jnt", bytes: byteCopy(joints) },
    { name: "wgt", bytes: byteCopy(weights) },
    { name: "ibm", bytes: byteCopy(ibm) },
    { name: "atm", bytes: byteCopy(animTimes) },
    { name: "aval", bytes: byteCopy(animValues) },
  ];
  const regions = new Map<string, BufferRegion>();
  let cursor = 0;
  for (const s of sections) {
    const aligned = (cursor + 3) & ~3;
    regions.set(s.name, { byteOffset: aligned, byteLength: s.bytes.byteLength });
    cursor = aligned + s.bytes.byteLength;
  }
  const binPad = (4 - (cursor % 4)) % 4;
  const binTotal = cursor + binPad;
  const bin = new Uint8Array(binTotal);
  for (const s of sections) {
    const r = regions.get(s.name)!;
    bin.set(s.bytes, r.byteOffset);
  }

  const r = (name: string) => regions.get(name)!;

  // ── JSON root ────────────────────────────────────────────────────────
  const root = {
    asset: { version: "2.0", generator: "augengine-skinned-cube-fixture" },
    scene: 0,
    scenes: [{ nodes: [0] }],
    // Node 0: mesh-bearing root. Node 1: skin joint 0 (root joint).
    // Node 2: skin joint 1 (child of joint 0). Mesh references skin.
    nodes: [
      { name: "cube", mesh: 0, skin: 0, children: [1] },
      { name: "joint_root", children: [2] },
      { name: "joint_tip", translation: [0, 0, 0] },
    ],
    meshes: [{
      name: "cube_mesh",
      primitives: [{
        attributes: {
          POSITION: 0,
          NORMAL: 1,
          JOINTS_0: 3,
          WEIGHTS_0: 4,
        },
        indices: 2,
      }],
    }],
    skins: [{
      joints: [1, 2],
      inverseBindMatrices: 5,
      skeleton: 1,
    }],
    animations: [{
      name: "tip_yaw_90",
      channels: [{
        sampler: 0,
        target: { node: 2, path: "rotation" },
      }],
      samplers: [{
        input: 6,
        output: 7,
        interpolation: "LINEAR",
      }],
    }],
    accessors: [
      // 0: POSITION  vec3 float32 × 8
      { bufferView: 0, componentType: 5126, count: 8, type: "VEC3",
        min: [-0.5, -0.5, -0.5], max: [0.5, 0.5, 0.5] },
      // 1: NORMAL    vec3 float32 × 8
      { bufferView: 1, componentType: 5126, count: 8, type: "VEC3" },
      // 2: indices   scalar uint16 × 36
      { bufferView: 2, componentType: 5123, count: 36, type: "SCALAR" },
      // 3: JOINTS_0  vec4 uint16 × 8
      { bufferView: 3, componentType: 5123, count: 8, type: "VEC4" },
      // 4: WEIGHTS_0 vec4 float32 × 8
      { bufferView: 4, componentType: 5126, count: 8, type: "VEC4" },
      // 5: IBM       mat4 float32 × 2
      { bufferView: 5, componentType: 5126, count: 2, type: "MAT4" },
      // 6: animTimes  scalar float32 × 2
      { bufferView: 6, componentType: 5126, count: 2, type: "SCALAR",
        min: [0], max: [1] },
      // 7: animValues vec4 float32 × 2 (quaternions)
      { bufferView: 7, componentType: 5126, count: 2, type: "VEC4" },
    ],
    bufferViews: [
      { buffer: 0, byteOffset: r("pos").byteOffset, byteLength: r("pos").byteLength },
      { buffer: 0, byteOffset: r("nrm").byteOffset, byteLength: r("nrm").byteLength },
      { buffer: 0, byteOffset: r("idx").byteOffset, byteLength: r("idx").byteLength },
      { buffer: 0, byteOffset: r("jnt").byteOffset, byteLength: r("jnt").byteLength },
      { buffer: 0, byteOffset: r("wgt").byteOffset, byteLength: r("wgt").byteLength },
      { buffer: 0, byteOffset: r("ibm").byteOffset, byteLength: r("ibm").byteLength },
      { buffer: 0, byteOffset: r("atm").byteOffset, byteLength: r("atm").byteLength },
      { buffer: 0, byteOffset: r("aval").byteOffset, byteLength: r("aval").byteLength },
    ],
    buffers: [{ byteLength: binTotal }],
  };

  // ── Pack the GLB container ──────────────────────────────────────────
  const jsonText = JSON.stringify(root);
  const jsonBytes = new TextEncoder().encode(jsonText);
  const jsonPad = (4 - (jsonBytes.length % 4)) % 4;
  const jsonChunkLen = jsonBytes.length + jsonPad;

  const total = 12 + 8 + jsonChunkLen + 8 + binTotal;
  const out = new Uint8Array(total);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, GLB_MAGIC, true);
  dv.setUint32(4, GLB_VERSION, true);
  dv.setUint32(8, total, true);
  dv.setUint32(12, jsonChunkLen, true);
  dv.setUint32(16, CHUNK_JSON, true);
  out.set(jsonBytes, 20);
  for (let i = 0; i < jsonPad; i++) out[20 + jsonBytes.length + i] = 0x20;
  const binOff = 20 + jsonChunkLen;
  dv.setUint32(binOff, binTotal, true);
  dv.setUint32(binOff + 4, CHUNK_BIN, true);
  out.set(bin, binOff + 8);

  return {
    bytes: out,
    meta: {
      jointCount: 2,
      vertexCount: 8,
      indexCount: 36,
      clipName: "tip_yaw_90",
      clipDuration: 1,
    },
  };
}

function byteCopy(view: ArrayBufferView): Uint8Array {
  return new Uint8Array(view.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength));
}
