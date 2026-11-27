import { loadGlb, type AugEngineAnimationFromGltf } from "../scene/loaders/gltf.js";
import { sampleGltfAnimationTrack } from "../scene/loaders/gltf-instance.js";
import type { Quat } from "../scene/types.js";

const GLB_MAGIC = 0x46546c67;
const GLB_VERSION = 2;
const CHUNK_JSON = 0x4e4f534a;
const CHUNK_BIN = 0x004e4942;

/**
 * A one-second punch rotation. The fight samples it at poseFrame/60,
 * which is the simulation frame, not wall-clock time.
 */
export function buildPunchGlb(): Uint8Array {
  const times = new Float32Array([0, 1]);
  const values = new Float32Array([
    0, 0, 0, 1,
    0, 0, Math.sin(Math.PI / 4), Math.cos(Math.PI / 4),
  ]);
  const bin = new Uint8Array(40);
  bin.set(new Uint8Array(times.buffer), 0);
  bin.set(new Uint8Array(values.buffer), 8);

  const root = {
    asset: { version: "2.0", generator: "augengine-fighter-slice" },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ name: "arm" }],
    animations: [
      {
        name: "punch",
        channels: [{ sampler: 0, target: { node: 0, path: "rotation" } }],
        samplers: [{ input: 0, output: 1, interpolation: "LINEAR" }],
      },
    ],
    accessors: [
      {
        bufferView: 0,
        componentType: 5126,
        count: 2,
        type: "SCALAR",
        min: [0],
        max: [1],
      },
      { bufferView: 1, componentType: 5126, count: 2, type: "VEC4" },
    ],
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: 8 },
      { buffer: 0, byteOffset: 8, byteLength: 32 },
    ],
    buffers: [{ byteLength: 40 }],
  };

  const jsonBytes = new TextEncoder().encode(JSON.stringify(root));
  const jsonPad = (4 - (jsonBytes.length % 4)) % 4;
  const jsonChunkLen = jsonBytes.length + jsonPad;
  const total = 12 + 8 + jsonChunkLen + 8 + bin.length;
  const out = new Uint8Array(total);
  const view = new DataView(out.buffer);
  view.setUint32(0, GLB_MAGIC, true);
  view.setUint32(4, GLB_VERSION, true);
  view.setUint32(8, total, true);
  view.setUint32(12, jsonChunkLen, true);
  view.setUint32(16, CHUNK_JSON, true);
  out.set(jsonBytes, 20);
  for (let i = 0; i < jsonPad; i++) out[20 + jsonBytes.length + i] = 0x20;
  const binAt = 20 + jsonChunkLen;
  view.setUint32(binAt, bin.length, true);
  view.setUint32(binAt + 4, CHUNK_BIN, true);
  out.set(bin, binAt + 8);
  return out;
}

let cached: AugEngineAnimationFromGltf | null = null;

export function punchClip(): AugEngineAnimationFromGltf {
  if (cached) return cached;
  const loaded = loadGlb(buildPunchGlb());
  const clip = loaded.animations[0];
  if (!clip || clip.name !== "punch") throw new Error("punch clip missing");
  cached = clip;
  return clip;
}

/** Sample the punch clip at a simulation frame. Time is frame/60. */
export function samplePunch(poseFrame: number): Quat {
  const clip = punchClip();
  const track = clip.tracks[0];
  if (!track) throw new Error("punch clip has no track");
  const sampled = sampleGltfAnimationTrack(track, poseFrame / 60);
  if (!sampled || sampled.length !== 4) throw new Error("punch sample was not a quaternion");
  return sampled;
}
