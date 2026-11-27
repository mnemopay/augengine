/**
 * Save/Load orchestration — bundles a deterministic SceneSnapshot + an AI
 * artifact manifest into a single SaveFile. Load reverses both halves and
 * returns a ready-to-run Scene + manifest pair.
 *
 * Format choice: JSON with optional gzip via CompressionStream when
 * available. For phase 1 we ship the JSON path; phase 9 adds a binary
 * format if save sizes become a problem (probably never for browser games).
 */

import { Scene } from "../scene/scene.js";
import {
  captureSnapshot,
  restoreSnapshot,
  worldStateHash,
  type SceneSnapshot,
  type SnapshotOptions,
} from "./state.js";
import { ArtifactManifest } from "./manifest.js";

export interface SaveFile {
  version: 1;
  savedAt: string;
  worldStateHash: string;
  snapshot: SceneSnapshot;
  manifest: ReturnType<ArtifactManifest["toJson"]>;
}

export interface SaveOptions extends SnapshotOptions {
  manifest?: ArtifactManifest;
}

/**
 * Capture the current Scene + provided manifest into a SaveFile object.
 * Hashes the snapshot, stamps the manifest's worldStateHash to match.
 */
export async function save(scene: Scene, opts: SaveOptions = {}): Promise<SaveFile> {
  const snapshot = captureSnapshot(scene, { rngSeed: opts.rngSeed ?? 0 });
  const hash = await worldStateHash(snapshot);
  const manifest = opts.manifest ?? new ArtifactManifest();
  manifest.setWorldStateHash(hash);
  return {
    version: 1,
    savedAt: new Date().toISOString(),
    worldStateHash: hash,
    snapshot,
    manifest: manifest.toJson(),
  };
}

/**
 * Restore a Scene + manifest from a SaveFile. Verifies the world-state hash
 * matches the snapshot before returning (defends against tampered save files).
 */
export async function load(file: SaveFile): Promise<{
  scene: Scene;
  rngSeed: number;
  manifest: ArtifactManifest;
}> {
  if (file.version !== 1) {
    throw new Error(`unsupported SaveFile.version: ${file.version}`);
  }
  const computed = await worldStateHash(file.snapshot);
  if (computed !== file.worldStateHash) {
    throw new Error(
      `SaveFile hash mismatch — snapshot=${computed}, file=${file.worldStateHash}`,
    );
  }
  const scene = restoreSnapshot(file.snapshot);
  const manifest = ArtifactManifest.fromJson(file.manifest);
  return { scene, rngSeed: file.snapshot.rngSeed, manifest };
}

/** Serialize a SaveFile to a UTF-8 byte array (JSON). Phase 1 unencrypted. */
export function serialize(save: SaveFile): Uint8Array {
  return new TextEncoder().encode(JSON.stringify(save));
}

/** Reverse of serialize. */
export function deserialize(bytes: Uint8Array): SaveFile {
  const text = new TextDecoder().decode(bytes);
  return JSON.parse(text) as SaveFile;
}

/**
 * gzip-compressed serialize via the browser-native CompressionStream API.
 * Falls back to identity (uncompressed) when CompressionStream is missing.
 * Browser-only (CompressionStream is not in older Node).
 */
export async function serializeGzip(save: SaveFile): Promise<Uint8Array> {
  const json = JSON.stringify(save);
  const buf = new TextEncoder().encode(json);
  const CS =
    (typeof globalThis !== "undefined" &&
      (globalThis as { CompressionStream?: typeof CompressionStream }).CompressionStream) ||
    null;
  if (!CS) return buf;
  const cs = new CS("gzip");
  const writer = cs.writable.getWriter();
  void writer.write(buf);
  void writer.close();
  const chunks: Uint8Array[] = [];
  const reader = cs.readable.getReader();
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (value) chunks.push(value);
  }
  let total = 0;
  for (const c of chunks) total += c.byteLength;
  const out = new Uint8Array(total);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.byteLength;
  }
  return out;
}

/** Reverse of serializeGzip. Detects gzip magic; falls back to plain JSON otherwise. */
export async function deserializeGzip(bytes: Uint8Array): Promise<SaveFile> {
  const isGzip = bytes.length >= 2 && bytes[0] === 0x1f && bytes[1] === 0x8b;
  if (!isGzip) return deserialize(bytes);
  const DS =
    (typeof globalThis !== "undefined" &&
      (globalThis as { DecompressionStream?: typeof DecompressionStream }).DecompressionStream) ||
    null;
  if (!DS) throw new Error("DecompressionStream unavailable; cannot decode gzip save");
  const ds = new DS("gzip");
  const writer = ds.writable.getWriter();
  void writer.write(bytes as BufferSource);
  void writer.close();
  const chunks: Uint8Array[] = [];
  const reader = ds.readable.getReader();
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (value) chunks.push(value);
  }
  let total = 0;
  for (const c of chunks) total += c.byteLength;
  const out = new Uint8Array(total);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.byteLength;
  }
  return deserialize(out);
}
