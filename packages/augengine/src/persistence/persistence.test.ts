/**
 * Phase 8 round-trip tests. Pure data, no GPU/audio APIs.
 */

import { describe, it, expect } from "vitest";
import {
  Scene,
  Tag,
  type EntityId,
  type SceneSpec,
} from "../scene/index.js";
import {
  ArtifactManifest,
  artifactKey,
  fnv1a64,
  captureSnapshot,
  restoreSnapshot,
  canonicalize,
  worldStateHash,
  diffEntities,
  save,
  load,
  serialize,
  deserialize,
} from "./index.js";

function buildScene(): Scene {
  const spec: SceneSpec = {
    schemaVersion: 1,
    source: { kind: "authored", path: "test" },
    entities: [
      {
        id: "cube" as EntityId,
        name: "Hero Cube",
        components: {
          [Tag.Transform]: {
            position: [0, 1, 0],
            rotation: [0, 0, 0, 1],
            scale: [1, 1, 1],
            parent: null,
          },
          [Tag.Persona]: {
            traits: ["curious"],
            voice: null,
            memoryId: "memory-cube",
            goals: [],
            mood: 0,
          },
        },
      },
    ],
  };
  return new Scene(spec);
}

describe("augengine / persistence — manifest", () => {
  it("fnv1a64 is deterministic across calls", () => {
    expect(fnv1a64("hello")).toBe(fnv1a64("hello"));
    expect(fnv1a64("hello")).not.toBe(fnv1a64("Hello"));
  });

  it("artifactKey composes inputs into a stable hash", () => {
    const k1 = artifactKey({
      personaId: "p1",
      promptHash: "abc",
      worldStateHash: "ws1",
    });
    const k2 = artifactKey({
      personaId: "p1",
      promptHash: "abc",
      worldStateHash: "ws1",
    });
    expect(k1).toBe(k2);
    const k3 = artifactKey({
      personaId: "p1",
      promptHash: "abc",
      worldStateHash: "ws2",
    });
    expect(k3).not.toBe(k1);
  });

  it("ArtifactManifest add/get/has roundtrip", () => {
    const m = new ArtifactManifest("ws-aaa");
    const ref = {
      hash: "h1",
      kind: "text" as const,
      url: "blob://1",
      generatedAt: "2026-05-08T00:00:00Z",
    };
    m.add(ref);
    expect(m.has("h1")).toBe(true);
    expect(m.get("h1")).toEqual(ref);
    const j = m.toJson();
    expect(j.artifacts).toHaveLength(1);
    expect(j.worldStateHash).toBe("ws-aaa");
    const m2 = ArtifactManifest.fromJson(j);
    expect(m2.get("h1")).toEqual(ref);
  });
});

describe("augengine / persistence — snapshot", () => {
  it("captures and restores entity state", () => {
    const scene = buildScene();
    const snap = captureSnapshot(scene, { rngSeed: 42 });
    expect(snap.rngSeed).toBe(42);
    expect(snap.spec.entities[0]?.id).toBe("cube");

    const restored = restoreSnapshot(snap);
    expect(restored.size()).toBe(1);
    expect(restored.get("cube" as EntityId)?.name).toBe("Hero Cube");
  });

  it("canonicalize is order-stable across property insertion order", () => {
    const a = {
      schemaVersion: 1,
      capturedAt: "ignored",
      rngSeed: 0,
      spec: { schemaVersion: 1, source: { kind: "authored", path: "x" }, entities: [] },
    };
    const b = {
      capturedAt: "different",
      schemaVersion: 1,
      spec: { entities: [], source: { path: "x", kind: "authored" }, schemaVersion: 1 },
      rngSeed: 0,
    };
    expect(canonicalize(a as never)).toBe(canonicalize(b as never));
  });

  it("worldStateHash returns deterministic hex", async () => {
    const scene = buildScene();
    const snap = captureSnapshot(scene);
    const h1 = await worldStateHash(snap);
    const h2 = await worldStateHash(snap);
    expect(h1).toMatch(/^[0-9a-f]{64}$/);
    expect(h1).toBe(h2);
  });

  it("diffEntities finds added + removed", () => {
    const sceneA = buildScene();
    const snapA = captureSnapshot(sceneA);
    const sceneB = buildScene();
    sceneB.add({
      id: "cube2" as EntityId,
      name: "Sidekick",
      components: {},
    });
    const snapB = captureSnapshot(sceneB);
    const d = diffEntities(snapA, snapB);
    expect(d.added).toEqual(["cube2"]);
    expect(d.removed).toEqual([]);
  });
});

describe("augengine / persistence — save/load", () => {
  it("round-trips a Scene through save → bytes → load", async () => {
    const scene = buildScene();
    const m = new ArtifactManifest();
    m.add({
      hash: "art-1",
      kind: "text",
      url: "blob://art-1",
      generatedAt: "2026-05-08T00:00:00Z",
    });
    const file = await save(scene, { rngSeed: 7, manifest: m });
    expect(file.version).toBe(1);
    expect(file.worldStateHash).toMatch(/^[0-9a-f]{64}$/);

    const bytes = serialize(file);
    expect(bytes).toBeInstanceOf(Uint8Array);
    expect(bytes.byteLength).toBeGreaterThan(0);

    const restored = await load(deserialize(bytes));
    expect(restored.scene.size()).toBe(1);
    expect(restored.rngSeed).toBe(7);
    expect(restored.manifest.size()).toBe(1);
    expect(restored.manifest.get("art-1")?.url).toBe("blob://art-1");
  });

  it("rejects tampered save files", async () => {
    const scene = buildScene();
    const file = await save(scene);
    file.worldStateHash = "0".repeat(64); // tamper
    await expect(load(file)).rejects.toThrow(/hash mismatch/);
  });
});
