/**
 * Smoke tests — Phase 1 surface checks. Pure data + math; no GPU/audio APIs
 * exercised here so vitest in Node passes. Browser-side correctness of the
 * runtimes is validated by the smoke-cube example.
 */

import { describe, it, expect } from "vitest";
import {
  Scene,
  Tag,
  v3Add,
  qIdentity,
  qMul,
  qFromAxisAngle,
  m4Create,
  m4FromTRS,
  type EntityId,
  type SceneSpec,
} from "./scene/index.js";
import {
  compileCharterYaml,
  parseYaml,
} from "./authoring/index.js";
import {
  AIScheduler,
  ObservationBus,
  ActionDispatcher,
  ensurePersona,
  renderPersonaSystem,
} from "./ai/index.js";
import {
  buildRetargetConfig,
  retargetFrame,
  makeIdentityFrame,
  type Skeleton,
  solveTwoBoneIK,
} from "./characters/index.js";
import { estimateVisemes } from "./audio/index.js";

describe("augengine / scene", () => {
  it("loads + computes world transforms", () => {
    const spec: SceneSpec = {
      schemaVersion: 1,
      source: { kind: "authored", path: "test" },
      entities: [
        {
          id: "root" as EntityId,
          name: "root",
          components: {
            [Tag.Transform]: {
              position: [1, 0, 0],
              rotation: [0, 0, 0, 1],
              scale: [1, 1, 1],
              parent: null,
            },
          },
        },
        {
          id: "child" as EntityId,
          name: "child",
          components: {
            [Tag.Transform]: {
              position: [0, 1, 0],
              rotation: [0, 0, 0, 1],
              scale: [1, 1, 1],
              parent: "root" as EntityId,
            },
          },
        },
      ],
    };
    const scene = new Scene(spec);
    const wt = scene.worldTransforms();
    const childIdx = wt.indexById.get("child" as EntityId)!;
    const childWorld = wt.worldMatrices.subarray(childIdx * 16, childIdx * 16 + 16);
    // World position should be (1, 1, 0) — root + child offset.
    expect(childWorld[12]).toBeCloseTo(1);
    expect(childWorld[13]).toBeCloseTo(1);
    expect(childWorld[14]).toBeCloseTo(0);
  });
});

describe("augengine / math", () => {
  it("vec3 add", () => {
    const out: [number, number, number] = [0, 0, 0];
    v3Add(out, [1, 2, 3], [4, 5, 6]);
    expect(out).toEqual([5, 7, 9]);
  });
  it("quaternion identity composes correctly", () => {
    const id = qIdentity();
    const r: [number, number, number, number] = [0, 0, 0, 1];
    qFromAxisAngle(r, [0, 1, 0], Math.PI / 2);
    const out: [number, number, number, number] = [0, 0, 0, 1];
    qMul(out, id, r);
    expect(out[3]).toBeCloseTo(r[3]);
  });
  it("mat4 from TRS produces correct translation", () => {
    const m = m4Create();
    m4FromTRS(m, [3, 4, 5], [0, 0, 0, 1], [1, 1, 1]);
    expect(m[12]).toBe(3);
    expect(m[13]).toBe(4);
    expect(m[14]).toBe(5);
  });
});

describe("augengine / authoring (charter YAML)", () => {
  it("parses + compiles a minimal charter", () => {
    const yaml = `name: smoke
description: smoke test
entities:
  - id: cube-1
    name: Hero
    position: [0, 0, 0]
    mesh:
      kind: primitive
      shape: box
      color: [1, 0.5, 0.2]
defaultCamera: camera-main
`;
    const doc = parseYaml(yaml) as { name: string };
    expect(doc.name).toBe("smoke");
    const spec = compileCharterYaml(yaml);
    expect(spec.entities[0]?.id).toBe("cube-1");
    expect(spec.defaultCamera).toBe("camera-main");
  });
});

describe("augengine / ai (scheduler + bus + dispatcher + persona)", () => {
  it("scheduler returns fallback when provider missing", async () => {
    const s = new AIScheduler();
    const r = await s.dispatch({
      task: "dialogue",
      payload: {},
      timeoutMs: 100,
      priority: 2,
      fallback: () => "fallback-hit",
    });
    expect(r).toBe("fallback-hit");
  });

  it("scheduler returns provider result when registered", async () => {
    const s = new AIScheduler();
    s.register("dialogue", async () => "real-result");
    const r = await s.dispatch({
      task: "dialogue",
      payload: {},
      timeoutMs: 1000,
      priority: 2,
      fallback: () => "fb",
    });
    expect(r).toBe("real-result");
  });

  it("scheduler respects deadline + falls back on timeout", async () => {
    const s = new AIScheduler();
    s.register("dialogue", () => new Promise((r) => setTimeout(() => r("late"), 200)));
    const r = await s.dispatch({
      task: "dialogue",
      payload: {},
      timeoutMs: 30,
      priority: 1,
      fallback: () => "deadline-fallback",
    });
    expect(r).toBe("deadline-fallback");
  });

  it("observation bus delivers to channel subscribers", () => {
    const bus = new ObservationBus();
    const got: unknown[] = [];
    const unsub = bus.subscribe("Visual", null, (e) => got.push(e.payload));
    bus.publish({
      channel: "Visual",
      ts: new Date().toISOString(),
      payload: { sighted: "actor-1" },
      source: null,
    });
    expect(got).toHaveLength(1);
    unsub();
  });

  it("action dispatcher validates + applies effects atomically", () => {
    const spec: SceneSpec = {
      schemaVersion: 1,
      source: { kind: "authored", path: "test" },
      entities: [
        {
          id: "actor" as EntityId,
          name: "actor",
          components: {},
        },
      ],
    };
    const scene = new Scene(spec);
    const d = new ActionDispatcher({ scene });
    d.registerVerb("Greet");
    let applied = false;
    d.registerEffect("logHello", () => {
      applied = true;
    });
    const result = d.dispatch({
      verb: "Greet",
      actor: "actor" as EntityId,
      params: {},
      preconditions: [],
      effects: ["logHello"],
      source: { kind: "script", scriptId: "test" },
    });
    expect(result.ok).toBe(true);
    expect(applied).toBe(true);
  });

  it("persona helpers compose system prompts", () => {
    const e = {
      id: "x" as EntityId,
      name: "x",
      components: {},
    };
    const p = ensurePersona(e, { traits: ["Curious", "Polite"], mood: 0.6 });
    const sys = renderPersonaSystem(p);
    expect(sys).toContain("Curious");
    expect(sys).toContain("Polite");
  });
});

describe("augengine / characters (retarget + IK)", () => {
  it("retargeting on identity skeletons preserves bind pose", () => {
    const skel: Skeleton = {
      joints: [
        { name: "Hips", parent: -1, bindPosition: [0, 1, 0], bindRotation: [0, 0, 0, 1] },
        { name: "Spine", parent: 0, bindPosition: [0, 0.3, 0], bindRotation: [0, 0, 0, 1] },
      ],
    };
    const config = buildRetargetConfig(skel, skel, { Hips: "Hips", Spine: "Spine" });
    const src = makeIdentityFrame(skel);
    const out = makeIdentityFrame(skel);
    retargetFrame(config, src, out);
    expect(out.localRotations[0]?.[3]).toBeCloseTo(1);
  });

  it("two-bone IK reaches a solvable target", () => {
    const r = solveTwoBoneIK({
      root: [0, 1, 0],
      mid: [0, 0.5, 0],
      tip: [0, 0, 0],
      target: [0.3, 0.3, 0],
      pole: [0, 1, 1],
    });
    expect(r.reached).toBe(true);
  });
});

describe("augengine / audio (lipsync formant detection)", () => {
  it("detects silence on empty FFT", () => {
    const fft = new Float32Array(512);
    const r = estimateVisemes(fft, 48000);
    expect(r.dominant).toBe("sil");
  });
  it("detects vowel-like content with energy", () => {
    const fft = new Float32Array(512);
    // Inject a peak around 800 Hz and 1200 Hz (the "aa" vowel).
    const binHz = 24000 / 512;
    const f1Bin = Math.round(800 / binHz);
    const f2Bin = Math.round(1200 / binHz);
    fft[f1Bin] = 1;
    fft[f2Bin] = 0.9;
    for (let i = 0; i < fft.length; i++) {
      if (i !== f1Bin && i !== f2Bin) fft[i] = 0.3;
    }
    const r = estimateVisemes(fft, 48000);
    expect(r.dominant).not.toBe("sil");
    expect(r.energy).toBeGreaterThan(0);
  });
});
