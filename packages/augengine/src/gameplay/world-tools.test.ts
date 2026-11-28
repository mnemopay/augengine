import { afterEach, describe, expect, it } from "vitest";
import { Scene } from "../scene/scene.js";
import { Tag, type EntityId, type Vec3 } from "../scene/types.js";
import { PhysicsSystem } from "../physics/system.js";
import { raycast } from "../physics/raycast.js";
import { KinematicCharacterController } from "../physics/character-controller.js";
import { parsePrefab, PrefabError, PrefabLibrary } from "./prefab.js";
import { CheckpointTracker, insideTrigger, Spawner, TriggerSystem } from "./triggers.js";
import { generateTerrain, grassTuftMesh, rockMesh, scatter } from "./terrain-gen.js";

const emptyScene = () => new Scene({ schemaVersion: 1, source: { kind: "authored", path: "t" }, entities: [] });

const GATE = {
  name: "gate",
  entities: [
    { id: "base", transform: { position: [0, 0, 0] }, components: { mesh: { geometry: { kind: "primitive", shape: "box" }, material: { kind: "unlit", color: [1, 1, 1] }, visible: true } } },
    { id: "pillarL", parent: "base", transform: { position: [-2, 1.5, 0], scale: [0.5, 3, 0.5] }, components: { collider: { shape: "box", extents: [0.25, 1.5, 0.25] } } },
    { id: "pillarR", parent: "base", transform: { position: [2, 1.5, 0], scale: [0.5, 3, 0.5] }, components: { collider: { shape: "box", extents: [0.25, 1.5, 0.25] } } },
    { id: "sign", parent: "pillarL", transform: { position: [0, 2, 0] } },
    { id: "zone", parent: "base", transform: { position: [0, 1, 1] }, gameplay: { trigger: { shape: "box", halfExtents: [2, 1, 1], event: "enter-arena", once: true } } },
  ],
};

describe("prefabs", () => {
  it("parses a valid prefab and reports every problem in an invalid one with its path", () => {
    expect(parsePrefab(GATE).name).toBe("gate");
    let err: PrefabError | null = null;
    try {
      parsePrefab({
        name: "",
        entities: [
          { id: "a", transform: { scale: [1, 0, 1], rotation: [0, 0, 0, 2] }, components: { meshh: {} } },
          { id: "a", parent: "nope" },
          { id: "b", gameplay: { trigger: { shape: "cone", event: "" }, spawner: { kind: "x", interval: -1 } } },
        ],
      });
    } catch (e) { err = e as PrefabError; }
    expect(err).toBeInstanceOf(PrefabError);
    const msg = err!.problems.join("\n");
    for (const needle of [
      "name: expected a non-empty string",
      "entities[0].transform.scale",
      "entities[0].transform.rotation",
      "entities[0].components.meshh: unknown component",
      'entities[1].id: duplicate "a"',
      'entities[1].parent: unknown local id "nope"',
      "entities[2].gameplay.trigger.shape",
      "entities[2].gameplay.trigger.event",
      "entities[2].gameplay.spawner.interval",
    ]) expect(msg).toContain(needle);
    expect(() => parsePrefab({ name: "c", entities: [{ id: "x", parent: "y" }, { id: "y", parent: "x" }] })).toThrow(/parent cycle/);
    expect(() => parsePrefab("nope")).toThrow(PrefabError);
  });

  it("instantiates at a placement: hierarchy kept for visuals, physics entities flattened to world roots", () => {
    const lib = new PrefabLibrary();
    lib.register(GATE);
    const scene = emptyScene();
    const inst = lib.instantiate(scene, "gate", { position: [10, 0, 5], yaw: 90 }, "g1");
    expect(inst.entityIds).toHaveLength(6);
    const pillarL = scene.get("g1/pillarL" as EntityId)!.components[Tag.Transform]!;
    expect(pillarL.parent).toBeNull(); // physics needs roots
    // yaw 90: local -X maps to world +Z → (10, 1.5, 5 + 2)
    expect(pillarL.position[0]).toBeCloseTo(10, 5);
    expect(pillarL.position[1]).toBeCloseTo(1.5, 5);
    expect(pillarL.position[2]).toBeCloseTo(7, 5);
    expect(pillarL.scale[1]).toBeCloseTo(3, 5);
    const sign = scene.get("g1/sign" as EntityId)!.components[Tag.Transform]!;
    expect(sign.parent).toBe("g1/pillarL"); // non-physical child keeps its parent
    // The sign inherits the pillar's 3x Y scale: 1.5 + 2 * 3.
    expect(scene.worldTransforms().worldMatrices[scene.worldTransforms().indexById.get("g1/sign" as EntityId)! * 16 + 13]).toBeCloseTo(7.5, 4);
    expect(scene.get("g1/base" as EntityId)!.components[Tag.Transform]!.parent).toBe("g1");
    expect(inst.triggers).toHaveLength(1);
    expect(inst.triggers[0]!.center[0]).toBeCloseTo(11, 5); // local (0,1,1) rotated 90° → +X
    expect(inst.triggers[0]!.center[2]).toBeCloseTo(5, 5);
  });

  it("nests prefabs, applies per-instance overrides by path, and rejects prefab cycles", () => {
    const lib = new PrefabLibrary();
    lib.register(GATE);
    lib.register({
      name: "fort",
      entities: [
        { id: "wall", components: { collider: { shape: "box", extents: [5, 1, 0.3] } } },
        { id: "entrance", transform: { position: [0, 0, 3] }, prefab: "gate" },
      ],
    });
    const scene = emptyScene();
    const inst = lib.instantiate(scene, "fort", { overrides: { "entrance/base": { mesh: { visible: false } } } }, "f");
    expect(inst.ids.get("entrance/pillarR")).toBe("f/entrance/pillarR");
    expect(scene.get("f/entrance/pillarR" as EntityId)!.components[Tag.Transform]!.position[2]).toBeCloseTo(3, 5);
    const mesh = scene.get("f/entrance/base" as EntityId)!.components[Tag.Mesh]!;
    expect(mesh.visible).toBe(false);
    expect(mesh.geometry).toEqual({ kind: "primitive", shape: "box" }); // shallow patch keeps the rest
    // Overrides don't leak into the registered spec.
    expect((GATE.entities[0]!.components as { mesh: { visible: boolean } }).mesh.visible).toBe(true);
    lib.register({ name: "loopA", entities: [{ id: "x", prefab: "loopB" }] });
    lib.register({ name: "loopB", entities: [{ id: "y", prefab: "loopA" }] });
    expect(() => lib.instantiate(emptyScene(), "loopA")).toThrow(/prefab cycle: loopA → loopB → loopA/);
    expect(() => lib.instantiate(emptyScene(), "missing")).toThrow(/not registered/);
  });

  it("instantiated colliders are valid physics roots", async () => {
    const lib = new PrefabLibrary();
    lib.register(GATE);
    const scene = emptyScene();
    lib.instantiate(scene, "gate", { position: [0, 0, 0] }, "g");
    const p = new PhysicsSystem();
    await p.init();
    try {
      p.stepFixed(scene, 1 / 60);
      expect(p.overlapShape({ kind: "sphere", radius: 0.1 }, [-2, 1.5, 0])).toEqual(["g/pillarL"]);
    } finally { p.dispose(); }
  });
});

describe("triggers, spawners, checkpoints", () => {
  const box = { id: "t1", shape: "box" as const, center: [0, 1, 0] as Vec3, halfExtents: [2, 1, 2] as Vec3, event: "zone" };

  it("point-in-volume for boxes and spheres", () => {
    expect(insideTrigger(box, [1.9, 0.1, -1.9])).toBe(true);
    expect(insideTrigger(box, [2.1, 1, 0])).toBe(false);
    expect(insideTrigger({ shape: "sphere", center: [0, 0, 0], radius: 1 }, [0.6, 0.6, 0.5])).toBe(true);
    expect(insideTrigger({ shape: "sphere", center: [0, 0, 0], radius: 1 }, [0.7, 0.7, 0.5])).toBe(false);
  });

  it("reports enter and exit per subject; once-triggers fire one enter then go quiet", () => {
    const ts = new TriggerSystem();
    ts.add(box);
    ts.add({ ...box, id: "t2", event: "once", once: true });
    expect(ts.update([{ id: "p", position: [5, 1, 0] }])).toEqual([]);
    const e1 = ts.update([{ id: "p", position: [0, 1, 0] }, { id: "q", position: [1, 1, 1] }]);
    expect(e1.map((e) => `${e.trigger}:${e.kind}:${e.subject}`)).toEqual(["t1:enter:p", "t1:enter:q", "t2:enter:p"]);
    expect(ts.occupants("t1").sort()).toEqual(["p", "q"]);
    const e2 = ts.update([{ id: "q", position: [1, 1, 1] }]);
    expect(e2.map((e) => `${e.trigger}:${e.kind}:${e.subject}`)).toEqual(["t1:exit:p"]);
    expect(ts.update([{ id: "p", position: [0, 1, 0] }]).map((e) => e.trigger + ":" + e.kind)).toEqual(["t1:enter", "t1:exit"]);
  });

  it("spawns on its interval up to maxAlive and count, deterministically, after activation", () => {
    const spec = { id: "s", kind: "grunt", worldPoints: [[0, 0, 0], [5, 0, 0], [10, 0, 0]] as Vec3[], interval: 1, maxAlive: 2, count: 5, activateOn: "go", seed: 42 };
    const a = new Spawner(spec), b = new Spawner(spec);
    expect(a.update(5, 0)).toEqual([]); // inactive
    a.activate(); b.activate();
    const r1 = a.update(0.01, 0);
    expect(r1).toHaveLength(1); // first spawn on activation
    expect(a.update(0.5, 1)).toEqual([]);
    expect(a.update(0.5, 1)).toHaveLength(1); // interval elapsed
    expect(a.update(3, 2)).toEqual([]); // capped at maxAlive
    expect(a.update(0.01, 0)).toHaveLength(1); // one interval of credit, not a burst
    expect(a.update(10, 0)).toHaveLength(2); // count = 5 reached
    expect(a.exhausted).toBe(true);
    expect(a.update(10, 0)).toEqual([]);
    // Same seed → same point order; each bag of 3 uses every point once.
    const pa: string[] = [], pb: string[] = [];
    const a2 = new Spawner({ ...spec, activateOn: undefined, maxAlive: 99, count: 6 }), b2 = new Spawner({ ...spec, activateOn: undefined, maxAlive: 99, count: 6 });
    for (let i = 0; i < 6; i++) { pa.push(...a2.update(1, 0).map((r) => String(r.position[0]))); pb.push(...b2.update(1, 0).map((r) => String(r.position[0]))); }
    expect(pa).toEqual(pb);
    expect(pa.slice(0, 3).sort()).toEqual(["0", "10", "5"]);
    void b;
  });

  it("checkpoints advance only forward and set the respawn point", () => {
    const cp = new CheckpointTracker([
      { id: "c2", order: 2, position: [20, 0, 0], radius: 2 },
      { id: "c1", order: 1, position: [10, 0, 0], radius: 2 },
    ], [0, 0, 0]);
    expect(cp.respawnPoint()).toEqual([0, 0, 0]);
    expect(cp.update([20.5, 0, 0])).toEqual(["c2"]);
    expect(cp.update([10, 0, 0])).toEqual([]); // lower order: ignored
    expect(cp.respawnPoint()).toEqual([20, 0, 0]);
  });
});

describe("procedural terrain + scatter", () => {
  const opts = { seed: 9, size: 40, cells: 32, amplitude: 3, flatRadius: 6, flatBlend: 6, rim: 2 };
  let physics: PhysicsSystem | null = null;
  afterEach(() => { physics?.dispose(); physics = null; });

  it("is deterministic per seed and flat inside the flat radius", () => {
    const a = generateTerrain(opts), b = generateTerrain(opts), c = generateTerrain({ ...opts, seed: 10 });
    expect(a.heights).toEqual(b.heights);
    expect(a.heights).not.toEqual(c.heights);
    // Exactly flat where every corner of the cell is inside the radius (radius − cell diagonal).
    for (const [x, z] of [[0, 0], [3, -2], [-2.5, 3]] as const) expect(a.heightAt(x, z)).toBe(0);
    expect(a.maxHeight).toBeGreaterThan(1);
    const m = a.mesh();
    expect(m.vertexCount).toBe(33 * 33);
    expect(m.indexCount).toBe(32 * 32 * 6);
    expect(m.colors!.length).toBe(33 * 33 * 4);
  });

  it("Rapier's heightfield, the engine raycast and heightAt agree", async () => {
    const t = generateTerrain(opts);
    const scene = emptyScene();
    scene.add({ id: "terrain" as EntityId, name: "terrain", components: { [Tag.Transform]: { position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1], parent: null }, [Tag.Collider]: t.collider } });
    physics = new PhysicsSystem();
    await physics.init();
    physics.stepFixed(scene, 1 / 60);
    const world = (physics as unknown as { world: { castRay(ray: unknown, max: number, solid: boolean): { timeOfImpact: number } | null } }).world;
    const R = await import("@dimforge/rapier3d-deterministic-compat");
    let worst = 0, worstRay = 0;
    for (let k = 0; k < 200; k++) {
      const x = -19.5 + ((k * 7.31) % 39), z = -19.5 + ((k * 3.97) % 39);
      const hit = world.castRay(new R.Ray({ x, y: 100, z }, { x: 0, y: -1, z: 0 }), 200, true);
      worst = Math.max(worst, Math.abs(100 - hit!.timeOfImpact - t.heightAt(x, z)));
      const rh = raycast(scene, [x, 100, z], [0, -1, 0], { maxDistance: 200 });
      worstRay = Math.max(worstRay, Math.abs(rh!.point[1] - t.heightAt(x, z)));
    }
    expect(worst).toBeLessThan(2e-3);
    expect(worstRay).toBeLessThan(2e-3);
  });

  it("the character controller walks up the terrain instead of through it", () => {
    const t = generateTerrain({ ...opts, flatRadius: 0, rim: 0, amplitude: 1.5 });
    const scene = emptyScene();
    scene.add({ id: "terrain" as EntityId, name: "terrain", components: { [Tag.Transform]: { position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1], parent: null }, [Tag.Collider]: t.collider } });
    scene.add({ id: "cap" as EntityId, name: "cap", components: { [Tag.Transform]: { position: [-15, t.heightAt(-15, 0) + 1, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1], parent: null }, [Tag.Collider]: { shape: "capsule", radius: 0.3, halfHeight: 0.5 } } });
    const kcc = new KinematicCharacterController(scene, { entityId: "cap" as EntityId, moveSpeed: 3, snapDistance: 0.5, maxGroundSlope: 0.5 });
    let worst = 0;
    for (let i = 0; i < 300; i++) {
      kcc.update(1 / 60, [1, 0, 0]);
      const p = scene.get("cap" as EntityId)!.components[Tag.Transform]!.position;
      if (i > 30) worst = Math.max(worst, Math.abs(p[1] - 0.8 - t.heightAt(p[0], p[2])));
    }
    expect(scene.get("cap" as EntityId)!.components[Tag.Transform]!.position[0]).toBeGreaterThan(-1.5); // 5 s at 3 m/s from x = -15
    expect(worst).toBeLessThan(0.12); // feet stay on the surface
  });

  it("scatter is seeded, respects spacing, slope, height and avoid zones", () => {
    const t = generateTerrain(opts);
    const o = { seed: 3, count: 150, minSpacing: 1.2, minNormalY: 0.9, avoid: [{ x: 0, z: 0, radius: 5 }], heightRange: [-1, 2.5] as [number, number], scaleRange: [0.8, 1.2] as [number, number] };
    const a = scatter(t, o), b = scatter(t, o);
    expect(a).toEqual(b);
    expect(a.length).toBeGreaterThan(80);
    for (let i = 0; i < a.length; i++) {
      const p = a[i]!.position;
      expect(t.normalAt(p[0], p[2])[1]).toBeGreaterThanOrEqual(0.9);
      expect(Math.hypot(p[0], p[2])).toBeGreaterThanOrEqual(5);
      expect(p[1]).toBeCloseTo(t.heightAt(p[0], p[2]), 5);
      expect(a[i]!.scale).toBeGreaterThanOrEqual(0.8);
      for (let j = 0; j < i; j++) expect(Math.hypot(p[0] - a[j]!.position[0], p[2] - a[j]!.position[2])).toBeGreaterThanOrEqual(1.2);
    }
    const tilted = scatter(t, { seed: 4, count: 30, minNormalY: 0, alignToNormal: 1 });
    for (const pl of tilted) expect(Math.hypot(...pl.rotation)).toBeCloseTo(1, 6);
  });

  it("scatter meshes are well-formed", () => {
    for (const m of [grassTuftMesh(1), rockMesh(2)]) {
      expect(m.vertices.length).toBe(m.vertexCount * 8);
      expect(m.colors!.length).toBe(m.vertexCount * 4);
      for (const i of m.indices) expect(i).toBeLessThan(m.vertexCount);
      for (const v of m.vertices) expect(Number.isFinite(v)).toBe(true);
    }
  });
});
