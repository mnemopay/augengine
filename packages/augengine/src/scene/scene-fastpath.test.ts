import { describe, expect, it } from "vitest";
import { Scene } from "./scene.js";
import { computeWorldTransforms } from "./transform.js";
import { Tag, type Entity, type EntityId, type Vec3 } from "./types.js";

const ent = (id: string, position: Vec3, parent: string | null = null): Entity => ({
  id: id as EntityId,
  name: id,
  components: { [Tag.Transform]: { position, rotation: [0, 0, 0, 1], scale: [1, 1, 1], parent: parent as EntityId | null } },
});

/** World matrices from a from-scratch pass, for comparison with the scene's cache. */
function fresh(scene: Scene): Map<string, number[]> {
  const c = computeWorldTransforms([...scene.entities.values()]);
  return new Map([...c.indexById].map(([id, i]) => [id, Array.from(c.worldMatrices.subarray(i * 16, i * 16 + 16))]));
}
function cached(scene: Scene): Map<string, number[]> {
  const c = scene.worldTransforms();
  return new Map([...c.indexById].map(([id, i]) => [id, Array.from(c.worldMatrices.subarray(i * 16, i * 16 + 16))]));
}

describe("Scene.setLocalTransform fast path for childless roots", () => {
  it("patches a childless root in place and still matches a full recompute", () => {
    const s = new Scene({ schemaVersion: 1, source: { kind: "authored", path: "t" }, entities: [ent("leaf", [0, 0, 0]), ent("root", [1, 0, 0]), ent("child", [0, 1, 0], "root")] });
    const before = s.worldTransforms();
    s.setLocalTransform("leaf" as EntityId, [5, 6, 7], [0, Math.SQRT1_2, 0, Math.SQRT1_2], [2, 2, 2]);
    expect(s.worldTransforms()).toBe(before); // same cache object, no full pass needed
    expect(cached(s)).toEqual(fresh(s));
  });

  it("moving a root that has children, or a child, still recomputes the subtree", () => {
    const s = new Scene({ schemaVersion: 1, source: { kind: "authored", path: "t" }, entities: [ent("root", [1, 0, 0]), ent("child", [0, 1, 0], "root")] });
    s.worldTransforms();
    s.setLocalTransform("root" as EntityId, [10, 0, 0], [0, 0, 0, 1], [1, 1, 1]);
    const c = s.worldTransforms();
    expect(c.worldMatrices[c.indexById.get("child" as EntityId)! * 16 + 12]).toBe(10);
    s.setLocalTransform("child" as EntityId, [0, 3, 0], [0, 0, 0, 1], [1, 1, 1]);
    expect(cached(s)).toEqual(fresh(s));
  });

  it("an entity that gains a child loses the fast path", () => {
    const s = new Scene({ schemaVersion: 1, source: { kind: "authored", path: "t" }, entities: [ent("a", [0, 0, 0])] });
    s.worldTransforms();
    s.add(ent("b", [0, 1, 0], "a"));
    s.worldTransforms();
    s.setLocalTransform("a" as EntityId, [4, 0, 0], [0, 0, 0, 1], [1, 1, 1]);
    const c = s.worldTransforms();
    expect(c.worldMatrices[c.indexById.get("b" as EntityId)! * 16 + 12]).toBe(4);
  });
});

describe("Scene world-transform refresh without re-sorting", () => {
  it("moving joints deep in a hierarchy matches a from-scratch pass", () => {
    const ents = [ent("root", [0, 0, 0]), ent("a", [0, 1, 0], "root"), ent("b", [0, 1, 0], "a"), ent("c", [1, 0, 0], "b"), ent("loose", [3, 3, 3])];
    const s = new Scene({ schemaVersion: 1, source: { kind: "authored", path: "t" }, entities: ents });
    s.worldTransforms();
    for (let k = 0; k < 5; k++) {
      s.setLocalTransform("a" as EntityId, [0, 1 + k, 0], [Math.sin(k / 2) * 0, Math.sin(k / 4), 0, Math.cos(k / 4)], [1, 1, 1]);
      s.setLocalTransform("c" as EntityId, [k, 0, 0], [0, 0, 0, 1], [1, 2, 1]);
      expect(cached(s)).toEqual(fresh(s));
    }
  });

  it("falls back to a full pass when a parent was changed behind its back", () => {
    const s = new Scene({ schemaVersion: 1, source: { kind: "authored", path: "t" }, entities: [ent("p1", [10, 0, 0]), ent("p2", [-10, 0, 0]), ent("kid", [0, 1, 0], "p1")] });
    s.worldTransforms();
    const kid = s.get("kid" as EntityId)!;
    kid.components[Tag.Transform] = { ...kid.components[Tag.Transform]!, parent: "p2" as EntityId };
    s.setLocalTransform("kid" as EntityId, [0, 2, 0], [0, 0, 0, 1], [1, 1, 1]);
    const c = s.worldTransforms();
    expect(c.worldMatrices[c.indexById.get("kid" as EntityId)! * 16 + 12]).toBe(-10);
    expect(cached(s)).toEqual(fresh(s));
  });
});
