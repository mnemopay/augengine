import { describe, expect, it } from "vitest";
import { Scene } from "../scene/scene.js";
import { Tag, type Entity, type EntityId, type Vec3 } from "../scene/types.js";
import { applyIkComponents, IK_TWO_BONE, LOOK_AT, lookAtDelta, solveTwoBoneWorld, twoBoneChainProblem } from "./ik-runtime.js";
import { qAngle, qRotP, vLen, vNorm, vSub, type V } from "./rig-math.js";

const close = (a: readonly number[], b: readonly number[], d = 4) => a.forEach((x, i) => expect(x).toBeCloseTo(b[i]!, d));

describe("solveTwoBoneWorld", () => {
  const a: V = [0, 1, 0], b: V = [0, 0.5, 0.05], c: V = [0, 0, 0];

  it("puts the tip on a reachable goal and keeps bone lengths", () => {
    const goal: V = [0.3, 0.3, 0.2];
    const s = solveTwoBoneWorld(a, b, c, goal, [0, 0.5, 1]);
    expect(s.reached).toBe(true);
    close(s.tip, goal);
    const newB = [a[0], a[1], a[2]].map((x, i) => x + qRotP(s.rootDelta, vSub(b, a))[i]!) as V;
    expect(vLen(vSub(newB, a))).toBeCloseTo(vLen(vSub(b, a)), 6);
    const c1 = newB.map((x, i) => x + qRotP(s.rootDelta, vSub(c, b))[i]!) as V;
    const newC = newB.map((x, i) => x + qRotP(s.midDelta, vSub(c1, newB))[i]!) as V;
    close(newC, goal);
    expect(vLen(vSub(newC, newB))).toBeCloseTo(vLen(vSub(c, b)), 6);
  });

  it("bends toward the pole", () => {
    const s = solveTwoBoneWorld(a, b, c, [0, 0.2, 0], [0, 0.5, 1]);
    const newB = qRotP(s.rootDelta, vSub(b, a));
    expect(newB[2]).toBeGreaterThan(0.1);
    const s2 = solveTwoBoneWorld(a, b, c, [0, 0.2, 0], [0, 0.5, -1]);
    expect(qRotP(s2.rootDelta, vSub(b, a))[2]).toBeLessThan(-0.1);
  });

  it("stretches toward an unreachable goal", () => {
    const s = solveTwoBoneWorld(a, b, c, [0, -5, 0]);
    expect(s.reached).toBe(false);
    expect(vLen(vSub(s.tip, a))).toBeCloseTo(vLen(vSub(b, a)) + vLen(vSub(c, b)), 3);
    close(vNorm(vSub(s.tip, a)), [0, -1, 0], 3);
  });
});

describe("lookAtDelta", () => {
  it("aims an axis at a goal and clamps the angle", () => {
    const q = lookAtDelta([0, 0, 0], [0, 0, 1], [1, 0, 0], Math.PI);
    close(qRotP(q, [0, 0, 1]), [1, 0, 0]);
    const clamped = lookAtDelta([0, 0, 0], [0, 0, 1], [0, 0, -1], Math.PI / 4);
    expect(qAngle(clamped)).toBeCloseTo(Math.PI / 4, 5);
  });
});

function tx(position: Vec3, parent: EntityId | null) {
  return { [Tag.Transform]: { position, rotation: [0, 0, 0, 1] as [number, number, number, number], scale: [1, 1, 1] as Vec3, parent } };
}

function legScene(extra: Record<string, unknown>): Scene {
  const ents: Entity[] = [
    { id: "body" as EntityId, name: "body", components: { ...tx([0, 0, 0], null), ...extra } },
    { id: "hip" as EntityId, name: "hip", components: tx([0, 1, 0], "body" as EntityId) },
    { id: "knee" as EntityId, name: "knee", components: tx([0, -0.5, 0.05], "hip" as EntityId) },
    { id: "ankle" as EntityId, name: "ankle", components: tx([0, -0.5, -0.05], "knee" as EntityId) },
    { id: "head" as EntityId, name: "head", components: tx([0, 1.6, 0], "body" as EntityId) },
    { id: "lamp" as EntityId, name: "lamp", components: tx([2, 1.6, 0], null) },
  ];
  return new Scene({ entities: ents } as never);
}

const worldPos = (s: Scene, id: string): V => {
  const wt = s.worldTransforms();
  const i = wt.indexById.get(id as EntityId)!;
  return [wt.worldMatrices[i * 16 + 12]!, wt.worldMatrices[i * 16 + 13]!, wt.worldMatrices[i * 16 + 14]!];
};

describe("applyIkComponents", () => {
  it("plants a foot on raised ground (terrain sampler)", () => {
    const s = legScene({ [IK_TWO_BONE]: { root: "hip", mid: "knee", tip: "ankle", mode: "ground", groundOffset: 0.08, pole: [0, 0.5, 1] } });
    expect(applyIkComponents(s, { heightAt: () => 0.3 })).toBe(1);
    expect(worldPos(s, "ankle")[1]).toBeCloseTo(0.38, 4);
    expect(worldPos(s, "knee")[2]).toBeGreaterThan(0);
  });

  it("reaches a target entity and blends by weight", () => {
    const s = legScene({ [IK_TWO_BONE]: { root: "hip", mid: "knee", tip: "ankle", mode: "target", target: "lamp", weight: 1 } });
    s.setLocalTransform("lamp" as EntityId, [0.4, 0.4, 0.3], [0, 0, 0, 1], [1, 1, 1]);
    applyIkComponents(s);
    close(worldPos(s, "ankle"), [0.4, 0.4, 0.3], 3);
  });

  it("turns the head toward a target", () => {
    const s = legScene({ [LOOK_AT]: { bone: "head", target: "lamp", axis: [0, 0, 1], maxAngleDeg: 120 } });
    applyIkComponents(s);
    const wt = s.worldTransforms();
    const i = wt.indexById.get("head" as EntityId)!;
    const m = wt.worldMatrices;
    const fwd = vNorm([m[i * 16 + 8]!, m[i * 16 + 9]!, m[i * 16 + 10]!]);
    close(fwd, [1, 0, 0], 3);
  });

  it("refuses a chain whose tip isn't under the mid joint", () => {
    const s = legScene({ [IK_TWO_BONE]: { root: "hip", mid: "knee", tip: "lamp", mode: "ground" } });
    expect(twoBoneChainProblem(s, { root: "hip" as EntityId, mid: "knee" as EntityId, tip: "lamp" as EntityId })).toMatch(/not under/);
    expect(applyIkComponents(s)).toBe(0);
  });
});
