import { describe, it, expect } from "vitest";
import {
  buildInstanceBatches,
  INSTANCE_BATCH_THRESHOLD,
  type InstanceCandidate,
} from "./instancing.js";
import type { EntityId, Mat4 } from "../../scene/types.js";

function id(n: number): EntityId {
  return `e${n}` as EntityId;
}

function identityMat(): Mat4 {
  const m = new Float32Array(16);
  m[0] = 1; m[5] = 1; m[10] = 1; m[15] = 1;
  return m;
}

function mk(n: number, meshKey = "m", materialKey = "mat"): InstanceCandidate {
  return { entityId: id(n), meshKey, materialKey, worldMatrix: identityMat() };
}

describe("instancing / threshold gate", () => {
  it("emits a batch when the group meets the threshold", () => {
    const cands: InstanceCandidate[] = [];
    for (let i = 0; i < INSTANCE_BATCH_THRESHOLD; i++) cands.push(mk(i));
    const r = buildInstanceBatches(cands);
    expect(r.batches).toHaveLength(1);
    expect(r.batches[0]!.entities).toHaveLength(INSTANCE_BATCH_THRESHOLD);
    expect(r.singletons).toHaveLength(0);
  });

  it("falls back to singletons when below threshold", () => {
    const cands = [mk(0), mk(1), mk(2)]; // 3 < 4
    const r = buildInstanceBatches(cands);
    expect(r.batches).toHaveLength(0);
    expect(r.singletons).toHaveLength(3);
  });

  it("splits across (mesh, material) groupings", () => {
    const cands: InstanceCandidate[] = [
      mk(0, "tree", "leaves"),
      mk(1, "tree", "leaves"),
      mk(2, "tree", "leaves"),
      mk(3, "tree", "leaves"),
      mk(4, "rock", "stone"),
      mk(5, "rock", "stone"),
    ];
    const r = buildInstanceBatches(cands);
    expect(r.batches).toHaveLength(1);
    expect(r.batches[0]!.meshKey).toBe("tree");
    expect(r.singletons).toHaveLength(2);
  });

  it("packs transforms as a 16-float column per entity", () => {
    const cands: InstanceCandidate[] = [mk(0), mk(1), mk(2), mk(3)];
    const r = buildInstanceBatches(cands);
    expect(r.batches[0]!.transforms).toHaveLength(64);
    expect(r.batches[0]!.transforms[0]).toBe(1); // identity m[0]
    expect(r.batches[0]!.transforms[15]).toBe(1); // identity m[15]
  });

  it("preserves input order within a batch (deterministic replays)", () => {
    const cands: InstanceCandidate[] = [mk(5), mk(2), mk(9), mk(1), mk(7)];
    const r = buildInstanceBatches(cands);
    expect(r.batches[0]!.entities).toEqual([id(5), id(2), id(9), id(1), id(7)]);
  });
});
