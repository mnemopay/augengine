/**
 * Phase 11 tests — collider intersections, ragdoll step, particle emit/step.
 */

import { describe, it, expect } from "vitest";
import {
  sphereSphere,
  sphereCapsule,
  capsuleCapsule,
  sphereMesh,
  sphereAabb,
  capsuleAabb,
  aabbForCollider,
  aabbOverlap,
  buildBvh,
  Ragdoll,
  ParticleBuffer,
  type AABB,
  type Sphere,
  type Capsule,
  type MeshCollider,
  type EmitterDef,
} from "./index.js";

describe("augengine / physics — sphere ↔ sphere", () => {
  it("detects overlap with correct depth", () => {
    const a: Sphere = { kind: "sphere", center: [0, 0, 0], radius: 1 };
    const b: Sphere = { kind: "sphere", center: [1.5, 0, 0], radius: 1 };
    const m = sphereSphere(a, b);
    expect(m).not.toBeNull();
    expect(m!.depth).toBeCloseTo(0.5, 4);
    expect(m!.normal[0]).toBeCloseTo(1, 4);
  });
  it("misses when separated", () => {
    const a: Sphere = { kind: "sphere", center: [0, 0, 0], radius: 1 };
    const b: Sphere = { kind: "sphere", center: [3, 0, 0], radius: 1 };
    expect(sphereSphere(a, b)).toBeNull();
  });
});

describe("augengine / physics — sphere ↔ capsule", () => {
  it("hits the capsule body midpoint", () => {
    const s: Sphere = { kind: "sphere", center: [0.5, 0.5, 0], radius: 0.4 };
    const c: Capsule = { kind: "capsule", p0: [0, 0, 0], p1: [0, 1, 0], radius: 0.3 };
    const m = sphereCapsule(s, c);
    expect(m).not.toBeNull();
    expect(m!.depth).toBeGreaterThan(0);
  });
  it("misses below capsule p0 hemisphere", () => {
    const s: Sphere = { kind: "sphere", center: [0, -3, 0], radius: 0.5 };
    const c: Capsule = { kind: "capsule", p0: [0, 0, 0], p1: [0, 1, 0], radius: 0.3 };
    expect(sphereCapsule(s, c)).toBeNull();
  });
});

describe("augengine / physics — capsule ↔ capsule", () => {
  it("detects parallel overlapping capsules", () => {
    const a: Capsule = { kind: "capsule", p0: [0, 0, 0], p1: [0, 2, 0], radius: 0.5 };
    const b: Capsule = { kind: "capsule", p0: [0.6, 0, 0], p1: [0.6, 2, 0], radius: 0.5 };
    const m = capsuleCapsule(a, b);
    expect(m).not.toBeNull();
  });
});

describe("augengine / physics — AABB", () => {
  it("aabbForCollider on a sphere", () => {
    const s: Sphere = { kind: "sphere", center: [0, 0, 0], radius: 1 };
    const aabb = aabbForCollider(s);
    expect(aabb.min).toEqual([-1, -1, -1]);
    expect(aabb.max).toEqual([1, 1, 1]);
  });
  it("aabbOverlap detects overlap", () => {
    expect(aabbOverlap(
      { min: [0, 0, 0], max: [1, 1, 1] },
      { min: [0.5, 0.5, 0.5], max: [2, 2, 2] },
    )).toBe(true);
    expect(aabbOverlap(
      { min: [0, 0, 0], max: [1, 1, 1] },
      { min: [2, 2, 2], max: [3, 3, 3] },
    )).toBe(false);
  });
});

describe("augengine / physics — sphere ↔ AABB", () => {
  const box: AABB = { min: [0, 0, 0], max: [2, 2, 2] };
  it("hits when sphere overlaps face", () => {
    const s: Sphere = { kind: "sphere", center: [-0.4, 1, 1], radius: 0.6 };
    const m = sphereAabb(s, box);
    expect(m).not.toBeNull();
    expect(m!.depth).toBeCloseTo(0.2, 4);
    expect(m!.normal[0]).toBeCloseTo(-1, 4);
  });
  it("hits corner with diagonal normal", () => {
    const s: Sphere = { kind: "sphere", center: [-0.3, -0.3, -0.3], radius: 0.7 };
    const m = sphereAabb(s, box);
    expect(m).not.toBeNull();
    // Normal should point away from corner (-x,-y,-z direction)
    expect(m!.normal[0]).toBeLessThan(0);
    expect(m!.normal[1]).toBeLessThan(0);
    expect(m!.normal[2]).toBeLessThan(0);
  });
  it("misses when separated", () => {
    const s: Sphere = { kind: "sphere", center: [5, 5, 5], radius: 0.5 };
    expect(sphereAabb(s, box)).toBeNull();
  });
  it("resolves with shallowest face when center inside", () => {
    const s: Sphere = { kind: "sphere", center: [1.9, 1, 1], radius: 0.3 };
    const m = sphereAabb(s, box);
    expect(m).not.toBeNull();
    expect(m!.normal[0]).toBeCloseTo(1, 4);
  });
});

describe("augengine / physics — capsule ↔ AABB", () => {
  it("walks a vertical capsule into a box face and reports contact", () => {
    const c: Capsule = {
      kind: "capsule",
      p0: [-0.3, 0.5, 1],
      p1: [-0.3, 1.5, 1],
      radius: 0.4,
    };
    const box: AABB = { min: [0, 0, 0], max: [2, 2, 2] };
    const m = capsuleAabb(c, box);
    expect(m).not.toBeNull();
    expect(m!.depth).toBeCloseTo(0.1, 3);
    expect(m!.normal[0]).toBeCloseTo(-1, 3);
  });
  it("misses when capsule is clear of the box", () => {
    const c: Capsule = {
      kind: "capsule",
      p0: [5, 0, 0],
      p1: [5, 1, 0],
      radius: 0.4,
    };
    const box: AABB = { min: [0, 0, 0], max: [1, 1, 1] };
    expect(capsuleAabb(c, box)).toBeNull();
  });
  it("hits the floor when capsule sits on the floor AABB", () => {
    // Floor: 36×36 thin slab. Capsule at typical player feet height.
    const c: Capsule = {
      kind: "capsule",
      p0: [0, 0.08, 0],
      p1: [0, 1.88, 0],
      radius: 0.4,
    };
    const floor: AABB = { min: [-18, -0.1, -18], max: [18, 0.1, 18] };
    const m = capsuleAabb(c, floor);
    expect(m).not.toBeNull();
    expect(m!.normal[1]).toBeGreaterThan(0); // pushes capsule up
  });
});

describe("augengine / physics — sphere ↔ mesh + BVH", () => {
  function makeQuadMesh(): MeshCollider {
    const vertices = new Float32Array([
      0, 0, 0,
      1, 0, 0,
      1, 1, 0,
      0, 1, 0,
    ]);
    const indices = new Uint32Array([0, 1, 2, 0, 2, 3]);
    return { kind: "mesh", vertices, indices };
  }

  it("hits the quad with a sphere centered above its midpoint", () => {
    const mesh = makeQuadMesh();
    const s: Sphere = { kind: "sphere", center: [0.5, 0.5, 0.4], radius: 0.5 };
    const m = sphereMesh(s, mesh);
    expect(m).not.toBeNull();
    expect(m!.depth).toBeGreaterThan(0);
  });

  it("misses when far above the quad", () => {
    const mesh = makeQuadMesh();
    const s: Sphere = { kind: "sphere", center: [0.5, 0.5, 5], radius: 0.5 };
    expect(sphereMesh(s, mesh)).toBeNull();
  });

  it("BVH-accelerated path returns the same hits", () => {
    const mesh = makeQuadMesh();
    mesh.bvh = buildBvh(mesh);
    const s: Sphere = { kind: "sphere", center: [0.5, 0.5, 0.4], radius: 0.5 };
    const m = sphereMesh(s, mesh);
    expect(m).not.toBeNull();
  });
});

describe("augengine / physics — ragdoll step", () => {
  it("falls under gravity", () => {
    const cap: Capsule = { kind: "capsule", p0: [0, 1, 0], p1: [0, 2, 0], radius: 0.2 };
    const ragdoll = new Ragdoll({
      bones: [{
        jointIndex: 0,
        capsule: cap,
        mass: 1,
        velocity: [0, 0, 0],
        prevMid: [0, 1.5, 0],
      }],
      joints: [],
    });
    ragdoll.step(0.1);
    // After 0.1s the midpoint should have dropped (gravity 9.81 m/s²).
    const yAfter = (cap.p0[1] + cap.p1[1]) / 2;
    expect(yAfter).toBeLessThan(1.5);
  });

  it("applyImpulse propagates to velocity", () => {
    const cap: Capsule = { kind: "capsule", p0: [0, 0, 0], p1: [0, 1, 0], radius: 0.1 };
    const r = new Ragdoll({
      bones: [{ jointIndex: 0, capsule: cap, mass: 2, velocity: [0, 0, 0], prevMid: [0, 0.5, 0] }],
      joints: [],
    });
    r.applyImpulse(0, [4, 0, 0]);
    expect(r.bones[0]!.velocity[0]).toBeCloseTo(2, 4);
  });
});

describe("augengine / physics — particle buffer", () => {
  it("emits + steps + ages out", () => {
    const pb = new ParticleBuffer(16, 12345);
    const emitter: EmitterDef = {
      origin: [0, 0, 0],
      direction: [0, 1, 0],
      rate: 100,
      initialSpeed: 5,
      spread: 0.2,
      lifetime: 0.5,
      size: 1,
      color: [1, 1, 1],
      alpha: 1,
      shape: { kind: "point" },
    };
    const emitted = pb.emit(emitter, 8);
    expect(emitted).toBe(8);
    let alive = pb.particles.filter((p) => p.age > 0).length;
    expect(alive).toBe(8);
    // Step 1 second — should age them all out.
    pb.step(1.0);
    alive = pb.particles.filter((p) => p.age > 0).length;
    expect(alive).toBe(0);
  });
});
