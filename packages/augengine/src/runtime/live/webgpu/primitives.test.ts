/**
 * Primitive-mesh smoke tests. Render verification belongs in a headless-WebGPU
 * harness; these tests pin the *data*: vertex count, index count, normal
 * direction, and triangle winding. Subtle off-by-ones in winding manifest as
 * "the floor doesn't render" — caught here, not at runtime.
 */

import { describe, it, expect } from "vitest";
import { cubeMesh, planeMesh, sphereMesh, cylinderMesh, primitiveMesh } from "./primitives.js";

function triNormalsCCW(vertices: Float32Array, indices: Uint32Array, stride = 8): boolean {
  // For each triangle, compute the geometric normal from CCW winding and
  // check it points the same direction as the per-vertex normal (within
  // hemispheric tolerance). Returns false if any triangle is wound the
  // wrong way relative to its declared normal.
  for (let t = 0; t < indices.length; t += 3) {
    const ia = indices[t]! * stride;
    const ib = indices[t + 1]! * stride;
    const ic = indices[t + 2]! * stride;
    const ax = vertices[ia]!, ay = vertices[ia + 1]!, az = vertices[ia + 2]!;
    const bx = vertices[ib]!, by = vertices[ib + 1]!, bz = vertices[ib + 2]!;
    const cx = vertices[ic]!, cy = vertices[ic + 1]!, cz = vertices[ic + 2]!;
    const ux = bx - ax, uy = by - ay, uz = bz - az;
    const vx = cx - ax, vy = cy - ay, vz = cz - az;
    const gx = uy * vz - uz * vy;
    const gy = uz * vx - ux * vz;
    const gz = ux * vy - uy * vx;
    const nx = vertices[ia + 3]!, ny = vertices[ia + 4]!, nz = vertices[ia + 5]!;
    if (gx * nx + gy * ny + gz * nz < 0) return false;
  }
  return true;
}

describe("primitives", () => {
  it("cubeMesh has 24 verts, 36 indices, consistent CCW winding", () => {
    const m = cubeMesh();
    expect(m.vertexCount).toBe(24);
    expect(m.indexCount).toBe(36);
    expect(m.indices.length).toBe(36);
    expect(m.vertices.length).toBe(24 * 8);
    expect(triNormalsCCW(m.vertices, m.indices)).toBe(true);
  });

  it("planeMesh is a single CCW quad with +Y normal", () => {
    const m = planeMesh();
    expect(m.vertexCount).toBe(4);
    expect(m.indexCount).toBe(6);
    // All four normals should be +Y.
    for (let i = 0; i < 4; i++) {
      expect(m.vertices[i * 8 + 3]).toBe(0);
      expect(m.vertices[i * 8 + 4]).toBe(1);
      expect(m.vertices[i * 8 + 5]).toBe(0);
    }
    expect(triNormalsCCW(m.vertices, m.indices)).toBe(true);
  });

  it("sphereMesh has consistent CCW winding and unit-length normals", () => {
    const m = sphereMesh(1, 8, 6);
    expect(m.indexCount).toBe(8 * 6 * 6);
    expect(triNormalsCCW(m.vertices, m.indices)).toBe(true);
    // Spot-check: a few vertex normals should be unit length.
    for (const v of [0, 5, 13, 22]) {
      const nx = m.vertices[v * 8 + 3]!;
      const ny = m.vertices[v * 8 + 4]!;
      const nz = m.vertices[v * 8 + 5]!;
      const len = Math.hypot(nx, ny, nz);
      expect(len).toBeGreaterThan(0.99);
      expect(len).toBeLessThan(1.01);
    }
  });

  it("cylinderMesh has CCW winding across sides + both caps", () => {
    const m = cylinderMesh(1, 2, 8);
    // sides = 8 quads × 2 tris = 16; top cap = 8 tris; bottom cap = 8 tris.
    // Total triangles = 32; indices = 96.
    expect(m.indexCount).toBe(96);
    expect(triNormalsCCW(m.vertices, m.indices)).toBe(true);
  });

  it("primitiveMesh dispatches by shape name", () => {
    expect(primitiveMesh("box").vertexCount).toBe(24);
    expect(primitiveMesh("plane").vertexCount).toBe(4);
    expect(primitiveMesh("sphere").vertexCount).toBeGreaterThan(100);
    expect(primitiveMesh("cylinder").vertexCount).toBeGreaterThan(10);
  });
});
