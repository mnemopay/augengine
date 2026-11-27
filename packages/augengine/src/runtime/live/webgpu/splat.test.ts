import { describe, it, expect } from "vitest";
import {
  packSplatInstances,
  packSplatUniforms,
  SPLAT_BILLBOARD_WGSL,
  SPLAT_INSTANCE_STRIDE_BYTES,
  SPLAT_QUAD_VERTS,
  SPLAT_UNIFORM_SIZE_BYTES,
  SplatCloudResources,
  type SplatDeviceLike,
} from "./splat.js";

describe("packSplatInstances", () => {
  it("emits 8 floats per splat (vec3 pos + pad + vec4 rgba)", () => {
    const packed = packSplatInstances({
      pointCount: 3,
      positions: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
      colors: new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
      alphas: new Float32Array([1, 0.5, 0.25]),
    });
    // v2 layout: 12 floats per splat (pos3 + maxScale1 + rgba4 + quat4).
    expect(packed.length).toBe(3 * 12);
    // First splat
    expect(packed[0]).toBe(1);
    expect(packed[1]).toBe(2);
    expect(packed[2]).toBe(3);
    // maxScale defaults to 1.0 when scales aren't supplied
    expect(packed[3]).toBe(1.0);
    expect(packed[4]).toBeCloseTo(0.1, 6);
    expect(packed[5]).toBeCloseTo(0.2, 6);
    expect(packed[6]).toBeCloseTo(0.3, 6);
    expect(packed[7]).toBe(1);
    // Quaternion defaults to identity (0, 0, 0, 1) when rotations are absent
    expect(packed[8]).toBe(0);
    expect(packed[9]).toBe(0);
    expect(packed[10]).toBe(0);
    expect(packed[11]).toBe(1);
    // Last splat alpha
    expect(packed[2 * 12 + 7]).toBe(0.25);
  });

  it("matches the declared stride", () => {
    expect(SPLAT_INSTANCE_STRIDE_BYTES).toBe(48);
    const packed = packSplatInstances({
      pointCount: 1,
      positions: new Float32Array([0, 0, 0]),
      colors: new Float32Array([1, 1, 1]),
      alphas: new Float32Array([1]),
    });
    expect(packed.byteLength).toBe(SPLAT_INSTANCE_STRIDE_BYTES);
  });

  it("v2 — per-splat maxScale takes the max of the three axis sigmas", () => {
    const packed = packSplatInstances({
      pointCount: 2,
      positions: new Float32Array([0, 0, 0, 0, 0, 0]),
      colors: new Float32Array([1, 1, 1, 1, 1, 1]),
      alphas: new Float32Array([1, 1]),
      scales: new Float32Array([0.1, 2.5, 1.0, 0.5, 0.5, 0.5]),
    });
    expect(packed[3]).toBe(2.5);
    expect(packed[12 + 3]).toBe(0.5);
  });

  it("v2 — quaternion is copied through when rotations are supplied", () => {
    const packed = packSplatInstances({
      pointCount: 1,
      positions: new Float32Array([0, 0, 0]),
      colors: new Float32Array([1, 1, 1]),
      alphas: new Float32Array([1]),
      rotations: new Float32Array([0.5, 0.5, 0.5, 0.5]),
    });
    expect(packed[8]).toBeCloseTo(0.5, 6);
    expect(packed[9]).toBeCloseTo(0.5, 6);
    expect(packed[10]).toBeCloseTo(0.5, 6);
    expect(packed[11]).toBeCloseTo(0.5, 6);
  });
});

describe("packSplatUniforms", () => {
  it("emits 20 floats (mat4 + vec4)", () => {
    const viewProj = new Float32Array(16);
    for (let i = 0; i < 16; i++) viewProj[i] = i;
    const out = packSplatUniforms({ viewProj, params: [4, 1920, 1080, 1] });
    expect(out.length).toBe(20);
    expect(out[0]).toBe(0);
    expect(out[15]).toBe(15);
    expect(out[16]).toBe(4);
    expect(out[17]).toBe(1920);
    expect(out[18]).toBe(1080);
    expect(out[19]).toBe(1);
    expect(out.byteLength).toBe(SPLAT_UNIFORM_SIZE_BYTES);
  });

  it("rejects mis-sized viewProj", () => {
    expect(() =>
      packSplatUniforms({ viewProj: new Float32Array(15), params: [1, 1, 1, 1] }),
    ).toThrow(/16 floats/);
  });
});

describe("SPLAT_QUAD_VERTS", () => {
  it("declares two triangles (6 vertices)", () => {
    expect(SPLAT_QUAD_VERTS.length).toBe(12); // 6 verts × 2 floats
  });

  it("every corner lies on the unit square", () => {
    for (let i = 0; i < SPLAT_QUAD_VERTS.length; i += 2) {
      expect(Math.abs(SPLAT_QUAD_VERTS[i] ?? 0)).toBe(1);
      expect(Math.abs(SPLAT_QUAD_VERTS[i + 1] ?? 0)).toBe(1);
    }
  });
});

describe("SPLAT_BILLBOARD_WGSL", () => {
  it("defines vs_main + fs_main entry points", () => {
    expect(SPLAT_BILLBOARD_WGSL).toContain("fn vs_main");
    expect(SPLAT_BILLBOARD_WGSL).toContain("fn fs_main");
  });

  it("declares SplatUniforms with view_proj + params", () => {
    expect(SPLAT_BILLBOARD_WGSL).toContain("view_proj: mat4x4<f32>");
    expect(SPLAT_BILLBOARD_WGSL).toContain("params: vec4<f32>");
  });

  it("discards fragments outside the unit disk", () => {
    expect(SPLAT_BILLBOARD_WGSL).toContain("if (r2 > 1.0) { discard; }");
  });

  it("applies the per-frame opacity scale from params.w", () => {
    expect(SPLAT_BILLBOARD_WGSL).toContain("u.params.w");
  });
});

function makeFakeDevice(): {
  device: SplatDeviceLike;
  buffers: { size: number; usage: number }[];
  writes: number;
  destroys: number;
} {
  const buffers: { size: number; usage: number }[] = [];
  let writes = 0;
  let destroys = 0;
  const device: SplatDeviceLike = {
    createBuffer(desc) {
      buffers.push({ size: desc.size, usage: desc.usage });
      return {
        destroy() {
          destroys++;
        },
      };
    },
    queue: {
      writeBuffer() {
        writes++;
      },
    },
  };
  return {
    device,
    buffers,
    get writes() {
      return writes;
    },
    get destroys() {
      return destroys;
    },
  };
}

describe("SplatCloudResources", () => {
  const cloud = {
    pointCount: 4,
    positions: new Float32Array(12),
    colors: new Float32Array(12),
    alphas: new Float32Array(4).fill(1),
  };

  it("allocates an instance buffer + a uniform buffer on construction", () => {
    const fake = makeFakeDevice();
    const res = new SplatCloudResources(fake.device, cloud);
    expect(fake.buffers).toHaveLength(2);
    expect(fake.buffers[0]!.size).toBe(4 * SPLAT_INSTANCE_STRIDE_BYTES);
    expect(fake.buffers[1]!.size).toBe(SPLAT_UNIFORM_SIZE_BYTES);
    expect(res.pointCount).toBe(4);
  });

  it("writes the instance data immediately + uniforms on updateUniforms", () => {
    const fake = makeFakeDevice();
    const res = new SplatCloudResources(fake.device, cloud);
    expect(fake.writes).toBe(1); // instance write
    res.updateUniforms({ viewProj: new Float32Array(16), params: [2, 800, 600, 1] });
    expect(fake.writes).toBe(2); // uniform write
  });

  it("destroy releases both buffers", () => {
    const fake = makeFakeDevice();
    const res = new SplatCloudResources(fake.device, cloud);
    res.destroy();
    expect(fake.destroys).toBe(2);
  });
});

describe("SPLAT_UNIFORM_SIZE_BYTES", () => {
  it("matches a mat4 (16 × 4) + vec4 (4 × 4) = 80 bytes", () => {
    expect(SPLAT_UNIFORM_SIZE_BYTES).toBe(80);
  });
});
