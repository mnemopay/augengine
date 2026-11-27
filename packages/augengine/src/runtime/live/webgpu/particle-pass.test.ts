/**
 * GPU particle pass tests.
 *
 * Two layers:
 *  1. Pure helpers (spec resolution, pool seeding, uniform packing, camera
 *     basis, blend state) — exercised directly.
 *  2. The real `ParticlePass` driven through a recording fake GPUDevice. Node
 *     has no WebGPU, so this cannot execute WGSL; what it DOES verify is every
 *     descriptor and command the class emits — the blend state, the depth
 *     configuration, dispatch counts, instance counts, buffer sizes and
 *     teardown. Those are where the bugs that silently break particles live.
 *
 * Shader execution itself is verified on real hardware by
 * `tests/integration/fire-particles.spec.ts`, which runs the compiled WGSL in
 * Chrome against a hardware adapter and reads the simulated pool back.
 */

import { describe, it, expect, beforeAll } from "vitest";
import {
  ParticlePass,
  FIRE_PRESET,
  SMOKE_PRESET,
  EMBER_PRESET,
  resolveEmitterSpec,
  seedParticlePool,
  packSimParams,
  packDrawParams,
  cameraBasisFromView,
  particleBlendState,
  particleGpuBytes,
  particleWorkgroupCount,
  expectedAliveCount,
  colorModeCode,
  PARTICLE_SIM_WGSL,
  PARTICLE_DRAW_WGSL,
  PARTICLE_STRIDE_BYTES,
  PARTICLE_STRIDE_FLOATS,
  PARTICLE_FIELD_OFFSETS,
  PARTICLE_WORKGROUP_SIZE,
  PARTICLE_VERTS_PER_QUAD,
  SIM_PARAMS_BYTES,
  DRAW_PARAMS_BYTES,
  MAX_PARTICLE_CAPACITY,
} from "./particle-pass.js";
import { m4Create, m4LookAt } from "../../../scene/math.js";
import type { Mat4 } from "../../../scene/types.js";

// The pass reads these WebGPU globals inside its constructor and methods.
// Node does not define them, so stand in with the spec's bit values.
beforeAll(() => {
  const g = globalThis as Record<string, unknown>;
  g.GPUShaderStage ??= { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };
  g.GPUBufferUsage ??= {
    MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16,
    VERTEX: 32, UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512,
  };
});

interface FakeBuffer {
  label?: string;
  size: number;
  usage: number;
  destroyed: boolean;
  destroy(): void;
}

function fakeGpu() {
  const buffers: FakeBuffer[] = [];
  const writes: { label?: string; offset: number; data: Float32Array }[] = [];
  const shaderModules: { label?: string; code: string }[] = [];
  const computePipelines: GPUComputePipelineDescriptor[] = [];
  const renderPipelines: GPURenderPipelineDescriptor[] = [];
  const bindGroups: GPUBindGroupDescriptor[] = [];

  const device = {
    createBuffer(desc: GPUBufferDescriptor): FakeBuffer {
      const b: FakeBuffer = {
        label: desc.label,
        size: desc.size,
        usage: desc.usage,
        destroyed: false,
        destroy() { this.destroyed = true; },
      };
      buffers.push(b);
      return b;
    },
    createShaderModule(desc: GPUShaderModuleDescriptor) {
      shaderModules.push({ label: desc.label, code: desc.code });
      return { label: desc.label };
    },
    createBindGroupLayout(desc: GPUBindGroupLayoutDescriptor) {
      return { label: desc.label, entries: desc.entries };
    },
    createPipelineLayout(desc: GPUPipelineLayoutDescriptor) {
      return { label: desc.label, bindGroupLayouts: desc.bindGroupLayouts };
    },
    createComputePipeline(desc: GPUComputePipelineDescriptor) {
      computePipelines.push(desc);
      return { label: desc.label, kind: "compute" };
    },
    createRenderPipeline(desc: GPURenderPipelineDescriptor) {
      renderPipelines.push(desc);
      return { label: desc.label, kind: "render" };
    },
    createBindGroup(desc: GPUBindGroupDescriptor) {
      bindGroups.push(desc);
      return { label: desc.label };
    },
    queue: {
      writeBuffer(buffer: FakeBuffer, offset: number, data: ArrayBufferView) {
        // Copy: the pass reuses one scratch array across emitters, and a
        // reference would make every record alias the last write.
        writes.push({
          label: buffer.label,
          offset,
          data: new Float32Array(
            (data as Float32Array).buffer.slice(
              (data as Float32Array).byteOffset,
              (data as Float32Array).byteOffset + (data as Float32Array).byteLength,
            ),
          ),
        });
      },
    },
  };

  return { device, buffers, writes, shaderModules, computePipelines, renderPipelines, bindGroups };
}

function fakeComputeEncoder() {
  const dispatches: number[] = [];
  let ended = false;
  const pass = {
    setPipeline() {},
    setBindGroup() {},
    dispatchWorkgroups(n: number) { dispatches.push(n); },
    end() { ended = true; },
  };
  return {
    encoder: { beginComputePass: () => pass },
    dispatches,
    get ended() { return ended; },
  };
}

function fakeRenderPass() {
  const draws: { verts: number; instances: number; pipeline: unknown }[] = [];
  let current: unknown = null;
  const setPipelineCalls: unknown[] = [];
  return {
    pass: {
      setPipeline(p: unknown) { current = p; setPipelineCalls.push(p); },
      setBindGroup() {},
      draw(verts: number, instances: number) { draws.push({ verts, instances, pipeline: current }); },
    },
    draws,
    setPipelineCalls,
  };
}

function makePass(overrides: Partial<{ sampleCount: 1 | 4; targetFormat: GPUTextureFormat }> = {}) {
  const gpu = fakeGpu();
  const pass = new ParticlePass(gpu.device as unknown as GPUDevice, {
    targetFormat: overrides.targetFormat ?? "rgba16float",
    sampleCount: overrides.sampleCount ?? 4,
  });
  return { ...gpu, pass };
}

// ──────────────────────────────────────────────────────────────────────

describe("particle layout constants", () => {
  it("keeps the struct stride a multiple of the 16-byte WGSL alignment", () => {
    expect(PARTICLE_STRIDE_BYTES % 16).toBe(0);
    expect(PARTICLE_STRIDE_FLOATS).toBe(PARTICLE_STRIDE_BYTES / 4);
  });

  it("gives every field a distinct slot inside one stride", () => {
    const offsets = Object.values(PARTICLE_FIELD_OFFSETS);
    expect(new Set(offsets).size).toBe(offsets.length);
    for (const o of offsets) expect(o).toBeLessThan(PARTICLE_STRIDE_FLOATS);
  });

  it("sizes the uniforms to whole vec4 blocks", () => {
    expect(SIM_PARAMS_BYTES % 16).toBe(0);
    expect(DRAW_PARAMS_BYTES % 16).toBe(0);
  });

  it("reports pool bytes and workgroup count from capacity", () => {
    expect(particleGpuBytes(1000)).toBe(48_000);
    expect(particleWorkgroupCount(64)).toBe(1);
    expect(particleWorkgroupCount(65)).toBe(2);
    expect(particleWorkgroupCount(4096)).toBe(64);
  });
});

describe("resolveEmitterSpec", () => {
  it("clamps capacity to the hard cap and floors it to an integer", () => {
    expect(resolveEmitterSpec({ capacity: 1e9 }).capacity).toBe(MAX_PARTICLE_CAPACITY);
    expect(resolveEmitterSpec({ capacity: 0 }).capacity).toBe(1);
    expect(resolveEmitterSpec({ capacity: 10.7 }).capacity).toBe(10);
  });

  it("sorts a reversed lifetime or size range instead of yielding a negative span", () => {
    const spec = resolveEmitterSpec({ lifetime: [2, 0.5], sizeRange: [0.4, 0.1] });
    expect(spec.lifetime).toEqual([0.5, 2]);
    expect(spec.sizeRange).toEqual([0.1, 0.4]);
  });

  it("falls back to the preset value for non-finite input rather than passing NaN to the GPU", () => {
    const spec = resolveEmitterSpec({
      emissionRate: Number.NaN,
      buoyancy: Number.POSITIVE_INFINITY,
      origin: [1, Number.NaN, 3],
    });
    expect(spec.emissionRate).toBe(FIRE_PRESET.emissionRate);
    // Infinity falls back to the preset rather than clamping to the ceiling:
    // a runaway value should not become a legal-but-absurd 1e4 buoyancy.
    expect(spec.buoyancy).toBe(FIRE_PRESET.buoyancy);
    expect(spec.origin).toEqual([1, FIRE_PRESET.origin[1], 3]);
    for (const v of Object.values(spec).flat()) {
      if (typeof v === "number") expect(Number.isFinite(v)).toBe(true);
    }
  });

  it("masks the seed into f32's exactly-representable integer range", () => {
    const spec = resolveEmitterSpec({ seed: 0xffffffff });
    expect(spec.seed).toBeLessThan(0x1000000);
    // Must survive the f32 uniform unchanged, or the RNG stream silently
    // differs from the seed the caller asked for.
    expect(Math.fround(spec.seed)).toBe(spec.seed);
  });

  it("keeps every shipped preset self-consistent under resolution", () => {
    for (const preset of [FIRE_PRESET, SMOKE_PRESET, EMBER_PRESET]) {
      const resolved = resolveEmitterSpec({}, preset);
      expect(resolved.lifetime[0]).toBeLessThanOrEqual(resolved.lifetime[1]);
      expect(resolved.sizeRange[0]).toBeLessThanOrEqual(resolved.sizeRange[1]);
      expect(resolved.capacity).toBeLessThanOrEqual(MAX_PARTICLE_CAPACITY);
      expect(resolved.alphaScale).toBeLessThanOrEqual(1);
      expect(Math.fround(resolved.seed)).toBe(resolved.seed);
    }
  });

  it("ships fire as additive HDR and smoke as alpha-blended and cool", () => {
    expect(FIRE_PRESET.blend).toBe("additive");
    expect(FIRE_PRESET.intensity).toBeGreaterThan(1); // must reach the bloom threshold
    expect(SMOKE_PRESET.blend).toBe("alpha");
    expect(SMOKE_PRESET.growth).toBeGreaterThan(FIRE_PRESET.growth);
    expect(EMBER_PRESET.gravity[1]).toBeLessThan(0); // embers fall back down
  });
});

describe("seedParticlePool", () => {
  it("allocates one stride per particle with every slot dead", () => {
    const spec = resolveEmitterSpec({ capacity: 8, emissionRate: 4 });
    const pool = seedParticlePool(spec);
    expect(pool.length).toBe(8 * PARTICLE_STRIDE_FLOATS);
    for (let i = 0; i < 8; i++) {
      expect(pool[i * PARTICLE_STRIDE_FLOATS + PARTICLE_FIELD_OFFSETS.life]).toBe(0);
    }
  });

  it("staggers first spawn as index / emissionRate so the pool does not pulse", () => {
    const spec = resolveEmitterSpec({ capacity: 5, emissionRate: 10 });
    const pool = seedParticlePool(spec);
    const respawns = [];
    for (let i = 0; i < 5; i++) {
      respawns.push(pool[i * PARTICLE_STRIDE_FLOATS + PARTICLE_FIELD_OFFSETS.respawn]!);
    }
    [0, 0.1, 0.2, 0.3, 0.4].forEach((expected, i) => {
      expect(respawns[i]!).toBeCloseTo(expected, 6);
    });
    for (let i = 1; i < respawns.length; i++) {
      expect(respawns[i]!).toBeGreaterThan(respawns[i - 1]!);
    }
  });

  it("does not divide by zero when the emitter rate is zero", () => {
    const pool = seedParticlePool(resolveEmitterSpec({ capacity: 4, emissionRate: 0 }));
    for (const v of pool) expect(Number.isFinite(v)).toBe(true);
  });
});

describe("packSimParams", () => {
  const spec = resolveEmitterSpec({
    capacity: 1000,
    emissionRate: 500,
    origin: [1, 2, 3],
    spawnExtent: [0.1, 0.2, 0.3],
    velocityMin: [-1, 0, -1],
    velocityMax: [1, 2, 1],
    gravity: [0, -9.81, 0],
    buoyancy: 3.5,
    wind: [0.5, 0, 0.25],
    drag: 1.25,
    lifetime: [0.5, 1.5],
    sizeRange: [0.1, 0.3],
    turbulence: 2,
    turbulenceFrequency: 1.5,
    coolRate: 1.75,
    growth: 0.9,
    active: true,
  });

  it("writes every field at its documented float offset", () => {
    const out = packSimParams(new Float32Array(SIM_PARAMS_BYTES / 4), spec, 1 / 60, 12.5, 42);
    expect(out[0]).toBeCloseTo(1 / 60, 6);
    expect(out[1]).toBeCloseTo(12.5, 6);
    expect(out[2]).toBeCloseTo(1000 / 500, 6); // period = capacity / rate
    expect(out[3]).toBe(1); // active
    expect([out[4], out[5], out[6]]).toEqual([1, 2, 3]);
    expect([out[8], out[9], out[10]].map((v) => Math.fround(v!))).toEqual([0.1, 0.2, 0.3].map(Math.fround));
    expect([out[12], out[13], out[14]]).toEqual([-1, 0, -1]);
    expect([out[16], out[17], out[18]]).toEqual([1, 2, 1]);
    expect(out[21]).toBeCloseTo(-9.81, 5);
    expect(out[23]).toBeCloseTo(3.5, 6); // buoyancy rides in gravity.w
    expect(out[27]).toBeCloseTo(1.25, 6); // drag rides in wind.w
    expect([out[28], out[29]]).toEqual([0.5, 1.5]);
    expect(out[32]).toBe(2);
    expect(out[34]).toBeCloseTo(1.75, 6);
    expect(out[36]).toBe(42);
  });

  it("leaves the reserved sim slot zero — growth is a draw-time value only", () => {
    const out = packSimParams(new Float32Array(SIM_PARAMS_BYTES / 4), spec, 0.016, 0, 1);
    expect(out[35]).toBe(0);
    expect(spec.growth).toBe(0.9); // the spec carries it; the sim uniform does not
  });

  it("flags an inactive emitter so the shader stops respawning", () => {
    const out = packSimParams(
      new Float32Array(SIM_PARAMS_BYTES / 4),
      resolveEmitterSpec({ active: false }),
      0.016,
      0,
      1,
    );
    expect(out[3]).toBe(0);
  });

  it("uses an effectively infinite respawn period when the rate is zero", () => {
    const out = packSimParams(
      new Float32Array(SIM_PARAMS_BYTES / 4),
      resolveEmitterSpec({ emissionRate: 0, capacity: 100 }),
      0.016,
      0,
      1,
    );
    expect(out[2]).toBeGreaterThan(1e8);
    expect(Number.isFinite(out[2]!)).toBe(true);
  });

  it("zeroes stale bytes so a reused scratch buffer cannot leak a prior emitter", () => {
    const scratch = new Float32Array(SIM_PARAMS_BYTES / 4).fill(7);
    packSimParams(scratch, resolveEmitterSpec({ origin: [0, 0, 0], gravity: [0, 0, 0] }), 0.016, 0, 1);
    expect(scratch[7]).toBe(0); // origin padding
    expect(scratch[39]).toBe(0); // misc padding
  });

  it("rejects an undersized destination instead of writing out of bounds", () => {
    expect(() => packSimParams(new Float32Array(8), spec, 0.016, 0, 1)).toThrow(/needs 40 floats/);
  });
});

describe("packDrawParams", () => {
  const spec = resolveEmitterSpec({
    tint: [0.25, 0.5, 0.75],
    intensity: 2.5,
    softness: 1.5,
    alphaScale: 0.8,
    growth: 1.25,
    colorMode: "ember",
  });

  it("copies the view-projection matrix and camera basis at their offsets", () => {
    const vp = new Float32Array(16).map((_, i) => i + 1) as unknown as Mat4;
    const out = packDrawParams(new Float32Array(DRAW_PARAMS_BYTES / 4), vp, [1, 0, 0], [0, 1, 0], spec);
    expect(Array.from(out.subarray(0, 16))).toEqual(Array.from(vp as unknown as Float32Array));
    expect([out[16], out[17], out[18]]).toEqual([1, 0, 0]);
    expect([out[20], out[21], out[22]]).toEqual([0, 1, 0]);
  });

  it("packs tint, intensity and the shaping vector including growth", () => {
    const vp = new Float32Array(16) as unknown as Mat4;
    const out = packDrawParams(new Float32Array(DRAW_PARAMS_BYTES / 4), vp, [1, 0, 0], [0, 1, 0], spec);
    expect([out[24], out[25], out[26]].map((v) => Math.fround(v!))).toEqual([0.25, 0.5, 0.75]);
    expect(out[27]).toBeCloseTo(2.5, 6); // intensity in tint.w
    expect(out[28]).toBeCloseTo(1.5, 6); // softness
    expect(out[29]).toBeCloseTo(0.8, 6); // alphaScale
    expect(out[30]).toBe(colorModeCode("ember"));
    expect(out[31]).toBeCloseTo(1.25, 6); // growth
  });

  it("gives each colour mode a distinct code", () => {
    const codes = (["fire", "flat", "ember"] as const).map(colorModeCode);
    expect(new Set(codes).size).toBe(3);
    expect(colorModeCode("fire")).toBe(0);
  });

  it("rejects an undersized destination", () => {
    const vp = new Float32Array(16) as unknown as Mat4;
    expect(() => packDrawParams(new Float32Array(4), vp, [1, 0, 0], [0, 1, 0], spec)).toThrow(/needs 32 floats/);
  });
});

describe("cameraBasisFromView", () => {
  it("recovers world right/up from a real m4LookAt view matrix", () => {
    // Camera at +Z looking at the origin with +Y up: right is +X, up is +Y.
    const view = m4LookAt(m4Create(), [0, 0, 5], [0, 0, 0], [0, 1, 0]);
    const { right, up } = cameraBasisFromView(view);
    expect(right[0]).toBeCloseTo(1, 5);
    expect(right[1]).toBeCloseTo(0, 5);
    expect(right[2]).toBeCloseTo(0, 5);
    expect(up[1]).toBeCloseTo(1, 5);
  });

  it("rotates the basis with the camera", () => {
    // Looking down -X from +X: camera right becomes -Z.
    const view = m4LookAt(m4Create(), [5, 0, 0], [0, 0, 0], [0, 1, 0]);
    const { right, up } = cameraBasisFromView(view);
    expect(right[2]).toBeCloseTo(-1, 5);
    expect(up[1]).toBeCloseTo(1, 5);
  });

  it("returns an orthonormal pair for an arbitrary camera", () => {
    const view = m4LookAt(m4Create(), [3, 4, 5], [1, 0, -2], [0, 1, 0]);
    const { right, up } = cameraBasisFromView(view);
    const dot = right[0] * up[0] + right[1] * up[1] + right[2] * up[2];
    expect(dot).toBeCloseTo(0, 5);
    expect(Math.hypot(...right)).toBeCloseTo(1, 5);
    expect(Math.hypot(...up)).toBeCloseTo(1, 5);
  });
});

describe("particleBlendState", () => {
  it("makes additive premultiplied one/one so draw order cannot matter", () => {
    const b = particleBlendState("additive");
    expect(b.color).toMatchObject({ srcFactor: "one", dstFactor: "one", operation: "add" });
    expect(b.alpha).toMatchObject({ srcFactor: "one", dstFactor: "one" });
  });

  it("makes alpha premultiplied source-over", () => {
    const b = particleBlendState("alpha");
    expect(b.color).toMatchObject({ srcFactor: "one", dstFactor: "one-minus-src-alpha" });
    expect(b.alpha).toMatchObject({ srcFactor: "one", dstFactor: "one-minus-src-alpha" });
  });
});

describe("expectedAliveCount", () => {
  it("is rate × mean lifetime while the pool has room", () => {
    const spec = resolveEmitterSpec({ capacity: 10_000, emissionRate: 100, lifetime: [1, 3] });
    expect(expectedAliveCount(spec)).toBeCloseTo(200, 6); // 100/s × 2s mean
  });

  it("clamps to the pool size when the emitter is pool-limited", () => {
    const spec = resolveEmitterSpec({ capacity: 64, emissionRate: 1000, lifetime: [2, 2] });
    expect(expectedAliveCount(spec)).toBe(64);
  });
});

// ──────────────────────────────────────────────────────────────────────

describe("ParticlePass pipelines", () => {
  it("compiles the compute pipeline from the simulation WGSL at construction", () => {
    const { computePipelines, shaderModules } = makePass();
    expect(computePipelines).toHaveLength(1);
    expect(computePipelines[0]!.compute.entryPoint).toBe("main");
    expect(shaderModules.some((m) => m.code === PARTICLE_SIM_WGSL)).toBe(true);
    expect(shaderModules.some((m) => m.code === PARTICLE_DRAW_WGSL)).toBe(true);
  });

  it("does not compile a render pipeline until an emitter needs one", () => {
    const { renderPipelines, pass } = makePass();
    expect(renderPipelines).toHaveLength(0);
    pass.attach("torch", {}, FIRE_PRESET);
    expect(renderPipelines).toHaveLength(1);
  });

  it("never writes depth, but still tests against opaque geometry", () => {
    const { renderPipelines, pass } = makePass();
    pass.attach("torch", {}, FIRE_PRESET);
    const ds = renderPipelines[0]!.depthStencil!;
    expect(ds.depthWriteEnabled).toBe(false);
    expect(ds.depthCompare).toBe("less");
    expect(ds.format).toBe("depth24plus");
  });

  it("matches the main pass target format and sample count", () => {
    const { renderPipelines, pass } = makePass({ sampleCount: 4, targetFormat: "rgba16float" });
    pass.attach("torch", {}, FIRE_PRESET);
    const p = renderPipelines[0]!;
    expect(p.multisample!.count).toBe(4);
    expect((p.fragment!.targets[0] as GPUColorTargetState).format).toBe("rgba16float");
    expect(p.primitive!.cullMode).toBe("none"); // billboards face the camera either way
  });

  it("honours a non-MSAA renderer", () => {
    const { renderPipelines, pass } = makePass({ sampleCount: 1 });
    pass.attach("torch", {}, FIRE_PRESET);
    expect(renderPipelines[0]!.multisample!.count).toBe(1);
  });

  it("compiles one pipeline per blend mode and reuses it across emitters", () => {
    const { renderPipelines, pass } = makePass();
    pass.attach("fire-a", {}, FIRE_PRESET);
    pass.attach("fire-b", {}, FIRE_PRESET);
    expect(renderPipelines).toHaveLength(1);
    pass.attach("smoke", {}, SMOKE_PRESET);
    expect(renderPipelines).toHaveLength(2);
    const blends = renderPipelines.map((p) => (p.fragment!.targets[0] as GPUColorTargetState).blend!.color.dstFactor);
    expect(new Set(blends)).toEqual(new Set(["one", "one-minus-src-alpha"]));
  });
});

describe("ParticlePass emitters", () => {
  it("allocates a pool sized capacity × stride plus the two uniforms", () => {
    const { buffers, pass } = makePass();
    pass.attach("torch", { capacity: 512 }, FIRE_PRESET);
    expect(buffers.map((b) => b.size)).toEqual([512 * PARTICLE_STRIDE_BYTES, SIM_PARAMS_BYTES, DRAW_PARAMS_BYTES]);
    expect(pass.gpuBytes()).toBe(512 * PARTICLE_STRIDE_BYTES);
  });

  it("gives each emitter its own uniform buffers so dispatches cannot alias", () => {
    const { bindGroups, pass } = makePass();
    pass.attach("a", { capacity: 64 }, FIRE_PRESET);
    pass.attach("b", { capacity: 64 }, FIRE_PRESET);
    const uniformResources = bindGroups.map(
      (bg) => ((bg.entries as GPUBindGroupEntry[])[1]!.resource as GPUBufferBinding).buffer,
    );
    expect(new Set(uniformResources).size).toBe(uniformResources.length);
  });

  it("seeds the pool on the GPU at attach time", () => {
    const { writes, pass } = makePass();
    pass.attach("torch", { capacity: 4, emissionRate: 2 }, FIRE_PRESET);
    expect(writes).toHaveLength(1);
    expect(writes[0]!.data.length).toBe(4 * PARTICLE_STRIDE_FLOATS);
    expect(writes[0]!.data[PARTICLE_FIELD_OFFSETS.respawn]).toBe(0);
    expect(writes[0]!.data[PARTICLE_STRIDE_FLOATS + PARTICLE_FIELD_OFFSETS.respawn]).toBeCloseTo(0.5, 6);
  });

  it("is idempotent — re-attaching an id replaces the old pool", () => {
    const { buffers, pass } = makePass();
    pass.attach("torch", { capacity: 64 }, FIRE_PRESET);
    pass.attach("torch", { capacity: 128 }, FIRE_PRESET);
    expect(pass.emitterCount).toBe(1);
    expect(buffers.filter((b) => b.destroyed)).toHaveLength(3); // the first emitter's three buffers
    expect(pass.getSpec("torch")!.capacity).toBe(128);
  });

  it("patches knobs in place without reallocating the pool", () => {
    const { buffers, pass } = makePass();
    pass.attach("torch", { capacity: 64, emissionRate: 100 }, FIRE_PRESET);
    const before = buffers.length;
    const next = pass.update("torch", { emissionRate: 400, intensity: 5 });
    expect(next!.emissionRate).toBe(400);
    expect(next!.intensity).toBe(5);
    expect(next!.capacity).toBe(64);
    expect(buffers.length).toBe(before);
    expect(buffers.some((b) => b.destroyed)).toBe(false);
  });

  it("reallocates when capacity changes, because the pool is fixed-size", () => {
    const { buffers, pass } = makePass();
    pass.attach("torch", { capacity: 64 }, FIRE_PRESET);
    pass.update("torch", { capacity: 256 });
    expect(pass.getSpec("torch")!.capacity).toBe(256);
    expect(buffers.filter((b) => b.destroyed)).toHaveLength(3);
  });

  it("returns null when patching or reading an unknown emitter", () => {
    const { pass } = makePass();
    expect(pass.update("nope", { intensity: 2 })).toBeNull();
    expect(pass.getSpec("nope")).toBeNull();
    expect(pass.has("nope")).toBe(false);
  });

  it("frees all three buffers on detach and tolerates an unknown id", () => {
    const { buffers, pass } = makePass();
    pass.attach("torch", { capacity: 64 }, FIRE_PRESET);
    pass.detach("torch");
    expect(pass.emitterCount).toBe(0);
    expect(pass.gpuBytes()).toBe(0);
    expect(buffers.every((b) => b.destroyed)).toBe(true);
    expect(() => pass.detach("torch")).not.toThrow();
  });

  it("frees every emitter on destroy", () => {
    const { buffers, pass } = makePass();
    pass.attach("a", { capacity: 64 }, FIRE_PRESET);
    pass.attach("b", { capacity: 64 }, SMOKE_PRESET);
    pass.destroy();
    expect(pass.emitterCount).toBe(0);
    expect(buffers.every((b) => b.destroyed)).toBe(true);
  });

  it("estimates alive counts only for active emitters and labels nothing as measured", () => {
    const { pass } = makePass();
    pass.attach("on", { capacity: 10_000, emissionRate: 100, lifetime: [1, 1] }, FIRE_PRESET);
    pass.attach("off", { capacity: 10_000, emissionRate: 100, lifetime: [1, 1], active: false }, FIRE_PRESET);
    expect(pass.estimatedAliveCount()).toBe(100);
  });
});

describe("ParticlePass.simulate", () => {
  it("dispatches ceil(capacity / workgroupSize) workgroups per emitter", () => {
    const { pass } = makePass();
    pass.attach("torch", { capacity: 1000 }, FIRE_PRESET);
    const { encoder, dispatches } = fakeComputeEncoder();
    pass.simulate(encoder as unknown as GPUCommandEncoder, 1 / 60);
    expect(dispatches).toEqual([Math.ceil(1000 / PARTICLE_WORKGROUP_SIZE)]);
  });

  it("dispatches once per emitter and closes the pass", () => {
    const { pass } = makePass();
    pass.attach("a", { capacity: 64 }, FIRE_PRESET);
    pass.attach("b", { capacity: 128 }, FIRE_PRESET);
    const h = fakeComputeEncoder();
    pass.simulate(h.encoder as unknown as GPUCommandEncoder, 1 / 60);
    expect(h.dispatches).toEqual([1, 2]);
    expect(h.ended).toBe(true);
  });

  it("does nothing with no emitters", () => {
    const { pass } = makePass();
    const h = fakeComputeEncoder();
    pass.simulate(h.encoder as unknown as GPUCommandEncoder, 1 / 60);
    expect(h.dispatches).toEqual([]);
  });

  it("skips a zero, negative or non-finite dt instead of advancing time", () => {
    const { pass, writes } = makePass();
    pass.attach("torch", { capacity: 64 }, FIRE_PRESET);
    const before = writes.length;
    for (const dt of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
      const h = fakeComputeEncoder();
      pass.simulate(h.encoder as unknown as GPUCommandEncoder, dt);
      expect(h.dispatches).toEqual([]);
    }
    expect(writes.length).toBe(before);
  });

  it("clamps a long stall to 0.1 s so particles cannot teleport", () => {
    const { pass, writes } = makePass();
    pass.attach("torch", { capacity: 64 }, FIRE_PRESET);
    const h = fakeComputeEncoder();
    pass.simulate(h.encoder as unknown as GPUCommandEncoder, 5);
    const sim = writes[writes.length - 1]!.data;
    expect(sim[0]).toBeCloseTo(0.1, 6);
  });

  it("advances elapsed time and the frame counter across calls", () => {
    const { pass, writes } = makePass();
    pass.attach("torch", { capacity: 64 }, FIRE_PRESET);
    for (let i = 0; i < 3; i++) {
      const h = fakeComputeEncoder();
      pass.simulate(h.encoder as unknown as GPUCommandEncoder, 0.02);
    }
    const sim = writes[writes.length - 1]!.data;
    expect(sim[1]).toBeCloseTo(0.06, 5); // elapsed
    expect(sim[36]).toBe(3); // frame
  });

  it("writes a distinct sim uniform per emitter in one pass", () => {
    const { pass, writes } = makePass();
    pass.attach("a", { capacity: 64, emissionRate: 100 }, FIRE_PRESET);
    pass.attach("b", { capacity: 64, emissionRate: 800 }, FIRE_PRESET);
    const h = fakeComputeEncoder();
    pass.simulate(h.encoder as unknown as GPUCommandEncoder, 1 / 60);
    const sims = writes.filter((w) => w.label?.includes("sim-params"));
    expect(sims).toHaveLength(2);
    expect(sims[0]!.data[2]).not.toBeCloseTo(sims[1]!.data[2]!, 6); // different periods
  });
});

describe("ParticlePass.encode", () => {
  const view = m4LookAt(m4Create(), [0, 1, 4], [0, 1, 0], [0, 1, 0]);
  const viewProj = m4Create();

  it("draws one instance per pool slot using a vertex-only quad", () => {
    const { pass } = makePass();
    pass.attach("torch", { capacity: 777 }, FIRE_PRESET);
    const h = fakeRenderPass();
    pass.encode(h.pass as unknown as GPURenderPassEncoder, view, viewProj);
    expect(h.draws).toEqual([
      { verts: PARTICLE_VERTS_PER_QUAD, instances: 777, pipeline: expect.anything() },
    ]);
  });

  it("draws nothing with no emitters", () => {
    const { pass } = makePass();
    const h = fakeRenderPass();
    pass.encode(h.pass as unknown as GPURenderPassEncoder, view, viewProj);
    expect(h.draws).toEqual([]);
    expect(h.setPipelineCalls).toEqual([]);
  });

  it("draws alpha-blended emitters before additive ones regardless of attach order", () => {
    const { pass, renderPipelines } = makePass();
    pass.attach("fire", { capacity: 64 }, FIRE_PRESET);
    pass.attach("smoke", { capacity: 32 }, SMOKE_PRESET);
    const h = fakeRenderPass();
    pass.encode(h.pass as unknown as GPURenderPassEncoder, view, viewProj);
    expect(h.draws.map((d) => d.instances)).toEqual([32, 64]); // smoke first
    const alphaPipelineLabel = renderPipelines.find(
      (p) => (p.fragment!.targets[0] as GPUColorTargetState).blend!.color.dstFactor === "one-minus-src-alpha",
    )!.label;
    expect((h.draws[0]!.pipeline as { label: string }).label).toBe(alphaPipelineLabel);
  });

  it("binds each blend pipeline once when emitters share a mode", () => {
    const { pass } = makePass();
    pass.attach("a", { capacity: 64 }, FIRE_PRESET);
    pass.attach("b", { capacity: 64 }, FIRE_PRESET);
    const h = fakeRenderPass();
    pass.encode(h.pass as unknown as GPURenderPassEncoder, view, viewProj);
    expect(h.setPipelineCalls).toHaveLength(1);
    expect(h.draws).toHaveLength(2);
  });

  it("writes the camera basis derived from the supplied view matrix", () => {
    const { pass, writes } = makePass();
    pass.attach("torch", { capacity: 64 }, FIRE_PRESET);
    const h = fakeRenderPass();
    pass.encode(h.pass as unknown as GPURenderPassEncoder, view, viewProj);
    const draw = writes.filter((w) => w.label?.includes("draw-params")).pop()!;
    expect(draw.data[16]).toBeCloseTo(1, 5); // right = +X for a camera on +Z
    expect(draw.data[21]).toBeCloseTo(1, 5); // up = +Y
  });
});

describe("particle WGSL source", () => {
  it("declares a read_write storage pool in compute and a read-only one in the draw stage", () => {
    // Vertex-stage storage MUST be read-only; read_write there is a WebGPU
    // validation error that only surfaces on a real device.
    expect(PARTICLE_SIM_WGSL).toContain("var<storage, read_write> particles");
    expect(PARTICLE_DRAW_WGSL).toContain("var<storage, read> particles");
    expect(PARTICLE_DRAW_WGSL).not.toContain("read_write");
  });

  it("uses the workgroup size the host dispatches against", () => {
    expect(PARTICLE_SIM_WGSL).toContain(`@compute @workgroup_size(${PARTICLE_WORKGROUP_SIZE})`);
  });

  it("bounds-checks the pool so a partial final workgroup cannot write past the end", () => {
    expect(PARTICLE_SIM_WGSL).toContain("arrayLength(&particles)");
  });

  it("declares the same Particle struct fields in both stages", () => {
    for (const field of ["position", "life", "velocity", "maxLife", "seed", "size", "temperature", "respawn"]) {
      expect(PARTICLE_SIM_WGSL).toContain(`${field}:`);
      expect(PARTICLE_DRAW_WGSL).toContain(`${field}:`);
    }
  });

  it("derives temperature and growth from life instead of integrating them", () => {
    expect(PARTICLE_SIM_WGSL).toContain("p.temperature = pow(lifeRatio, params.shaping.z)");
    expect(PARTICLE_DRAW_WGSL).toContain("params.shaping.w * ageRatio");
  });

  it("clips dead particles rather than drawing degenerate geometry", () => {
    expect(PARTICLE_DRAW_WGSL).toContain("vec4<f32>(0.0, 0.0, 2.0, 1.0)");
  });

  it("emits premultiplied colour, matching both blend states", () => {
    expect(PARTICLE_DRAW_WGSL).toContain("rgb * params.tint.w * a");
  });

  it("avoids WGSL reserved words as identifiers", () => {
    // Regression guard. `active` is a reserved WGSL keyword and Tint rejected
    // `let active = ...` outright — caught only by the real-adapter probe, not
    // by any Node test, so pin the whole reserved list we have tripped over.
    // `half`, `filter` and `sample` are reserved for future use.
    const reserved = ["active", "half", "filter", "sample", "common", "binding", "typedef"];
    for (const src of [PARTICLE_SIM_WGSL, PARTICLE_DRAW_WGSL]) {
      for (const word of reserved) {
        expect(src, `WGSL reserved word "${word}" used as an identifier`).not.toMatch(
          new RegExp(`\\b(let|var|const)\\s+${word}\\b`),
        );
        expect(src, `WGSL reserved word "${word}" used as a function name`).not.toMatch(
          new RegExp(`\\bfn\\s+${word}\\b`),
        );
      }
    }
  });
});
