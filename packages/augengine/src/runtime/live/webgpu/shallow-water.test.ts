/**
 * Shallow-water solver tests.
 *
 * Three layers:
 *  1. Pure helpers (spec resolution, CFL, seeding, face gating, normals,
 *     absorption, Fresnel, uniform packing) — exercised directly.
 *  2. PHYSICS INVARIANTS of the CPU reference solver. These are the tests that
 *     matter: volume conservation, wave celerity against sqrt(g·h), still
 *     water staying still over a sloped bed, symmetry preservation, run-up
 *     onto a dry beach, friction removing energy but not volume, and
 *     finiteness after thousands of steps. The scheme is the executable
 *     specification the WGSL mirrors, so asserting here is asserting the
 *     physics the GPU runs.
 *  3. `ShallowWaterSim` driven through a recording fake GPUDevice. Node has no
 *     WebGPU so this cannot execute WGSL; what it DOES verify is every
 *     descriptor and command — buffer sizes, dispatch counts, substep counts,
 *     teardown.
 *
 * Shader execution itself is verified on real hardware by
 * `tests/integration/water-simulation.spec.ts`, which runs the compiled WGSL
 * in Chrome against a hardware adapter, reads the field back, and compares it
 * cell by cell against this CPU reference. That comparison is what stops the
 * two implementations drifting apart.
 *
 * Tolerance note: the cell array is Float32Array, so a quantity that is
 * conserved exactly in real arithmetic drifts by float rounding. The measured
 * drift over 3000 steps on a 48×48 grid is ~7.6e-8 relative; the assertions
 * below sit an order of magnitude above that at 1e-6, not at a number tuned
 * to just barely pass.
 */

import { describe, it, expect, beforeAll } from "vitest";
import {
  ShallowWaterField,
  ShallowWaterSim,
  POOL_PRESET,
  SHORE_PRESET,
  resolveShallowWaterSpec,
  shallowWaveSpeed,
  maxStableTimeStep,
  waterCellCount,
  waterGpuBytes,
  waterWorkgroupCount,
  seedWaterField,
  slopedBed,
  packWaterSimParams,
  packWaterImpulseParams,
  splashFalloff,
  splashVolumeM3,
  waterFaceIsOpen,
  waterSurfaceNormal,
  waterTransmittance,
  waterPathLength,
  fresnelSchlick,
  WATER_SIM_WGSL,
  WATER_CELL_STRUCT_WGSL,
  WATER_CELL_STRIDE_BYTES,
  WATER_CELL_STRIDE_FLOATS,
  WATER_CELL_FIELD_OFFSETS,
  WATER_SIM_PARAMS_BYTES,
  WATER_IMPULSE_PARAMS_BYTES,
  WATER_WORKGROUP_SIZE,
  WATER_GRAVITY,
  MAX_WATER_GRID,
  DEFAULT_CFL_SAFETY,
} from "./shallow-water.js";

const O = WATER_CELL_FIELD_OFFSETS;

beforeAll(() => {
  const g = globalThis as Record<string, unknown>;
  g.GPUShaderStage ??= { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };
  g.GPUBufferUsage ??= {
    MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16,
    VERTEX: 32, UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512,
  };
});

// ──────────────────────────────────────────────────────────────────────
// Fakes
// ──────────────────────────────────────────────────────────────────────

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
    createBindGroup(desc: GPUBindGroupDescriptor) {
      bindGroups.push(desc);
      return { label: desc.label };
    },
    queue: {
      writeBuffer(buffer: FakeBuffer, offset: number, data: ArrayBufferView) {
        // Copy: the sim reuses one scratch array, and a reference would make
        // every record alias the last write.
        const view = data as Float32Array;
        writes.push({
          label: buffer.label,
          offset,
          data: new Float32Array(
            view.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength),
          ),
        });
      },
    },
  };
  return { device, buffers, writes, shaderModules, computePipelines, bindGroups };
}

function fakeComputeEncoder() {
  const dispatches: { pipeline: unknown; x: number; y: number }[] = [];
  let current: unknown = null;
  let ended = false;
  const pass = {
    setPipeline(p: unknown) { current = p; },
    setBindGroup() {},
    dispatchWorkgroups(x: number, y: number) { dispatches.push({ pipeline: current, x, y }); },
    end() { ended = true; },
  };
  return {
    encoder: { beginComputePass: () => pass } as unknown as GPUCommandEncoder,
    dispatches,
    get ended() { return ended; },
  };
}

/** A flat-bed field forced to an exact uniform depth, bypassing the preset. */
function flatField(
  width: number,
  height: number,
  cellSize: number,
  depth: number,
  options: Parameters<typeof resolveShallowWaterSpec>[0] & { damping?: number } = {},
): ShallowWaterField {
  const field = new ShallowWaterField(
    {
      width,
      height,
      cellSize,
      origin: [0, 0],
      surfaceLevel: 0,
      bedLevel: -depth,
      damping: 0,
      ...options,
    },
    POOL_PRESET,
  );
  return field;
}

// ──────────────────────────────────────────────────────────────────────
// Layout
// ──────────────────────────────────────────────────────────────────────

describe("water cell layout", () => {
  it("keeps the TS stride and the WGSL struct the same size", () => {
    expect(WATER_CELL_STRIDE_FLOATS).toBe(WATER_CELL_STRIDE_BYTES / 4);
    // Six f32 members: WGSL alignment is 4, so the struct is exactly 24 bytes
    // with no padding and the array stride matches the TS stride.
    const members = WATER_CELL_STRUCT_WGSL.match(/^\s+\w+: f32,$/gm) ?? [];
    expect(members).toHaveLength(WATER_CELL_STRIDE_FLOATS);
  });

  it("gives every field a distinct slot inside one stride", () => {
    const offsets = Object.values(WATER_CELL_FIELD_OFFSETS);
    expect(new Set(offsets).size).toBe(offsets.length);
    for (const offset of offsets) expect(offset).toBeLessThan(WATER_CELL_STRIDE_FLOATS);
  });

  it("declares the TS field names in the WGSL struct, in order", () => {
    const names = Object.entries(WATER_CELL_FIELD_OFFSETS)
      .sort((a, b) => a[1] - b[1])
      .map(([name]) => name);
    const declared = [...WATER_CELL_STRUCT_WGSL.matchAll(/^\s+(\w+): f32,$/gm)].map((m) => m[1]);
    expect(declared).toEqual(names);
  });

  it("sizes the uniforms to whole vec4 blocks", () => {
    expect(WATER_SIM_PARAMS_BYTES % 16).toBe(0);
    expect(WATER_IMPULSE_PARAMS_BYTES % 16).toBe(0);
  });

  it("reports cell bytes and workgroup counts from the grid", () => {
    expect(waterCellCount({ width: 128, height: 64 })).toBe(8192);
    expect(waterGpuBytes({ width: 128, height: 128 })).toBe(128 * 128 * 24);
    expect(waterWorkgroupCount(WATER_WORKGROUP_SIZE)).toBe(1);
    expect(waterWorkgroupCount(WATER_WORKGROUP_SIZE + 1)).toBe(2);
    expect(waterWorkgroupCount(128)).toBe(16);
  });

  it("keeps the biggest allowed field inside the playable GPU budget", () => {
    // content/budget.ts caps estimated GPU bytes at 32 MB for the Intel-UHD
    // playable floor. One max-size field must not eat the whole thing.
    expect(waterGpuBytes({ width: MAX_WATER_GRID, height: MAX_WATER_GRID })).toBeLessThan(8_000_000);
  });
});

// ──────────────────────────────────────────────────────────────────────
// Spec resolution
// ──────────────────────────────────────────────────────────────────────

describe("resolveShallowWaterSpec", () => {
  it("clamps the grid to the hard cap and floors it to integers", () => {
    expect(resolveShallowWaterSpec({ width: 1e9 }).width).toBe(MAX_WATER_GRID);
    expect(resolveShallowWaterSpec({ height: 1e9 }).height).toBe(MAX_WATER_GRID);
    expect(resolveShallowWaterSpec({ width: 64.7 }).width).toBe(64);
  });

  it("requires two cells across X but allows a one-row 1-D channel", () => {
    expect(resolveShallowWaterSpec({ width: 0 }).width).toBe(2);
    expect(resolveShallowWaterSpec({ height: 1 }).height).toBe(1);
    expect(resolveShallowWaterSpec({ height: 0 }).height).toBe(1);
  });

  it("falls back to the base value for non-finite input instead of passing NaN to the GPU", () => {
    const spec = resolveShallowWaterSpec({
      cellSize: Number.NaN,
      gravity: Number.POSITIVE_INFINITY,
      damping: Number.NaN,
      surfaceLevel: Number.NaN,
      bedLevel: Number.NaN,
      origin: [Number.NaN, 4],
    });
    expect(spec.cellSize).toBe(POOL_PRESET.cellSize);
    expect(spec.gravity).toBe(POOL_PRESET.gravity);
    expect(spec.damping).toBe(POOL_PRESET.damping);
    expect(spec.surfaceLevel).toBe(POOL_PRESET.surfaceLevel);
    expect(spec.bedLevel).toBe(POOL_PRESET.bedLevel);
    expect(spec.origin).toEqual([POOL_PRESET.origin[0], 4]);
  });

  it("permits zero gravity, which simply freezes propagation", () => {
    expect(resolveShallowWaterSpec({ gravity: 0 }).gravity).toBe(0);
    expect(maxStableTimeStep(0.25, 0, 1)).toBe(Number.POSITIVE_INFINITY);
  });

  it("ships presets that are actually wet — a dry default draws nothing", () => {
    for (const preset of [POOL_PRESET, SHORE_PRESET]) {
      expect(preset.bedLevel).toBeLessThan(preset.surfaceLevel);
    }
    expect(new ShallowWaterField().maxDepth()).toBeCloseTo(1.5, 6);
  });
});

// ──────────────────────────────────────────────────────────────────────
// CFL
// ──────────────────────────────────────────────────────────────────────

describe("wave speed and the Courant limit", () => {
  it("computes celerity as sqrt(g·h)", () => {
    expect(shallowWaveSpeed(WATER_GRAVITY, 1)).toBeCloseTo(Math.sqrt(WATER_GRAVITY), 10);
    expect(shallowWaveSpeed(9.8, 4)).toBeCloseTo(Math.sqrt(39.2), 10);
    expect(shallowWaveSpeed(9.8, 0)).toBe(0);
    // Negative depth is nonsense, not a negative speed.
    expect(shallowWaveSpeed(9.8, -4)).toBe(0);
  });

  it("puts the timestep exactly at safety × dx / (√2 · c)", () => {
    // Re-derive the Courant number from the returned step rather than
    // restating the formula: safety = dt · c · √2 / dx must come back out.
    for (const [dx, g, h, safety] of [
      [0.25, WATER_GRAVITY, 1.5, 0.5],
      [1, 9.8, 10, 0.9],
      [0.05, 3.7, 0.3, 0.25],
    ] as const) {
      const dt = maxStableTimeStep(dx, g, h, safety);
      const courant = (dt * shallowWaveSpeed(g, h) * Math.SQRT2) / dx;
      expect(courant).toBeCloseTo(safety, 10);
    }
  });

  it("tightens the step as the water deepens", () => {
    const shallow = maxStableTimeStep(0.25, WATER_GRAVITY, 0.5);
    const deep = maxStableTimeStep(0.25, WATER_GRAVITY, 8);
    expect(deep).toBeLessThan(shallow);
  });

  it("defaults to a safety factor strictly inside the stability limit", () => {
    expect(DEFAULT_CFL_SAFETY).toBeGreaterThan(0);
    expect(DEFAULT_CFL_SAFETY).toBeLessThan(1);
  });
});

// ──────────────────────────────────────────────────────────────────────
// Seeding and bed
// ──────────────────────────────────────────────────────────────────────

describe("seedWaterField", () => {
  it("fills to the surface level over the flat bed and starts at rest", () => {
    const spec = resolveShallowWaterSpec({ width: 4, height: 3, surfaceLevel: 2, bedLevel: -1 });
    const cells = seedWaterField(spec);
    expect(cells).toHaveLength(4 * 3 * WATER_CELL_STRIDE_FLOATS);
    for (let c = 0; c < cells.length; c += WATER_CELL_STRIDE_FLOATS) {
      expect(cells[c + O.height]).toBeCloseTo(3, 6);
      expect(cells[c + O.bed]).toBeCloseTo(-1, 6);
      expect(cells[c + O.velocityX]).toBe(0);
      expect(cells[c + O.velocityZ]).toBe(0);
      expect(cells[c + O.fluxX]).toBe(0);
      expect(cells[c + O.fluxZ]).toBe(0);
    }
  });

  it("starts dry where the bed rises above the fill level", () => {
    const spec = resolveShallowWaterSpec({ width: 11, height: 1, surfaceLevel: 0 });
    const cells = seedWaterField(spec, slopedBed(11, -2, 2));
    const depths = [];
    for (let i = 0; i < 11; i++) depths.push(cells[i * WATER_CELL_STRIDE_FLOATS + O.height]!);
    // Bed runs -2 → +2, so the lower half is wet and the upper half dry.
    expect(depths[0]).toBeCloseTo(2, 6);
    expect(depths[5]).toBeCloseTo(0, 6);
    expect(depths[10]).toBe(0);
    expect(depths.filter((d) => d > 0).length).toBe(5);
  });

  it("never seeds a negative depth", () => {
    const spec = resolveShallowWaterSpec({ width: 8, height: 8, surfaceLevel: -5, bedLevel: 0 });
    for (const value of seedWaterField(spec)) expect(Number.isFinite(value)).toBe(true);
    const cells = seedWaterField(spec);
    for (let c = 0; c < cells.length; c += WATER_CELL_STRIDE_FLOATS) {
      expect(cells[c + O.height]).toBe(0);
    }
  });

  it("interpolates slopedBed linearly between its endpoints", () => {
    const bed = slopedBed(5, 0, 4);
    expect(bed(0, 0)).toBeCloseTo(0, 10);
    expect(bed(2, 0)).toBeCloseTo(2, 10);
    expect(bed(4, 0)).toBeCloseTo(4, 10);
  });
});

// ──────────────────────────────────────────────────────────────────────
// Face gating
// ──────────────────────────────────────────────────────────────────────

describe("waterFaceIsOpen", () => {
  const dry = 1e-5;

  it("opens between two wet columns", () => {
    expect(waterFaceIsOpen(1, -1, 1, -1, dry)).toBe(true);
  });

  it("closes between two dry columns", () => {
    expect(waterFaceIsOpen(0, 0, 0, 1, dry)).toBe(false);
  });

  it("closes a lake against a beach whose bed is above the water line", () => {
    // Wet cell: 1 m of water on a bed at -1, so its surface is 0.
    // Dry cell: bed at +0.5, above that surface. Water cannot climb it.
    expect(waterFaceIsOpen(1, -1, 0, 0.5, dry)).toBe(false);
  });

  it("opens once a wave lifts the surface above the dry bed — this is run-up", () => {
    // Same beach, but the wave has raised the wet column to a surface of +0.8.
    expect(waterFaceIsOpen(1.8, -1, 0, 0.5, dry)).toBe(true);
  });

  it("is symmetric under swapping the two sides", () => {
    const cases: [number, number, number, number][] = [
      [1, -1, 0, 0.5],
      [1.8, -1, 0, 0.5],
      [0, 0, 0, 1],
      [1, -1, 2, -2],
      [0.5, 0.2, 0, 0.3],
    ];
    for (const [ha, ba, hb, bb] of cases) {
      expect(waterFaceIsOpen(ha, ba, hb, bb, dry)).toBe(waterFaceIsOpen(hb, bb, ha, ba, dry));
    }
  });
});

// ──────────────────────────────────────────────────────────────────────
// Splash shaping
// ──────────────────────────────────────────────────────────────────────

describe("splashFalloff", () => {
  it("is 1 at the centre, 0 at the radius and beyond", () => {
    expect(splashFalloff(0, 2)).toBeCloseTo(1, 10);
    expect(splashFalloff(2, 2)).toBeCloseTo(0, 10);
    expect(splashFalloff(5, 2)).toBeCloseTo(0, 10);
  });

  it("is the smoothstep half-way value at half the radius", () => {
    expect(splashFalloff(1, 2)).toBeCloseTo(0.5, 10);
  });

  it("decreases monotonically with distance", () => {
    let previous = Number.POSITIVE_INFINITY;
    for (let d = 0; d <= 2; d += 0.05) {
      const value = splashFalloff(d, 2);
      expect(value).toBeLessThanOrEqual(previous + 1e-12);
      previous = value;
    }
  });

  it("reports zero for a degenerate radius rather than dividing by it", () => {
    expect(splashFalloff(1, 0)).toBe(0);
    expect(splashFalloff(0, -1)).toBe(0);
  });

  it("predicts the volume a splash injects", () => {
    const field = flatField(32, 32, 0.25, 1);
    const spec = field.spec;
    const predicted = splashVolumeM3(spec, 4, 4, 1, 0.1);
    const actual = field.splash(4, 4, 1, 0.1);
    expect(actual).toBeCloseTo(predicted, 6);
    expect(predicted).toBeGreaterThan(0);
    // Negative amplitude removes water.
    expect(splashVolumeM3(spec, 4, 4, 1, -0.1)).toBeCloseTo(-predicted, 6);
  });
});

// ──────────────────────────────────────────────────────────────────────
// Normals
// ──────────────────────────────────────────────────────────────────────

describe("waterSurfaceNormal", () => {
  it("points straight up on a flat surface", () => {
    const n = waterSurfaceNormal(() => 3, 4, 4, 9, 9, 0.5);
    expect(n[0]).toBeCloseTo(0, 12);
    expect(n[1]).toBeCloseTo(1, 12);
    expect(n[2]).toBeCloseTo(0, 12);
  });

  it("is always a unit vector", () => {
    const eta = (i: number, j: number) => Math.sin(i * 0.7) * 0.4 + Math.cos(j * 0.3) * 0.9;
    for (let j = 0; j < 9; j++) {
      for (let i = 0; i < 9; i++) {
        const n = waterSurfaceNormal(eta, i, j, 9, 9, 0.25);
        expect(Math.hypot(...n)).toBeCloseTo(1, 10);
      }
    }
  });

  it("matches the exact normal of a linear ramp", () => {
    // eta = slope · x. The true normal is normalize(-slope, 1, 0), and a
    // central difference of a linear function is exact, so this must be too.
    const slope = 0.4;
    const cellSize = 0.25;
    const eta = (i: number) => slope * i * cellSize;
    const n = waterSurfaceNormal(eta, 4, 4, 9, 9, cellSize);
    const length = Math.hypot(slope, 1, 0);
    expect(n[0]).toBeCloseTo(-slope / length, 9);
    expect(n[1]).toBeCloseTo(1 / length, 9);
    expect(n[2]).toBeCloseTo(0, 12);
  });

  it("agrees with a finite difference of the same elevation field", () => {
    // The whole point of deriving normals from the displaced surface is that
    // they cannot disagree with it. Compare against an independent
    // finite-difference gradient of the same sampler.
    const cellSize = 0.2;
    const eta = (i: number, j: number) =>
      0.3 * Math.sin(i * cellSize * 1.3) + 0.2 * Math.cos(j * cellSize * 0.9);
    for (const [i, j] of [[3, 3], [5, 2], [7, 6]] as const) {
      const n = waterSurfaceNormal(eta, i, j, 12, 12, cellSize);
      const dEtaDx = (eta(i + 1, j) - eta(i - 1, j)) / (2 * cellSize);
      const dEtaDz = (eta(i, j + 1) - eta(i, j - 1)) / (2 * cellSize);
      const length = Math.hypot(dEtaDx, 1, dEtaDz);
      expect(n[0]).toBeCloseTo(-dEtaDx / length, 9);
      expect(n[1]).toBeCloseTo(1 / length, 9);
      expect(n[2]).toBeCloseTo(-dEtaDz / length, 9);
    }
  });

  it("tilts away from rising water", () => {
    // Elevation rising with +x must tip the normal toward -x.
    const n = waterSurfaceNormal((i) => i * 0.5, 4, 4, 9, 9, 0.25);
    expect(n[0]).toBeLessThan(0);
    const m = waterSurfaceNormal((i) => -i * 0.5, 4, 4, 9, 9, 0.25);
    expect(m[0]).toBeGreaterThan(0);
  });

  it("falls back to a one-sided difference at the grid edge", () => {
    const slope = 0.6;
    const cellSize = 0.5;
    const eta = (i: number) => slope * i * cellSize;
    const edge = waterSurfaceNormal(eta, 0, 0, 9, 9, cellSize);
    const interior = waterSurfaceNormal(eta, 4, 4, 9, 9, cellSize);
    // A one-sided difference of a linear function equals the central one.
    expect(edge[0]).toBeCloseTo(interior[0], 9);
    expect(Math.hypot(...edge)).toBeCloseTo(1, 10);
  });

  it("returns +Y for a single-cell grid instead of dividing by a zero span", () => {
    const n = waterSurfaceNormal(() => 1, 0, 0, 1, 1, 0.25);
    expect(n.map((value) => value === 0 ? 0 : value)).toEqual([0, 1, 0]);
  });
});

// ──────────────────────────────────────────────────────────────────────
// Optics helpers
// ──────────────────────────────────────────────────────────────────────

describe("waterTransmittance", () => {
  it("is 1 through no water at all", () => {
    expect(waterTransmittance(0.5, 0)).toBe(1);
    expect(waterTransmittance(0, 10)).toBe(1);
  });

  it("decreases strictly as the path lengthens — deep water darkens", () => {
    let previous = Number.POSITIVE_INFINITY;
    for (let d = 0; d <= 20; d += 0.25) {
      const t = waterTransmittance(0.4, d);
      expect(t).toBeLessThanOrEqual(previous);
      if (d > 0) expect(t).toBeLessThan(1);
      previous = t;
    }
  });

  it("decreases strictly as extinction rises", () => {
    expect(waterTransmittance(0.8, 2)).toBeLessThan(waterTransmittance(0.2, 2));
  });

  it("stays inside (0, 1] for every input", () => {
    for (const [k, d] of [[0.3, 5], [10, 50], [1e3, 1e3], [-1, 5], [0.3, -5]] as const) {
      const t = waterTransmittance(k, d);
      expect(t).toBeGreaterThanOrEqual(0);
      expect(t).toBeLessThanOrEqual(1);
    }
  });

  it("is exactly Beer-Lambert", () => {
    expect(waterTransmittance(0.75, 3)).toBeCloseTo(Math.exp(-2.25), 12);
  });
});

describe("waterPathLength", () => {
  it("equals the depth looking straight down", () => {
    expect(waterPathLength(2.5, 1)).toBeCloseTo(2.5, 12);
    expect(waterPathLength(2.5, -1)).toBeCloseTo(2.5, 12);
  });

  it("lengthens at a grazing angle", () => {
    expect(waterPathLength(1, 0.5)).toBeCloseTo(2, 12);
    expect(waterPathLength(1, 0.25)).toBeCloseTo(4, 12);
  });

  it("stays finite at the horizon instead of producing a black band", () => {
    const grazing = waterPathLength(1, 0, 0.08);
    expect(Number.isFinite(grazing)).toBe(true);
    expect(grazing).toBeCloseTo(1 / 0.08, 9);
  });

  it("is zero through zero depth whatever the angle", () => {
    expect(waterPathLength(0, 0.01)).toBe(0);
  });
});

describe("fresnelSchlick", () => {
  it("returns F0 looking straight down the normal", () => {
    expect(fresnelSchlick(1, 0.02)).toBeCloseTo(0.02, 12);
  });

  it("reaches total reflection at a grazing angle", () => {
    expect(fresnelSchlick(0, 0.02)).toBeCloseTo(1, 12);
  });

  it("rises monotonically as the view gets more grazing", () => {
    let previous = -1;
    for (let c = 1; c >= 0; c -= 0.05) {
      const f = fresnelSchlick(c, 0.02);
      expect(f).toBeGreaterThanOrEqual(previous);
      previous = f;
    }
  });

  it("clamps a cosine outside [0,1] rather than extrapolating", () => {
    expect(fresnelSchlick(1.5, 0.02)).toBeCloseTo(0.02, 12);
    expect(fresnelSchlick(-2, 0.02)).toBeCloseTo(1, 12);
  });
});

// ──────────────────────────────────────────────────────────────────────
// Uniform packing
// ──────────────────────────────────────────────────────────────────────

describe("packWaterSimParams", () => {
  const spec = resolveShallowWaterSpec({
    width: 64, height: 32, cellSize: 0.5, gravity: 9.8, damping: 0.25, dryDepth: 2e-5,
  });

  it("packs the documented layout", () => {
    const out = packWaterSimParams(new Float32Array(WATER_SIM_PARAMS_BYTES / 4), spec, 0.01);
    expect(out[0]).toBe(Math.fround(0.01));
    expect(out[1]).toBe(Math.fround(9.8));
    expect(out[2]).toBeCloseTo(0.5, 10);
    expect(out[4]).toBe(64);
    expect(out[5]).toBe(32);
    expect(out[6]).toBe(Math.fround(2e-5));
  });

  it("precomputes friction as exp(-damping·dt) so the shader has one definition", () => {
    const out = packWaterSimParams(new Float32Array(8), spec, 0.02);
    expect(out[3]).toBe(Math.fround(Math.exp(-0.25 * 0.02)));
    // Zero damping must be exactly 1, not 0.9999…, or a long run would decay.
    const undamped = resolveShallowWaterSpec({ damping: 0 }, spec);
    expect(packWaterSimParams(new Float32Array(8), undamped, 0.02)[3]).toBe(1);
  });

  it("leaves the alignment padding slot zero", () => {
    // Pinned so a future parameter cannot be smuggled into the padding and
    // then applied nowhere in the shader.
    const out = packWaterSimParams(new Float32Array(8), spec, 0.01);
    expect(out[7]).toBe(0);
  });

  it("throws rather than writing past a short buffer", () => {
    expect(() => packWaterSimParams(new Float32Array(4), spec, 0.01)).toThrow(/needs 8 floats/);
  });
});

describe("packWaterImpulseParams", () => {
  const spec = resolveShallowWaterSpec({ width: 32, height: 16, cellSize: 0.25, origin: [-4, -2] });

  it("converts the splash centre to grid-relative metres", () => {
    const out = packWaterImpulseParams(new Float32Array(8), spec, 1, 3, 0.75, 0.2);
    expect(out[0]).toBeCloseTo(5, 10);
    expect(out[1]).toBeCloseTo(5, 10);
    expect(out[2]).toBeCloseTo(0.75, 10);
    expect(out[3]).toBe(Math.fround(0.2));
    expect(out[4]).toBe(32);
    expect(out[5]).toBe(16);
    expect(out[6]).toBeCloseTo(0.25, 10);
    expect(out[7]).toBe(0);
  });

  it("refuses a negative radius", () => {
    expect(packWaterImpulseParams(new Float32Array(8), spec, 0, 0, -3, 1)[2]).toBe(0);
  });

  it("throws rather than writing past a short buffer", () => {
    expect(() => packWaterImpulseParams(new Float32Array(4), spec, 0, 0, 1, 1)).toThrow(/needs 8 floats/);
  });
});

// ══════════════════════════════════════════════════════════════════════
// PHYSICS INVARIANTS
// ══════════════════════════════════════════════════════════════════════

describe("shallow-water physics: volume conservation", () => {
  it("conserves volume to float rounding over 3000 steps on a closed domain", () => {
    const field = flatField(48, 48, 0.3, 2);
    field.splash(7.2, 7.2, 1.5, 0.3);
    const start = field.totalVolumeM3();
    for (let s = 0; s < 3000; s++) field.stepOnce(0.004);
    const end = field.totalVolumeM3();
    // Nothing dried out, so the clamp never fired and the only error is the
    // Float32 rounding of each per-cell update.
    expect(field.clampedVolumeM3).toBe(0);
    expect(Math.abs(end - start) / start).toBeLessThan(1e-6);
    expect(field.nonFiniteCount()).toBe(0);
  });

  it("accounts for every cubic metre the dry clamp invents", () => {
    // A violent splash into shallow water over a beach WILL dry cells out and
    // the clamp will create water. The invariant is not that it never happens
    // — it is that the solver reports exactly how much.
    const width = 60;
    const field = new ShallowWaterField(
      {
        width, height: 4, cellSize: 0.25, origin: [0, 0], surfaceLevel: 0,
        damping: 0.02, bed: slopedBed(width, -2, 1),
      },
      POOL_PRESET,
    );
    const start = field.totalVolumeM3();
    field.splash(2, 0.5, 1.2, 0.55);
    for (let s = 0; s < 1200; s++) field.stepOnce(0.003);
    const drift = field.totalVolumeM3() - start - field.injectedVolumeM3;
    expect(field.clampedVolumeM3).toBeGreaterThan(0);
    expect(Math.abs(drift - field.clampedVolumeM3)).toBeLessThan(1e-4);
    expect(field.nonFiniteCount()).toBe(0);
  });

  it("never lets a column go negative", () => {
    const field = flatField(32, 32, 0.25, 0.3);
    field.splash(4, 4, 1, -0.9);
    for (let s = 0; s < 600; s++) field.stepOnce(0.003);
    for (let j = 0; j < 32; j++) {
      for (let i = 0; i < 32; i++) expect(field.depthAt(i, j)).toBeGreaterThanOrEqual(0);
    }
  });

  it("holds water in: no discharge crosses a domain wall", () => {
    const field = flatField(24, 24, 0.25, 1);
    field.splash(1, 1, 0.8, 0.4);
    for (let s = 0; s < 400; s++) {
      field.stepOnce(0.004);
      for (let j = 0; j < 24; j++) {
        // The +X face of the last column and the +Z face of the last row are
        // the reflecting boundary and must carry nothing, ever.
        const lastColumn = (j * 24 + 23) * WATER_CELL_STRIDE_FLOATS;
        expect(field.cells[lastColumn + O.fluxX]).toBe(0);
        expect(field.cells[lastColumn + O.velocityX]).toBe(0);
      }
      for (let i = 0; i < 24; i++) {
        const lastRow = (23 * 24 + i) * WATER_CELL_STRIDE_FLOATS;
        expect(field.cells[lastRow + O.fluxZ]).toBe(0);
        expect(field.cells[lastRow + O.velocityZ]).toBe(0);
      }
    }
  });
});

describe("shallow-water physics: wave propagation", () => {
  it("propagates a crest at the shallow-water celerity sqrt(g·h)", () => {
    // 1-D channel, 40 m long, 1 m deep, no friction. A small bump splits into
    // two counter-propagating halves; track the right-going crest between two
    // times so the initial splash radius cancels out of the speed.
    const width = 401;
    const cellSize = 0.1;
    const field = flatField(width, 1, cellSize, 1, { dryDepth: 0 });
    const centre = 200 * cellSize;
    field.splash(centre, 0, 0.5, 0.02);

    const crestOffset = (): number => {
      let best = -1;
      let bestDepth = Number.NEGATIVE_INFINITY;
      // Start clear of the initial bump so the stationary centre peak is not
      // mistaken for the travelling crest.
      for (let i = 210; i < width; i++) {
        const depth = field.depthAt(i, 0);
        if (depth > bestDepth) { bestDepth = depth; best = i; }
      }
      return best * cellSize - centre;
    };

    const dt = 0.005;
    const steps = Math.round(1 / dt);
    for (let s = 0; s < steps; s++) field.stepOnce(dt);
    const first = crestOffset();
    for (let s = 0; s < steps; s++) field.stepOnce(dt);
    const second = crestOffset();

    const measured = second - first; // metres per second, over exactly 1 s
    const expected = shallowWaveSpeed(WATER_GRAVITY, 1);
    // Measured 3.100 vs 3.132 m/s: ~1 % low from numerical dispersion, and
    // the crest position is quantised to one cell (0.1 m ≈ 3 %) over the 1 s
    // window. 8 % is a real constraint, not a rubber stamp.
    expect(measured / expected).toBeGreaterThan(0.92);
    expect(measured / expected).toBeLessThan(1.08);
    // It really did travel, and it stayed inside the numerical light cone
    // (one cell per step is the absolute maximum this scheme can move).
    expect(second).toBeGreaterThan(first);
    expect(measured).toBeLessThan(cellSize / dt);
  });

  it("speeds waves up in deeper water", () => {
    const run = (depth: number): number => {
      const width = 301;
      const cellSize = 0.1;
      const field = flatField(width, 1, cellSize, depth, { dryDepth: 0 });
      const centre = 150 * cellSize;
      field.splash(centre, 0, 0.4, depth * 0.01);
      for (let s = 0; s < 150; s++) field.stepOnce(0.002);
      let best = -1;
      let bestDepth = Number.NEGATIVE_INFINITY;
      for (let i = 155; i < width; i++) {
        const d = field.depthAt(i, 0);
        if (d > bestDepth) { bestDepth = d; best = i; }
      }
      return best * cellSize - centre;
    };
    // 4× the depth is 2× the celerity, so in the same time the crest gets
    // meaningfully further. This is shoaling, the physical reason waves slow
    // and steepen as they approach a shore.
    expect(run(4)).toBeGreaterThan(run(1) * 1.5);
  });

  it("reflects off a wall instead of absorbing or leaking", () => {
    const width = 201;
    const cellSize = 0.1;
    const field = flatField(width, 1, cellSize, 1, { dryDepth: 0 });
    const start = field.totalVolumeM3();
    field.splash(10, 0, 0.5, 0.05);
    const injected = field.injectedVolumeM3;
    // Long enough for the right-going crest to hit the far wall and come back.
    for (let s = 0; s < 3000; s++) field.stepOnce(0.004);
    // The disturbance is still in the domain: nothing left through the walls.
    expect(Math.abs(field.totalVolumeM3() - start - injected) / start).toBeLessThan(1e-6);
    expect(field.disturbanceM3()).toBeGreaterThan(0);
  });
});

describe("shallow-water physics: equilibrium and symmetry", () => {
  it("leaves still water over a SLOPED bed exactly still", () => {
    // The scheme takes its pressure gradient on the surface elevation, not on
    // the depth, and gates faces against a dry bed above the water line. Both
    // are needed: either one missing and a lake spontaneously generates a
    // current down the slope or pushes itself off the beach.
    const width = 40;
    const field = new ShallowWaterField(
      {
        width, height: 8, cellSize: 0.25, origin: [0, 0], surfaceLevel: 0,
        damping: 0.1, bed: slopedBed(width, -2, 1),
      },
      POOL_PRESET,
    );
    const start = field.totalVolumeM3();
    for (let s = 0; s < 500; s++) {
      field.stepOnce(0.004);
      for (let c = 0; c < field.cells.length; c += WATER_CELL_STRIDE_FLOATS) {
        expect(field.cells[c + O.velocityX]).toBe(0);
        expect(field.cells[c + O.velocityZ]).toBe(0);
      }
    }
    expect(field.totalVolumeM3()).toBe(start);
    expect(field.clampedVolumeM3).toBe(0);
    // The slope really did produce a shoreline rather than a uniform pool.
    const wet = Array.from({ length: width }, (_, i) => field.depthAt(i, 0)).filter((d) => d > 0);
    expect(wet.length).toBeGreaterThan(0);
    expect(wet.length).toBeLessThan(width);
  });

  it("leaves still water over a flat bed exactly still", () => {
    const field = flatField(16, 16, 0.25, 1, { damping: 0.3 });
    for (let s = 0; s < 200; s++) field.stepOnce(0.004);
    for (let j = 0; j < 16; j++) {
      for (let i = 0; i < 16; i++) expect(field.depthAt(i, j)).toBe(1);
    }
    expect(field.disturbanceM3()).toBe(0);
  });

  it("preserves the symmetry of a centred splash exactly", () => {
    // Mirroring the grid mirrors the arithmetic: IEEE addition is commutative
    // and a - b is a + (-b), so the mirrored update is bit-identical, not
    // merely close. Any change that breaks this has broken the scheme's
    // isotropy.
    const width = 65;
    const cellSize = 0.25;
    const field = flatField(width, width, cellSize, 1);
    const centre = 32 * cellSize;
    field.splash(centre, centre, 1, 0.05);
    for (let s = 0; s < 300; s++) field.stepOnce(0.004);
    let mirrorX = 0;
    let mirrorZ = 0;
    let transpose = 0;
    for (let j = 0; j < width; j++) {
      for (let i = 0; i < width; i++) {
        const here = field.depthAt(i, j);
        mirrorX = Math.max(mirrorX, Math.abs(here - field.depthAt(width - 1 - i, j)));
        mirrorZ = Math.max(mirrorZ, Math.abs(here - field.depthAt(i, width - 1 - j)));
        transpose = Math.max(transpose, Math.abs(here - field.depthAt(j, i)));
      }
    }
    expect(mirrorX).toBe(0);
    expect(mirrorZ).toBe(0);
    expect(transpose).toBe(0);
  });
});

describe("shallow-water physics: friction and stability", () => {
  it("lets friction remove the disturbance while leaving the volume alone", () => {
    const field = flatField(64, 64, 0.25, 1.5, { damping: 0.8 });
    const start = field.totalVolumeM3();
    field.splash(8, 8, 1, 0.4);
    const injected = field.injectedVolumeM3;
    const before = field.disturbanceM3();
    for (let s = 0; s < 5000; s++) field.stepOnce(0.004);
    const after = field.disturbanceM3();
    expect(before).toBeGreaterThan(0);
    // Measured: 0.747 → 0.0008 m³ of surface deviation. Energy goes, water stays.
    expect(after).toBeLessThan(before * 0.01);
    expect(Math.abs(field.totalVolumeM3() - start - injected) / start).toBeLessThan(1e-6);
  });

  it("keeps an undamped field running instead of silently decaying", () => {
    const field = flatField(48, 48, 0.25, 1, { damping: 0 });
    field.splash(6, 6, 1, 0.2);
    const before = field.disturbanceM3();
    for (let s = 0; s < 2000; s++) field.stepOnce(0.004);
    // Without friction the wave energy stays in the domain. It redistributes,
    // so this is not an exact equality, but it must not have collapsed.
    expect(field.disturbanceM3()).toBeGreaterThan(before * 0.3);
  });

  it("stays finite after 5000 steps with a violent initial condition", () => {
    const field = flatField(64, 64, 0.25, 1.5, { damping: 0.1 });
    for (let k = 0; k < 8; k++) field.splash(2 + k * 1.7, 3 + k * 1.1, 0.8, 0.5);
    for (let s = 0; s < 5000; s++) field.stepOnce(0.004);
    expect(field.nonFiniteCount()).toBe(0);
    // Bounded, not merely finite: nothing has run away.
    expect(field.maxDepth()).toBeLessThan(20);
  });
});

describe("shallow-water physics: run-up onto a dry bed", () => {
  it("wets cells that started dry when a wave arrives", () => {
    const width = 60;
    const field = new ShallowWaterField(
      {
        width, height: 4, cellSize: 0.25, origin: [0, 0], surfaceLevel: 0,
        damping: 0.02, bed: slopedBed(width, -2, 1),
      },
      POOL_PRESET,
    );
    let firstDry = -1;
    for (let i = 0; i < width; i++) {
      if (field.depthAt(i, 0) <= 0) { firstDry = i; break; }
    }
    expect(firstDry).toBeGreaterThan(0);
    field.splash(2, 0.5, 1.2, 0.55);
    for (let s = 0; s < 1200; s++) field.stepOnce(0.003);
    let wetted = 0;
    for (let i = firstDry; i < width; i++) {
      if (field.depthAt(i, 0) > 1e-4) wetted++;
    }
    // A height field cannot break, but it can run water up a beach.
    expect(wetted).toBeGreaterThan(0);
  });
});

describe("shallow-water timing", () => {
  it("substeps a host frame at the CFL step through the shared fixed clock", () => {
    const field = flatField(32, 32, 0.25, 1, { maxSubsteps: 8 });
    expect(field.substepSeconds).toBeGreaterThan(0);
    expect(field.substepSeconds).toBeLessThanOrEqual(1 / 60);
    const report = field.advance(field.substepSeconds * 3);
    expect(report.steps).toBe(3);
    expect(report.droppedSeconds).toBe(0);
  });

  it("reports overload instead of hiding it in a bigger dt", () => {
    const field = flatField(32, 32, 0.25, 1, { maxSubsteps: 4 });
    const report = field.advance(1);
    expect(report.steps).toBe(4);
    // 1 s asked for, 4 substeps granted — the rest is booked, not absorbed.
    expect(report.droppedSeconds).toBeCloseTo(1 - field.substepSeconds * 4, 6);
    expect(field.droppedSeconds).toBe(report.droppedSeconds);
  });

  it("takes no step at all when the field is paused", () => {
    const field = flatField(16, 16, 0.25, 1, { running: false });
    field.splash(2, 2, 0.5, 0.3);
    const snapshot = field.cells.slice();
    expect(field.advance(1 / 60).steps).toBe(0);
    expect(field.cells).toEqual(snapshot);
  });

  it("ignores a non-finite or negative frame dt rather than throwing", () => {
    const field = flatField(16, 16, 0.25, 1);
    expect(field.advance(Number.NaN).steps).toBe(0);
    expect(field.advance(-1).steps).toBe(0);
    expect(field.nonFiniteCount()).toBe(0);
  });

  it("ignores a non-finite substep instead of poisoning the field", () => {
    const field = flatField(8, 8, 0.25, 1);
    const snapshot = field.cells.slice();
    field.stepOnce(Number.NaN);
    field.stepOnce(0);
    field.stepOnce(-0.01);
    expect(field.cells).toEqual(snapshot);
  });
});

describe("ShallowWaterField accessors", () => {
  it("reports elevation as bed + depth", () => {
    const field = flatField(8, 8, 0.25, 1.25);
    expect(field.bedAt(3, 3)).toBeCloseTo(-1.25, 6);
    expect(field.depthAt(3, 3)).toBeCloseTo(1.25, 6);
    expect(field.elevationAt(3, 3)).toBeCloseTo(0, 6);
  });

  it("reports a flat still surface as a +Y normal", () => {
    const field = flatField(8, 8, 0.25, 1);
    expect(field.normalAt(4, 4).map((value) => value === 0 ? 0 : value)).toEqual([0, 1, 0]);
  });

  it("tilts the normal where a splash has raised the surface", () => {
    const field = flatField(32, 32, 0.25, 1);
    field.splash(4, 4, 1, 0.3);
    const n = field.normalAt(20, 16);
    expect(Math.hypot(...n)).toBeCloseTo(1, 6);
    // Uphill is toward the splash at smaller i, so the normal tips toward +x.
    expect(n[0]).toBeGreaterThan(0);
  });

  it("refuses a splash with a degenerate radius or a non-finite amplitude", () => {
    const field = flatField(8, 8, 0.25, 1);
    expect(field.splash(1, 1, 0, 0.2)).toBe(0);
    expect(field.splash(1, 1, -1, 0.2)).toBe(0);
    expect(field.splash(1, 1, 1, Number.NaN)).toBe(0);
    expect(field.injectedVolumeM3).toBe(0);
  });
});

// ──────────────────────────────────────────────────────────────────────
// WGSL text guards
// ──────────────────────────────────────────────────────────────────────

describe("solver WGSL", () => {
  it("avoids WGSL reserved words as identifiers", () => {
    // Regression guard. `active` is a reserved WGSL keyword and Tint rejected
    // `let active = ...` outright — caught only by the real-adapter probe, not
    // by any Node test, so pin the whole reserved list we have tripped over.
    // `half`, `filter` and `sample` are reserved for future use.
    const reserved = ["active", "half", "filter", "sample", "common", "binding", "typedef"];
    for (const word of reserved) {
      expect(WATER_SIM_WGSL, `WGSL reserved word "${word}" used as an identifier`).not.toMatch(
        new RegExp(`\\b(let|var|const)\\s+${word}\\b`),
      );
      expect(WATER_SIM_WGSL, `WGSL reserved word "${word}" used as a function name`).not.toMatch(
        new RegExp(`\\bfn\\s+${word}\\b`),
      );
    }
  });

  it("declares the three entry points the host dispatches", () => {
    for (const entry of ["flux_main", "height_main", "impulse_main"]) {
      expect(WATER_SIM_WGSL).toContain(`fn ${entry}(`);
    }
  });

  it("uses the 2-D workgroup size the host dispatches against", () => {
    expect(WATER_SIM_WGSL).toContain(
      `@compute @workgroup_size(${WATER_WORKGROUP_SIZE}, ${WATER_WORKGROUP_SIZE})`,
    );
    // Once per entry point.
    const matches = WATER_SIM_WGSL.match(/@compute @workgroup_size/g) ?? [];
    expect(matches).toHaveLength(3);
  });

  it("bounds-checks the grid so a partial final workgroup cannot write past the end", () => {
    const guards = WATER_SIM_WGSL.match(/if \(i >= width \|\| j >= gridHeight\)/g) ?? [];
    expect(guards.length).toBeGreaterThanOrEqual(2);
  });

  it("takes the pressure gradient on the SURFACE, not on the depth", () => {
    // Using depth here is the classic bug that makes a sloped bed generate a
    // spurious current. The elevation must include the bed.
    expect(WATER_SIM_WGSL).toContain("let etaNext = bedNext + depthNext;");
    expect(WATER_SIM_WGSL).toContain("let etaHere = bedHere + depthHere;");
    expect(WATER_SIM_WGSL).toContain("gravity * dt * (etaNext - etaHere) / cellSize");
  });

  it("mirrors the CPU reference's wet/dry face gate", () => {
    expect(WATER_SIM_WGSL).toContain("fn face_is_open(");
    expect(WATER_SIM_WGSL).toContain("depthA + bedA > bedB");
    expect(WATER_SIM_WGSL).toContain("depthB + bedB > bedA");
  });

  it("applies friction as the precomputed exponential factor, not 1 - k·dt", () => {
    const occurrences = WATER_SIM_WGSL.match(/\* dampingFactor/g) ?? [];
    expect(occurrences).toHaveLength(2);
    expect(WATER_SIM_WGSL).not.toContain("1.0 - damping");
  });

  it("writes h only from discharges, never from a neighbour's height", () => {
    // This is what makes the two passes race-free in place. The height pass
    // must read fluxX/fluxZ and nothing else from its neighbours.
    const heightPass = WATER_SIM_WGSL.slice(
      WATER_SIM_WGSL.indexOf("fn height_main("),
      WATER_SIM_WGSL.indexOf("fn splash_falloff("),
    );
    expect(heightPass).toContain(".fluxX");
    expect(heightPass).toContain(".fluxZ");
    expect(heightPass).not.toContain("].height;");
    expect(heightPass).toContain("max(next, 0.0)");
  });

  it("zeroes the boundary faces so the domain holds water in", () => {
    const fluxPass = WATER_SIM_WGSL.slice(
      WATER_SIM_WGSL.indexOf("fn flux_main("),
      WATER_SIM_WGSL.indexOf("fn height_main("),
    );
    expect(fluxPass).toContain("cells[here].fluxX = 0.0;");
    expect(fluxPass).toContain("cells[here].fluxZ = 0.0;");
  });

  it("mirrors the TS splash falloff so CPU and GPU inject the same bump", () => {
    expect(WATER_SIM_WGSL).toContain("let t = 1.0 - clamp(distance / radius, 0.0, 1.0);");
    expect(WATER_SIM_WGSL).toContain("return t * t * (3.0 - 2.0 * t);");
  });
});

// ──────────────────────────────────────────────────────────────────────
// GPU solver descriptors
// ──────────────────────────────────────────────────────────────────────

function makeSim() {
  const gpu = fakeGpu();
  const sim = new ShallowWaterSim(gpu.device as unknown as GPUDevice);
  return { ...gpu, sim };
}

describe("ShallowWaterSim pipelines", () => {
  it("compiles all three compute pipelines from one shader module", () => {
    const { computePipelines, shaderModules } = makeSim();
    expect(computePipelines).toHaveLength(3);
    expect(shaderModules).toHaveLength(1);
    expect(shaderModules[0]!.code).toBe(WATER_SIM_WGSL);
    expect(computePipelines.map((p) => p.compute.entryPoint)).toEqual([
      "flux_main", "height_main", "impulse_main",
    ]);
  });

  it("shares one pipeline layout so bind groups survive a pipeline switch", () => {
    // The sim alternates flux → height inside a single compute pass without
    // re-binding. That is only legal while the layouts are compatible.
    const { computePipelines } = makeSim();
    const layouts = new Set(computePipelines.map((p) => p.layout));
    expect(layouts.size).toBe(1);
  });
});

describe("ShallowWaterSim fields", () => {
  it("allocates a cell buffer of width × height × stride plus two uniforms", () => {
    const { buffers, sim } = makeSim();
    sim.attach("pool", { width: 32, height: 16 });
    expect(buffers.map((b) => b.size)).toEqual([
      32 * 16 * WATER_CELL_STRIDE_BYTES,
      WATER_SIM_PARAMS_BYTES,
      WATER_IMPULSE_PARAMS_BYTES,
    ]);
  });

  it("marks the cell buffer copyable so the field can be verified on hardware", () => {
    const { buffers, sim } = makeSim();
    sim.attach("pool", { width: 8, height: 8 });
    const usage = buffers[0]!.usage;
    expect(usage & GPUBufferUsage.STORAGE).toBeTruthy();
    expect(usage & GPUBufferUsage.COPY_SRC).toBeTruthy();
    expect(usage & GPUBufferUsage.COPY_DST).toBeTruthy();
  });

  it("uploads a seeded field rather than leaving the buffer zeroed", () => {
    const { writes, sim } = makeSim();
    sim.attach("pool", { width: 4, height: 4, surfaceLevel: 0, bedLevel: -2 });
    const seed = writes.find((w) => w.label === "water/cells[pool]");
    expect(seed).toBeTruthy();
    expect(seed!.data).toHaveLength(4 * 4 * WATER_CELL_STRIDE_FLOATS);
    expect(seed!.data[O.height]).toBeCloseTo(2, 6);
    expect(seed!.data[O.bed]).toBeCloseTo(-2, 6);
  });

  it("is idempotent: re-attaching the same id replaces the field", () => {
    const { buffers, sim } = makeSim();
    sim.attach("pool", { width: 8, height: 8 });
    sim.attach("pool", { width: 16, height: 16 });
    expect(sim.fieldCount).toBe(1);
    // The first field's three buffers were released.
    expect(buffers.slice(0, 3).every((b) => b.destroyed)).toBe(true);
  });

  it("reports per-field bytes and the CFL substep", () => {
    const { sim } = makeSim();
    sim.attach("pool", { width: 32, height: 32, cellSize: 0.25, surfaceLevel: 0, bedLevel: -1 });
    const stats = sim.getStats("pool")!;
    expect(stats.gpuBytes).toBe(32 * 32 * WATER_CELL_STRIDE_BYTES);
    expect(stats.substepSeconds).toBeGreaterThan(0);
    expect(stats.substepSeconds).toBeLessThanOrEqual(1 / 60);
    expect(stats.droppedSeconds).toBe(0);
    expect(sim.gpuBytes()).toBe(stats.gpuBytes);
  });

  it("releases every buffer on detach and destroy", () => {
    const { buffers, sim } = makeSim();
    sim.attach("a", { width: 8, height: 8 });
    sim.attach("b", { width: 8, height: 8 });
    sim.detach("a");
    expect(sim.fieldCount).toBe(1);
    sim.destroy();
    expect(sim.fieldCount).toBe(0);
    expect(buffers.every((b) => b.destroyed)).toBe(true);
  });

  it("tolerates detaching an unknown id", () => {
    const { sim } = makeSim();
    expect(() => sim.detach("nope")).not.toThrow();
    expect(sim.getSpec("nope")).toBeNull();
    expect(sim.cellBufferFor("nope")).toBeNull();
    expect(sim.getStats("nope")).toBeNull();
    expect(sim.splash("nope", 0, 0, 1, 1)).toBe(false);
  });

  it("reallocates when the grid geometry changes and patches in place otherwise", () => {
    const { sim } = makeSim();
    sim.attach("pool", { width: 16, height: 16 });
    const before = sim.cellBufferFor("pool");
    expect(sim.update("pool", { damping: 0.5 })!.damping).toBe(0.5);
    expect(sim.cellBufferFor("pool")).toBe(before);
    expect(sim.update("pool", { width: 32 })!.width).toBe(32);
    expect(sim.cellBufferFor("pool")).not.toBe(before);
    expect(sim.update("missing", { damping: 0.1 })).toBeNull();
  });
});

describe("ShallowWaterSim dispatch", () => {
  it("preserves an authored shoreline when water level restarts the field", () => {
    const { sim, writes } = makeSim();
    const bed = (x: number) => x * 0.1;
    const spec = sim.attach("shore", { width: 8, height: 8, origin: [0, 0], bed, surfaceLevel: 1 });
    sim.update("shore", { surfaceLevel: 2 });
    const seed = writes.filter((write) => write.label === "water/cells[shore]").at(-1)!.data;
    expect(seed[6 + O.bed]).toBeCloseTo(0.1, 6);
    expect(seed[6 + O.height]).toBeCloseTo(1.9, 6);
  });
  it("dispatches a flux/height pair per substep over the whole grid", () => {
    const { sim } = makeSim();
    const spec = sim.attach("pool", { width: 32, height: 16, maxSubsteps: 8 });
    const substep = sim.getStats("pool")!.substepSeconds;
    const recording = fakeComputeEncoder();
    const { encoder, dispatches } = recording;
    sim.simulate(encoder, substep * 3);
    expect(sim.getStats("pool")!.lastSteps).toBe(3);
    // 3 substeps × 2 passes.
    expect(dispatches).toHaveLength(6);
    for (const d of dispatches) {
      expect(d.x).toBe(waterWorkgroupCount(spec.width));
      expect(d.y).toBe(waterWorkgroupCount(spec.height));
    }
    // Alternating pipelines, flux first.
    expect(dispatches[0]!.pipeline).not.toBe(dispatches[1]!.pipeline);
    expect(dispatches[0]!.pipeline).toBe(dispatches[2]!.pipeline);
    expect(recording.ended).toBe(true);
  });

  it("writes the sim uniform once per frame, not once per substep", () => {
    const { writes, sim } = makeSim();
    sim.attach("pool", { width: 16, height: 16 });
    const substep = sim.getStats("pool")!.substepSeconds;
    const before = writes.filter((w) => w.label === "water/sim-params[pool]").length;
    sim.simulate(fakeComputeEncoder().encoder, substep * 4);
    const after = writes.filter((w) => w.label === "water/sim-params[pool]").length;
    expect(after - before).toBe(1);
  });

  it("encodes nothing for a paused field but still applies a queued splash", () => {
    const { sim } = makeSim();
    sim.attach("pool", { width: 16, height: 16, running: false });
    const first = fakeComputeEncoder();
    sim.simulate(first.encoder, 1 / 60);
    expect(first.dispatches).toHaveLength(0);
    expect(sim.splash("pool", 0, 0, 1, 0.2)).toBe(true);
    const second = fakeComputeEncoder();
    sim.simulate(second.encoder, 1 / 60);
    // One impulse dispatch, no integration.
    expect(second.dispatches).toHaveLength(1);
  });

  it("drains the impulse queue so a splash is applied exactly once", () => {
    const { sim } = makeSim();
    sim.attach("pool", { width: 16, height: 16 });
    sim.splash("pool", 1, 1, 0.5, 0.3);
    sim.splash("pool", 2, 2, 0.5, 0.3);
    const first = fakeComputeEncoder();
    sim.simulate(first.encoder, 0);
    expect(first.dispatches).toHaveLength(2);
    const second = fakeComputeEncoder();
    sim.simulate(second.encoder, 0);
    expect(second.dispatches).toHaveLength(0);
  });

  it("rejects a non-finite splash argument", () => {
    const { sim } = makeSim();
    sim.attach("pool", { width: 8, height: 8 });
    expect(sim.splash("pool", Number.NaN, 0, 1, 1)).toBe(false);
    expect(sim.splash("pool", 0, 0, 0, 1)).toBe(false);
    expect(sim.splash("pool", 0, 0, 1, Number.POSITIVE_INFINITY)).toBe(false);
  });

  it("clamps a pathological frame dt instead of dispatching forever", () => {
    const { sim } = makeSim();
    sim.attach("pool", { width: 16, height: 16, maxSubsteps: 4 });
    const { encoder, dispatches } = fakeComputeEncoder();
    sim.simulate(encoder, 1e6);
    expect(dispatches).toHaveLength(8); // 4 substeps × 2 passes, and no more.
    expect(sim.getStats("pool")!.droppedSeconds).toBeGreaterThan(0);
  });

  it("encodes nothing at all when there are no fields", () => {
    const { sim } = makeSim();
    const { encoder, dispatches, ended } = fakeComputeEncoder();
    sim.simulate(encoder, 1 / 60);
    expect(dispatches).toHaveLength(0);
    expect(ended).toBe(false);
  });

  it("takes no step for a zero or non-finite frame dt", () => {
    const { sim } = makeSim();
    sim.attach("pool", { width: 16, height: 16 });
    const { encoder, dispatches } = fakeComputeEncoder();
    sim.simulate(encoder, 0);
    sim.simulate(encoder, Number.NaN);
    expect(dispatches).toHaveLength(0);
  });
});
