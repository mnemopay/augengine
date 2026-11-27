import { describe, expect, it } from "vitest";
import { DRAW_RECORD_FLOATS, INSTANCE_MIN_GROUP, instanceCapacity, packDrawRecord, planInstanceGroups } from "./instanced-draws.js";
import { buildWgsl, canInstance, pipelineKeyString, type PipelineKey } from "./pipeline-factory.js";

describe("instanced draws: grouping", () => {
  it("batches groups at or above the threshold and leaves small ones per entity", () => {
    const c = (key: string, id: string) => ({ key, item: id });
    const plan = planInstanceGroups([
      c("box", "a"), c("pillar", "p1"), c("box", "b"), c("box", "c"),
      c("dome", "d1"), c("box", "d"), c("pillar", "p2"), c("box", "e"),
    ]);
    expect(INSTANCE_MIN_GROUP).toBe(4);
    expect(plan.groups).toEqual([{ key: "box", items: ["a", "b", "c", "d", "e"] }]);
    // Small groups fall back, in input order.
    expect(plan.singles).toEqual(["p1", "p2", "d1"]);
  });

  it("a hidden (culled) entity is simply not offered, so it never reaches the instance buffer", () => {
    const visible = ["a", "b", "c", "d", "e"].filter((id) => id !== "c");
    const plan = planInstanceGroups(visible.map((id) => ({ key: "box", item: id })));
    expect(plan.groups[0]!.items).toEqual(["a", "b", "d", "e"]);
    expect(plan.groups[0]!.items).not.toContain("c");
  });

  it("grows storage capacity by powers of two (min 16)", () => {
    expect(instanceCapacity(1)).toBe(16);
    expect(instanceCapacity(16)).toBe(16);
    expect(instanceCapacity(17)).toBe(32);
    expect(instanceCapacity(300)).toBe(512);
  });

  it("packs a record in the same layout as the per-entity draw uniform", () => {
    const out = new Float32Array(2 * DRAW_RECORD_FLOATS);
    const model = new Float32Array(16).map((_, i) => i + 1);
    packDrawRecord(out, 1, model, [0.1, 0.2, 0.3], [0.5, 0.6, 0, 0], [2, 0, 0]);
    const rec = out.subarray(DRAW_RECORD_FLOATS);
    expect(Array.from(rec.subarray(0, 16))).toEqual(Array.from(model));
    expect(rec[16]).toBeCloseTo(0.1); expect(rec[19]).toBe(1);
    expect(rec[20]).toBeCloseTo(0.5); expect(rec[21]).toBeCloseTo(0.6);
    expect(rec[24]).toBe(2); expect(rec[27]).toBe(0);
    // Slot 0 untouched.
    expect(Array.from(out.subarray(0, DRAW_RECORD_FLOATS)).every((v) => v === 0)).toBe(true);
  });
});

describe("instanced draws: pipeline variant", () => {
  const base: PipelineKey = { materialKind: "pbr", vertexStride: 8, sampleCount: 1 };

  it("has its own cache key", () => {
    expect(pipelineKeyString({ ...base, instanced: true })).toBe(`${pipelineKeyString(base)}|instanced`);
  });

  it("only exists for rigid, untextured unlit/pbr/cel", () => {
    expect(canInstance(base)).toBe(true);
    expect(canInstance({ ...base, materialKind: "unlit" })).toBe(true);
    expect(canInstance({ ...base, materialKind: "cel" })).toBe(true);
    expect(canInstance({ ...base, materialKind: "outline" })).toBe(false);
    expect(canInstance({ ...base, materialKind: "vrm-mtoon" })).toBe(false);
    expect(canInstance({ ...base, hasMorphTargets: true })).toBe(false);
    expect(canInstance({ ...base, hasVertexColors: true })).toBe(true);
    expect(canInstance({ ...base, hasVertexColors: true, hasMorphTargets: true })).toBe(false);
    expect(canInstance({ ...base, hasPbrBaseColorTexture: true })).toBe(false);
    expect(canInstance({ ...base, vertexStride: 16 })).toBe(false);
    expect(() => buildWgsl({ ...base, materialKind: "outline", instanced: true })).toThrow();
  });

  it("vertex-coloured instanced variant keeps the colour stream and its own varying slots", () => {
    for (const materialKind of ["pbr", "unlit", "cel"] as const) {
      const inst = buildWgsl({ ...base, materialKind, hasVertexColors: true, instanced: true });
      expect(inst).toContain("@location(5) color:    vec4<f32>");
      expect(inst).toContain("out.color = in.color;");
      expect(inst).toContain("@location(6) @interpolate(flat) i_color");
      // The per-vertex colour varying must not collide with the instanced ones (6-8).
      const vertexOut = inst.slice(inst.indexOf("struct VertexOut {"), inst.indexOf("}", inst.indexOf("struct VertexOut {")));
      const slot = Number(/@location\((\d+)\) color:/.exec(vertexOut)?.[1]);
      expect(slot).toBeGreaterThanOrEqual(0);
      expect(slot).toBeLessThan(6);
    }
  });

  it("reads the draw record from a storage array by instance_index in both stages", () => {
    for (const materialKind of ["pbr", "unlit", "cel"] as const) {
      for (const vertexStride of [6, 8] as const) {
        const plain = buildWgsl({ ...base, materialKind, vertexStride });
        const inst = buildWgsl({ ...base, materialKind, vertexStride, instanced: true });
        expect(plain).toContain("var<uniform> u_draw: DrawUniforms;");
        expect(inst).not.toContain("var<uniform> u_draw");
        expect(inst).toContain("@group(1) @binding(0) var<storage, read> u_draws: array<DrawUniforms>;");
        expect(inst).toContain("var<private> u_draw: DrawUniforms;");
        expect(inst).toContain("@builtin(instance_index) inst: u32");
        // The storage record is loaded once, per vertex; the fragment stage gets flat varyings.
        expect(inst.match(/u_draws\[/g)?.length).toBe(1);
        expect(inst).toContain("@location(6) @interpolate(flat) i_color: vec4<f32>");
        expect(inst).toContain("@location(8) @interpolate(flat) i_emissive: vec4<f32>");
        const fs = inst.slice(inst.indexOf("fn fs_main"));
        expect(fs).not.toContain("u_draws");
        expect(fs).toContain("u_draw.color = in.i_color;");
        // The shading itself is unchanged: same u_draw.* reads as the per-entity shader.
        const fields = (s: string) => [...new Set(s.match(/u_draw\.[a-z]+/g) ?? [])].sort().join(",");
        // Every field the per-entity shader reads is still provided; nothing new is read.
        const got = new Set(fields(inst).split(","));
        for (const f of fields(plain).split(",")) expect(got.has(f), f).toBe(true);
        for (const f of got) expect(["u_draw.color", "u_draw.params", "u_draw.emissive", "u_draw.model"]).toContain(f);
      }
    }
  });
});
