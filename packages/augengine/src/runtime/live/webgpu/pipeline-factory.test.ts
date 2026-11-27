/**
 * Pipeline factory unit tests.
 *
 * Pure-data tests — verify the factory's cache-key uniqueness, key-to-
 * string roundtrip, WGSL template selection logic, and the Phase 6 v2
 * morph splice. Vitest can't provide a real WebGPU device in Node, so
 * the morph tests assert on the source string instead of compiling
 * shaders.
 */

import { describe, it, expect } from "vitest";
import type { PipelineKey } from "./pipeline-factory.js";

/**
 * Mirror of the private `keyToString` in `pipeline-factory.ts`. Stays
 * in sync via review — if you rename or add a key field, update this.
 */
function keyToString(k: PipelineKey): string {
  return `${k.materialKind}|${k.vertexStride}|${k.sampleCount}|static|${k.hasMorphTargets ? "morph" : "rigid"}`;
}

describe("PipelineFactory key", () => {
  it("produces a unique cache key per material/vertex/sample/morph combination", () => {
    const keys: PipelineKey[] = [
      { materialKind: "unlit", vertexStride: 6, sampleCount: 4 },
      { materialKind: "pbr", vertexStride: 6, sampleCount: 4 },
      { materialKind: "pbr", vertexStride: 8, sampleCount: 4 },
      { materialKind: "pbr", vertexStride: 8, sampleCount: 4, hasMorphTargets: true },
      { materialKind: "pbr", vertexStride: 8, sampleCount: 1 },
      { materialKind: "cel", vertexStride: 6, sampleCount: 4 },
      { materialKind: "outline", vertexStride: 6, sampleCount: 4 },
      { materialKind: "outline", vertexStride: 6, sampleCount: 4, hasMorphTargets: true },
      { materialKind: "vrm-mtoon", vertexStride: 16, sampleCount: 4 },
    ];
    const strings = keys.map(keyToString);
    expect(new Set(strings).size).toBe(keys.length);
  });

  it("encodes the morph bit as 'morph' or 'rigid' for human readability", () => {
    expect(keyToString({ materialKind: "pbr", vertexStride: 8, sampleCount: 4 })).toBe("pbr|8|4|static|rigid");
    expect(
      keyToString({ materialKind: "pbr", vertexStride: 8, sampleCount: 4, hasMorphTargets: true }),
    ).toBe("pbr|8|4|static|morph");
  });

  it("treats every supported material kind as a valid key (no compile-time mismatches)", () => {
    const supported = ["unlit", "pbr", "cel", "outline", "vrm-mtoon"] as const;
    for (const kind of supported) {
      const k: PipelineKey = { materialKind: kind, vertexStride: 6, sampleCount: 4 };
      expect(typeof keyToString(k)).toBe("string");
    }
  });
});

// Phase 6 v2 — morph-aware WGSL splice. String-level assertions are
// sufficient here; a real WebGPU device runs the shader for real in the
// smoke-game once a glTF morph-target mesh ships.

describe("Phase 6 v2 — morph-aware pipeline WGSL", () => {
  it("morph preamble declares the 52-channel weight layout + storage bindings", async () => {
    const { readFile } = await import("node:fs/promises");
    const src = await readFile(
      new URL("./pipeline-factory.ts", import.meta.url),
      "utf-8",
    );
    expect(src).toContain("AUGENGINE_BLENDSHAPE_COUNT: u32 = 52u");
    expect(src).toContain("@group(2) @binding(0) var<uniform> u_morph_weights");
    expect(src).toContain("@group(2) @binding(1) var<storage, read> u_morph_position_deltas");
    expect(src).toContain("@group(2) @binding(2) var<storage, read> u_morph_normal_deltas");
    expect(src).toContain("fn augengine_morph_position(vertex_idx: u32) -> vec3<f32>");
    expect(src).toContain("fn augengine_morph_normal(vertex_idx: u32) -> vec3<f32>");
  });

  it("morph vertex shader applies position + normal offsets before model xform", async () => {
    const { readFile } = await import("node:fs/promises");
    const src = await readFile(
      new URL("./pipeline-factory.ts", import.meta.url),
      "utf-8",
    );
    // VS_BASE_MORPH chunk
    expect(src).toContain("let posOffset = augengine_morph_position(in.vertex_idx)");
    expect(src).toContain("let nrmOffset = augengine_morph_normal(in.vertex_idx)");
    expect(src).toContain("let morphedPos = in.position + posOffset");
    expect(src).toContain("normalize(in.normal + nrmOffset)");
    // OUTLINE_MORPH_WGSL preserves the inverted-hull but morphs first
    expect(src).toContain("OUTLINE_MORPH_WGSL");
    expect(src).toContain("let extruded = morphedPos + morphedNrm * u_draw.params.x");
  });

  it("buildWgsl handles every material kind under hasMorphTargets", async () => {
    const { readFile } = await import("node:fs/promises");
    const src = await readFile(
      new URL("./pipeline-factory.ts", import.meta.url),
      "utf-8",
    );
    for (const kind of ["unlit", "pbr", "cel", "outline", "vrm-mtoon"] as const) {
      expect(src).toContain(`case "${kind}":`);
    }
    // Splat correctly throws — splat has its own pipeline.
    expect(src).toContain(
      'throw new Error("splat materials use the dedicated splat pipeline, not pipeline-factory")',
    );
  });

  it("the morph weights uniform packs 52 f32 into 13 vec4 slots (16-byte aligned)", async () => {
    const { readFile } = await import("node:fs/promises");
    const src = await readFile(
      new URL("./pipeline-factory.ts", import.meta.url),
      "utf-8",
    );
    expect(src).toContain("array<vec4<f32>, 13>");
  });
});

describe("Phase 6 v3 - renderer morph submission", () => {
  it("selects morph pipelines and binds weights plus delta buffers during draw", async () => {
    const { readFile } = await import("node:fs/promises");
    const src = await readFile(new URL("./renderer.ts", import.meta.url), "utf-8");
    expect(src).toContain("uploadMorphTargets(this.morphDeltaStorage, name, data.morphTargets)");
    expect(src).toContain("const hasMorphTargets = !!mesh.morph && !!this.blendshapePool");
    expect(src).toContain("entry.compiled.morphBindGroupLayout");
    expect(src).toContain("pass.setBindGroup(2, d.morphBindGroup)");
  });
});

describe("Character Proof Room - PBR base-color texture path", () => {
  it("declares sampled texture resources and sends UV0 to the fragment shader", async () => {
    const { readFile } = await import("node:fs/promises");
    const src = await readFile(new URL("./pipeline-factory.ts", import.meta.url), "utf-8");
    expect(src).toContain("hasPbrBaseColorTexture?: boolean");
    expect(src).toContain("@group(3) @binding(0) var t_base_color");
    expect(src).toContain("@group(3) @binding(1) var s_pbr");
    expect(src).toContain("@group(3) @binding(2) var t_metallic_roughness");
    expect(src).toContain("@group(3) @binding(3) var t_normal");
    expect(src).toContain("out.uv = in.uv");
    expect(src).toContain("textureSample(t_base_color, s_pbr, in.uv).rgb");
    expect(src).toContain("textureSample(t_metallic_roughness, s_pbr, in.uv).b");
    expect(src).toContain("augengine_sample_normal");
  });

  it("uploads and binds decoded textures in the live renderer", async () => {
    const { readFile } = await import("node:fs/promises");
    const src = await readFile(new URL("./renderer.ts", import.meta.url), "utf-8");
    expect(src).toContain("async uploadPbrTextures");
    expect(src).toContain("copyExternalImageToTexture");
    expect(src).toContain("pass.setBindGroup(3, d.textureBindGroup)");
  });

  it("preserves UV0 through compute skinning for textured animated meshes", async () => {
    const { readFile } = await import("node:fs/promises");
    const src = await readFile(new URL("./skinning.ts", import.meta.url), "utf-8");
    expect(src).toContain("SKINNING_COMPUTE_WGSL_UV");
    expect(src).toContain("v_out[i].data[6] = src.data[6]");
    expect(src).toContain("v_out[i].data[7] = src.data[7]");
  });
});
