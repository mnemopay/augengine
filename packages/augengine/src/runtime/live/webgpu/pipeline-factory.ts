/**
 * Phase 9 GPU close-out — pipeline factory.
 *
 * The single hard-coded pipeline in `renderer.ts` is replaced by a registry
 * keyed on (MaterialKind, vertex layout, sample count). Each unique combo
 * gets compiled once and cached. The renderer's draw loop looks up the
 * right pipeline per entity at draw time.
 *
 * The live renderer selects these variants per entity and binds optional
 * morph and PBR texture resources during draw submission.
 */

import type { MaterialKind } from "../../../scene/material.js";
import { MAX_EXTRA_DIRECTIONAL, MAX_LOCAL_LIGHTS, PCSS_ALONG_FILTER_TEXELS, PCSS_MIN_FILTER_TEXELS, RPDB_MAX_GRADIENT } from "./lights.js";
import { PROBE_RING } from "./probe-ring.js";

export interface PipelineKey {
  materialKind: MaterialKind["kind"];
  /** Vertex attributes: pos+normal=6, pos+normal+uv=8, pos+normal+uv+joints+weights=16. */
  vertexStride: 6 | 8 | 16;
  sampleCount: 1 | 4;
  /**
   * Phase 6 — when true, the compiled vertex shader splices
   * `BLENDSHAPE_WGSL_HELPER` and adds a morph-offset displacement step
   * before the model-matrix transform. Default false. Requires the
   * caller to bind the per-entity blendshape uniform (52 weights) and
   * a storage buffer of per-vertex position/normal deltas.
   */
  hasMorphTargets?: boolean;
  hasVertexColors?: boolean;
  /** PBR-only variant that samples an sRGB base-color texture using UV0. */
  hasPbrBaseColorTexture?: boolean;
  /** PBR-only variant that samples the glTF green/blue roughness/metalness map. */
  hasPbrMetallicRoughnessTexture?: boolean;
  /** PBR-only variant that samples a tangent-space normal map. */
  hasPbrNormalTexture?: boolean;
  /**
   * Instanced variant: group(1) binding 0 is a read-only storage array of
   * DrawUniforms indexed by @builtin(instance_index), so one drawIndexed call
   * draws many entities that share a mesh. Only for rigid, untextured,
   * vertex-colour-free unlit/pbr/cel draws (the renderer gates this).
   */
  instanced?: boolean;
  /**
   * Natural-light variant: compiles the sky/HDR image-based lighting, SH9
   * irradiance and aerial-perspective code. Without it those paths are left
   * out of the shader entirely (the renderer picks this variant whenever
   * setNaturalLight is on), so the legacy look pays nothing for them.
   */
  natural?: boolean;
  /** PCSS variant: compiles the contact-hardening shadow filter ("soft" qualities). */
  pcss?: boolean;
  /**
   * glTF alphaMode MASK (PBR with a base-colour texture only): discard the
   * fragment when the base colour alpha is below u_draw.params.z (the cutoff
   * divided by the factor alpha).
   */
  alphaMask?: boolean;
  /** glTF doubleSided: no back-face culling; back faces use the flipped normal. */
  doubleSided?: boolean;
}

export interface CompiledPipeline {
  key: PipelineKey;
  pipeline: GPURenderPipeline;
  /** Per-pipeline bind-group layouts the caller uses to build draw bind groups. */
  frameBindGroupLayout: GPUBindGroupLayout;
  drawBindGroupLayout: GPUBindGroupLayout;
  /** Present only for morph-target pipelines (`hasMorphTargets: true`).
   *  Caller binds this as group(2) with [weights uniform, position deltas
   *  storage, normal deltas storage] in that order. */
  morphBindGroupLayout?: GPUBindGroupLayout;
  /** Present only for PBR pipelines sampling an albedo/base-color texture. */
  pbrTextureBindGroupLayout?: GPUBindGroupLayout;
}

export class PipelineFactory {
  private device: GPUDevice;
  private format: GPUTextureFormat;
  private cache = new Map<string, CompiledPipeline>();

  constructor(device: GPUDevice, format: GPUTextureFormat) {
    this.device = device;
    this.format = format;
  }

  /** Lookup or compile a pipeline for the given key. */
  get(key: PipelineKey): CompiledPipeline {
    const cacheKey = this.keyToString(key);
    let cached = this.cache.get(cacheKey);
    if (cached) return cached;
    cached = this.compile(key);
    this.cache.set(cacheKey, cached);
    return cached;
  }

  /** Wipe all cached pipelines (e.g., on canvas resize / format change). */
  invalidate(): void {
    this.cache.clear();
  }

  private compile(key: PipelineKey): CompiledPipeline {
    const wgsl = buildWgsl(key);
    const module = this.device.createShaderModule({ code: wgsl });

    // group(0): frame uniform + the key light's shadow map and its comparison
    // sampler. The renderer binds a 1×1 depth texture when shadows are off, so
    // every pipeline shares one frame layout whatever the shadow setting.
    const frameLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "depth" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "comparison" } },
        // Streamed irradiance probe ring (probe-ring.ts), fixed size.
        { binding: 3, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
        // Prefiltered specular environment cube (env-lighting.ts) + its sampler;
        // a 1×1 cube is bound while natural light is off.
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float", viewDimension: "cube" } },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
        // Split-sum DFG LUT (ibl-brdf.ts): u = N·V, v = roughness.
        { binding: 6, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
      ],
    });
    const drawLayoutEntries: GPUBindGroupLayoutEntry[] = [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: key.instanced ? "read-only-storage" : "uniform" },
      },
    ];
    const drawLayout = this.device.createBindGroupLayout({ entries: drawLayoutEntries });

    // Phase 6 — morph-target pipelines bind a third group:
    //   group(2) binding 0: uniform   weights[52]
    //   group(2) binding 1: storage   position_deltas[]   read-only
    //   group(2) binding 2: storage   normal_deltas[]     read-only
    // Renderer assembles + binds this group per-entity at draw time.
    const morphLayout: GPUBindGroupLayout | null = key.hasMorphTargets
      ? this.device.createBindGroupLayout({
          entries: [
            { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
            { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
            { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
          ],
        })
      : null;
    const hasPbrTextures = key.materialKind === "pbr" && (
      key.hasPbrBaseColorTexture ||
      key.hasPbrMetallicRoughnessTexture ||
      key.hasPbrNormalTexture
    );
    const pbrTextureLayout: GPUBindGroupLayout | null =
      hasPbrTextures
        ? this.device.createBindGroupLayout({
            entries: [
              ...(key.hasPbrBaseColorTexture
                ? [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" as const } }]
                : []),
              { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
              ...(key.hasPbrMetallicRoughnessTexture
                ? [{ binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" as const } }]
                : []),
              ...(key.hasPbrNormalTexture
                ? [{ binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" as const } }]
                : []),
            ],
          })
        : null;
    const layoutGroups: GPUBindGroupLayout[] = [frameLayout, drawLayout];
    if (morphLayout || pbrTextureLayout) {
      layoutGroups.push(morphLayout ?? this.device.createBindGroupLayout({ entries: [] }));
    }
    if (pbrTextureLayout) layoutGroups.push(pbrTextureLayout);

    const pipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: layoutGroups,
    });

    const vertexStrideBytes = key.vertexStride * 4;
    const vertexAttrs: GPUVertexAttribute[] = [
      { shaderLocation: 0, offset: 0, format: "float32x3" },                  // position
      { shaderLocation: 1, offset: 12, format: "float32x3" },                 // normal
    ];
    if (key.vertexStride >= 8) {
      vertexAttrs.push({ shaderLocation: 2, offset: 24, format: "float32x2" }); // uv
    }
    if (key.vertexStride === 16) {
      vertexAttrs.push({ shaderLocation: 3, offset: 32, format: "uint16x4" });  // joints
      vertexAttrs.push({ shaderLocation: 4, offset: 40, format: "float32x4" }); // weights
    }

    const buffers: GPUVertexBufferLayout[] = [
      { arrayStride: vertexStrideBytes, attributes: vertexAttrs }
    ];
    if (key.hasVertexColors) {
      buffers.push({
        arrayStride: 16, // float32x4 RGBA vertex colors
        attributes: [{ shaderLocation: 5, offset: 0, format: "float32x4" }]
      });
    }

    const pipeline = this.device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: {
        module,
        entryPoint: "vs_main",
        buffers,
      },
      fragment: {
        module,
        entryPoint: "fs_main",
        targets: [{ format: this.format }],
      },
      primitive: {
        topology: "triangle-list",
        cullMode: key.materialKind === "outline" ? "front" : key.doubleSided ? "none" : "back",
      },
      depthStencil: {
        depthWriteEnabled: true,
        depthCompare: "less",
        format: "depth24plus",
      },
      multisample: { count: key.sampleCount },
    });

    const compiled: CompiledPipeline = {
      key,
      pipeline,
      frameBindGroupLayout: frameLayout,
      drawBindGroupLayout: drawLayout,
    };
    if (morphLayout) compiled.morphBindGroupLayout = morphLayout;
    if (pbrTextureLayout) compiled.pbrTextureBindGroupLayout = pbrTextureLayout;
    return compiled;
  }

  private keyToString(k: PipelineKey): string {
    return pipelineKeyString(k);
  }
}

/** Cache key for a pipeline variant. The renderer uses the same string. */
export function pipelineKeyString(k: PipelineKey): string {
  return `${k.materialKind}|${k.vertexStride}|${k.sampleCount}|static|${k.hasMorphTargets ? "morph" : "rigid"}|${k.hasVertexColors ? "color" : "nocolor"}|${k.hasPbrBaseColorTexture ? "basecolor" : "flat"}|${k.hasPbrMetallicRoughnessTexture ? "mr" : "nomr"}|${k.hasPbrNormalTexture ? "normal" : "nonormal"}${k.instanced ? "|instanced" : ""}${k.natural ? "|natural" : ""}${k.pcss ? "|pcss" : ""}${k.alphaMask ? "|mask" : ""}${k.doubleSided ? "|2side" : ""}`;
}

/** True when `buildWgsl` can produce an instanced variant for this key. */
export function canInstance(k: PipelineKey): boolean {
  return (k.materialKind === "pbr" || k.materialKind === "unlit" || k.materialKind === "cel") &&
    k.vertexStride !== 16 &&
    // Vertex colours are a per-vertex stream shared by every instance (slot 1),
    // so they instance fine; their varying sits below the instanced ones (6-8).
    !k.hasMorphTargets && !k.alphaMask && !k.doubleSided &&
    !k.hasPbrBaseColorTexture && !k.hasPbrMetallicRoughnessTexture && !k.hasPbrNormalTexture;
}

/**
 * Rewrite a per-draw WGSL module into its instanced form: the uniform
 * `u_draw` becomes a private copy of `u_draws[instance_index]`, loaded once
 * per vertex. The fragment stage never touches the storage buffer (a 112-byte
 * load per pixel is expensive on integrated GPUs with screen-filling floors):
 * colour, params and emissive ride to it as flat varyings and refill the
 * private copy. Every existing `u_draw.*` reference keeps working unchanged.
 */
function instancedWgsl(src: string): string {
  const uniformDecl = "@group(1) @binding(0) var<uniform> u_draw: DrawUniforms;";
  const vsHead = "fn vs_main(in: VertexIn) -> VertexOut {\n  var out: VertexOut;";
  const fsHead = "fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {";
  for (const needle of [uniformDecl, vsHead, fsHead, "struct VertexIn {\n", "struct VertexOut {\n"]) {
    if (!src.includes(needle)) throw new Error(`instanced WGSL: missing anchor ${JSON.stringify(needle.slice(0, 40))}`);
  }
  return src
    .replace(uniformDecl, "@group(1) @binding(0) var<storage, read> u_draws: array<DrawUniforms>;\nvar<private> u_draw: DrawUniforms;")
    .replace("struct VertexIn {\n", "struct VertexIn {\n  @builtin(instance_index) inst: u32,\n")
    .replace(
      "struct VertexOut {\n",
      "struct VertexOut {\n" +
        "  @location(6) @interpolate(flat) i_color: vec4<f32>,\n" +
        "  @location(7) @interpolate(flat) i_params: vec4<f32>,\n" +
        "  @location(8) @interpolate(flat) i_emissive: vec4<f32>,\n",
    )
    .replace(
      vsHead,
      `${vsHead}\n  u_draw = u_draws[in.inst];\n  out.i_color = u_draw.color;\n  out.i_params = u_draw.params;\n  out.i_emissive = u_draw.emissive;`,
    )
    .replace(fsHead, `${fsHead}\n  u_draw.color = in.i_color;\n  u_draw.params = in.i_params;\n  u_draw.emissive = in.i_emissive;`);
}

/**
 * Strip the feature blocks a variant does not use. WGSL has no preprocessor
 * and override constants cannot remove code (a dead branch still costs
 * registers and compile size on Intel UHD), so feature code sits between
 * `// @<feature>-begin` and `// @<feature>-end` lines and is cut out here.
 */
export function specializeWgsl(src: string, features: { natural: boolean; pcss: boolean }): string {
  let out = src;
  for (const [name, on] of [["natural", features.natural], ["pcss", features.pcss]] as const) {
    const re = new RegExp(`^[ \\t]*// @${name}-begin\\n[\\s\\S]*?^[ \\t]*// @${name}-end\\n`, "gm");
    out = on ? out.replace(new RegExp(`^[ \\t]*// @${name}-(begin|end)\\n`, "gm"), "") : out.replace(re, "");
  }
  return out;
}

export function buildWgsl(key: PipelineKey): string {
  return specializeWgsl(buildWgslFull(key), { natural: key.natural === true, pcss: key.pcss === true });
}

function buildWgslFull(key: PipelineKey): string {
  if (key.instanced) {
    if (!canInstance(key)) throw new Error(`no instanced variant for ${pipelineKeyString(key)}`);
    return instancedWgsl(buildWgslFull({ ...key, instanced: false }));
  }
  const morph = key.hasMorphTargets === true;
  const morphHead = morph ? `// AUGENGINE_MORPH_TARGETS\n${MORPH_PREAMBLE_WGSL}\n` : "";
  const textureHead = key.materialKind === "pbr"
    ? getPbrTexturePreamble(key)
    : "";

  switch (key.materialKind) {
    case "outline":
      return morphHead + FRAME_UNIFORM_BLOCK + (morph ? OUTLINE_MORPH_WGSL : OUTLINE_WGSL);
    case "splat":
      throw new Error("splat materials use the dedicated splat pipeline, not pipeline-factory");
    case "unlit":
    case "pbr":
    case "cel":
    case "vrm-mtoon": {
      const vs = getVertexShader(key);
      if (key.alphaMask && !(key.materialKind === "pbr" && key.hasPbrBaseColorTexture)) {
        throw new Error("alphaMask pipelines need a PBR base-colour texture");
      }
      let fs = getFragmentShader(
        key.materialKind,
        key.hasVertexColors === true,
        key.hasPbrBaseColorTexture === true,
        key.hasPbrMetallicRoughnessTexture === true,
        key.hasPbrNormalTexture === true,
        key.alphaMask === true,
      );
      if (key.doubleSided) fs = doubleSidedFragment(fs);
      return morphHead + textureHead + FRAME_UNIFORM_BLOCK + vs + fs;
    }
  }
}

function getVertexShader(key: PipelineKey): string {
  const morph = key.hasMorphTargets === true;
  const hasColors = key.hasVertexColors === true;
  const hasPbrTexture = key.materialKind === "pbr" && (
    key.hasPbrBaseColorTexture === true ||
    key.hasPbrMetallicRoughnessTexture === true ||
    key.hasPbrNormalTexture === true
  );
  if (hasPbrTexture && key.vertexStride < 8) {
    throw new Error("PBR base-color texture pipeline requires UV0 vertex data");
  }

  let vertexIn = `struct VertexIn {
  @location(0) position: vec3<f32>,
  @location(1) normal:   vec3<f32>,
`;
  if (morph) {
    vertexIn += `  @builtin(vertex_index) vertex_idx: u32,\n`;
  }
  if (hasPbrTexture) {
    vertexIn += `  @location(2) uv:       vec2<f32>,\n`;
  }
  if (hasColors) {
    vertexIn += `  @location(5) color:    vec4<f32>,\n`;
  }
  vertexIn += `};\n`;

  // @invariant: the motion pass (motion.ts) redraws moving meshes with the
  // same expression and depth-tests `less-equal` against this pass's depth.
  let vertexOut = `struct VertexOut {
  @builtin(position) @invariant clipPos: vec4<f32>,
  @location(0) worldNormal: vec3<f32>,
  @location(1) worldPos: vec3<f32>,
  // Irradiance-probe L1 coefficients sampled at the vertex (see augengine_probe_coeffs).
  @location(9) probeR: vec4<f32>,
  @location(10) probeG: vec4<f32>,
  @location(11) probeB: vec4<f32>,
  @location(12) probeCov: f32,
`;
  if (hasColors) {
    vertexOut += `  @location(${hasPbrTexture ? 3 : 2}) color:       vec4<f32>,\n`;
  }
  if (hasPbrTexture) {
    vertexOut += `  @location(2) uv:          vec2<f32>,\n`;
  }
  vertexOut += `};\n`;

  let vsMain = "";
  if (morph) {
    vsMain = `@vertex
fn vs_main(in: VertexIn) -> VertexOut {
  var out: VertexOut;
  let posOffset = augengine_morph_position(in.vertex_idx);
  let nrmOffset = augengine_morph_normal(in.vertex_idx);
  let morphedPos = in.position + posOffset;
  let morphedNrm = normalize(in.normal + nrmOffset);
  let world = u_draw.model * vec4<f32>(morphedPos, 1.0);
  out.clipPos = u_frame.viewProj * world;
  out.worldNormal = normalize((u_draw.model * vec4<f32>(morphedNrm, 0.0)).xyz);
  out.worldPos = world.xyz;
  let pc = augengine_probe_coeffs(world.xyz);
  out.probeR = pc.r;
  out.probeG = pc.g;
  out.probeB = pc.b;
  out.probeCov = pc.cov;
  ${hasColors ? "out.color = in.color;" : ""}
  ${hasPbrTexture ? "out.uv = in.uv;" : ""}
  return out;
}
`;
  } else {
    vsMain = `@vertex
fn vs_main(in: VertexIn) -> VertexOut {
  var out: VertexOut;
  let world = u_draw.model * vec4<f32>(in.position, 1.0);
  out.clipPos = u_frame.viewProj * world;
  out.worldNormal = normalize((u_draw.model * vec4<f32>(in.normal, 0.0)).xyz);
  out.worldPos = world.xyz;
  let pc = augengine_probe_coeffs(world.xyz);
  out.probeR = pc.r;
  out.probeG = pc.g;
  out.probeB = pc.b;
  out.probeCov = pc.cov;
  ${hasColors ? "out.color = in.color;" : ""}
  ${hasPbrTexture ? "out.uv = in.uv;" : ""}
  return out;
}
`;
  }

  return vertexIn + vertexOut + vsMain;
}

/**
 * Double-sided variant of a lit fragment shader: back faces flip the
 * interpolated normal before any lighting reads it (glTF doubleSided).
 */
function doubleSidedFragment(fs: string): string {
  const head = "fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {";
  if (!fs.includes(head)) throw new Error("double-sided WGSL: missing fs_main anchor");
  return fs
    .replace(head, "fn fs_main(inRaw: VertexOut, @builtin(front_facing) frontFacing: bool) -> @location(0) vec4<f32> {\n  var in = inRaw;\n  in.worldNormal = select(-inRaw.worldNormal, inRaw.worldNormal, frontFacing);");
}

function getFragmentShader(
  kind: MaterialKind["kind"],
  hasColors: boolean,
  hasBaseColorTexture: boolean,
  hasMetallicRoughnessTexture: boolean,
  hasNormalTexture: boolean,
  alphaMask = false,
): string {
  const colorMult = hasColors ? " * in.color.rgb" : "";
  switch (kind) {
    case "unlit":
      return `
@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
  return vec4<f32>((u_draw.color.rgb${colorMult} + u_draw.emissive.rgb) * u_frame.exposure.a, 1.0);
}
`;
    case "pbr":
      return `
fn fresnel_schlick(cosT: f32, F0: vec3<f32>) -> vec3<f32> {
  return F0 + (vec3<f32>(1.0) - F0) * pow(1.0 - cosT, 5.0);
}
fn distribution_ggx(NdotH: f32, roughness: f32) -> f32 {
  let a = roughness * roughness;
  let a2 = a * a;
  let denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
  return a2 / max(3.14159265 * denom * denom, 1e-5);
}
fn geometry_schlick_ggx(NdotV: f32, roughness: f32) -> f32 {
  let r = roughness + 1.0;
  let k = (r * r) / 8.0;
  return NdotV / (NdotV * (1.0 - k) + k);
}
// Cook-Torrance response to one light arriving along L, times N·L.
fn augengine_pbr_light(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, albedo: vec3<f32>, metallic: f32, roughness: f32, F0: vec3<f32>) -> vec3<f32> {
  let H = normalize(V + L);
  let NdotL = max(dot(N, L), 0.0);
  let NdotV = max(dot(N, V), 1e-4);
  let NdotH = max(dot(N, H), 0.0);
  let HdotV = max(dot(H, V), 0.0);
  let F = fresnel_schlick(HdotV, F0);
  let D = distribution_ggx(NdotH, roughness);
  let G = geometry_schlick_ggx(NdotV, roughness) * geometry_schlick_ggx(NdotL, roughness);
  let spec = (D * G * F) / max(4.0 * NdotV * NdotL, 1e-5);
  let kd = (vec3<f32>(1.0) - F) * (1.0 - metallic);
  return (kd * albedo / 3.14159265 + spec) * NdotL;
}
@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
  ${hasBaseColorTexture ? "let baseSample = textureSample(t_base_color, s_pbr, in.uv);" : ""}
  ${alphaMask ? "// glTF alphaMode MASK: params.z = alphaCutoff / baseColorFactor.a.\n  if (baseSample.a < u_draw.params.z) { discard; }" : ""}
  let albedo = u_draw.color.rgb${colorMult}${hasBaseColorTexture ? " * baseSample.rgb" : ""};
  let metallic = u_draw.params.x${hasMetallicRoughnessTexture ? " * textureSample(t_metallic_roughness, s_pbr, in.uv).b" : ""};
  let roughness = u_draw.params.y${hasMetallicRoughnessTexture ? " * textureSample(t_metallic_roughness, s_pbr, in.uv).g" : ""};
  var N = normalize(in.worldNormal);
  ${hasNormalTexture ? "N = augengine_sample_normal(N, in.worldPos, in.uv);" : ""}
  let V = normalize(u_frame.cameraPos.xyz - in.worldPos);
  let F0 = mix(vec3<f32>(0.04), albedo, metallic);
  let Lkey = normalize(-u_frame.lightDir.xyz);
  var lit = augengine_pbr_light(N, V, Lkey, albedo, metallic, roughness, F0)
    * u_frame.lightColor.rgb * augengine_shadow(in.worldPos, N, in.clipPos.xy);
  for (var i = 0u; i < u32(u_frame.counts.x); i = i + 1u) {
    let d = u_frame.dirLights[i];
    lit = lit + augengine_pbr_light(N, V, normalize(-d.dir.xyz), albedo, metallic, roughness, F0) * d.radiance.rgb;
  }
  for (var i = 0u; i < u32(u_frame.counts.y); i = i + 1u) {
    let la = augengine_local_light(i, in.worldPos);
    lit = lit + augengine_pbr_light(N, V, la.xyz, albedo, metallic, roughness, F0) * u_frame.localLights[i].radianceInner.rgb * la.w;
  }
  let amb = augengine_ambient(N, in.probeR, in.probeG, in.probeB, in.probeCov);
  // @natural-begin
  if (u_frame.envParams.x > 0.5) {
    lit = lit + augengine_env_ibl(N, V, albedo, metallic, roughness, F0, amb);
  } else {
  // @natural-end
    // With sky specular on, metals take their ambient from the reflection
    // instead of the diffuse term; with it off this is the old flat ambient.
    let ambientW = select(1.0, 1.0 - metallic, u_frame.skySpecSH[0].a > 0.5);
    lit = lit + albedo * amb * ambientW;
    lit = lit + augengine_sky_specular(N, V, F0, roughness);
  // @natural-begin
  }
  // @natural-end
  let fogged = augengine_fog_apply(lit, in.worldPos);
  return vec4<f32>(fogged * u_frame.exposure.rgb + u_draw.emissive.rgb * u_frame.exposure.a, 1.0);
}
`;
    case "cel":
      return `
@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
  let N = normalize(in.worldNormal);
  let L = normalize(-u_frame.lightDir.xyz);
  let NdotL = max(dot(N, L), 0.0) * augengine_shadow(in.worldPos, N, in.clipPos.xy);
  let steps = max(u_draw.params.x, 2.0);
  let stepped = floor(NdotL * steps) / max(steps - 1.0, 1.0);
  let baseColor = u_draw.color.rgb${colorMult};
  var lightTerm = u_frame.lightColor.rgb * stepped;
  for (var i = 0u; i < u32(u_frame.counts.x); i = i + 1u) {
    let d = u_frame.dirLights[i];
    let s = floor(max(dot(N, normalize(-d.dir.xyz)), 0.0) * steps) / max(steps - 1.0, 1.0);
    lightTerm = lightTerm + d.radiance.rgb * s;
  }
  for (var i = 0u; i < u32(u_frame.counts.y); i = i + 1u) {
    let la = augengine_local_light(i, in.worldPos);
    let s = floor(max(dot(N, la.xyz), 0.0) * la.w * steps) / max(steps - 1.0, 1.0);
    lightTerm = lightTerm + u_frame.localLights[i].radianceInner.rgb * s;
  }
  let lit = baseColor * (augengine_ambient(N, in.probeR, in.probeG, in.probeB, in.probeCov) + lightTerm);
  let fogged = augengine_fog_apply(lit, in.worldPos);
  return vec4<f32>(fogged * u_frame.exposure.rgb + u_draw.emissive.rgb * u_frame.exposure.a, 1.0);
}
`;
    case "vrm-mtoon":
      return `
@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
  let N = normalize(in.worldNormal);
  let L = normalize(-u_frame.lightDir.xyz);
  let lambert = dot(N, L) * augengine_shadow(in.worldPos, N, in.clipPos.xy);
  let shifted = clamp((lambert - u_draw.params.x) / max(u_draw.params.y, 1e-4), 0.0, 1.0);
  let baseColor = u_draw.color.rgb${colorMult};
  var lit = mix(augengine_ambient(N, in.probeR, in.probeG, in.probeB, in.probeCov), baseColor * u_frame.lightColor.rgb, shifted);
  for (var i = 0u; i < u32(u_frame.counts.x); i = i + 1u) {
    let d = u_frame.dirLights[i];
    lit = lit + baseColor * d.radiance.rgb * max(dot(N, normalize(-d.dir.xyz)), 0.0);
  }
  for (var i = 0u; i < u32(u_frame.counts.y); i = i + 1u) {
    let la = augengine_local_light(i, in.worldPos);
    lit = lit + baseColor * u_frame.localLights[i].radianceInner.rgb * max(dot(N, la.xyz), 0.0) * la.w;
  }
  let fogged = augengine_fog_apply(lit, in.worldPos);
  return vec4<f32>(fogged * u_frame.exposure.rgb + u_draw.emissive.rgb * u_frame.exposure.a, 1.0);
}
`;
    default:
      return "";
  }
}

function getPbrTexturePreamble(key: PipelineKey): string {
  const hasTextures = key.hasPbrBaseColorTexture ||
    key.hasPbrMetallicRoughnessTexture ||
    key.hasPbrNormalTexture;
  if (!hasTextures) return "";
  let source = "// AUGENGINE_PBR_TEXTURES\n";
  if (key.hasPbrBaseColorTexture) {
    source += "@group(3) @binding(0) var t_base_color: texture_2d<f32>;\n";
  }
  source += "@group(3) @binding(1) var s_pbr: sampler;\n";
  if (key.hasPbrMetallicRoughnessTexture) {
    source += "@group(3) @binding(2) var t_metallic_roughness: texture_2d<f32>;\n";
  }
  if (key.hasPbrNormalTexture) {
    source += `@group(3) @binding(3) var t_normal: texture_2d<f32>;
fn augengine_sample_normal(normal: vec3<f32>, worldPos: vec3<f32>, uv: vec2<f32>) -> vec3<f32> {
  let mapped = textureSample(t_normal, s_pbr, uv).xyz * 2.0 - vec3<f32>(1.0);
  let posDx = dpdx(worldPos);
  let posDy = dpdy(worldPos);
  let uvDx = dpdx(uv);
  let uvDy = dpdy(uv);
  let tangent = normalize(posDx * uvDy.y - posDy * uvDx.y);
  let bitangent = normalize(-posDx * uvDy.x + posDy * uvDx.x);
  return normalize(mat3x3<f32>(tangent, bitangent, normal) * mapped);
}
`;
  }
  return source;
}

const MORPH_PREAMBLE_WGSL = /* wgsl */ `
const AUGENGINE_BLENDSHAPE_COUNT: u32 = 52u;

struct AugEngineMorphWeights { values: array<vec4<f32>, 13> };
// 52 floats packed into 13 vec4s — uniform buffers must be 16-byte aligned
// per element. Index access: weights.values[k >> 2][k & 3].
@group(2) @binding(0) var<uniform> u_morph_weights: AugEngineMorphWeights;
@group(2) @binding(1) var<storage, read> u_morph_position_deltas: array<vec3<f32>>;
@group(2) @binding(2) var<storage, read> u_morph_normal_deltas: array<vec3<f32>>;

fn augengine_morph_weight(k: u32) -> f32 {
  let v = u_morph_weights.values[k >> 2u];
  let lane = k & 3u;
  if (lane == 0u) { return v.x; }
  if (lane == 1u) { return v.y; }
  if (lane == 2u) { return v.z; }
  return v.w;
}

fn augengine_morph_position(vertex_idx: u32) -> vec3<f32> {
  var offset = vec3<f32>(0.0);
  for (var k: u32 = 0u; k < AUGENGINE_BLENDSHAPE_COUNT; k = k + 1u) {
    let w = augengine_morph_weight(k);
    if (w == 0.0) { continue; }
    let idx = vertex_idx * AUGENGINE_BLENDSHAPE_COUNT + k;
    offset = offset + w * u_morph_position_deltas[idx];
  }
  return offset;
}

fn augengine_morph_normal(vertex_idx: u32) -> vec3<f32> {
  var offset = vec3<f32>(0.0);
  for (var k: u32 = 0u; k < AUGENGINE_BLENDSHAPE_COUNT; k = k + 1u) {
    let w = augengine_morph_weight(k);
    if (w == 0.0) { continue; }
    let idx = vertex_idx * AUGENGINE_BLENDSHAPE_COUNT + k;
    offset = offset + w * u_morph_normal_deltas[idx];
  }
  return offset;
}
`;

/** A JS number as a WGSL f32 literal (always with a decimal point). */
export function wgslF32(x: number): string {
  const t = String(x);
  return /[.eE]/.test(t) ? t : `${t}.0`;
}

// Layout MUST match FRAME_LAYOUT in lights.ts (float offsets in comments).
const FRAME_UNIFORM_BLOCK = /* wgsl */ `
struct DirLight { dir: vec4<f32>, radiance: vec4<f32>, };
struct LocalLight {
  posRange: vec4<f32>,       // xyz world position, w range
  radianceInner: vec4<f32>,  // rgb colour × intensity, a = cos(inner cone)
  spotDirOuter: vec4<f32>,   // xyz cone direction, w = cos(outer cone); -2 = point light
};
struct FrameUniforms {
  viewProj: mat4x4<f32>,     // 0
  lightDir: vec4<f32>,       // 16 — key light travel direction
  ambient:  vec4<f32>,       // 20
  cameraPos: vec4<f32>,      // 24
  // fog.rgb = fog color (typically the sky horizon color).
  // fog.a   = density. 0 disables fog. Typical values 0.005–0.04 for
  //           outdoor scenes; the exp falloff makes 0.02 read as a
  //           gentle haze and 0.05 as a dense overcast.
  fog: vec4<f32>,            // 28
  lightColor: vec4<f32>,     // 32 — key radiance, a = shadow on
  counts: vec4<f32>,         // 36 — x extra directional, y local lights
  shadowViewProj: mat4x4<f32>, // 40
  shadowParams: vec4<f32>,   // 56 — x texel (uv), y depth bias, z normal offset, w PCF taps/axis
  dirLights: array<DirLight, ${MAX_EXTRA_DIRECTIONAL}>,   // 60
  localLights: array<LocalLight, ${MAX_LOCAL_LIGHTS}>,
  // Sky-shaped ambient: [0].rgb k0, [0].a enabled, [1].rgb k1, [2].rgb k2 (ibl.ts).
  ambientSH: array<vec4<f32>, 3>,
  // Far shadow cascade (counts.z == 2): right half of a 2:1 atlas.
  shadowViewProj2: mat4x4<f32>,
  shadowParams2: vec4<f32>,  // x texel (uv), y depth bias, z normal offset, w blend fraction
  // Sky specular (ibl.ts): raw-radiance zonal SH [0].rgb c0 + [0].a enabled, [1].rgb c1, [2].rgb c2.
  skySpecSH: array<vec4<f32>, 3>,
  // Irradiance probes: x on, y column spacing, z layer-0 height above ground, w layer spacing.
  probeParams: vec4<f32>,
  // Natural light (env-lighting.ts): irradiance/π SH9, basis constants folded in.
  envSH: array<vec4<f32>, 9>,
  // x env on, y specular mips − 1, z specular intensity, w multiple scattering on.
  envParams: vec4<f32>,
  // rgb pre-exposure × white balance on lit output, a emissive/unlit scale.
  exposure: vec4<f32>,
  // PCSS: x tan(light angular radius)·softness, y/z cascade widths (m), w taps (0 = off).
  shadowSoft: vec4<f32>,
  // x frame index mod 64, y blocker search height (m).
  shadowSoft2: vec4<f32>,
  // Natural atmosphere: x on, y HG g, z sun in-scatter strength, w legacy-fog veil distance (m, 0 = full legacy fog).
  atmos: vec4<f32>,
  // rgb aerial extinction (1/m).
  atmosSigma: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u_frame: FrameUniforms;
@group(0) @binding(1) var t_shadow: texture_depth_2d;
@group(0) @binding(2) var s_shadow: sampler_comparison;
// Probe ring: per probe r, g, b (L1 relative ambient k0 kx ky kz) + meta (gi, gj, groundY, valid).
@group(0) @binding(3) var<storage, read> u_probes: array<vec4<f32>>;
@group(0) @binding(4) var t_env: texture_cube<f32>;
@group(0) @binding(5) var s_env: sampler;
@group(0) @binding(6) var t_dfg: texture_2d<f32>;
const AUGENGINE_PROBE_RING: i32 = ${PROBE_RING};

fn augengine_probe_base(i: i32, j: i32, k: i32) -> u32 {
  let si = ((i % AUGENGINE_PROBE_RING) + AUGENGINE_PROBE_RING) % AUGENGINE_PROBE_RING;
  let sj = ((j % AUGENGINE_PROBE_RING) + AUGENGINE_PROBE_RING) % AUGENGINE_PROBE_RING;
  return u32((k * AUGENGINE_PROBE_RING + sj) * AUGENGINE_PROBE_RING + si) * 4u;
}

fn augengine_probe_ok(m: vec4<f32>, i: i32, j: i32) -> bool {
  return m.w > 0.5 && i32(round(m.x)) == i && i32(round(m.y)) == j;
}

// Probe L1 coefficients at P (per channel k0, kx, ky, kz) + coverage. Mirrors
// probe-ring.ts sampleProbeRing: ground base bilinear over the valid layer-0
// columns, then trilinear over the 8 probes around P, renormalised over the
// valid ones, so tiles blend across their seams and a missing neighbour fades
// rather than cuts. Run per VERTEX: L1 irradiance is linear in the normal, so
// interpolated coefficients dotted with the per-pixel normal are exact at the
// vertices — a per-pixel fetch cost ~3 ms on Intel UHD (2026-11-23).
struct AugengineProbeCoeffs { r: vec4<f32>, g: vec4<f32>, b: vec4<f32>, cov: f32 };
fn augengine_probe_coeffs(P: vec3<f32>) -> AugengineProbeCoeffs {
  var out: AugengineProbeCoeffs;
  out.r = vec4<f32>(0.0);
  out.g = vec4<f32>(0.0);
  out.b = vec4<f32>(0.0);
  out.cov = 0.0;
  if (u_frame.probeParams.x < 0.5) { return out; }
  let gx = P.x / u_frame.probeParams.y - 0.5;
  let gz = P.z / u_frame.probeParams.y - 0.5;
  let fx0 = floor(gx);
  let fz0 = floor(gz);
  let fx = gx - fx0;
  let fz = gz - fz0;
  let i0 = i32(fx0);
  let j0 = i32(fz0);
  var baseSum = 0.0;
  var baseW = 0.0;
  for (var c = 0; c < 4; c = c + 1) {
    let di = c & 1;
    let dj = c >> 1u;
    let m = u_probes[augengine_probe_base(i0 + di, j0 + dj, 0) + 3u];
    if (!augengine_probe_ok(m, i0 + di, j0 + dj)) { continue; }
    let w = select(1.0 - fx, fx, di == 1) * select(1.0 - fz, fz, dj == 1);
    baseSum = baseSum + w * m.z;
    baseW = baseW + w;
  }
  if (baseW < 1e-4) { return out; }
  let gy = clamp((P.y - baseSum / baseW - u_frame.probeParams.z) / u_frame.probeParams.w, 0.0, 1.0);
  var wsum = 0.0;
  for (var c = 0; c < 8; c = c + 1) {
    let di = c & 1;
    let dj = (c >> 1u) & 1;
    let k = c >> 2u;
    let b = augengine_probe_base(i0 + di, j0 + dj, k);
    let m = u_probes[b + 3u];
    if (!augengine_probe_ok(m, i0 + di, j0 + dj)) { continue; }
    let w = select(1.0 - fx, fx, di == 1) * select(1.0 - fz, fz, dj == 1) * select(1.0 - gy, gy, k == 1);
    out.r = out.r + w * u_probes[b];
    out.g = out.g + w * u_probes[b + 1u];
    out.b = out.b + w * u_probes[b + 2u];
    wsum = wsum + w;
  }
  if (wsum < 1e-4) { return out; }
  out.r = out.r / wsum;
  out.g = out.g / wsum;
  out.b = out.b / wsum;
  out.cov = clamp(baseW * 2.0, 0.0, 1.0);
  return out;
}

// @pcss-begin
// Receiver-plane depth bias (Isidoro 2006): the change in shadow depth per
// unit of shadow-map uv across this pixel's surface, from the screen-space
// derivatives of its light-space position. A PCSS tap at uv offset o then
// compares against depth + dot(g, o), the receiver's own plane at the tap,
// instead of the centre depth. Without it a receiver at a grazing angle to
// the sun (a wall under a high sun) sees its own texels up-slope as blockers,
// so the filter has to stay about one texel wide, and a one-texel filter
// cannot hide the shadow-map staircase that grazing angle stretches into
// streaks. Both cascades share the light basis and scale uv and depth by the
// same radius, so one gradient serves both. Mirrors lights.ts
// receiverPlaneDepthGradient. Must run in uniform control flow (derivatives).
fn augengine_receiver_plane(worldPos: vec3<f32>) -> vec2<f32> {
  let m = u_frame.shadowViewProj;
  let m3 = mat3x3<f32>(m[0].xyz, m[1].xyz, m[2].xyz);
  let ldx = m3 * dpdx(worldPos);
  let ldy = m3 * dpdy(worldPos);
  // Clip xy -> uv: u = x/2 + 1/2, v = 1/2 - y/2.
  let ux = vec2<f32>(0.5 * ldx.x, -0.5 * ldx.y);
  let uy = vec2<f32>(0.5 * ldy.x, -0.5 * ldy.y);
  let det = ux.x * uy.y - ux.y * uy.x;
  if (abs(det) < 1e-14) { return vec2<f32>(0.0); }
  let g = vec2<f32>(uy.y * ldx.z - ux.y * ldy.z, ux.x * ldy.z - uy.x * ldx.z) / det;
  // Edge-on receivers (and quads straddling a silhouette) give unbounded
  // gradients; cap the slope (lights.ts RPDB_MAX_GRADIENT).
  let len = length(g);
  return select(g, g * (AUGENGINE_RPDB_MAX / len), len > AUGENGINE_RPDB_MAX);
}
const AUGENGINE_RPDB_MAX: f32 = ${wgslF32(RPDB_MAX_GRADIENT)};
// Smallest PCSS filter radius in shadow texels (lights.ts PCSS_MIN_FILTER_TEXELS).
// Under one texel the filter cannot smooth the map's staircase; the receiver
// plane makes a wider floor safe at grazing angles. Fixed PCF is unchanged.
const AUGENGINE_PCSS_MIN_TEXELS: f32 = ${wgslF32(PCSS_MIN_FILTER_TEXELS)};
const AUGENGINE_PCSS_ALONG_TEXELS: f32 = ${wgslF32(PCSS_ALONG_FILTER_TEXELS)};

// Contact-hardening shadow (Fernando 2005, PCSS) for an orthographic
// cascade: average the blockers in a search disc, turn the receiver–blocker
// distance (depth × the cascade's metric depth span) into a penumbra width
// with the light's angular size, then PCF over that width. Samples form a
// Vogel (golden-angle) disc generated by incremental rotation — no sample
// table (a per-invocation private array cost ~2 ms on Intel UHD) and no trig
// per tap — outermost first, rotated per pixel by interleaved gradient noise
// that changes every frame so TAA integrates it. Every tap compares against
// the receiver plane (rp) at its own offset. Mirrors lights.ts
// pcssPenumbraWidth / vogelDisc / receiverPlaneDepthGradient.
fn augengine_pcss(uv: vec2<f32>, depth: f32, texel: f32, slot: f32, cascades: f32, worldSize: f32, fragXY: vec2<f32>, rp: vec2<f32>) -> f32 {
  let taps = i32(u_frame.shadowSoft.w);
  let dims = vec2<f32>(textureDimensions(t_shadow));
  let tanL = u_frame.shadowSoft.x;
  let depthRange = 1.995 * worldSize;
  let a = fract(52.9829189 * fract(dot(fragXY + vec2<f32>(u_frame.shadowSoft2.x * 5.588238), vec2<f32>(0.06711056, 0.00583715)))) * 6.2831853;
  let cs = vec2<f32>(cos(a), sin(a));
  let searchUv = clamp(2.0 * tanL * u_frame.shadowSoft2.y / worldSize, 1.5 * texel, 24.0 * texel);
  // A tap reads the texel it lands in, whose stored depth is the plane at
  // that texel's centre, up to ~0.7 texel away: allow for that much slope.
  let quant = (abs(rp.x) + abs(rp.y)) * texel * 0.7;
  let golden = vec2<f32>(-0.7373688, 0.6754903); // cos, sin of 2.39996323 rad
  let invTaps = 1.0 / f32(taps);
  var blockSum = 0.0;
  var blockN = 0.0;
  var dir = cs;
  for (var i = 0; i < taps; i = i + 1) {
    // Most of a sunlit frame has no blocker at all: after the first four
    // (outermost, spread by the golden angle) taps find none, call it lit.
    if (i == 4 && blockN < 0.5) { return 1.0; }
    let p = dir * sqrt(1.0 - (f32(i) + 0.5) * invTaps);
    dir = vec2<f32>(dir.x * golden.x - dir.y * golden.y, dir.x * golden.y + dir.y * golden.x);
    let s = clamp(uv + p * searchUv, vec2<f32>(texel), vec2<f32>(1.0 - texel));
    let d = textureLoad(t_shadow, vec2<i32>(vec2<f32>((s.x + slot) / cascades, s.y) * dims), 0);
    let refD = depth + dot(rp, s - uv) - quant;
    if (d < refD) {
      blockSum = blockSum + (refD - d);
      blockN = blockN + 1.0;
    }
  }
  if (blockN < 0.5) { return 1.0; }
  // Every search tap blocked: deep umbra, no penumbra to filter.
  if (blockN > f32(taps) - 0.5) { return 0.0; }
  let penumbraUv = (blockSum / blockN) * depthRange * 2.0 * tanL / worldSize;
  let radius = clamp(penumbraUv * 0.5, AUGENGINE_PCSS_MIN_TEXELS * texel, 24.0 * texel);
  // Anisotropic filter floor. On a receiver tilted away from the light, the
  // map's texels stretch on screen along the receiver's slope (rp), by about
  // 1/N·L, while the staircase steps of a shadow edge repeat along the edge.
  // So widen the kernel along the receiver's iso-depth line (t), where blur
  // is not magnified, up to AUGENGINE_PCSS_ALONG_TEXELS as the slope grows,
  // and keep it tight along the slope (n) so the edge stays crisp.
  let rpLen = length(rp);
  let n = select(vec2<f32>(1.0, 0.0), rp / max(rpLen, 1e-6), rpLen > 1e-3);
  let t = vec2<f32>(-n.y, n.x);
  let radiusT = max(radius, mix(AUGENGINE_PCSS_MIN_TEXELS, AUGENGINE_PCSS_ALONG_TEXELS, clamp(rpLen * 0.5, 0.0, 1.0)) * texel);
  var lit = 0.0;
  dir = cs;
  for (var i = 0; i < taps; i = i + 1) {
    let p = dir * sqrt(1.0 - (f32(i) + 0.5) * invTaps);
    dir = vec2<f32>(dir.x * golden.x - dir.y * golden.y, dir.x * golden.y + dir.y * golden.x);
    let s = clamp(uv + t * (p.x * radiusT) + n * (p.y * radius), vec2<f32>(texel), vec2<f32>(1.0 - texel));
    lit = lit + textureSampleCompareLevel(t_shadow, s_shadow, vec2<f32>((s.x + slot) / cascades, s.y), depth + dot(rp, s - uv) - quant);
  }
  return lit / f32(taps);
}

// @pcss-end
fn augengine_shadow_cascade(lp: vec4<f32>, texel: f32, bias: f32, slot: f32, cascades: f32, worldSize: f32, fragXY: vec2<f32>, rp: vec2<f32>) -> f32 {
  let uv = vec2<f32>(lp.x * 0.5 + 0.5, 0.5 - lp.y * 0.5);
  let depth = lp.z - bias;
  if (uv.x <= 0.0 || uv.x >= 1.0 || uv.y <= 0.0 || uv.y >= 1.0) { return -1.0; }
  if (depth >= 1.0) { return 1.0; }
  // Keep PCF taps inside this cascade's half of the atlas.
  let inset = clamp(uv, vec2<f32>(texel * 1.5), vec2<f32>(1.0 - texel * 1.5));
  // @pcss-begin
  if (u_frame.shadowSoft.w > 0.5) { return augengine_pcss(inset, depth + dot(rp, inset - uv), texel, slot, cascades, worldSize, fragXY, rp); }
  // @pcss-end
  let auv = vec2<f32>((inset.x + slot) / cascades, inset.y);
  let taps = i32(u_frame.shadowParams.w);
  if (taps <= 1) {
    return textureSampleCompareLevel(t_shadow, s_shadow, auv, depth);
  }
  let step = vec2<f32>(texel / cascades, texel);
  var sum = 0.0;
  for (var y = -1; y <= 1; y = y + 1) {
    for (var x = -1; x <= 1; x = x + 1) {
      sum = sum + textureSampleCompareLevel(t_shadow, s_shadow, auv + vec2<f32>(f32(x), f32(y)) * step, depth);
    }
  }
  return sum / 9.0;
}

fn augengine_shadow(worldPos: vec3<f32>, N: vec3<f32>, fragXY: vec2<f32>) -> f32 {
  if (u_frame.lightColor.a < 0.5) { return 1.0; }
  // Receiver plane for PCSS, taken before any per-pixel branch (derivatives
  // need uniform control flow). Fixed-PCF qualities skip it.
  var rp = vec2<f32>(0.0);
  // @pcss-begin
  if (u_frame.shadowSoft.w > 0.5) { rp = augengine_receiver_plane(worldPos); }
  // @pcss-end
  let L = normalize(-u_frame.lightDir.xyz);
  // Facing away from the light: every caller multiplies by N·L ≤ 0 (or, for
  // toon, treats it as the dark side), so skip the shadow taps entirely.
  if (dot(N, L) <= 0.0) { return 1.0; }
  let grazing = 1.0 - clamp(dot(N, L), 0.0, 1.0);
  let cascades = select(1.0, 2.0, u_frame.counts.z > 1.5);
  let p0 = worldPos + N * (u_frame.shadowParams.z * (0.25 + grazing));
  let lp0 = u_frame.shadowViewProj * vec4<f32>(p0, 1.0);
  let near = augengine_shadow_cascade(lp0, u_frame.shadowParams.x, u_frame.shadowParams.y, 0.0, cascades, u_frame.shadowSoft.y, fragXY, rp);
  if (cascades < 1.5) { return select(near, 1.0, near < 0.0); }
  // Far cascade where the near one does not reach, and a blend across the
  // near cascade's outer rim so the resolution change has no visible seam.
  let p1 = worldPos + N * (u_frame.shadowParams2.z * (0.25 + grazing));
  let lp1 = u_frame.shadowViewProj2 * vec4<f32>(p1, 1.0);
  if (near < 0.0) {
    let far = augengine_shadow_cascade(lp1, u_frame.shadowParams2.x, u_frame.shadowParams2.y, 1.0, cascades, u_frame.shadowSoft.z, fragXY, rp);
    return select(far, 1.0, far < 0.0);
  }
  let edge = max(abs(lp0.x), abs(lp0.y));
  let w = smoothstep(1.0 - u_frame.shadowParams2.w, 1.0, edge);
  if (w <= 0.0) { return near; }
  let far = augengine_shadow_cascade(lp1, u_frame.shadowParams2.x, u_frame.shadowParams2.y, 1.0, cascades, u_frame.shadowSoft.z, fragXY, rp);
  return mix(near, select(far, 1.0, far < 0.0), w);
}

// Ambient for a surface normal: the flat frame ambient, or — when the sky-shaped
// ambient is on — the zonal SH of the sky: k0 + k1·n.y + k2·(3·n.y² − 1).
fn augengine_ambient_sky(N: vec3<f32>) -> vec3<f32> {
  // @natural-begin
  if (u_frame.envParams.x > 0.5) { return augengine_env_irradiance(N); }
  // @natural-end
  if (u_frame.ambientSH[0].a < 0.5) { return u_frame.ambient.rgb; }
  let y = N.y;
  return max(u_frame.ambientSH[0].rgb + u_frame.ambientSH[1].rgb * y + u_frame.ambientSH[2].rgb * (3.0 * y * y - 1.0), vec3<f32>(0.0));
}

// @natural-begin
// Natural-light diffuse: SH9 of the sky / HDR environment, irradiance / π.
fn augengine_env_irradiance(N: vec3<f32>) -> vec3<f32> {
  var r = u_frame.envSH[0].rgb + u_frame.envSH[1].rgb * N.y + u_frame.envSH[2].rgb * N.z + u_frame.envSH[3].rgb * N.x;
  r = r + u_frame.envSH[4].rgb * (N.x * N.y) + u_frame.envSH[5].rgb * (N.y * N.z) + u_frame.envSH[6].rgb * (3.0 * N.z * N.z - 1.0);
  r = r + u_frame.envSH[7].rgb * (N.x * N.z) + u_frame.envSH[8].rgb * (N.x * N.x - N.y * N.y);
  return max(r, vec3<f32>(0.0));
}

// Natural-light image-based lighting for PBR: split-sum specular from the
// prefiltered cube (lod = roughness × (mips − 1)), the integrated DFG LUT, and
// Fdez-Agüera's (2019) multiple-scattering compensation so rough and metallic
// surfaces do not lose the energy single-scatter GGX drops. irr = diffuse
// irradiance / π (sky SH9 or probes). Mirrors ibl-brdf.ts.
fn augengine_env_ibl(N: vec3<f32>, V: vec3<f32>, albedo: vec3<f32>, metallic: f32, roughness: f32, F0: vec3<f32>, irr: vec3<f32>) -> vec3<f32> {
  let R = reflect(-V, N);
  let nDotV = clamp(dot(N, V), 1e-4, 1.0);
  let pre = textureSampleLevel(t_env, s_env, R, roughness * u_frame.envParams.y).rgb * u_frame.envParams.z;
  let AB = textureSampleLevel(t_dfg, s_env, vec2<f32>(nDotV, roughness), 0.0).rg;
  let FssEss = F0 * AB.x + AB.y;
  let Ems = clamp(1.0 - (AB.x + AB.y), 0.0, 1.0);
  let Favg = F0 + (vec3<f32>(1.0) - F0) / 21.0;
  let multi = FssEss * Favg / (vec3<f32>(1.0) - Ems * Favg) * Ems * u_frame.envParams.w;
  let kd = albedo * (1.0 - metallic) * max(vec3<f32>(1.0) - FssEss - multi, vec3<f32>(0.0));
  // Specular occlusion from the GI probes: where they see less light than the
  // open sky (under a tree, in a doorway), dim the sky reflection by the same
  // ratio. Without probes irr is the sky irradiance and this is 1.
  let lw = vec3<f32>(0.2126, 0.7152, 0.0722);
  let specOcc = clamp(dot(irr, lw) / max(dot(augengine_env_irradiance(N), lw), 1e-4), 0.0, 1.0);
  return pre * FssEss * specOcc + irr * (multi + kd);
}

// @natural-end
// With irradiance probes on, the probes (relative to the flat ambient; their
// L1 coefficients come from the vertex stage) replace the sky/flat ambient
// where they cover the surface; outside the streamed area it fades back to
// augengine_ambient_sky.
fn augengine_ambient(N: vec3<f32>, pr: vec4<f32>, pg: vec4<f32>, pb: vec4<f32>, cov: f32) -> vec3<f32> {
  let base = augengine_ambient_sky(N);
  if (cov <= 0.0) { return base; }
  let n4 = vec4<f32>(1.0, N);
  let rel = max(vec3<f32>(dot(pr, n4), dot(pg, n4), dot(pb, n4)), vec3<f32>(0.0));
  return mix(base, u_frame.ambient.rgb * rel, cov);
}

// Split-sum sky specular (ibl.ts): the sky's zonal SH damped per band by a
// GGX-equivalent lobe for this roughness, evaluated at the reflection
// direction, times Karis's analytic environment BRDF. Zero when disabled.
fn augengine_sky_specular(N: vec3<f32>, V: vec3<f32>, F0: vec3<f32>, roughness: f32) -> vec3<f32> {
  if (u_frame.skySpecSH[0].a < 0.5) { return vec3<f32>(0.0); }
  let R = reflect(-V, N);
  let a = max(roughness * roughness, 1e-3);
  let n = max(2.0 / (a * a) - 2.0, 0.0);
  let h1 = select(0.0, exp(-1.0 / (2.0 * n)), n > 0.0);
  let h2 = select(0.0, exp(-4.0 / (2.0 * n)), n > 0.0);
  let y = R.y;
  let L = max(u_frame.skySpecSH[0].rgb + u_frame.skySpecSH[1].rgb * (h1 * y) + u_frame.skySpecSH[2].rgb * (h2 * (3.0 * y * y - 1.0)), vec3<f32>(0.0));
  let nDotV = clamp(dot(N, V), 0.0, 1.0);
  let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
  let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
  let r = roughness * c0 + c1;
  let a004 = min(r.x * r.x, exp2(-9.28 * nDotV)) * r.x + r.y;
  let AB = vec2<f32>(-1.04, 1.04) * a004 + r.zw;
  return L * (F0 * AB.x + AB.y);
}

// Point/spot light i: xyz = unit direction from the surface toward the light,
// w = attenuation (windowed inverse-square × spot cone). Zero past range.
fn augengine_local_light(i: u32, worldPos: vec3<f32>) -> vec4<f32> {
  let l = u_frame.localLights[i];
  let toLight = l.posRange.xyz - worldPos;
  let dist = length(toLight);
  let L = toLight / max(dist, 1e-4);
  let range = max(l.posRange.w, 1e-3);
  let x = clamp(1.0 - pow(dist / range, 4.0), 0.0, 1.0);
  var atten = (x * x) / (dist * dist + 1.0);
  if (l.spotDirOuter.w > -1.5) {
    let cd = dot(-L, normalize(l.spotDirOuter.xyz));
    atten = atten * smoothstep(l.spotDirOuter.w, l.radianceInner.a, cd);
  }
  return vec4<f32>(L, atten);
}
struct DrawUniforms {
  model: mat4x4<f32>,
  color: vec4<f32>,
  params: vec4<f32>,
  // emissive.rgb = self-illumination added to the shaded result (HDR, linear).
  // Values >1 survive into the bloom-extract pass, so this is the intentional
  // "glow" channel — independent of scene lighting. emissive.a reserved.
  emissive: vec4<f32>,
};
@group(1) @binding(0) var<uniform> u_draw: DrawUniforms;

// Exponential fog factor: 1 - exp(-optical depth). Returns a value in
// [0, 1] suitable for mix(lit, fogColor, factor). With density=0 the factor
// is exactly 0 so fog becomes a no-op for callers that don't opt in.
// ambient.a = height falloff k (1/m). k = 0 is uniform fog (density * distance);
// k > 0 is exponential height fog, density * e^(-k*y), integrated in closed
// form along the camera ray, so it pools low and thins with altitude.
// Fog / atmosphere applied to a lit colour. Legacy: mix toward the fog
// colour by augengine_fog_factor. Natural light: the scene's height fog plus
// per-channel aerial extinction from the sky model's Rayleigh + aerosol
// optical depths, and in-scattered light = sky ambient (fog.rgb, the horizon
// radiance) + the sun through a Henyey–Greenstein phase function, so haze
// glows toward the sun and blues with distance.
fn augengine_fog_apply(lit: vec3<f32>, worldPos: vec3<f32>) -> vec3<f32> {
  let f = augengine_fog_factor(worldPos);
  // @natural-begin
  if (u_frame.atmos.x > 0.5) { return augengine_atmosphere_apply(lit, worldPos, f); }
  // @natural-end
  return mix(lit, u_frame.fog.rgb, f);
}

// @natural-begin
fn augengine_atmosphere_apply(lit: vec3<f32>, worldPos: vec3<f32>, f: f32) -> vec3<f32> {
  let toP = worldPos - u_frame.cameraPos.xyz;
  let dist = length(toP);
  // Legacy (scene) fog under natural light: as a far "veil" it fades in by
  // smoothstep over atmos.w metres, so the aerial extinction alone sets the
  // near and middle distance and the scene keeps its contrast (lights.ts
  // renderer.ts legacyFogVeil).
  var optical = -log(max(1.0 - f, 1e-6));
  if (u_frame.atmos.w > 0.0) { optical = optical * smoothstep(0.0, u_frame.atmos.w, dist); }
  let T = exp(-(vec3<f32>(optical) + u_frame.atmosSigma.rgb * dist));
  let V = toP / max(dist, 1e-4);
  let g = u_frame.atmos.y;
  let cosT = dot(V, -u_frame.lightDir.xyz);
  let phase = (1.0 - g * g) / (12.5663706 * pow(max(1.0 + g * g - 2.0 * g * cosT, 1e-4), 1.5));
  let inscatter = u_frame.fog.rgb + u_frame.lightColor.rgb * (phase * u_frame.atmos.z);
  return lit * T + inscatter * (vec3<f32>(1.0) - T);
}

// @natural-end

fn augengine_fog_factor(worldPos: vec3<f32>) -> f32 {
  let density = u_frame.fog.a;
  if (density <= 0.0) { return 0.0; }
  let toP = worldPos - u_frame.cameraPos.xyz;
  let dist = length(toP);
  let k = u_frame.ambient.a;
  var optical = density * dist;
  if (k > 0.0) {
    let kdy = k * toP.y;
    let shape = select((1.0 - exp(-kdy)) / kdy, 1.0, abs(kdy) < 1e-4);
    optical = density * dist * exp(-k * u_frame.cameraPos.y) * shape;
  }
  return clamp(1.0 - exp(-optical), 0.0, 1.0);
}
`;

const OUTLINE_WGSL = /* wgsl */ `
struct VertexIn { @location(0) position: vec3<f32>, @location(1) normal: vec3<f32> };
struct VertexOut { @builtin(position) clipPos: vec4<f32> };
@vertex
fn vs_main(in: VertexIn) -> VertexOut {
  var out: VertexOut;
  // Inverted-hull: extrude vertex along normal by params.x = width.
  let extruded = in.position + in.normal * u_draw.params.x;
  let world = u_draw.model * vec4<f32>(extruded, 1.0);
  out.clipPos = u_frame.viewProj * world;
  return out;
}
@fragment
fn fs_main() -> @location(0) vec4<f32> { return u_draw.color; }
`;

const OUTLINE_MORPH_WGSL = /* wgsl */ `
struct VertexIn {
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @builtin(vertex_index) vertex_idx: u32,
};
struct VertexOut { @builtin(position) clipPos: vec4<f32> };
@vertex
fn vs_main(in: VertexIn) -> VertexOut {
  var out: VertexOut;
  let posOffset = augengine_morph_position(in.vertex_idx);
  let nrmOffset = augengine_morph_normal(in.vertex_idx);
  let morphedPos = in.position + posOffset;
  let morphedNrm = normalize(in.normal + nrmOffset);
  // Inverted-hull: extrude morphed vertex along morphed normal.
  let extruded = morphedPos + morphedNrm * u_draw.params.x;
  let world = u_draw.model * vec4<f32>(extruded, 1.0);
  out.clipPos = u_frame.viewProj * world;
  return out;
}
@fragment
fn fs_main() -> @location(0) vec4<f32> { return u_draw.color; }
`;
