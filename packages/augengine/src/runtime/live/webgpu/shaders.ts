/** WGSL shaders for Phase 1 — minimal unlit + simple lambert. */

/** Per-frame uniforms shared by all draws in a pass. */
export const FRAME_UNIFORM_WGSL = /* wgsl */ `
struct FrameUniforms {
  viewProj: mat4x4<f32>,
  lightDir: vec4<f32>,    // xyz = direction (normalized), w unused
  ambient:  vec4<f32>,    // rgb = ambient color, a unused
};

@group(0) @binding(0) var<uniform> u_frame: FrameUniforms;
`;

/** Per-draw uniforms — model matrix + base color. */
export const DRAW_UNIFORM_WGSL = /* wgsl */ `
struct DrawUniforms {
  model: mat4x4<f32>,
  color: vec4<f32>,       // rgb = base color, a = unlit/lit flag (0=unlit, 1=lit)
};

@group(1) @binding(0) var<uniform> u_draw: DrawUniforms;
`;

export const VERTEX_WGSL = /* wgsl */ `
struct VertexIn {
  @location(0) position: vec3<f32>,
  @location(1) normal:   vec3<f32>,
};

struct VertexOut {
  @builtin(position) clipPos: vec4<f32>,
  @location(0) worldNormal:   vec3<f32>,
};

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
  var out: VertexOut;
  let world = u_draw.model * vec4<f32>(in.position, 1.0);
  out.clipPos = u_frame.viewProj * world;
  // Cheap normal transform; assumes uniform scale (Phase 1 shortcut).
  let nw = u_draw.model * vec4<f32>(in.normal, 0.0);
  out.worldNormal = normalize(nw.xyz);
  return out;
}
`;

export const FRAGMENT_WGSL = /* wgsl */ `
@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
  let base = u_draw.color.rgb;
  if (u_draw.color.a < 0.5) {
    return vec4<f32>(base, 1.0);
  }
  let nDotL = max(dot(in.worldNormal, normalize(u_frame.lightDir.xyz)), 0.0);
  let lit = base * (u_frame.ambient.rgb + nDotL);
  return vec4<f32>(lit, 1.0);
}
`;

/** Concatenated module source for createShaderModule. */
export const AUGENGINE_SHADER_MODULE_WGSL =
  FRAME_UNIFORM_WGSL +
  DRAW_UNIFORM_WGSL +
  VERTEX_WGSL +
  FRAGMENT_WGSL;
