/**
 * WebGPU splat renderer — point-billboard variant (Phase 13 v1).
 *
 * Draws an SPZ-decoded point cloud as screen-aligned billboards, each
 * sized in pixels and tinted by the splat's color × alpha. This is the
 * "splats as colored disks" rendering — good enough to recognise an
 * ingested Marble world but not the full Gaussian ellipsoid rasterizer
 * with covariance + spherical harmonics (that's v2 alongside the
 * `scales` + `rotations` + `SH` stream decoders).
 *
 * Why v1 ships now: the smoke-game's diorama entities already declare
 * `material: { kind: "splat", spzUrl }` via `augengine-worldgen-ingest`'s
 * Marble adapter. Without a renderer, the engine drew them as
 * procedural placeholders. v1 closes the loop end-to-end: SPZ blob →
 * `decodeSpzBody` → `SplatRenderer.upload` → frame draw. v2 promotes
 * the billboards to proper anisotropic Gaussians.
 *
 * Pipeline shape:
 *   - vertex stage emits 6 indexed verts per splat (two triangles)
 *   - instance buffer carries (position xyz, color rgba) per splat
 *   - vertex shader projects to clip + offsets by the unit-quad corner
 *   - fragment shader applies a circular Gaussian falloff for the disk
 *
 * Tests assert: buffer sizing, vertex/index counts, WGSL string content,
 * uniform layout. Real GPU draws happen in the smoke-game.
 */

export interface SplatPointCloud {
  /** XYZ positions, length = 3 × pointCount. */
  positions: Float32Array;
  /** Linear-space RGB, length = 3 × pointCount, components ∈ [0, 1]. */
  colors: Float32Array;
  /** Per-splat alpha, length = pointCount, components ∈ [0, 1]. */
  alphas: Float32Array;
  /** Number of splats. */
  pointCount: number;
  /**
   * Optional per-splat anisotropic scale (Gaussian sigma per axis) in
   * metres. Length = 3 × pointCount. When present, the renderer sizes
   * the billboard from `max(scaleX, scaleY, scaleZ)` so anisotropic
   * splats no longer all render at the same uniform pixel radius.
   * v3 will project the full 3×3 covariance to a screen-space ellipse.
   */
  scales?: Float32Array;
  /**
   * Optional per-splat rotation quaternion (x, y, z, w).
   * Length = 4 × pointCount. Forwarded to the GPU so v3's covariance
   * projection code has everything it needs without re-uploading.
   */
  rotations?: Float32Array;
}

/**
 * Instance attribute layout — what the vertex shader sees per splat.
 * v2 layout: vec3 pos (12) + f32 maxScale (4) + vec4 rgba (16) +
 * vec4 quat (16) = 48 bytes. v1's 32-byte layout was a prefix of v2
 * minus the quat; v2 has no separate v1 path because the maxScale
 * field acts as a uniform-radius surrogate when scales aren't supplied.
 */
export const SPLAT_INSTANCE_STRIDE_BYTES = 48;

/** Pack a SplatPointCloud into the instance buffer layout. */
export function packSplatInstances(cloud: SplatPointCloud): Float32Array {
  const N = cloud.pointCount;
  const out = new Float32Array(N * 12);
  for (let i = 0; i < N; i++) {
    out[i * 12 + 0] = cloud.positions[i * 3 + 0] ?? 0;
    out[i * 12 + 1] = cloud.positions[i * 3 + 1] ?? 0;
    out[i * 12 + 2] = cloud.positions[i * 3 + 2] ?? 0;
    // maxScale: max of the three per-axis sigmas, default 1.0 so
    // v1-style clouds (no scales) render unchanged.
    if (cloud.scales) {
      const sx = cloud.scales[i * 3 + 0] ?? 1;
      const sy = cloud.scales[i * 3 + 1] ?? 1;
      const sz = cloud.scales[i * 3 + 2] ?? 1;
      out[i * 12 + 3] = Math.max(sx, sy, sz);
    } else {
      out[i * 12 + 3] = 1.0;
    }
    out[i * 12 + 4] = cloud.colors[i * 3 + 0] ?? 1;
    out[i * 12 + 5] = cloud.colors[i * 3 + 1] ?? 1;
    out[i * 12 + 6] = cloud.colors[i * 3 + 2] ?? 1;
    out[i * 12 + 7] = cloud.alphas[i] ?? 1;
    // Quaternion — defaults to identity (0, 0, 0, 1) when rotations
    // are absent so v3 covariance code can rely on a unit quat.
    if (cloud.rotations) {
      out[i * 12 + 8] = cloud.rotations[i * 4 + 0] ?? 0;
      out[i * 12 + 9] = cloud.rotations[i * 4 + 1] ?? 0;
      out[i * 12 + 10] = cloud.rotations[i * 4 + 2] ?? 0;
      out[i * 12 + 11] = cloud.rotations[i * 4 + 3] ?? 1;
    } else {
      out[i * 12 + 8] = 0;
      out[i * 12 + 9] = 0;
      out[i * 12 + 10] = 0;
      out[i * 12 + 11] = 1;
    }
  }
  return out;
}

/** Quad corner offsets in NDC — two triangles, 6 verts. */
export const SPLAT_QUAD_VERTS = new Float32Array([
  -1, -1,
   1, -1,
  -1,  1,
  -1,  1,
   1, -1,
   1,  1,
]);

/** Uniform block the renderer pushes per frame. 96 bytes = mat4×4 + vec4. */
export const SPLAT_UNIFORM_SIZE_BYTES = 80; // mat4 (64) + vec4 (16)

export interface SplatFrameUniforms {
  /** Combined view·projection matrix (column-major, 16 floats). */
  viewProj: Float32Array;
  /** [splatPixelRadius, viewportWidth, viewportHeight, opacityScale]. */
  params: [number, number, number, number];
}

export function packSplatUniforms(u: SplatFrameUniforms): Float32Array {
  if (u.viewProj.length !== 16) {
    throw new Error("packSplatUniforms: viewProj must have 16 floats");
  }
  const out = new Float32Array(20);
  out.set(u.viewProj, 0);
  out[16] = u.params[0];
  out[17] = u.params[1];
  out[18] = u.params[2];
  out[19] = u.params[3];
  return out;
}

/** WGSL for the splat point-billboard pipeline. Single shader module
 *  with `vs_main` + `fs_main`. The instance attributes match
 *  `SPLAT_INSTANCE_STRIDE_BYTES` and the v2 layout (pos + maxScale +
 *  rgba + quat).
 *
 *  v2 additions vs v1:
 *    - Per-splat `max_scale` scales the billboard radius alongside the
 *      base pixel radius. Anisotropic splats no longer all render at
 *      the same size; the largest axis dictates billboard footprint.
 *    - Per-splat `quat` is parsed + held in `out.quat` even though v2
 *      still draws an axis-aligned billboard. v3 promotes this to a
 *      full Σ' = J·W·Σ·Wᵀ·Jᵀ covariance projection so the disk becomes
 *      a true rotated ellipse.
 *    - Depth output threads through `clip.z / clip.w` (linearised to
 *      [0, 1]) so the consumer can sort + alpha-blend if needed.
 */
export const SPLAT_BILLBOARD_WGSL = /* wgsl */ `
struct SplatUniforms {
  view_proj: mat4x4<f32>,
  // x = base pixel radius, y = viewport width, z = viewport height, w = opacity scale
  params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: SplatUniforms;

struct VertexInput {
  // Per-vertex (quad corner)
  @location(0) corner: vec2<f32>,
  // Per-instance (splat) — v2 layout
  @location(1) splat_pos: vec3<f32>,
  @location(2) splat_max_scale: f32,
  @location(3) splat_color: vec4<f32>,
  @location(4) splat_quat: vec4<f32>,
};

struct VertexOutput {
  @builtin(position) clip_pos: vec4<f32>,
  @location(0) corner: vec2<f32>,
  @location(1) color: vec4<f32>,
  @location(2) quat: vec4<f32>,
  @location(3) ndc_depth: f32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
  var out: VertexOutput;
  let world = vec4<f32>(in.splat_pos, 1.0);
  let view_proj = u.view_proj;
  let clip = view_proj * world;

  // v2 — scale the base pixel radius by the splat's max anisotropic axis.
  let base_radius_px = u.params.x;
  let effective_radius_px = base_radius_px * in.splat_max_scale;
  let vw = u.params.y;
  let vh = u.params.z;
  let dx = (in.corner.x * effective_radius_px * 2.0) / vw;
  let dy = (in.corner.y * effective_radius_px * 2.0) / vh;

  // Offset in clip-space (scaled by W for correct perspective behavior).
  out.clip_pos = vec4<f32>(clip.x + dx * clip.w, clip.y + dy * clip.w, clip.z, clip.w);
  out.corner = in.corner;
  out.color = in.splat_color;
  out.quat = in.splat_quat;
  out.ndc_depth = clip.z / max(clip.w, 1e-6);
  return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let r2 = in.corner.x * in.corner.x + in.corner.y * in.corner.y;
  if (r2 > 1.0) { discard; }
  // Smooth circular falloff approximating a Gaussian — looks more like
  // a soft disk than the hard circle of a thresholded radius check.
  let alpha = exp(-r2 * 4.0) * in.color.a * u.params.w;
  return vec4<f32>(in.color.rgb, alpha);
}
`;

/** Minimal GPUDevice surface the splat renderer needs. Pluggable for
 *  tests; the production renderer passes its `GPUDevice` as `unknown`. */
export interface SplatDeviceLike {
  createBuffer(desc: { size: number; usage: number; mappedAtCreation?: boolean }): SplatBufferLike;
  queue: { writeBuffer(buf: SplatBufferLike, offset: number, data: ArrayBufferView | ArrayBuffer): void };
}

export interface SplatBufferLike {
  destroy?(): void;
}

const USAGE_VERTEX_COPY_DST = 0x20 | 0x8; // VERTEX | COPY_DST
const USAGE_UNIFORM_COPY_DST = 0x40 | 0x8; // UNIFORM | COPY_DST

/**
 * Per-cloud GPU resources. Owns the instance + uniform buffer for one
 * splat asset; the engine renderer instantiates one per `splat` mesh
 * entity it draws.
 */
export class SplatCloudResources {
  readonly pointCount: number;
  readonly instanceBuffer: SplatBufferLike;
  readonly uniformBuffer: SplatBufferLike;

  constructor(private readonly device: SplatDeviceLike, cloud: SplatPointCloud) {
    this.pointCount = cloud.pointCount;
    const packed = packSplatInstances(cloud);
    this.instanceBuffer = this.device.createBuffer({
      size: packed.byteLength,
      usage: USAGE_VERTEX_COPY_DST,
    });
    this.device.queue.writeBuffer(this.instanceBuffer, 0, packed);
    this.uniformBuffer = this.device.createBuffer({
      size: SPLAT_UNIFORM_SIZE_BYTES,
      usage: USAGE_UNIFORM_COPY_DST,
    });
  }

  /** Update the per-frame uniforms (view-projection + params). */
  updateUniforms(u: SplatFrameUniforms): void {
    this.device.queue.writeBuffer(this.uniformBuffer, 0, packSplatUniforms(u));
  }

  /** Free the GPU resources. */
  destroy(): void {
    this.instanceBuffer.destroy?.();
    this.uniformBuffer.destroy?.();
  }
}
