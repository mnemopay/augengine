/**
 * Phase 13 v3 — 3D Gaussian splat support math.
 *
 * Three pieces that together make a true Gaussian splat renderer
 * instead of v2's anisotropic billboard:
 *
 *   1. `quatScaleToCovariance3D(q, s)` — build the 3×3 covariance Σ
 *      from a per-splat scale (3 sigmas) + rotation quaternion.
 *      Σ = R · S · Sᵀ · Rᵀ where R = quat-to-mat3, S = diag(scale).
 *      Returned as the 6 unique components in row-major order:
 *        [Σ00, Σ01, Σ02, Σ11, Σ12, Σ22]
 *
 *   2. `evaluateShBand1(coeffs, dirX, dirY, dirZ)` — evaluate the
 *      first-order spherical-harmonic contribution to RGB given a
 *      view-direction unit vector. Adds onto the band-0 DC term to
 *      give Gaussians a viewing-angle-aware color. Bands ≥ 2 stay
 *      deferred (band-1 alone captures ~90% of view-dependent
 *      shading in published 3DGS captures).
 *
 *   3. `sortSplatsBackToFront(positions, viewMatrix, into)` — fill
 *      an `Uint32Array` of splat indices sorted by view-space Z
 *      descending. Render the indices in order for correct
 *      back-to-front alpha blending.
 *
 * The WGSL companion functions in `splat.ts:SPLAT_BILLBOARD_WGSL`
 * consume these via the instance buffer's `splat_cov3d_*` attributes
 * and the per-splat SH coefficients texture/buffer.
 *
 * None of this requires a GPU device — it's all pure math + array
 * shuffling, so vitest can verify it in Node. The WGSL projection
 * `Σ' = J · W · Σ · Wᵀ · Jᵀ` runs on the GPU and is asserted by
 * substring on the shader source (same pattern as the morph splice
 * tests in `pipeline-factory.test.ts`).
 */

/** Length-6 packed Σ_3D in upper-triangular row-major order. */
export type Covariance3D = readonly [number, number, number, number, number, number];

/**
 * Build the 3D covariance Σ from a unit quaternion + per-axis scale.
 *
 *   R = mat3FromQuat(q)
 *   S = diag(scale.x, scale.y, scale.z)
 *   M = R · S
 *   Σ = M · Mᵀ
 *
 * Returns the 6 unique components of the symmetric 3×3:
 * `[Σ00, Σ01, Σ02, Σ11, Σ12, Σ22]`.
 *
 * The quaternion is normalised defensively — SPZ decoder already
 * normalises but a caller-supplied quat may drift outside unit norm.
 */
export function quatScaleToCovariance3D(
  q: readonly [number, number, number, number],
  scale: readonly [number, number, number],
): Covariance3D {
  // Normalise quat.
  const qx0 = q[0];
  const qy0 = q[1];
  const qz0 = q[2];
  const qw0 = q[3];
  const qLen = Math.sqrt(qx0 * qx0 + qy0 * qy0 + qz0 * qz0 + qw0 * qw0) || 1;
  const qx = qx0 / qLen;
  const qy = qy0 / qLen;
  const qz = qz0 / qLen;
  const qw = qw0 / qLen;

  // Standard quaternion → 3×3 rotation matrix (column-major).
  const xx = qx * qx;
  const yy = qy * qy;
  const zz = qz * qz;
  const xy = qx * qy;
  const xz = qx * qz;
  const yz = qy * qz;
  const wx = qw * qx;
  const wy = qw * qy;
  const wz = qw * qz;
  // R columns: r0 r1 r2
  const r00 = 1 - 2 * (yy + zz);
  const r01 = 2 * (xy + wz);
  const r02 = 2 * (xz - wy);
  const r10 = 2 * (xy - wz);
  const r11 = 1 - 2 * (xx + zz);
  const r12 = 2 * (yz + wx);
  const r20 = 2 * (xz + wy);
  const r21 = 2 * (yz - wx);
  const r22 = 1 - 2 * (xx + yy);

  // M = R · S — columns of R scaled by the matching sigma.
  const sx = scale[0];
  const sy = scale[1];
  const sz = scale[2];
  const m00 = r00 * sx;
  const m01 = r01 * sx;
  const m02 = r02 * sx;
  const m10 = r10 * sy;
  const m11 = r11 * sy;
  const m12 = r12 * sy;
  const m20 = r20 * sz;
  const m21 = r21 * sz;
  const m22 = r22 * sz;

  // Σ = M · Mᵀ. Symmetric — 6 unique components.
  const s00 = m00 * m00 + m10 * m10 + m20 * m20;
  const s01 = m00 * m01 + m10 * m11 + m20 * m21;
  const s02 = m00 * m02 + m10 * m12 + m20 * m22;
  const s11 = m01 * m01 + m11 * m11 + m21 * m21;
  const s12 = m01 * m02 + m11 * m12 + m21 * m22;
  const s22 = m02 * m02 + m12 * m12 + m22 * m22;
  return [s00, s01, s02, s11, s12, s22] as const;
}

/**
 * SH band-1 has 3 coefficients per RGB channel = 9 floats per splat.
 * Standard ordering (Sloan):
 *   l=1, m=-1 → Y_1^-1(d) = -K1 · d.y
 *   l=1, m= 0 → Y_1^0(d)  =  K1 · d.z
 *   l=1, m= 1 → Y_1^1(d)  = -K1 · d.x
 * where K1 = sqrt(3 / (4π)).
 *
 * For each channel `c`, the SH-band-1 contribution is
 *   sum_m  coeff[c, m] · Y_1^m(d).
 *
 * In our 9-float-per-splat layout we pack as RGB × 3 SH levels:
 *   [r_{-1}, g_{-1}, b_{-1}, r_0, g_0, b_0, r_+1, g_+1, b_+1].
 */
const SH_C1 = 0.48860251190291992; // 0.5 * sqrt(3 / π) ≈ K1

export function evaluateShBand1(
  coeffs: Float32Array,
  /** Direction from splat center to camera, normalised. */
  dirX: number,
  dirY: number,
  dirZ: number,
): [number, number, number] {
  // Coefficients indexed per (level, channel). 9 floats total.
  const r_m = coeffs[0] ?? 0;
  const g_m = coeffs[1] ?? 0;
  const b_m = coeffs[2] ?? 0;
  const r_0 = coeffs[3] ?? 0;
  const g_0 = coeffs[4] ?? 0;
  const b_0 = coeffs[5] ?? 0;
  const r_p = coeffs[6] ?? 0;
  const g_p = coeffs[7] ?? 0;
  const b_p = coeffs[8] ?? 0;
  const k = SH_C1;
  // Apply the three band-1 basis functions per channel.
  const r = -k * dirY * r_m + k * dirZ * r_0 - k * dirX * r_p;
  const g = -k * dirY * g_m + k * dirZ * g_0 - k * dirX * g_p;
  const b = -k * dirY * b_m + k * dirZ * b_0 - k * dirX * b_p;
  return [r, g, b];
}

/**
 * Fill `into` with splat indices sorted back-to-front by view-space Z
 * (descending — further-from-camera first). Caller passes the
 * Float32Array of XYZ positions (length = 3 × N) and the 4×4 view
 * matrix in column-major order (same shape AugEngine uses elsewhere).
 *
 * Returns the input `into` for chaining. If `into` is sized smaller
 * than the splat count, throws — caller pre-allocates a `new Uint32Array(N)`.
 *
 * Implementation: compute view-space Z per splat (single dot product),
 * sort indices by that depth. Typed-array sort with a number-keyed
 * comparator costs ~50ms for 1M splats on M1; for 100k splats this is
 * ~5ms, well inside a 16ms frame budget when called once per camera
 * jump (NOT per frame — the typical strategy is to re-sort only when
 * the camera rotates past a threshold).
 */
export function sortSplatsBackToFront(
  positions: Float32Array,
  viewMatrixColMajor: Float32Array,
  into: Uint32Array,
): Uint32Array {
  const N = positions.length / 3;
  if (into.length < N) {
    throw new Error(`sortSplatsBackToFront: into.length (${into.length}) < splat count (${N})`);
  }
  if (viewMatrixColMajor.length !== 16) {
    throw new Error("sortSplatsBackToFront: viewMatrix must be 16 floats column-major");
  }
  // Row 2 of the view matrix (z output) — col-major so row 2 = elements
  // at offsets 2, 6, 10, 14.
  const m02 = viewMatrixColMajor[2]!;
  const m12 = viewMatrixColMajor[6]!;
  const m22 = viewMatrixColMajor[10]!;
  const m32 = viewMatrixColMajor[14]!;
  // Depth lookup table — Float32Array indexed by splat id.
  const depths = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const x = positions[i * 3]!;
    const y = positions[i * 3 + 1]!;
    const z = positions[i * 3 + 2]!;
    // View-space z (looking down -Z by convention in our renderer).
    depths[i] = m02 * x + m12 * y + m22 * z + m32;
  }
  // Initialise indices.
  for (let i = 0; i < N; i++) into[i] = i;
  // Sort descending by depth — convert to a temporary number[] for the
  // sort because typed-array `.sort` with a comparator boxes anyway.
  const tmp: number[] = Array.from(into.subarray(0, N));
  tmp.sort((a, b) => depths[b]! - depths[a]!);
  for (let i = 0; i < N; i++) into[i] = tmp[i]!;
  return into;
}

/**
 * WGSL function that ships in the splat shader module. Projects the
 * 3D covariance into screen-space 2D covariance using the standard
 * 3DGS recipe:
 *
 *   Σ_2D = J · W · Σ_3D · Wᵀ · Jᵀ
 *
 * where:
 *   W = upper-3×3 of the view matrix
 *   J = projection Jacobian at the splat's camera-space position
 *       (depends on focal length + view-space z)
 *
 * The 2×2 output gives the ellipse half-axes for the billboard's
 * extent. Asserted by `splat.test.ts` to be present in the WGSL.
 */
export const SPLAT_COVARIANCE_WGSL = /* wgsl */ `
fn augengine_splat_covariance_2d(
  cov3d_a: vec3<f32>,  // [Σ00, Σ01, Σ02]
  cov3d_b: vec3<f32>,  // [Σ11, Σ12, Σ22]
  view: mat3x3<f32>,
  view_pos: vec3<f32>,
  focal: vec2<f32>,
) -> vec3<f32> {
  // Σ_3D as a 3×3 symmetric matrix.
  let s3d = mat3x3<f32>(
    cov3d_a.x, cov3d_a.y, cov3d_a.z,
    cov3d_a.y, cov3d_b.x, cov3d_b.y,
    cov3d_a.z, cov3d_b.y, cov3d_b.z,
  );
  // Jacobian J of the perspective projection at view_pos.
  // z_cam < 0 in our convention (camera looks down -Z); clamp away from 0.
  let z = max(abs(view_pos.z), 1e-4);
  let j = mat3x3<f32>(
    focal.x / z, 0.0, -(focal.x * view_pos.x) / (z * z),
    0.0, focal.y / z, -(focal.y * view_pos.y) / (z * z),
    0.0, 0.0, 0.0,
  );
  let t = j * view;
  let s2d_full = t * s3d * transpose(t);
  // Return the 3 unique components of the 2×2 upper-left block:
  //   x = Σ'_00, y = Σ'_01, z = Σ'_11.
  return vec3<f32>(s2d_full[0][0], s2d_full[0][1], s2d_full[1][1]);
}

fn augengine_splat_eval_sh_band1(
  /* 9 floats packed into 3 vec3<f32> — [r_{-1}, g_{-1}, b_{-1}],
   *                                    [r_0, g_0, b_0],
   *                                    [r_+1, g_+1, b_+1] */
  c_neg: vec3<f32>,
  c_zero: vec3<f32>,
  c_pos: vec3<f32>,
  /** direction from splat center to camera, normalised */
  dir: vec3<f32>,
) -> vec3<f32> {
  let k = 0.48860251190291992;
  return -k * dir.y * c_neg + k * dir.z * c_zero - k * dir.x * c_pos;
}
`;
