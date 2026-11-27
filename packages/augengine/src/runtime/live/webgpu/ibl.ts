/**
 * Sky-shaped ambient light: the diffuse half of image-based lighting for a
 * procedural sky.
 *
 * The sky in this renderer is a vertical gradient (sky.ts), so its radiance
 * depends only on a direction's elevation y. Its irradiance therefore lives
 * entirely in the zonal (m = 0) spherical-harmonic bands; L0–L2 capture
 * ~99% of a smooth gradient's irradiance (Ramamoorthi & Hanrahan 2001). We
 * project L(y) onto Y00, Y10, Y20 numerically, convolve with the clamped
 * cosine (A0 = π, A1 = 2π/3, A2 = π/4), and divide by π so the result is the
 * same "albedo × ambient" quantity the material shaders already use.
 *
 * The result is then RESCALED so its average over all normals equals the
 * scene's existing flat ambient: brightness stays where the scene was tuned,
 * and what changes is direction — surfaces facing up get the sky's colour,
 * surfaces facing down get the darker ground bounce.
 *
 * Specular (packSkySpecular): the split-sum approximation (Karis 2013) with
 * both halves analytic, because the sky is a function of elevation only:
 *  - prefiltered radiance: the same zonal L0–L2 coefficients of RAW radiance,
 *    each band damped by a GGX-equivalent Phong lobe, h_l = exp(−l²/2n) with
 *    n = 2/α² − 2 (a Gaussian lobe in SH). Rough metals see the L0 average,
 *    smooth ones the full gradient (as band-limited as the diffuse term).
 *  - environment BRDF: Karis's analytic fit of the DFG LUT ("Physically Based
 *    Shading on Mobile", 2014) instead of a sampled LUT texture.
 *
 * Not included: anything that depends on azimuth (a sun disc, buildings) and
 * local reflection/irradiance probes — the sky is the only environment.
 */

import type { Vec3 } from "../../../scene/types.js";
import type { SkyGradient } from "./sky.js";

const Y00 = 0.28209479177387814;
const Y1 = 0.4886025119029199;
const Y2 = 0.31539156525252005;
const A0 = Math.PI;
const A1 = (2 * Math.PI) / 3;
const A2 = Math.PI / 4;

/** Radiance of the gradient sky along a direction with elevation y (sky.ts math). */
export function skyRadiance(g: SkyGradient, y: number, groundBounce: number): Vec3 {
  if (y < 0) {
    // Below the horizon: the ground, lit by the sky, reflects a fraction of the
    // horizon colour (a sky.ts render shows the bare horizon colour there).
    return [g.horizon[0] * groundBounce, g.horizon[1] * groundBounce, g.horizon[2] * groundBounce];
  }
  const up = Math.min(1, y);
  const t = Math.pow(up, Math.max(g.exponent, 0.05));
  const glow = Math.exp(-up * Math.max(g.glowFalloff, 0.01));
  return [
    g.horizon[0] + (g.zenith[0] - g.horizon[0]) * t + g.horizonGlow[0] * glow,
    g.horizon[1] + (g.zenith[1] - g.horizon[1]) * t + g.horizonGlow[1] * glow,
    g.horizon[2] + (g.zenith[2] - g.horizon[2]) * t + g.horizonGlow[2] * glow,
  ];
}

/**
 * Zonal SH (L0..L2) of the cosine-convolved radiance, divided by π:
 * ambient(n) = k0 + k1·n.y + k2·(3·n.y² − 1). Returns [k0, k1, k2] per channel.
 * `radiance(y)` is integrated over the sphere with `samples` midpoint steps in y
 * (dω = 2π dy for a function of elevation only).
 */
export function zonalAmbientSH(radiance: (y: number) => Vec3, samples = 256): [Vec3, Vec3, Vec3] {
  const c0 = [0, 0, 0], c1 = [0, 0, 0], c2 = [0, 0, 0];
  const dy = 2 / samples;
  for (let i = 0; i < samples; i++) {
    const y = -1 + (i + 0.5) * dy;
    const L = radiance(y);
    const w0 = 2 * Math.PI * Y00 * dy;
    const w1 = 2 * Math.PI * Y1 * y * dy;
    const w2 = 2 * Math.PI * Y2 * (3 * y * y - 1) * dy;
    for (let ch = 0; ch < 3; ch++) {
      c0[ch]! += L[ch]! * w0;
      c1[ch]! += L[ch]! * w1;
      c2[ch]! += L[ch]! * w2;
    }
  }
  // E(n) = Σ A_l c_l Y_l(n);  ambient = E / π.
  const k0 = c0.map((c) => (A0 * c * Y00) / Math.PI) as unknown as Vec3;
  const k1 = c1.map((c) => (A1 * c * Y1) / Math.PI) as unknown as Vec3;
  const k2 = c2.map((c) => (A2 * c * Y2) / Math.PI) as unknown as Vec3;
  return [k0, k1, k2];
}

/** Evaluate the packed zonal ambient for a normal with vertical component ny. */
export function evalZonalAmbient(k: [Vec3, Vec3, Vec3], ny: number): Vec3 {
  const p2 = 3 * ny * ny - 1;
  return [0, 1, 2].map((c) => k[0][c]! + k[1][c]! * ny + k[2][c]! * p2) as unknown as Vec3;
}

/**
 * Sky-shaped ambient, rescaled per channel so its sphere average equals
 * `ambient` (the average of the L1 and L2 terms over the sphere is 0, so the
 * average is exactly k0). Packed as 12 floats: k0.rgb + enabled flag, k1.rgb,
 * k2.rgb — the FRAME_LAYOUT.ambientSH block.
 */
export function packSkyAmbient(out: Float32Array, offset: number, gradient: SkyGradient | null, ambient: Vec3, groundBounce = 0.35): void {
  // Without a sky, a neutral dome: full ambient overhead, the bounce below.
  const g: SkyGradient = gradient ?? { zenith: [1, 1, 1], horizon: [0.8, 0.8, 0.8], horizonGlow: [0, 0, 0], exponent: 1, glowFalloff: 1 };
  const k = zonalAmbientSH((y) => skyRadiance(g, y, groundBounce));
  for (let c = 0; c < 3; c++) {
    const scale = k[0][c]! > 1e-8 ? ambient[c]! / k[0][c]! : 0;
    out[offset + c] = k[0][c]! * scale;
    out[offset + 4 + c] = k[1][c]! * scale;
    out[offset + 8 + c] = k[2][c]! * scale;
  }
  out[offset + 3] = 1;
  out[offset + 7] = 0;
  out[offset + 11] = 0;
}

/**
 * Zonal SH (L0..L2) of the RAW sky radiance (no cosine lobe): the shader damps
 * each band by the GGX-equivalent lobe for the surface roughness and evaluates
 * at the reflection direction. Returns [c0, c1, c2] already multiplied by
 * their Y_l normalisation, so L(y) ≈ c0 + c1·y + c2·(3y² − 1).
 */
export function zonalRadianceSH(radiance: (y: number) => Vec3, samples = 256): [Vec3, Vec3, Vec3] {
  const c0 = [0, 0, 0], c1 = [0, 0, 0], c2 = [0, 0, 0];
  const dy = 2 / samples;
  for (let i = 0; i < samples; i++) {
    const y = -1 + (i + 0.5) * dy;
    const L = radiance(y);
    for (let ch = 0; ch < 3; ch++) {
      c0[ch]! += L[ch]! * 2 * Math.PI * Y00 * dy;
      c1[ch]! += L[ch]! * 2 * Math.PI * Y1 * y * dy;
      c2[ch]! += L[ch]! * 2 * Math.PI * Y2 * (3 * y * y - 1) * dy;
    }
  }
  return [
    c0.map((c) => c * Y00) as unknown as Vec3,
    c1.map((c) => c * Y1) as unknown as Vec3,
    c2.map((c) => c * Y2) as unknown as Vec3,
  ];
}

/** GGX-equivalent Phong-lobe damping of SH band l for roughness (perceptual) r. */
export function specularBandWeight(l: number, roughness: number): number {
  const a = Math.max(roughness * roughness, 1e-3);
  const n = Math.max(2 / (a * a) - 2, 0);
  return n <= 0 ? (l === 0 ? 1 : 0) : Math.exp(-(l * l) / (2 * n));
}

/**
 * Karis's analytic environment BRDF (mobile fit of the split-sum DFG LUT):
 * returns [A, B] with specular = prefiltered × (F0·A + B).
 */
export function envBrdfApprox(roughness: number, nDotV: number): [number, number] {
  const c0 = [-1, -0.0275, -0.572, 0.022];
  const c1 = [1, 0.0425, 1.04, -0.04];
  const r = [roughness * c0[0]! + c1[0]!, roughness * c0[1]! + c1[1]!, roughness * c0[2]! + c1[2]!, roughness * c0[3]! + c1[3]!];
  const a004 = Math.min(r[0]! * r[0]!, Math.pow(2, -9.28 * nDotV)) * r[0]! + r[1]!;
  return [a004 * -1.04 + r[2]!, a004 * 1.04 + r[3]!];
}

/**
 * Sky specular coefficients for the frame uniform (FRAME_LAYOUT.skySpecSH):
 * raw-radiance zonal SH scaled by the same per-channel factor the diffuse
 * ambient was normalised with, so specular and diffuse stay consistent with
 * the scene's tuned ambient level. [0].a = enabled.
 */
export function packSkySpecular(out: Float32Array, offset: number, gradient: SkyGradient | null, ambient: Vec3, groundBounce = 0.35): void {
  const g: SkyGradient = gradient ?? { zenith: [1, 1, 1], horizon: [0.8, 0.8, 0.8], horizonGlow: [0, 0, 0], exponent: 1, glowFalloff: 1 };
  const rad = (y: number) => skyRadiance(g, y, groundBounce);
  const kd = zonalAmbientSH(rad);
  const ks = zonalRadianceSH(rad);
  for (let c = 0; c < 3; c++) {
    const scale = kd[0][c]! > 1e-8 ? ambient[c]! / kd[0][c]! : 0;
    out[offset + c] = ks[0][c]! * scale;
    out[offset + 4 + c] = ks[1][c]! * scale;
    out[offset + 8 + c] = ks[2][c]! * scale;
  }
  out[offset + 3] = 1;
  out[offset + 7] = 0;
  out[offset + 11] = 0;
}
