/**
 * Order-2 (9-coefficient) real spherical harmonics for diffuse environment
 * lighting — the irradiance half of image-based lighting.
 *
 * Ramamoorthi & Hanrahan, "An Efficient Representation for Irradiance
 * Environment Maps" (SIGGRAPH 2001): irradiance is the radiance convolved
 * with the clamped cosine, whose SH transfer is A0 = π, A1 = 2π/3, A2 = π/4,
 * and L0–L2 keep the average error under ~3% for any lighting.
 *
 * Coefficients are stored ALREADY convolved and divided by π, so evaluating
 * them at a normal gives the quantity a Lambertian BRDF multiplies by albedo:
 * outgoing radiance = albedo × evalSH9(c, n). A uniform environment of
 * radiance L evaluates to exactly L at every normal.
 *
 * Basis order (world axes, +y up):
 *   0: Y00            1: Y1,-1 ∝ y      2: Y1,0 ∝ z      3: Y1,1 ∝ x
 *   4: Y2,-2 ∝ xy     5: Y2,-1 ∝ yz     6: Y2,0 ∝ 3z²−1  7: Y2,1 ∝ xz
 *   8: Y2,2 ∝ x²−y²
 * Any orthonormal real basis works as long as projection and evaluation (CPU
 * and the WGSL in SH9_WGSL) agree; these are the standard Sloan constants.
 */

import type { Vec3 } from "../../../scene/types.js";

export const SH9_COUNT = 9;

const K0 = 0.282094791773878;
const K1 = 0.488602511902920;
const K2 = 1.092548430592079;
const K20 = 0.315391565252520;
const K22 = 0.546274215296039;

/** The 9 basis values at a unit direction. */
export function sh9Basis(x: number, y: number, z: number, out: Float64Array | number[] = new Array(9)): Float64Array | number[] {
  out[0] = K0;
  out[1] = K1 * y;
  out[2] = K1 * z;
  out[3] = K1 * x;
  out[4] = K2 * x * y;
  out[5] = K2 * y * z;
  out[6] = K20 * (3 * z * z - 1);
  out[7] = K2 * x * z;
  out[8] = K22 * (x * x - y * y);
  return out;
}

/** Per-band clamped-cosine transfer divided by π (A_l / π). */
const BAND_DIFFUSE = [1, 2 / 3, 2 / 3, 2 / 3, 1 / 4, 1 / 4, 1 / 4, 1 / 4, 1 / 4];

/** RGB SH9: 27 numbers, coefficient-major ([c0.r, c0.g, c0.b, c1.r, …]). */
export type Sh9 = Float64Array;

export function sh9Zero(): Sh9 {
  return new Float64Array(27);
}

/**
 * Project a radiance function onto SH9 (RAW radiance, not convolved) by
 * stratified Fibonacci-sphere sampling with equal solid-angle weights.
 */
export function projectSh9(radiance: (dir: Vec3) => Vec3, samples = 2048): Sh9 {
  const out = sh9Zero();
  const basis = new Float64Array(9);
  const golden = Math.PI * (3 - Math.sqrt(5));
  const w = (4 * Math.PI) / samples;
  for (let i = 0; i < samples; i++) {
    const y = 1 - ((i + 0.5) / samples) * 2;
    const r = Math.sqrt(Math.max(0, 1 - y * y));
    const phi = i * golden;
    const x = Math.cos(phi) * r, z = Math.sin(phi) * r;
    const L = radiance([x, y, z]);
    sh9Basis(x, y, z, basis);
    for (let k = 0; k < 9; k++) {
      const b = basis[k]! * w;
      out[k * 3] = out[k * 3]! + L[0] * b;
      out[k * 3 + 1] = out[k * 3 + 1]! + L[1] * b;
      out[k * 3 + 2] = out[k * 3 + 2]! + L[2] * b;
    }
  }
  return out;
}

/**
 * Project an equirectangular radiance image (RGB floats, row 0 = straight up)
 * onto SH9 with exact per-row solid-angle weights. Mapping: u = 0 at −z
 * (north), increasing toward +x (east); see equirectDirection.
 * `step` subsamples large images (every step-th pixel in both axes).
 */
export function projectEquirectSh9(data: Float32Array, width: number, height: number, channels = 3, step = 1, rotation = 0): Sh9 {
  const out = sh9Zero();
  const basis = new Float64Array(9);
  const s = Math.max(1, Math.floor(step));
  const dTheta = (Math.PI / height) * s;
  const dPhi = ((2 * Math.PI) / width) * s;
  for (let row = 0; row < height; row += s) {
    const theta = ((row + 0.5) / height) * Math.PI;
    const st = Math.sin(theta);
    const w = st * dTheta * dPhi;
    for (let col = 0; col < width; col += s) {
      const d = equirectDirection((col + 0.5) / width, (row + 0.5) / height, rotation);
      sh9Basis(d[0], d[1], d[2], basis);
      const p = (row * width + col) * channels;
      const r = data[p]!, g = data[p + 1]!, b = data[p + 2]!;
      for (let k = 0; k < 9; k++) {
        const bw = basis[k]! * w;
        out[k * 3] = out[k * 3]! + r * bw;
        out[k * 3 + 1] = out[k * 3 + 1]! + g * bw;
        out[k * 3 + 2] = out[k * 3 + 2]! + b * bw;
      }
    }
  }
  return out;
}

/**
 * Direction for equirect texture coordinates (u right, v down). u = 0.5 looks
 * along −z (north), u = 0.75 along +x (east); v = 0 is the zenith.
 * `rotation` (radians) spins the image about +y.
 */
export function equirectDirection(u: number, v: number, rotation = 0): Vec3 {
  const phi = (u - 0.5) * 2 * Math.PI + rotation;
  const theta = v * Math.PI;
  const st = Math.sin(theta);
  return [st * Math.sin(phi), Math.cos(theta), -st * Math.cos(phi)];
}

/** Convolve raw-radiance SH9 with the clamped cosine and divide by π, in place. */
export function convolveSh9Diffuse(sh: Sh9): Sh9 {
  for (let k = 0; k < 9; k++) {
    for (let c = 0; c < 3; c++) sh[k * 3 + c]! *= BAND_DIFFUSE[k]!;
  }
  return sh;
}

/** Evaluate SH9 at a unit direction (per channel). */
export function evalSh9(sh: Sh9, n: Vec3): Vec3 {
  const b = sh9Basis(n[0], n[1], n[2], new Float64Array(9));
  const out: [number, number, number] = [0, 0, 0];
  for (let k = 0; k < 9; k++) {
    out[0] += sh[k * 3]! * b[k]!;
    out[1] += sh[k * 3 + 1]! * b[k]!;
    out[2] += sh[k * 3 + 2]! * b[k]!;
  }
  return out;
}

/** Scale every coefficient (e.g. an environment intensity). */
export function scaleSh9(sh: Sh9, s: number): Sh9 {
  for (let i = 0; i < 27; i++) sh[i]! *= s;
  return sh;
}

/**
 * Pack into 9 vec4s for the frame uniform: vec4 k = (r, g, b, 0) of
 * coefficient k with the basis constant folded in, so the shader evaluates
 * Σ c_k · poly_k(n) with poly = [1, y, z, x, xy, yz, 3z²−1, xz, x²−y²].
 */
export function packSh9(out: Float32Array, offset: number, sh: Sh9): void {
  const K = [K0, K1, K1, K1, K2, K2, K20, K2, K22];
  for (let k = 0; k < 9; k++) {
    out[offset + k * 4] = sh[k * 3]! * K[k]!;
    out[offset + k * 4 + 1] = sh[k * 3 + 1]! * K[k]!;
    out[offset + k * 4 + 2] = sh[k * 3 + 2]! * K[k]!;
    out[offset + k * 4 + 3] = 0;
  }
}

/** WGSL evaluation of a packSh9 block (array<vec4<f32>, 9>). */
export const SH9_WGSL = /* wgsl */ `
fn augengine_eval_sh9(c: array<vec4<f32>, 9>, n: vec3<f32>) -> vec3<f32> {
  var r = c[0].rgb;
  r = r + c[1].rgb * n.y + c[2].rgb * n.z + c[3].rgb * n.x;
  r = r + c[4].rgb * (n.x * n.y) + c[5].rgb * (n.y * n.z) + c[6].rgb * (3.0 * n.z * n.z - 1.0);
  r = r + c[7].rgb * (n.x * n.z) + c[8].rgb * (n.x * n.x - n.y * n.y);
  return r;
}
`;
