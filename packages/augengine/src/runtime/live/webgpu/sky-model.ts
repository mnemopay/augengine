/**
 * Physical sun + sky model (CPU side). Pure math, no GPU: the WGSL in
 * physical-sky.ts / env-lighting.ts evaluates the same Perez function from the
 * coefficients packed here.
 *
 * Units are photometric throughout:
 *   - sky radiance  → luminance in cd/m² (nits), per linear-sRGB channel
 *   - sun           → illuminance on a surface facing it, in lux
 *   - a Lambertian surface of albedo ρ under illuminance E has L = ρ·E/π cd/m²
 * which is what the material shaders compute once natural light is on.
 *
 * Sky: Preetham, Shirley & Smits, "A Practical Analytic Model for Daylight",
 * SIGGRAPH 1999 — the Perez et al. (1993) all-weather luminance distribution
 * with turbidity-dependent coefficients (paper appendix A.2) for Y, x and y,
 * and the paper's absolute zenith luminance (kcd/m²) and zenith chromaticity.
 * Hosek-Wilkie (2012) is the better model near the horizon, but its
 * coefficient tables are ~30 KB; Preetham is a dozen numbers and is what this
 * tier ships. Its known weaknesses (too bright/too white for low suns, no
 * twilight) are handled explicitly below and documented in docs/LIGHTING.md.
 *
 * Overcast: CIE standard overcast sky (Moon & Spencer 1942),
 * L(θ) = Lz·(1 + 2 cos θ)/3, with the zenith luminance from Krochmann's
 * overcast fit Lz = 123 + 8600·sin(α) cd/m². `cloudCover` blends clear→overcast
 * and hides the sun disc.
 *
 * Sun: extraterrestrial illuminance 127.5 klux (the IES value) attenuated by
 * Rayleigh (Leckner 1978, τ = 0.008735 λ^−4.08) and aerosol (Ångström,
 * β from Preetham's turbidity relation, α = 1.3) optical depth along the
 * Kasten & Young (1989) air mass, evaluated per channel at 650/550/450 nm.
 *
 * Sun position: declination (Cooper 1969), hour angle from local SOLAR time,
 * latitude → elevation/azimuth. Good to ~1°, plenty for lighting.
 *
 * World convention: +y up, +x east, −z north.
 */

import type { Vec3 } from "../../../scene/types.js";

const DEG = Math.PI / 180;

/** Extraterrestrial solar illuminance (lux), IES Lighting Handbook value. */
export const SOLAR_ILLUMINANCE_EXTRATERRESTRIAL = 127_500;
/** Mean angular diameter of the sun (degrees). */
export const SUN_ANGULAR_DIAMETER_DEG = 0.533;

export interface SunPositionInput {
  /** Geographic latitude in degrees (+north). */
  latitudeDeg: number;
  /** Day of year, 1–365. */
  dayOfYear: number;
  /** Local SOLAR time in hours (12 = solar noon). */
  solarHour: number;
}

export interface SunPosition {
  /** Elevation above the horizon, radians (negative = below). */
  elevation: number;
  /** Azimuth clockwise from north, radians (π/2 = east). */
  azimuth: number;
  /** Unit vector from the scene TOWARD the sun (world, +y up, −z north). */
  dir: Vec3;
}

/** Solar declination (radians) for a day of year (Cooper 1969). */
export function solarDeclination(dayOfYear: number): number {
  return 23.44 * DEG * Math.sin((2 * Math.PI * (284 + dayOfYear)) / 365);
}

export function sunPosition(input: SunPositionInput): SunPosition {
  const phi = input.latitudeDeg * DEG;
  const delta = solarDeclination(input.dayOfYear);
  const H = (input.solarHour - 12) * 15 * DEG;
  // East / north / up components of the direction to the sun.
  const e = -Math.cos(delta) * Math.sin(H);
  const n = Math.cos(phi) * Math.sin(delta) - Math.sin(phi) * Math.cos(delta) * Math.cos(H);
  const u = Math.sin(phi) * Math.sin(delta) + Math.cos(phi) * Math.cos(delta) * Math.cos(H);
  return sunFromEnu(e, n, u);
}

/** Direction to the sun from elevation + azimuth (degrees, azimuth clockwise from north). */
export function sunFromAngles(elevationDeg: number, azimuthDeg: number): SunPosition {
  const el = elevationDeg * DEG;
  const az = azimuthDeg * DEG;
  return sunFromEnu(Math.cos(el) * Math.sin(az), Math.cos(el) * Math.cos(az), Math.sin(el));
}

function sunFromEnu(e: number, n: number, u: number): SunPosition {
  const len = Math.hypot(e, n, u) || 1;
  const ex = e / len, nx = n / len, ux = u / len;
  const elevation = Math.asin(Math.max(-1, Math.min(1, ux)));
  let azimuth = Math.atan2(ex, nx);
  if (azimuth < 0) azimuth += 2 * Math.PI;
  return { elevation, azimuth, dir: [ex, ux, -nx] };
}

// ─── Preetham clear sky ────────────────────────────────────────────────

export interface PerezCoeffs { A: number; B: number; C: number; D: number; E: number }

/** Perez coefficients for luminance Y and chromaticity x, y (Preetham 1999, A.2). */
export function preethamCoefficients(turbidity: number): { Y: PerezCoeffs; x: PerezCoeffs; y: PerezCoeffs } {
  const T = turbidity;
  return {
    Y: { A: 0.1787 * T - 1.463, B: -0.3554 * T + 0.4275, C: -0.0227 * T + 5.3251, D: 0.1206 * T - 2.5771, E: -0.067 * T + 0.3703 },
    x: { A: -0.0193 * T - 0.2592, B: -0.0665 * T + 0.0008, C: -0.0004 * T + 0.2125, D: -0.0641 * T - 0.8989, E: -0.0033 * T + 0.0452 },
    y: { A: -0.0167 * T - 0.2608, B: -0.095 * T + 0.0092, C: -0.0079 * T + 0.2102, D: -0.0441 * T - 1.6537, E: -0.0109 * T + 0.0529 },
  };
}

/** Perez distribution F(θ, γ); θ = view zenith angle, γ = angle to the sun. */
export function perez(cosTheta: number, gamma: number, c: PerezCoeffs): number {
  const ct = Math.max(cosTheta, 0.01);
  const cg = Math.cos(gamma);
  return (1 + c.A * Math.exp(c.B / ct)) * (1 + c.C * Math.exp(c.D * gamma) + c.E * cg * cg);
}

/**
 * Zenith luminance (cd/m²) and chromaticity for sun zenith angle θs.
 * Preetham's Yz is in kcd/m²; returned ×1000.
 */
export function preethamZenith(turbidity: number, thetaS: number): { Y: number; x: number; y: number } {
  const T = turbidity;
  const chi = (4 / 9 - T / 120) * (Math.PI - 2 * thetaS);
  const Yk = (4.0453 * T - 4.971) * Math.tan(chi) - 0.2155 * T + 2.4192;
  const t3 = thetaS ** 3, t2 = thetaS ** 2, t1 = thetaS;
  const T2 = T * T;
  const x =
    T2 * (0.00166 * t3 - 0.00375 * t2 + 0.00209 * t1) +
    T * (-0.02903 * t3 + 0.06377 * t2 - 0.03202 * t1 + 0.00394) +
    (0.11693 * t3 - 0.21196 * t2 + 0.06052 * t1 + 0.25886);
  const y =
    T2 * (0.00275 * t3 - 0.0061 * t2 + 0.00317 * t1) +
    T * (-0.04214 * t3 + 0.0897 * t2 - 0.04153 * t1 + 0.00516) +
    (0.15346 * t3 - 0.26756 * t2 + 0.0667 * t1 + 0.26688);
  return { Y: Math.max(Yk, 0) * 1000, x, y };
}

/** CIE xyY → linear sRGB (D65), Y in the caller's units. */
export function xyYToLinearSrgb(x: number, y: number, Y: number): Vec3 {
  if (y <= 1e-6) return [0, 0, 0];
  const X = (x / y) * Y;
  const Z = ((1 - x - y) / y) * Y;
  return [
    3.2406 * X - 1.5372 * Y - 0.4986 * Z,
    -0.9689 * X + 1.8758 * Y + 0.0415 * Z,
    0.0557 * X - 0.204 * Y + 1.057 * Z,
  ];
}

/** Rec.709 / sRGB luminance of a linear colour. */
export function luminance(c: ArrayLike<number>): number {
  return 0.2126 * c[0]! + 0.7152 * c[1]! + 0.0722 * c[2]!;
}

// ─── Sky parameters → evaluable model ──────────────────────────────────

export interface SkyParams {
  /** Unit vector toward the sun. */
  sunDir: Vec3;
  /** Atmospheric turbidity, 1.7 (very clear) … 10 (hazy). Default 2.5. */
  turbidity?: number;
  /** 0 = clear (Preetham), 1 = CIE overcast with the sun hidden. */
  cloudCover?: number;
  /** Ground albedo used for the lower hemisphere and the ground bounce. Default 0.2. */
  groundAlbedo?: number;
  /**
   * Multiplier on Preetham's absolute clear-sky luminance. Default
   * PREETHAM_CALIBRATION (matches the IES clear-sky horizontal illuminance at
   * a 45° sun); pass 1 for the paper's raw values.
   */
  calibration?: number;
}

/**
 * Preetham's zenith-luminance fit overestimates absolute level: integrated
 * over the dome, its horizontal sky illuminance is 1.75–2.0× the IES
 * clear-sky equation for suns at 15°–75° (T = 2.5; measured by
 * natural-light.test.ts). One constant, fitted at a 45° sun, brings it onto
 * the reference; the test checks the remaining 15°–75° spread stays ±15%.
 */
export const PREETHAM_CALIBRATION = 0.532;

export interface SkyModel {
  sunDir: Vec3;
  turbidity: number;
  cloudCover: number;
  groundAlbedo: number;
  /** Sun zenith angle actually fed to Preetham (clamped just above the horizon). */
  thetaS: number;
  coeffs: { Y: PerezCoeffs; x: PerezCoeffs; y: PerezCoeffs };
  zenith: { Y: number; x: number; y: number };
  /** Perez F(0, θs) per Y/x/y — the zenith normalisation. */
  f0: { Y: number; x: number; y: number };
  /**
   * Clear-sky luminance multiplier: the calibration in daylight, fading
   * through civil twilight once the sun is below the horizon (Preetham has no
   * twilight).
   */
  clearScale: number;
  /** CIE overcast zenith luminance (cd/m²). */
  overcastZenith: number;
  /** Floor luminance (cd/m²) so a night sky is dim, not black. */
  nightFloor: number;
  /** Sun illuminance (lux) on a surface facing it, per channel, after clouds. */
  sunIlluminance: Vec3;
  /** Luminance of the sun disc (cd/m²), per channel (0 when hidden). */
  sunDiscRadiance: Vec3;
  /** Radiance of the ground (lower hemisphere), cd/m² per channel. */
  groundRadiance: Vec3;
}

/** Kasten & Young (1989) relative optical air mass for a zenith angle (radians). */
export function airMass(zenith: number): number {
  const zd = Math.min(zenith / DEG, 93.9);
  return 1 / (Math.cos(zd * DEG) + 0.50572 * Math.pow(96.07995 - zd, -1.6364));
}

const CHANNEL_WAVELENGTH_UM: Vec3 = [0.65, 0.55, 0.45];

/**
 * Direct-normal sun illuminance (lux) per linear-sRGB channel for a clear sky.
 * The channels are scaled so their luminance is the photopic (550 nm)
 * transmittance × 127.5 klux; colour comes from the per-wavelength ratio.
 */
export function sunIlluminanceClear(sunElevation: number, turbidity: number): Vec3 {
  if (sunElevation <= -1 * DEG) return [0, 0, 0];
  const m = airMass(Math.PI / 2 - Math.max(sunElevation, -0.5 * DEG));
  const beta = 0.04608365822050 * turbidity - 0.04586025928522;
  const t = CHANNEL_WAVELENGTH_UM.map((lambda) => {
    const tauR = 0.008735 * Math.pow(lambda, -4.08);
    const tauA = Math.max(beta, 0) * Math.pow(lambda, -1.3);
    return Math.exp(-(tauR + tauA) * m);
  }) as unknown as Vec3;
  // Normalise colour to luminance 1, then scale by the 550 nm transmittance.
  const lum = luminance(t);
  const scale = lum > 1e-9 ? (SOLAR_ILLUMINANCE_EXTRATERRESTRIAL * t[1]) / lum : 0;
  // Fade the last degree so the sun does not pop at the horizon.
  const fade = smoothstep(-1 * DEG, 0.5 * DEG, sunElevation);
  return [t[0] * scale * fade, t[1] * scale * fade, t[2] * scale * fade];
}

function smoothstep(a: number, b: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

/** Build the evaluable sky model for a sun direction + atmosphere. */
export function buildSkyModel(p: SkyParams): SkyModel {
  const turbidity = Math.min(10, Math.max(1.7, p.turbidity ?? 2.5));
  const cloudCover = Math.min(1, Math.max(0, p.cloudCover ?? 0));
  const groundAlbedo = Math.min(1, Math.max(0, p.groundAlbedo ?? 0.2));
  const len = Math.hypot(p.sunDir[0], p.sunDir[1], p.sunDir[2]) || 1;
  const sunDir: Vec3 = [p.sunDir[0] / len, p.sunDir[1] / len, p.sunDir[2] / len];
  const elevation = Math.asin(Math.max(-1, Math.min(1, sunDir[1])));
  // Preetham is only defined for a sun above the horizon; hold it at +0.5°
  // and fade the result through civil twilight (sun to −6°).
  const thetaS = Math.min(Math.PI / 2 - 0.5 * DEG, Math.PI / 2 - elevation);
  const twilight = elevation >= 0 ? 1 : Math.pow(10, (elevation / DEG) * 0.5) * smoothstep(-8 * DEG, 0, elevation);
  const clearScale = twilight * (p.calibration ?? PREETHAM_CALIBRATION);
  const coeffs = preethamCoefficients(turbidity);
  const zenith = preethamZenith(turbidity, thetaS);
  const f0 = { Y: perez(1, thetaS, coeffs.Y), x: perez(1, thetaS, coeffs.x), y: perez(1, thetaS, coeffs.y) };
  const sinAlt = Math.max(Math.sin(elevation), 0);
  const overcastZenith = (123 + 8600 * sinAlt) * smoothstep(-8 * DEG, 2 * DEG, elevation);
  const clearSun = sunIlluminanceClear(elevation, turbidity);
  // Thin cloud dims the sun faster than it brightens the sky; fully hidden at 1.
  const sunVis = Math.pow(1 - cloudCover, 2);
  const sunIlluminance: Vec3 = [clearSun[0] * sunVis, clearSun[1] * sunVis, clearSun[2] * sunVis];
  const discSolidAngle = 2 * Math.PI * (1 - Math.cos((SUN_ANGULAR_DIAMETER_DEG / 2) * DEG));
  const sunDiscRadiance: Vec3 = [sunIlluminance[0] / discSolidAngle, sunIlluminance[1] / discSolidAngle, sunIlluminance[2] / discSolidAngle];
  const model: SkyModel = {
    sunDir, turbidity, cloudCover, groundAlbedo, thetaS, coeffs, zenith, f0, clearScale,
    overcastZenith, nightFloor: 0.02, sunIlluminance, sunDiscRadiance, groundRadiance: [0, 0, 0],
  };
  // Ground: Lambertian, lit by sun + the upper sky hemisphere.
  const skyE = horizontalSkyIlluminance(model, 24, 48);
  const sunE = sunIlluminance.map((c) => c * Math.max(sunDir[1], 0));
  model.groundRadiance = [0, 1, 2].map((i) => (groundAlbedo * (skyE[i]! + sunE[i]!)) / Math.PI) as unknown as Vec3;
  return model;
}

/**
 * Sky luminance (cd/m², linear sRGB) along a view direction. Excludes the sun
 * disc (the renderer shades the sun as the key light). Below the horizon:
 * the ground radiance.
 */
export function skyRadianceAt(m: SkyModel, dir: Vec3): Vec3 {
  const len = Math.hypot(dir[0], dir[1], dir[2]) || 1;
  const d: Vec3 = [dir[0] / len, dir[1] / len, dir[2] / len];
  if (d[1] < 0) return [...m.groundRadiance] as Vec3;
  const cosTheta = d[1];
  const cosGamma = Math.max(-1, Math.min(1, d[0] * m.sunDir[0] + d[1] * m.sunDir[1] + d[2] * m.sunDir[2]));
  const gamma = Math.acos(cosGamma);
  const Y = (m.zenith.Y * perez(cosTheta, gamma, m.coeffs.Y)) / m.f0.Y;
  const x = (m.zenith.x * perez(cosTheta, gamma, m.coeffs.x)) / m.f0.x;
  const y = (m.zenith.y * perez(cosTheta, gamma, m.coeffs.y)) / m.f0.y;
  const clear = xyYToLinearSrgb(x, y, Math.max(Y, 0) * m.clearScale);
  // CIE overcast: neutral (D65 white in linear sRGB), brighter at the zenith.
  const ov = (m.overcastZenith * (1 + 2 * cosTheta)) / 3;
  const c = m.cloudCover;
  const out: Vec3 = [
    Math.max(clear[0], 0) * (1 - c) + ov * c,
    Math.max(clear[1], 0) * (1 - c) + ov * c,
    Math.max(clear[2], 0) * (1 - c) + ov * c,
  ];
  return [out[0] + m.nightFloor * 0.6, out[1] + m.nightFloor * 0.8, out[2] + m.nightFloor * 1.3];
}

/**
 * Illuminance (lux) on an upward-facing horizontal surface from the SKY
 * dome only (no sun), by midpoint integration over the upper hemisphere.
 */
export function horizontalSkyIlluminance(m: SkyModel, thetaSteps = 48, phiSteps = 96): Vec3 {
  const acc: [number, number, number] = [0, 0, 0];
  const dTheta = (Math.PI / 2) / thetaSteps;
  const dPhi = (2 * Math.PI) / phiSteps;
  for (let i = 0; i < thetaSteps; i++) {
    const theta = (i + 0.5) * dTheta;
    const st = Math.sin(theta), ct = Math.cos(theta);
    for (let j = 0; j < phiSteps; j++) {
      const phi = (j + 0.5) * dPhi;
      const L = skyRadianceAt(m, [st * Math.cos(phi), ct, st * Math.sin(phi)]);
      const w = ct * st * dTheta * dPhi;
      acc[0] += L[0] * w; acc[1] += L[1] * w; acc[2] += L[2] * w;
    }
  }
  return acc;
}

/**
 * IES clear-sky reference values (klux → lux), the daylight equations used by
 * the IES Lighting Handbook / RP-5 for clear skies:
 *   horizontal sky illuminance  E_sky = 0.8 + 15.5·sin(α)^0.5 klux
 *   direct-normal sun           E_dn  = 127.5·exp(−0.21·m) klux
 */
export function iesClearSky(sunElevation: number): { skyHorizontal: number; sunNormal: number } {
  const s = Math.max(Math.sin(sunElevation), 0);
  const m = airMass(Math.PI / 2 - sunElevation);
  return { skyHorizontal: (0.8 + 15.5 * Math.sqrt(s)) * 1000, sunNormal: 127.5 * Math.exp(-0.21 * m) * 1000 };
}

// ─── GPU packing ───────────────────────────────────────────────────────

/** vec4 count of the WGSL `SkyModelU` struct (see SKY_MODEL_WGSL). */
export const SKY_MODEL_VEC4S = 11;
export const SKY_MODEL_FLOATS = SKY_MODEL_VEC4S * 4;

/**
 * Pack a model for SKY_MODEL_WGSL. `preExposureWb` (rgb) multiplies the
 * background output; it is ignored by the environment bake.
 */
export function packSkyModel(out: Float32Array, offset: number, m: SkyModel, preExposureWb: Vec3 = [1, 1, 1], mode = 0, intensity = 1): void {
  const o = offset;
  const c = m.coeffs;
  out.set([m.sunDir[0], m.sunDir[1], m.sunDir[2], Math.cos((SUN_ANGULAR_DIAMETER_DEG / 2) * DEG)], o);
  out.set([c.Y.A, c.x.A, c.y.A, m.cloudCover], o + 4);
  out.set([c.Y.B, c.x.B, c.y.B, m.overcastZenith], o + 8);
  out.set([c.Y.C, c.x.C, c.y.C, m.clearScale], o + 12);
  out.set([c.Y.D, c.x.D, c.y.D, m.groundAlbedo], o + 16);
  out.set([c.Y.E, c.x.E, c.y.E, m.nightFloor], o + 20);
  out.set([m.zenith.Y, m.zenith.x, m.zenith.y, 0], o + 24);
  out.set([m.f0.Y, m.f0.x, m.f0.y, 0], o + 28);
  out.set([m.sunDiscRadiance[0], m.sunDiscRadiance[1], m.sunDiscRadiance[2], 0], o + 32);
  out.set([preExposureWb[0], preExposureWb[1], preExposureWb[2], mode], o + 36);
  out.set([m.groundRadiance[0], m.groundRadiance[1], m.groundRadiance[2], intensity], o + 40);
}

/**
 * WGSL mirror of skyRadianceAt + the packed struct. `sky_radiance(dir)` is
 * the dome without the sun disc; `sun_disc(dir)` adds the limb-darkened disc
 * for the visible background.
 */
export const SKY_MODEL_WGSL = /* wgsl */ `
struct SkyModelU {
  sunDir: vec4<f32>,   // xyz toward sun, w = cos(sun angular radius)
  pA: vec4<f32>,       // Perez A for Y,x,y; w = cloud cover
  pB: vec4<f32>,       // w = CIE overcast zenith luminance
  pC: vec4<f32>,       // w = clear-sky twilight scale
  pD: vec4<f32>,       // w = ground albedo
  pE: vec4<f32>,       // w = night floor luminance
  zenith: vec4<f32>,   // Yz (cd/m²), xz, yz
  f0: vec4<f32>,       // Perez F(0, θs) for Y,x,y
  sunDisc: vec4<f32>,  // disc luminance rgb
  expo: vec4<f32>,     // rgb pre-exposure × white balance, w = mode (0 sky, 1 equirect)
  ground: vec4<f32>,   // ground radiance rgb, w = environment intensity
};
fn sky_perez(cosTheta: f32, gamma: f32, i: i32) -> f32 {
  let ct = max(cosTheta, 0.01);
  let cg = cos(gamma);
  return (1.0 + sky.pA[i] * exp(sky.pB[i] / ct)) * (1.0 + sky.pC[i] * exp(sky.pD[i] * gamma) + sky.pE[i] * cg * cg);
}
fn sky_xyY_to_rgb(x: f32, y: f32, Y: f32) -> vec3<f32> {
  if (y <= 1e-6) { return vec3<f32>(0.0); }
  let X = x / y * Y;
  let Z = (1.0 - x - y) / y * Y;
  return vec3<f32>(3.2406 * X - 1.5372 * Y - 0.4986 * Z,
                   -0.9689 * X + 1.8758 * Y + 0.0415 * Z,
                   0.0557 * X - 0.2040 * Y + 1.0570 * Z);
}
fn sky_radiance(dirIn: vec3<f32>) -> vec3<f32> {
  let d = normalize(dirIn);
  if (d.y < 0.0) { return sky.ground.rgb; }
  let gamma = acos(clamp(dot(d, sky.sunDir.xyz), -1.0, 1.0));
  let Y = sky.zenith.x * sky_perez(d.y, gamma, 0) / sky.f0.x;
  let x = sky.zenith.y * sky_perez(d.y, gamma, 1) / sky.f0.y;
  let y = sky.zenith.z * sky_perez(d.y, gamma, 2) / sky.f0.z;
  let clear = max(sky_xyY_to_rgb(x, y, max(Y, 0.0) * sky.pC.w), vec3<f32>(0.0));
  let ov = sky.pB.w * (1.0 + 2.0 * d.y) / 3.0;
  let c = sky.pA.w;
  return clear * (1.0 - c) + vec3<f32>(ov * c) + sky.pE.w * vec3<f32>(0.6, 0.8, 1.3);
}
// Limb-darkened sun disc (Neckel & Labs 1994 style: I(μ)/I(1) ≈ 1 − u(1 − μ^α)).
fn sun_disc(dirIn: vec3<f32>) -> vec3<f32> {
  let d = normalize(dirIn);
  let c = dot(d, sky.sunDir.xyz);
  let cosR = sky.sunDir.w;
  if (c < cosR) { return vec3<f32>(0.0); }
  let r = sqrt(max(1.0 - c * c, 0.0)) / sqrt(max(1.0 - cosR * cosR, 1e-9));
  let mu = sqrt(max(1.0 - r * r, 0.0));
  let limb = 1.0 - 0.6 * (1.0 - pow(mu, 0.8));
  // Normalised so the disc integrates to the sun illuminance: the area-mean of
  // the limb term is 1 − 0.6·(1 − 1/1.4) = 0.8286.
  return sky.sunDisc.rgb * limb / 0.8286;
}
`;
