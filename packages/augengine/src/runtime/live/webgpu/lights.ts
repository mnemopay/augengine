/**
 * Scene lights → renderer frame uniforms, and directional shadow fitting.
 *
 * Before this module the live renderer lit everything with ONE hard-coded
 * directional light (`RenderFrameOptions.lightDir`, white, intensity 1) and a
 * constant ambient; the `Tag.Light` entities a scene declared were ignored.
 * Now the renderer reads them every frame:
 *
 *   - directional: the brightest one is the KEY light (it casts the shadow);
 *     up to MAX_EXTRA_DIRECTIONAL more add unshadowed contribution.
 *   - point / spot: up to MAX_LOCAL_LIGHTS, nearest-to-camera first when a
 *     scene declares more. Smooth windowed inverse-square falloff to `range`
 *     (default DEFAULT_LOCAL_RANGE); spots add a smoothstep cone.
 *   - ambient: summed colour × intensity replaces the frame ambient.
 *
 * A scene with no light entities renders exactly as before (key light =
 * `lightDir`, colour 1), so existing maps are unchanged.
 *
 * Direction convention for directional and spot lights: a light whose
 * transform is ROTATED shines along its local −Z. A light with an identity
 * rotation shines from its position toward the world origin (that is how the
 * authored rooms place key/fill/rim lights); at the origin it points straight
 * down.
 */

import type { Mat4, Vec3 } from "../../../scene/types.js";
import { Tag } from "../../../scene/types.js";
import type { Scene } from "../../../scene/scene.js";

export const MAX_EXTRA_DIRECTIONAL = 3;
export const MAX_LOCAL_LIGHTS = 8;
export const DEFAULT_LOCAL_RANGE = 10;

export interface DirectionalLightData {
  /** Direction the light TRAVELS (from light toward scene), unit length. */
  dir: Vec3;
  /** colour × intensity, linear. */
  radiance: Vec3;
}

export interface LocalLightData {
  position: Vec3;
  range: number;
  radiance: Vec3;
  /** Spot only: direction the cone points, unit length. */
  spotDir?: Vec3;
  cosInner?: number;
  cosOuter?: number;
}

export interface FrameLights {
  key: DirectionalLightData;
  extraDirectional: DirectionalLightData[];
  local: LocalLightData[];
  /** Null when the scene declares no ambient light (caller's ambient stands). */
  ambient: Vec3 | null;
  /** True when the key light came from a scene entity (not the fallback). */
  keyFromScene: boolean;
}

function normalize(v: Vec3): Vec3 {
  const l = Math.hypot(v[0], v[1], v[2]);
  return l > 1e-8 ? [v[0] / l, v[1] / l, v[2] / l] : [0, -1, 0];
}

/**
 * Direction a directional/spot light travels, from its world matrix and local
 * rotation quaternion (see the convention in the module header).
 */
export function lightDirection(world: ArrayLike<number>, localRotation: ArrayLike<number> | undefined): Vec3 {
  const q = localRotation ?? [0, 0, 0, 1];
  const identity = Math.abs(q[0]!) < 1e-6 && Math.abs(q[1]!) < 1e-6 && Math.abs(q[2]!) < 1e-6;
  if (!identity) {
    // Local −Z in world space = −(third column of the world matrix).
    return normalize([-world[8]!, -world[9]!, -world[10]!]);
  }
  const p: Vec3 = [world[12]!, world[13]!, world[14]!];
  if (Math.hypot(p[0], p[1], p[2]) < 1e-6) return [0, -1, 0];
  return normalize([-p[0], -p[1], -p[2]]);
}

/**
 * Collect the scene's light entities into frame-ready data.
 *
 * @param fallbackDir   key-light direction when the scene has no directional light
 * @param cameraPos     used to keep the nearest local lights when over the cap
 */
export function collectSceneLights(scene: Scene, fallbackDir: Vec3, cameraPos: Vec3): FrameLights {
  const wt = scene.worldTransforms();
  const directional: DirectionalLightData[] = [];
  const directionalIntensity: number[] = [];
  const local: Array<LocalLightData & { d2: number }> = [];
  let ambient: Vec3 | null = null;

  for (const e of scene.entities.values()) {
    const light = e.components[Tag.Light];
    if (!light) continue;
    const idx = wt.indexById.get(e.id);
    if (idx === undefined) continue;
    const world = wt.worldMatrices.subarray(idx * 16, idx * 16 + 16);
    const intensity = Number.isFinite(light.intensity) ? Math.max(0, light.intensity) : 0;
    const radiance: Vec3 = [light.color[0] * intensity, light.color[1] * intensity, light.color[2] * intensity];
    const rotation = e.components[Tag.Transform]?.rotation;
    if (light.kind === "ambient") {
      ambient = ambient ? [ambient[0] + radiance[0], ambient[1] + radiance[1], ambient[2] + radiance[2]] : radiance;
    } else if (light.kind === "directional") {
      directional.push({ dir: lightDirection(world, rotation), radiance });
      directionalIntensity.push(intensity * Math.max(light.color[0], light.color[1], light.color[2]));
    } else {
      const position: Vec3 = [world[12]!, world[13]!, world[14]!];
      const d2 = (position[0] - cameraPos[0]) ** 2 + (position[1] - cameraPos[1]) ** 2 + (position[2] - cameraPos[2]) ** 2;
      const entry: LocalLightData & { d2: number } = {
        position,
        range: light.range && light.range > 0 ? light.range : DEFAULT_LOCAL_RANGE,
        radiance,
        d2,
      };
      if (light.kind === "spot") {
        const outer = light.outerConeRad ?? Math.PI / 4;
        const inner = Math.min(light.innerConeRad ?? outer * 0.8, outer);
        entry.spotDir = lightDirection(world, rotation);
        entry.cosInner = Math.cos(inner);
        entry.cosOuter = Math.cos(outer);
      }
      local.push(entry);
    }
  }

  // Brightest directional light is the key (the shadow caster).
  let keyIdx = -1;
  for (let i = 0; i < directional.length; i++) {
    if (keyIdx < 0 || directionalIntensity[i]! > directionalIntensity[keyIdx]!) keyIdx = i;
  }
  const key = keyIdx >= 0 ? directional[keyIdx]! : { dir: normalize(fallbackDir), radiance: [1, 1, 1] as Vec3 };
  const extraDirectional = directional.filter((_, i) => i !== keyIdx).slice(0, MAX_EXTRA_DIRECTIONAL);
  local.sort((a, b) => a.d2 - b.d2);
  return {
    key,
    extraDirectional,
    local: local.slice(0, MAX_LOCAL_LIGHTS).map(({ d2: _d2, ...l }) => l),
    ambient,
    keyFromScene: keyIdx >= 0,
  };
}

// ─── Frame uniform packing ─────────────────────────────────────────────

/**
 * Float offsets inside the frame uniform (see FRAME_UNIFORM_BLOCK in
 * pipeline-factory.ts — the two MUST agree).
 */
export const FRAME_LAYOUT = {
  viewProj: 0, // mat4
  lightDir: 16, // key light travel direction
  ambient: 20,
  cameraPos: 24,
  fog: 28,
  lightColor: 32, // key radiance.rgb, a = shadow on (1) / off (0)
  counts: 36, // x = extra directional, y = local lights, z = shadow cascades (0/1 = one map, 2 = atlas)
  shadowViewProj: 40, // mat4
  shadowParams: 56, // x = texel size (uv), y = depth bias, z = normal offset (world), w = PCF taps per axis
  dirLights: 60, // MAX_EXTRA_DIRECTIONAL × (dir vec4, radiance vec4)
  localLights: 60 + MAX_EXTRA_DIRECTIONAL * 8, // MAX_LOCAL_LIGHTS × (posRange, radiance+cosInner, spotDir+cosOuter)
  // Sky-shaped ambient (ibl.ts): 3 × vec4 = zonal SH k0.rgb + enabled flag, k1.rgb, k2.rgb.
  ambientSH: 60 + MAX_EXTRA_DIRECTIONAL * 8 + MAX_LOCAL_LIGHTS * 12,
  // Far shadow cascade (counts.z = 2): mat4, then texel (uv), depth bias,
  // normal offset (world), blend width as a fraction of the near cascade.
  shadowViewProj2: 60 + MAX_EXTRA_DIRECTIONAL * 8 + MAX_LOCAL_LIGHTS * 12 + 12,
  shadowParams2: 60 + MAX_EXTRA_DIRECTIONAL * 8 + MAX_LOCAL_LIGHTS * 12 + 28,
  // Sky specular (ibl.ts packSkySpecular): raw-radiance zonal SH c0.rgb + enabled, c1.rgb, c2.rgb.
  skySpecSH: 60 + MAX_EXTRA_DIRECTIONAL * 8 + MAX_LOCAL_LIGHTS * 12 + 32,
  // Streamed irradiance probes (probe-ring.ts): x = on, y = column spacing,
  // z = layer-0 height above ground, w = layer spacing.
  probeParams: 60 + MAX_EXTRA_DIRECTIONAL * 8 + MAX_LOCAL_LIGHTS * 12 + 44,
  // Natural-light tier (env-lighting.ts). Diffuse irradiance / π as SH9:
  // 9 × vec4 (rgb, 0) with the basis constants folded in (sh9.ts packSh9).
  envSH: 60 + MAX_EXTRA_DIRECTIONAL * 8 + MAX_LOCAL_LIGHTS * 12 + 48,
  // x = 1 when the env cube + SH9 drive ambient/specular, y = specular mip
  // count − 1, z = specular intensity, w = multiple-scattering compensation (0/1).
  envParams: 60 + MAX_EXTRA_DIRECTIONAL * 8 + MAX_LOCAL_LIGHTS * 12 + 84,
  // rgb = pre-exposure × white-balance gain applied to lit output (1 = off),
  // a = scale on emissive + unlit colour (1 = display-referred, as before).
  exposure: 60 + MAX_EXTRA_DIRECTIONAL * 8 + MAX_LOCAL_LIGHTS * 12 + 88,
  // Contact-hardening (PCSS) shadows: x = tan(light angular radius) × softness,
  // y = near cascade width (m), z = far cascade width (m), w = Poisson taps (0 = off).
  shadowSoft: 60 + MAX_EXTRA_DIRECTIONAL * 8 + MAX_LOCAL_LIGHTS * 12 + 92,
  // x = frame index mod 64 (rotates the per-pixel sample pattern for TAA),
  // y = blocker search height (m), zw reserved.
  shadowSoft2: 60 + MAX_EXTRA_DIRECTIONAL * 8 + MAX_LOCAL_LIGHTS * 12 + 96,
  // Natural atmosphere (sky-model.ts aerialExtinction): x = on, y = HG
  // anisotropy g, z = sun in-scatter strength, w reserved.
  atmos: 60 + MAX_EXTRA_DIRECTIONAL * 8 + MAX_LOCAL_LIGHTS * 12 + 100,
  // rgb = aerial extinction σ (1/m) per channel, w reserved.
  atmosSigma: 60 + MAX_EXTRA_DIRECTIONAL * 8 + MAX_LOCAL_LIGHTS * 12 + 104,
} as const;

export const FRAME_UNIFORM_FLOATS = FRAME_LAYOUT.atmosSigma + 4;
export const FRAME_UNIFORM_SIZE = FRAME_UNIFORM_FLOATS * 4;

export interface ShadowUniform {
  enabled: boolean;
  viewProj: Mat4 | Float32Array;
  texelSize: number;
  depthBias: number;
  normalOffset: number;
  pcfTaps: number;
  /** Second (far) cascade; its map sits in the right half of a 2:1 atlas. */
  far?: { viewProj: Mat4 | Float32Array; texelSize: number; depthBias: number; normalOffset: number; blend: number };
  /** Contact-hardening soft shadows (PCSS). */
  soft?: { tanLight: number; worldSize: number; worldSizeFar: number; taps: number; frame: number; searchHeight: number };
}

/** Write the light + shadow portion of the frame uniform. */
export function packFrameLights(out: Float32Array, lights: FrameLights, shadow: ShadowUniform | null): void {
  const L = FRAME_LAYOUT;
  out[L.lightDir] = lights.key.dir[0];
  out[L.lightDir + 1] = lights.key.dir[1];
  out[L.lightDir + 2] = lights.key.dir[2];
  out[L.lightDir + 3] = 0;
  out[L.lightColor] = lights.key.radiance[0];
  out[L.lightColor + 1] = lights.key.radiance[1];
  out[L.lightColor + 2] = lights.key.radiance[2];
  out[L.lightColor + 3] = shadow?.enabled ? 1 : 0;
  out[L.counts] = lights.extraDirectional.length;
  out[L.counts + 1] = lights.local.length;
  out[L.counts + 2] = shadow?.enabled && shadow.far ? 2 : 1;
  out[L.counts + 3] = 0;
  if (shadow?.enabled) {
    out.set(shadow.viewProj as ArrayLike<number>, L.shadowViewProj);
    out[L.shadowParams] = shadow.texelSize;
    out[L.shadowParams + 1] = shadow.depthBias;
    out[L.shadowParams + 2] = shadow.normalOffset;
    out[L.shadowParams + 3] = shadow.pcfTaps;
  } else {
    out.fill(0, L.shadowViewProj, L.shadowParams + 4);
  }
  if (shadow?.enabled && shadow.soft && shadow.soft.taps > 0) {
    out[L.shadowSoft] = shadow.soft.tanLight;
    out[L.shadowSoft + 1] = shadow.soft.worldSize;
    out[L.shadowSoft + 2] = shadow.soft.worldSizeFar;
    out[L.shadowSoft + 3] = Math.min(16, Math.max(4, Math.round(shadow.soft.taps)));
    out[L.shadowSoft2] = shadow.soft.frame % 64;
    out[L.shadowSoft2 + 1] = shadow.soft.searchHeight;
    out[L.shadowSoft2 + 2] = 0;
    out[L.shadowSoft2 + 3] = 0;
  } else {
    out.fill(0, L.shadowSoft, L.shadowSoft2 + 4);
  }
  if (shadow?.enabled && shadow.far) {
    out.set(shadow.far.viewProj as ArrayLike<number>, L.shadowViewProj2);
    out[L.shadowParams2] = shadow.far.texelSize;
    out[L.shadowParams2 + 1] = shadow.far.depthBias;
    out[L.shadowParams2 + 2] = shadow.far.normalOffset;
    out[L.shadowParams2 + 3] = shadow.far.blend;
  } else {
    out.fill(0, L.shadowViewProj2, L.shadowParams2 + 4);
  }
  for (let i = 0; i < MAX_EXTRA_DIRECTIONAL; i++) {
    const o = L.dirLights + i * 8;
    const d = lights.extraDirectional[i];
    if (!d) { out.fill(0, o, o + 8); continue; }
    out[o] = d.dir[0]; out[o + 1] = d.dir[1]; out[o + 2] = d.dir[2]; out[o + 3] = 0;
    out[o + 4] = d.radiance[0]; out[o + 5] = d.radiance[1]; out[o + 6] = d.radiance[2]; out[o + 7] = 0;
  }
  for (let i = 0; i < MAX_LOCAL_LIGHTS; i++) {
    const o = L.localLights + i * 12;
    const l = lights.local[i];
    if (!l) { out.fill(0, o, o + 12); continue; }
    out[o] = l.position[0]; out[o + 1] = l.position[1]; out[o + 2] = l.position[2]; out[o + 3] = l.range;
    out[o + 4] = l.radiance[0]; out[o + 5] = l.radiance[1]; out[o + 6] = l.radiance[2];
    out[o + 7] = l.cosInner ?? -2;
    const sd = l.spotDir ?? [0, 0, 0];
    out[o + 8] = sd[0]; out[o + 9] = sd[1]; out[o + 10] = sd[2];
    // cosOuter = -2 marks a point light (no cone).
    out[o + 11] = l.spotDir ? (l.cosOuter ?? -2) : -2;
  }
}

// ─── Directional shadow fitting ────────────────────────────────────────

/**
 * Orthographic light view-projection covering a sphere (center, radius), with
 * WebGPU clip depth in [0, 1]. The sphere centre is snapped to whole shadow
 * texels in light space so the map does not shimmer as the camera moves.
 *
 * Returns a column-major mat4 (same convention as scene/math).
 */
export function fitDirectionalShadow(dirIn: Vec3, center: Vec3, radius: number, mapSize: number, out?: Float32Array): Float32Array {
  const dir = normalize(dirIn);
  // Light basis: z points back toward the light (−dir), x/y span the map.
  const z: Vec3 = [-dir[0], -dir[1], -dir[2]];
  const upRef: Vec3 = Math.abs(z[1]) > 0.99 ? [0, 0, 1] : [0, 1, 0];
  const x = normalize([upRef[1] * z[2] - upRef[2] * z[1], upRef[2] * z[0] - upRef[0] * z[2], upRef[0] * z[1] - upRef[1] * z[0]]);
  const y: Vec3 = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]];

  // Snap the centre to the texel grid in light space.
  const texelWorld = (2 * radius) / mapSize;
  let cx = center[0] * x[0] + center[1] * x[1] + center[2] * x[2];
  let cy = center[0] * y[0] + center[1] * y[1] + center[2] * y[2];
  const cz = center[0] * z[0] + center[1] * z[1] + center[2] * z[2];
  cx = Math.floor(cx / texelWorld) * texelWorld;
  cy = Math.floor(cy / texelWorld) * texelWorld;

  // View matrix rows are the basis; translation puts the snapped centre at the
  // origin of light space, pulled back by 2·radius along +z (toward the light).
  const tx = -cx, ty = -cy, tz = -cz - 2 * radius;
  const near = radius * 0.01; // light-space distance in front of the eye
  const far = radius * 4;
  // Ortho: x,y in [-r, r] → [-1, 1]; view-space z in [-near, -far] → depth [0, 1].
  const sx = 1 / radius, sy = 1 / radius;
  const sz = -1 / (far - near);
  const oz = -near / (far - near);
  const m = out && out.length >= 16 ? out : new Float32Array(16);
  // m = Ortho * View, written directly (column-major).
  m[0] = sx * x[0]; m[4] = sx * x[1]; m[8] = sx * x[2]; m[12] = sx * tx;
  m[1] = sy * y[0]; m[5] = sy * y[1]; m[9] = sy * y[2]; m[13] = sy * ty;
  m[2] = sz * z[0]; m[6] = sz * z[1]; m[10] = sz * z[2]; m[14] = sz * tz + oz;
  m[3] = 0; m[7] = 0; m[11] = 0; m[15] = 1;
  return m;
}

/** Apply a column-major mat4 to a point (w = 1). Exported for tests. */
export function transformPoint(m: ArrayLike<number>, p: Vec3): [number, number, number] {
  return [
    m[0]! * p[0] + m[4]! * p[1] + m[8]! * p[2] + m[12]!,
    m[1]! * p[0] + m[5]! * p[1] + m[9]! * p[2] + m[13]!,
    m[2]! * p[0] + m[6]! * p[1] + m[10]! * p[2] + m[14]!,
  ];
}

// ─── Shadow quality ────────────────────────────────────────────────────

/**
 * "low" / "high": fixed-width PCF (hardware 2×2 / 3×3 over it). "soft" /
 * "soft-high": contact-hardening PCSS — penumbra width from the light's
 * angular size and the blocker distance, so shadows are sharp where an object
 * meets the ground and widen with height, as sunlight does.
 */
export type ShadowQuality = "off" | "low" | "high" | "soft" | "soft-high";

export interface ShadowSettings {
  mapSize: number;
  /** PCF taps per axis (1 = single hardware-filtered compare, 3 = 3×3). */
  pcfTaps: number;
  /** 1 = one map; 2 = near + far cascade side by side in a 2:1 atlas. */
  cascades: 1 | 2;
  /** PCSS Poisson taps for blocker search and filter (0 = fixed PCF). */
  pcss: number;
}

export function shadowSettingsFor(quality: ShadowQuality, cascades: 1 | 2 = 1): ShadowSettings | null {
  switch (quality) {
    case "off": return null;
    case "high": return { mapSize: 2048, pcfTaps: 3, cascades, pcss: 0 };
    case "soft": return { mapSize: 1024, pcfTaps: 1, cascades, pcss: 8 };
    case "soft-high": return { mapSize: 2048, pcfTaps: 1, cascades, pcss: 16 };
    default: return { mapSize: 1024, pcfTaps: 1, cascades, pcss: 0 };
  }
}

/**
 * PCSS penumbra width (world units) for a directional light: the light's
 * angular diameter seen across the receiver–blocker distance,
 * w = 2 · d · tan(θ/2). Mirrors the shader (pipeline-factory augengine_pcss).
 */
export function pcssPenumbraWidth(blockerToReceiver: number, tanHalfAngle: number): number {
  return 2 * Math.max(blockerToReceiver, 0) * tanHalfAngle;
}

/**
 * The PCSS sample pattern (CPU mirror of the shader): a Vogel disc, golden
 * angle between taps, outermost first, starting at `angle`.
 */
export function vogelDisc(taps: number, angle = 0): Array<[number, number]> {
  const out: Array<[number, number]> = [];
  for (let i = 0; i < taps; i++) {
    const r = Math.sqrt(1 - (i + 0.5) / taps);
    const a = angle + i * 2.39996323;
    out.push([Math.cos(a) * r, Math.sin(a) * r]);
  }
  return out;
}

/** Depth span (world units) of a cascade fitted by fitDirectionalShadow: far − near = 3.99 r, width = 2 r. */
export function cascadeDepthRange(worldSize: number): number {
  return 1.995 * worldSize;
}

/**
 * Bounding spheres (centre distance along the view direction, radius) for the
 * near and far cascades. The near one is the existing single-map fit
 * (diameter `range`, starting at the camera); the far one covers `farScale`×
 * that diameter, starting at the same place, so it contains the near one and
 * the blend region is always inside both.
 */
export function cascadeSpheres(range: number, farScale: number): Array<{ centerDist: number; radius: number }> {
  const r0 = range * 0.5;
  const r1 = range * farScale * 0.5;
  return [
    { centerDist: r0, radius: r0 },
    { centerDist: r1, radius: r1 },
  ];
}
