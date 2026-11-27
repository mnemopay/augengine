/**
 * Backend B: three.js WebGPURenderer behind the LiveRenderer adapter.
 *
 * Reads the same Scene the WebGPU backend reads (entities + world transforms)
 * every frame and mirrors it into a three.js scene. Nothing the game depends
 * on lives here — physics, ECS, personas and streaming keep running on the
 * Scene exactly as they do with backend A.
 *
 * Lighting is matched to backend A's shaders rather than to three's defaults:
 *   - A's diffuse is albedo/π · N·L · radiance, the same as three's
 *     physically based lights, so directional/point radiance maps 1:1.
 *   - A adds ambient as albedo · ambient (no 1/π); three's AmbientLight goes
 *     through the Lambert BRDF (albedo/π), so ambient intensity is scaled by π.
 *   - A's fog is 1 − exp(−density · distance) in world space; three's FogExp2
 *     squares the exponent, so fog is a custom TSL factor instead.
 *   - Local lights: A's falloff is window² / (d² + 1); three's is
 *     window² / max(d², 0.01). Close but brighter within ~1 m of a light.
 *
 * Not implemented (no-ops here): GPU particles, shallow-water surfaces, the
 * outline material (drawn unlit), splats (drawn unlit). Skinning uses
 * three.SkinnedMesh with bones driven from the joint entities' world matrices.
 */

import * as THREE from "three/webgpu";
import {
  pass,
  uniform,
  float,
  exp,
  fog,
  positionWorld,
  cameraPosition,
  positionWorldDirection,
  pow,
  max,
  mix,
  smoothstep,
  clamp,
  renderOutput,
  select,
} from "three/tsl";
import { bloom } from "three/addons/tsl/display/BloomNode.js";
import { fxaa } from "three/addons/tsl/display/FXAANode.js";

import { primitiveMesh, type MeshData, type PrimitiveShape } from "../webgpu/primitives.js";
import { collectSceneLights, shadowSettingsFor, type ShadowQuality } from "../webgpu/lights.js";
import { OVERCLOCK_SKY, type SkyGradient } from "../webgpu/sky.js";
import type { LiveRendererOptions, LookOptions, RenderFrameOptions, RendererFrameStats } from "../webgpu/renderer.js";
import type { LiveRenderer } from "../renderer-api.js";
import type { GpuMsSource, PassTimingSample } from "../../perf/timestamp-queries.js";
import { Tag, type Vec3 } from "../../../scene/types.js";
import type { Scene } from "../../../scene/scene.js";
import type { MaterialKind } from "../../../scene/material.js";

/** Same threshold as backend A's instancing (instanced-draws.ts INSTANCE_MIN_GROUP). */
const INSTANCE_MIN_GROUP = 4;
const MAX_EXTRA_DIRECTIONAL = 3;
const MAX_LOCAL_LIGHTS = 8;

export interface ThreeLiveRendererOptions extends LiveRendererOptions {
  /** Force three's WebGL2 backend (the automatic fallback when WebGPU is missing). */
  forceWebGL?: boolean;
}

interface MeshEntry {
  data: MeshData;
  stride: 6 | 8 | 16;
  geometry: THREE.BufferGeometry | null;
  pbrFactors?: { baseColor: Vec3; metallic: number; roughness: number };
  textures: { map?: THREE.Texture; metalRough?: THREE.Texture; normal?: THREE.Texture };
  /** Bumped when textures change so cached materials rebuild. */
  textureGen: number;
}

interface EntityObject {
  obj: THREE.Mesh;
  meshKey: string;
  matKey: string;
  skinned: boolean;
  seen: number;
}

interface InstanceGroup {
  mesh: THREE.InstancedMesh;
  capacity: number;
  seen: number;
}

interface Candidate {
  entityId: string;
  meshKey: string;
  entry: MeshEntry;
  matKind: MaterialKind["kind"];
  material: Record<string, unknown>;
  world: Float32Array;
  castShadow: boolean;
}

export class ThreeLiveRenderer implements LiveRenderer {
  readonly backend = "three" as const;
  ready = false;

  private canvas: HTMLCanvasElement | OffscreenCanvas;
  private opts: ThreeLiveRendererOptions;
  private renderer: THREE.WebGPURenderer | null = null;
  private scene3 = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(60, 1, 0.1, 100);
  private meshes = new Map<string, MeshEntry>();
  private entities = new Map<string, EntityObject>();
  private groups = new Map<string, InstanceGroup>();
  private materials = new Map<string, THREE.Material>();
  private blendshapes = new Map<string, Float32Array>();
  private frame = 0;
  private sampleCount: 1 | 4 = 1;

  // Lights
  private keyLight = new THREE.DirectionalLight(0xffffff, 1);
  private extraLights: THREE.DirectionalLight[] = [];
  private localLights: Array<THREE.PointLight | THREE.SpotLight> = [];
  private ambientLight = new THREE.AmbientLight(0xffffff, 0);

  // Look
  private skyEnabled = false;
  private skyGradient: SkyGradient = OVERCLOCK_SKY;
  private skyUniforms = {
    zenith: uniform(new THREE.Color()),
    horizon: uniform(new THREE.Color()),
    glow: uniform(new THREE.Color()),
    exponent: uniform(1.4),
    glowFalloff: uniform(6),
  };
  private fogColor = uniform(new THREE.Color());
  private fogDensity = uniform(0);
  private iblEnabled = false;
  private iblIrradiance: Vec3 = [0.22, 0.24, 0.28];
  private iblIntensity = 1;
  private shadowQuality: ShadowQuality = "off";
  private shadowRange = 12;
  private postChain: "full" | "tonemap" = "full";
  private look: LookOptions = {};
  private fogHeight = uniform(0);
  private pipeline: THREE.RenderPipeline | null = null;
  private pipelineChain: "full" | "tonemap" | null = null;
  private instancingEnabled = true;
  private frameStats: RendererFrameStats = { drawCalls: 0, instancedDraws: 0, instancedEntities: 0, entitiesDrawn: 0, triangles: 0 };

  // Timing
  private timestampSupport = { supported: false, reason: "not probed" };
  private gpuMsListener: ((sample: { gpu_ms: number; source: GpuMsSource }) => void) | null = null;
  private passListener: ((sample: PassTimingSample) => void) | null = null;
  private resolving = false;

  // Scratch
  private tmpMat = new THREE.Matrix4();
  private tmpColor = new THREE.Color();
  private candidates: Candidate[] = [];
  private groupBuckets = new Map<string, Candidate[]>();

  /** Same dispatch as backend A's `meshKeyForEntity` — override per app if needed. */
  meshKeyForEntity = (entityId: string, scene?: Scene): string => {
    if (!scene) return "cube";
    const e = scene.entities.get(entityId as never);
    const m = e?.components[Tag.Mesh];
    if (!m) return "cube";
    const g = m.geometry;
    if (g.kind === "primitive") {
      const mat = m.material;
      if (mat && mat.kind === "pbr" && mat.baseColorTexture) return `primitive:${g.shape}:${mat.baseColorTexture}`;
      return `primitive:${g.shape}`;
    }
    if (g.kind === "gltf") return `gltf:${g.url}#${g.primitiveIndex ?? 0}`;
    if (g.kind === "procedural") return `proc:${g.generator}:${g.seed}`;
    return "cube";
  };

  constructor(canvas: HTMLCanvasElement | OffscreenCanvas, opts: ThreeLiveRendererOptions = {}) {
    this.canvas = canvas;
    this.opts = opts;
    this.scene3.add(this.keyLight, this.keyLight.target, this.ambientLight);
    this.scene3.matrixWorldAutoUpdate = false;
    this.camera.matrixAutoUpdate = false;
    this.keyLight.matrixAutoUpdate = false;
    this.keyLight.target.matrixAutoUpdate = false;
  }

  getSampleCount(): 1 | 4 {
    return this.sampleCount;
  }

  async init(): Promise<void> {
    this.autosizeCanvas();
    const renderer = new THREE.WebGPURenderer({
      canvas: this.canvas as HTMLCanvasElement,
      antialias: false,
      trackTimestamp: true,
      ...(this.opts.forceWebGL ? { forceWebGL: true } : {}),
    });
    await renderer.init();
    this.renderer = renderer;
    renderer.setPixelRatio(1);
    renderer.setSize(this.canvas.width, this.canvas.height, false);
    // A's post chain: ACES at exposure 1.1 (its saturation knob has no three equivalent).
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.shadowMap.enabled = false;
    renderer.shadowMap.type = THREE.PCFShadowMap;

    const isWebGPU = (renderer.backend as unknown as { isWebGPUBackend?: boolean }).isWebGPUBackend === true;
    let tsOk = false;
    try { tsOk = renderer.hasFeature("timestamp-query"); } catch { tsOk = false; }
    this.timestampSupport = tsOk
      ? { supported: true, reason: "" }
      : { supported: false, reason: `${isWebGPU ? "WebGPU" : "WebGL2"} backend without timestamp queries` };

    // No MSAA on this backend (antialias: false); "full" post uses FXAA like A
    // does on integrated GPUs. Report 1 so captures are labelled honestly.
    this.sampleCount = 1;

    if (typeof window !== "undefined") {
      (window as unknown as { __augengineRendererBackend?: unknown }).__augengineRendererBackend = {
        backend: "three",
        api: isWebGPU ? "webgpu" : "webgl2",
        three: THREE.REVISION,
        timestampQuery: tsOk,
      };
    }

    // Same optical depth as A's augengine_fog_factor: uniform when the height
    // falloff k is 0, otherwise density * e^(-k*y) integrated along the ray.
    const toP = positionWorld.sub(cameraPosition);
    const dist = toP.length();
    const kdy = this.fogHeight.mul(toP.y);
    const shape = select(kdy.abs().lessThan(1e-4), float(1), float(1).sub(exp(kdy.negate())).div(kdy));
    const heightOptical = this.fogDensity.mul(dist).mul(exp(this.fogHeight.negate().mul(cameraPosition.y))).mul(shape);
    const optical = select(this.fogHeight.greaterThan(0), heightOptical, this.fogDensity.mul(dist));
    this.scene3.fogNode = fog(this.fogColor, clamp(float(1).sub(exp(optical.negate())), 0, 1));
    this.applyLook();
    this.applySky();
    this.applyShadowQuality();
    if (typeof ResizeObserver !== "undefined" && "clientWidth" in this.canvas) {
      new ResizeObserver(() => this.autosizeCanvas()).observe(this.canvas as HTMLCanvasElement);
    }
    this.ready = true;
  }

  // ─── Mesh registry ────────────────────────────────────────────────
  uploadMesh(name: string, data: MeshData, vertexStride: 6 | 8 | 16 = 6): void {
    if (this.meshes.has(name)) return;
    const entry: MeshEntry = { data, stride: vertexStride, geometry: null, textures: {}, textureGen: 0 };
    if (data.pbrMaterial) {
      entry.pbrFactors = {
        baseColor: data.pbrMaterial.baseColor,
        metallic: data.pbrMaterial.metallic,
        roughness: data.pbrMaterial.roughness,
      };
    }
    this.meshes.set(name, entry);
  }

  /**
   * Free a mesh (streamed meshes that leave with their tile). Objects and
   * instance groups built from it are dropped too, so a later upload under the
   * same key gets fresh geometry instead of the disposed one.
   */
  releaseMesh(name: string): void {
    const entry = this.meshes.get(name);
    if (!entry) return;
    this.meshes.delete(name);
    for (const [id, eo] of this.entities) {
      if (eo.meshKey !== name) continue;
      this.scene3.remove(eo.obj);
      this.entities.delete(id);
    }
    for (const [gk, g] of this.groups) {
      if (!gk.startsWith(`${name}|`)) continue;
      this.scene3.remove(g.mesh);
      g.mesh.dispose();
      this.groups.delete(gk);
    }
    entry.geometry?.dispose();
    entry.textures.map?.dispose();
    entry.textures.metalRough?.dispose();
    entry.textures.normal?.dispose();
  }

  ensurePrimitive(shape: PrimitiveShape): void {
    this.uploadMesh(`primitive:${shape}`, primitiveMesh(shape), 8);
  }

  async uploadPbrTextures(
    meshName: string,
    sources: {
      baseColor?: { bytes?: Uint8Array; mimeType?: string; uri?: string };
      metallicRoughness?: { bytes?: Uint8Array; mimeType?: string; uri?: string };
      normal?: { bytes?: Uint8Array; mimeType?: string; uri?: string };
    },
  ): Promise<boolean> {
    const entry = this.meshes.get(meshName);
    if (!entry || entry.stride < 8) return false;
    const load = async (src: { bytes?: Uint8Array; mimeType?: string; uri?: string }, srgb: boolean) => {
      let blob: Blob;
      if (src.bytes) blob = new Blob([src.bytes.slice().buffer], { type: src.mimeType ?? "image/png" });
      else if (src.uri) {
        const res = await fetch(src.uri);
        if (!res.ok) throw new Error(`texture fetch failed: ${res.status} ${src.uri}`);
        blob = await res.blob();
      } else return undefined;
      // glTF UVs are top-left origin; keep the image unflipped to match backend A.
      const bitmap = await createImageBitmap(blob, { imageOrientation: "none" } as ImageBitmapOptions);
      const tex = new THREE.Texture(bitmap as unknown as HTMLImageElement);
      tex.flipY = false;
      tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
      if (srgb) tex.colorSpace = THREE.SRGBColorSpace;
      tex.needsUpdate = true;
      return tex;
    };
    let ok = false;
    if (sources.baseColor) { const t = await load(sources.baseColor, true); if (t) { entry.textures.map = t; ok = true; } }
    if (sources.metallicRoughness) { const t = await load(sources.metallicRoughness, false); if (t) { entry.textures.metalRough = t; ok = true; } }
    if (sources.normal) { const t = await load(sources.normal, false); if (t) { entry.textures.normal = t; ok = true; } }
    if (ok) entry.textureGen++;
    return ok;
  }

  private geometryFor(entry: MeshEntry): THREE.BufferGeometry {
    if (entry.geometry) return entry.geometry;
    const d = entry.data;
    const geo = new THREE.BufferGeometry();
    const ib = new THREE.InterleavedBuffer(d.vertices, entry.stride);
    geo.setAttribute("position", new THREE.InterleavedBufferAttribute(ib, 3, 0));
    geo.setAttribute("normal", new THREE.InterleavedBufferAttribute(ib, 3, 3));
    if (entry.stride >= 8) geo.setAttribute("uv", new THREE.InterleavedBufferAttribute(ib, 2, 6));
    geo.setIndex(new THREE.BufferAttribute(d.indices, 1));
    if (d.colors) geo.setAttribute("color", new THREE.BufferAttribute(d.colors, 4));
    if (d.joints && d.weights) {
      geo.setAttribute("skinIndex", new THREE.Uint16BufferAttribute(d.joints, 4));
      geo.setAttribute("skinWeight", new THREE.BufferAttribute(d.weights, 4));
    }
    if (d.morphTargets && d.morphTargets.length > 0) {
      geo.morphTargetsRelative = true;
      geo.morphAttributes.position = d.morphTargets.map((t) => new THREE.BufferAttribute(t.positions, 3));
      if (d.morphTargets.every((t) => t.normals)) {
        geo.morphAttributes.normal = d.morphTargets.map((t) => new THREE.BufferAttribute(t.normals!, 3));
      }
    }
    geo.computeBoundingSphere();
    entry.geometry = geo;
    return geo;
  }

  // ─── Materials ────────────────────────────────────────────────────
  private materialFor(
    kind: MaterialKind["kind"],
    mat: Record<string, unknown>,
    entry: MeshEntry,
    opts: { instanced: boolean; morph: boolean; skinned: boolean },
  ): { material: THREE.Material; key: string } {
    const color = (mat.baseColor ?? mat.color ?? [0.7, 0.7, 0.75]) as Vec3;
    const emissive = (mat.emissive ?? [0, 0, 0]) as Vec3;
    const metallic = Number(mat.metallic ?? 0);
    const roughness = Number(mat.roughness ?? 0.5);
    const vcol = !!entry.data.colors;
    const tex = entry.textures;
    // Instanced groups carry colour per instance, so the shared material stays white.
    const c = opts.instanced ? "inst" : `${color[0]},${color[1]},${color[2]}`;
    const key = `${kind}|${c}|${metallic}|${roughness}|${emissive.join(",")}|${vcol ? "vc" : ""}|${opts.morph ? "m" : ""}|${opts.skinned ? "s" : ""}|${tex.map ? entry.textureGen : ""}${tex.map ? `|${Object.keys(tex).join("+")}` : ""}`;
    let m = this.materials.get(key);
    if (m) return { material: m, key };
    const base = opts.instanced ? new THREE.Color(1, 1, 1) : new THREE.Color(color[0], color[1], color[2]);
    if (kind === "pbr") {
      const s = new THREE.MeshStandardNodeMaterial({ color: base, metalness: metallic, roughness });
      s.emissive.setRGB(emissive[0], emissive[1], emissive[2]);
      if (tex.map) s.map = tex.map;
      if (tex.metalRough) { s.metalnessMap = tex.metalRough; s.roughnessMap = tex.metalRough; }
      if (tex.normal) s.normalMap = tex.normal;
      m = s;
    } else if (kind === "cel" || kind === "vrm-mtoon") {
      m = new THREE.MeshToonNodeMaterial({ color: base });
    } else {
      // unlit / outline / splat: flat colour, like backend A's unlit fallback.
      const b = new THREE.MeshBasicNodeMaterial({ color: base });
      if (!opts.instanced) b.color.setRGB(color[0] + emissive[0], color[1] + emissive[1], color[2] + emissive[2]);
      m = b;
    }
    if (vcol) (m as THREE.MeshStandardNodeMaterial).vertexColors = true;
    this.materials.set(key, m);
    return { material: m, key };
  }

  // ─── Look setters ─────────────────────────────────────────────────
  setSky(enabled: boolean, gradient?: Partial<SkyGradient>): void {
    this.skyEnabled = enabled;
    if (gradient) this.skyGradient = { ...this.skyGradient, ...gradient };
    this.applySky();
  }

  private applySky(): void {
    const g = this.skyGradient;
    this.skyUniforms.zenith.value.setRGB(g.zenith[0], g.zenith[1], g.zenith[2]);
    this.skyUniforms.horizon.value.setRGB(g.horizon[0], g.horizon[1], g.horizon[2]);
    this.skyUniforms.glow.value.setRGB(g.horizonGlow[0], g.horizonGlow[1], g.horizonGlow[2]);
    this.skyUniforms.exponent.value = g.exponent;
    this.skyUniforms.glowFalloff.value = (g as unknown as { glowFalloff?: number }).glowFalloff ?? 6;
    if (this.skyEnabled) {
      // Same gradient as backend A's sky.ts, evaluated per view ray.
      const dir = positionWorldDirection;
      const up = clamp(dir.y, 0, 1);
      const t = pow(up, max(this.skyUniforms.exponent, 0.05));
      const glow = exp(up.negate().mul(max(this.skyUniforms.glowFalloff, 0.01))).mul(smoothstep(-0.15, 0.05, dir.y));
      this.scene3.backgroundNode = mix(this.skyUniforms.horizon, this.skyUniforms.zenith, t).add(
        this.skyUniforms.glow.mul(glow),
      );
    } else {
      this.scene3.backgroundNode = null;
    }
  }

  setIbl(enabled: boolean, opts?: { irradiance?: Vec3; intensity?: number }): void {
    this.iblEnabled = enabled;
    if (opts?.irradiance) this.iblIrradiance = opts.irradiance;
    if (opts?.intensity !== undefined) this.iblIntensity = opts.intensity;
  }

  setShadowQuality(quality: ShadowQuality): void {
    this.shadowQuality = quality;
    this.applyShadowQuality();
  }

  private applyShadowQuality(): void {
    const s = shadowSettingsFor(this.shadowQuality);
    if (!this.renderer) return;
    this.renderer.shadowMap.enabled = !!s;
    this.keyLight.castShadow = !!s;
    if (s) {
      this.keyLight.shadow.mapSize.set(s.mapSize, s.mapSize);
      this.keyLight.shadow.map?.dispose();
      this.keyLight.shadow.map = null;
      this.keyLight.shadow.bias = -0.0006;
      this.keyLight.shadow.normalBias = (this.shadowRange / s.mapSize) * 1.5;
    }
  }

  setShadowRange(range: number): void {
    if (Number.isFinite(range) && range > 0.5) this.shadowRange = range;
  }

  setPostChain(chain: "full" | "tonemap"): void {
    this.postChain = chain;
  }

  setInstancing(enabled: boolean): void {
    this.instancingEnabled = enabled;
  }

  /** Tonemapper + exposure map onto three; the display grade is A-only for now. */
  setLook(look: LookOptions): void {
    this.look = { ...this.look, ...look };
    this.applyLook();
  }

  private applyLook(): void {
    const r = this.renderer;
    if (!r) return;
    const t = this.look.tonemapper ?? "aces";
    r.toneMapping = t === "agx" ? THREE.AgXToneMapping : t === "neutral" ? THREE.NeutralToneMapping : THREE.ACESFilmicToneMapping;
    r.toneMappingExposure = this.look.exposure ?? 1.1;
  }

  getFrameStats(): Readonly<RendererFrameStats> {
    return this.frameStats;
  }

  setBlendshapeWeights(entityId: string, weights: Float32Array): void {
    let w = this.blendshapes.get(entityId);
    if (!w || w.length !== weights.length) { w = new Float32Array(weights.length); this.blendshapes.set(entityId, w); }
    w.set(weights);
  }

  // Not implemented on this backend (see header).
  addParticleEmitter(): null { return null; }
  addWaterSurface(): null { return null; }

  // ─── Timing ───────────────────────────────────────────────────────
  enableGpuTimestamps(listener: ((sample: { gpu_ms: number; source: GpuMsSource }) => void) | null): void {
    this.gpuMsListener = listener;
  }

  getTimestampSupport(): { supported: boolean; reason: string } {
    return this.timestampSupport;
  }

  enablePassTimings(listener: ((sample: PassTimingSample) => void) | null): boolean {
    this.passListener = listener;
    return listener !== null && this.timestampSupport.supported;
  }

  private resolveTimestamps(): void {
    if (!this.renderer || this.resolving || !this.timestampSupport.supported) return;
    if (!this.gpuMsListener && !this.passListener) return;
    this.resolving = true;
    const passListener = this.passListener;
    const frameListener = this.gpuMsListener;
    this.renderer.resolveTimestampsAsync(THREE.TimestampQuery.RENDER).then((ms) => {
      this.resolving = false;
      if (ms === undefined || !Number.isFinite(ms)) return;
      // three sums every render pass of the last resolved frame; it does not
      // label them, so the whole frame is reported as "main".
      passListener?.({ frame_ms: ms, sum_ms: ms, passes: { main: ms } });
      frameListener?.({ gpu_ms: ms, source: "timestamp-query" });
    }).catch(() => { this.resolving = false; });
  }

  // ─── Frame ────────────────────────────────────────────────────────
  render(scene: Scene, opts: RenderFrameOptions = {}): void {
    const renderer = this.renderer;
    if (!this.ready || !renderer) return;
    this.frame++;
    const frame = this.frame;
    if (renderer.domElement.width !== this.canvas.width || renderer.domElement.height !== this.canvas.height) {
      renderer.setSize(this.canvas.width, this.canvas.height, false);
    }

    const camPos = this.syncCamera(scene);
    const wt = scene.worldTransforms();

    // Lights — same resolution as backend A (brightest directional = key).
    const lightDir = opts.lightDir ?? [-0.4, -1.0, -0.2];
    const ambientFallback = opts.ambient ?? (this.iblEnabled
      ? [this.iblIrradiance[0] * this.iblIntensity, this.iblIrradiance[1] * this.iblIntensity, this.iblIrradiance[2] * this.iblIntensity] as Vec3
      : [0.18, 0.18, 0.22] as Vec3);
    const lights = collectSceneLights(scene, lightDir, camPos);
    const amb = lights.ambient ?? ambientFallback;
    this.ambientLight.color.setRGB(amb[0], amb[1], amb[2]);
    this.ambientLight.intensity = Math.PI;
    this.placeDirectional(this.keyLight, lights.key.dir, lights.key.radiance, camPos, true);
    for (let i = 0; i < MAX_EXTRA_DIRECTIONAL; i++) {
      const d = lights.extraDirectional[i];
      let l = this.extraLights[i];
      if (!l) { l = new THREE.DirectionalLight(); l.matrixAutoUpdate = false; l.target.matrixAutoUpdate = false; this.scene3.add(l, l.target); this.extraLights[i] = l; }
      l.visible = !!d;
      if (d) this.placeDirectional(l, d.dir, d.radiance, camPos, false);
    }
    this.syncLocalLights(lights.local);

    const fogColor = opts.fogColor ?? opts.clearColor ?? [0, 0, 0];
    this.fogColor.value.setRGB(fogColor[0], fogColor[1], fogColor[2]);
    this.fogDensity.value = opts.fogDensity ?? 0;
    this.fogHeight.value = Math.max(0, opts.fogHeightFalloff ?? 0);
    const clear = opts.clearColor ?? [0.05, 0.06, 0.09];
    renderer.setClearColor(this.tmpColor.setRGB(clear[0], clear[1], clear[2], THREE.LinearSRGBColorSpace), 1);

    // ─── Entities → three objects ─────────────────────────────────
    const shadowOn = this.keyLight.castShadow;
    const candidates = this.candidates;
    candidates.length = 0;
    let singles = 0;
    for (const e of scene.entities.values()) {
      const meshComp = e.components[Tag.Mesh];
      if (!meshComp || !meshComp.visible) continue;
      if (shadowOn && e.id.endsWith(".shadow")) continue;
      const meshKey = this.meshKeyForEntity(e.id, scene);
      let entry = this.meshes.get(meshKey);
      if (!entry && meshKey.startsWith("primitive:")) {
        this.uploadMesh(meshKey, primitiveMesh(meshKey.split(":")[1] as PrimitiveShape), 8);
        entry = this.meshes.get(meshKey);
        const mat = meshComp.material;
        if (entry && mat.kind === "pbr" && mat.baseColorTexture) {
          void this.uploadPbrTextures(meshKey, {
            baseColor: { uri: mat.baseColorTexture },
            ...(mat.metallicRoughnessTexture ? { metallicRoughness: { uri: mat.metallicRoughnessTexture } } : {}),
            ...(mat.normalTexture ? { normal: { uri: mat.normalTexture } } : {}),
          });
        }
      }
      if (!entry) continue;
      const idx = wt.indexById.get(e.id);
      if (idx === undefined) continue;
      const world = wt.worldMatrices.subarray(idx * 16, idx * 16 + 16);
      const matKind = meshComp.material.kind as MaterialKind["kind"];
      const material = (entry.pbrFactors && matKind === "pbr"
        ? { kind: "pbr", ...entry.pbrFactors }
        : meshComp.material) as Record<string, unknown>;
      const skin = e.components[Tag.Skin];
      const skinned = !!skin && !!entry.data.joints;
      const morph = !!entry.data.morphTargets?.length;
      const castShadow = shadowOn && !e.id.endsWith(".shadow") && matKind !== "unlit" && matKind !== "outline" && matKind !== "splat";
      if (
        this.instancingEnabled && !skinned && !morph && !entry.textures.map &&
        (matKind === "pbr" || matKind === "unlit" || matKind === "cel") && entry.stride !== 16
      ) {
        candidates.push({ entityId: e.id, meshKey, entry, matKind, material, world, castShadow });
        continue;
      }
      this.syncSingle(e.id, meshKey, entry, matKind, material, world, castShadow, frame, skinned ? skin! : null, scene, morph);
      singles++;
    }

    // Instancing: group by mesh + material (colour goes per instance).
    const buckets = this.groupBuckets;
    for (const b of buckets.values()) b.length = 0;
    for (const c of candidates) {
      const m = c.material;
      const gk = `${c.meshKey}|${c.matKind}|${Number(m.metallic ?? 0)}|${Number(m.roughness ?? 0.5)}|${((m.emissive ?? [0, 0, 0]) as Vec3).join(",")}|${c.castShadow ? 1 : 0}`;
      let b = buckets.get(gk);
      if (!b) { b = []; buckets.set(gk, b); }
      b.push(c);
    }
    let instancedDraws = 0;
    let instancedEntities = 0;
    for (const [gk, items] of buckets) {
      if (items.length === 0) continue;
      if (items.length < INSTANCE_MIN_GROUP) {
        for (const c of items) {
          this.syncSingle(c.entityId, c.meshKey, c.entry, c.matKind, c.material, c.world, c.castShadow, frame, null, scene, false);
          singles++;
        }
        continue;
      }
      this.syncGroup(gk, items, frame);
      instancedDraws++;
      instancedEntities += items.length;
    }

    // Hide (or drop) objects not touched this frame.
    for (const [id, eo] of this.entities) {
      if (eo.seen === frame) continue;
      if (!scene.entities.has(id as never)) { this.scene3.remove(eo.obj); this.entities.delete(id); }
      else eo.obj.visible = false;
    }
    for (const g of this.groups.values()) if (g.seen !== frame) g.mesh.visible = false;

    // ─── Draw ─────────────────────────────────────────────────────
    this.renderFrame(renderer);
    const info = renderer.info.render;
    this.frameStats = {
      drawCalls: info.drawCalls,
      instancedDraws,
      instancedEntities,
      entitiesDrawn: singles + instancedEntities,
      triangles: info.triangles,
    };
    this.resolveTimestamps();
  }

  private renderFrame(renderer: THREE.WebGPURenderer): void {
    if (this.postChain === "tonemap") {
      renderer.render(this.scene3, this.camera);
      return;
    }
    if (!this.pipeline || this.pipelineChain !== this.postChain) {
      // A's "full" chain: bloom (threshold 0.9, intensity 0.6) → ACES → FXAA.
      const scenePass = pass(this.scene3, this.camera);
      const color = scenePass.getTextureNode("output");
      const withBloom = color.add(bloom(color, 0.6, 0.4, 0.9));
      this.pipeline?.dispose();
      this.pipeline = new THREE.RenderPipeline(renderer);
      this.pipeline.outputColorTransform = false;
      this.pipeline.outputNode = fxaa(renderOutput(withBloom));
      this.pipelineChain = this.postChain;
    }
    this.pipeline.render();
  }

  private syncSingle(
    id: string,
    meshKey: string,
    entry: MeshEntry,
    matKind: MaterialKind["kind"],
    material: Record<string, unknown>,
    world: Float32Array,
    castShadow: boolean,
    frame: number,
    skin: { jointEntities: readonly string[]; inverseBindMatrices: Float32Array } | null,
    scene: Scene,
    morph: boolean,
  ): void {
    const { material: mat, key: matKey } = this.materialFor(matKind, material, entry, { instanced: false, morph, skinned: !!skin });
    let eo = this.entities.get(id);
    if (!eo || eo.meshKey !== meshKey || eo.skinned !== !!skin) {
      if (eo) this.scene3.remove(eo.obj);
      const geo = this.geometryFor(entry);
      let obj: THREE.Mesh;
      if (skin) {
        const bones = skin.jointEntities.map(() => { const b = new THREE.Bone(); b.matrixAutoUpdate = false; return b; });
        const inverses = skin.jointEntities.map((_, i) => new THREE.Matrix4().fromArray(skin.inverseBindMatrices, i * 16));
        const sm = new THREE.SkinnedMesh(geo, mat);
        sm.bindMode = THREE.DetachedBindMode;
        sm.bind(new THREE.Skeleton(bones, inverses), new THREE.Matrix4());
        sm.frustumCulled = false;
        obj = sm;
      } else {
        obj = new THREE.Mesh(geo, mat);
      }
      obj.matrixAutoUpdate = false;
      this.scene3.add(obj);
      eo = { obj, meshKey, matKey, skinned: !!skin, seen: frame };
      this.entities.set(id, eo);
    }
    if (eo.matKey !== matKey) { eo.obj.material = mat; eo.matKey = matKey; }
    eo.seen = frame;
    eo.obj.visible = true;
    eo.obj.castShadow = castShadow;
    eo.obj.receiveShadow = matKind !== "unlit";
    if (skin) {
      // Vertices skin straight into world space (backend A does the same):
      // bones carry each joint's world matrix, the mesh itself stays identity.
      const wt = scene.worldTransforms();
      const sk = (eo.obj as THREE.SkinnedMesh).skeleton;
      skin.jointEntities.forEach((jid, i) => {
        const ji = wt.indexById.get(jid as never);
        if (ji !== undefined) sk.bones[i]!.matrixWorld.fromArray(wt.worldMatrices, ji * 16);
      });
      sk.update();
      eo.obj.matrixWorld.identity();
    } else {
      eo.obj.matrixWorld.fromArray(world);
    }
    if (morph) {
      const w = this.blendshapes.get(id);
      const infl = eo.obj.morphTargetInfluences;
      if (infl && w) for (let i = 0; i < infl.length; i++) infl[i] = w[i] ?? 0;
    }
  }

  private syncGroup(gk: string, items: Candidate[], frame: number): void {
    const first = items[0]!;
    const { material } = this.materialFor(first.matKind, first.material, first.entry, { instanced: true, morph: false, skinned: false });
    let g = this.groups.get(gk);
    if (!g || g.capacity < items.length) {
      if (g) { this.scene3.remove(g.mesh); g.mesh.dispose(); }
      const capacity = Math.max(16, 1 << Math.ceil(Math.log2(items.length)));
      const mesh = new THREE.InstancedMesh(this.geometryFor(first.entry), material, capacity);
      mesh.matrixAutoUpdate = false;
      mesh.frustumCulled = false; // instances move every frame; A does not cull either
      mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      mesh.setColorAt(0, this.tmpColor.setRGB(1, 1, 1));
      this.scene3.add(mesh);
      g = { mesh, capacity, seen: frame };
      this.groups.set(gk, g);
    }
    const mesh = g.mesh;
    for (let i = 0; i < items.length; i++) {
      const c = items[i]!;
      this.tmpMat.fromArray(c.world);
      mesh.setMatrixAt(i, this.tmpMat);
      const col = (c.material.baseColor ?? c.material.color ?? [0.7, 0.7, 0.75]) as Vec3;
      mesh.setColorAt(i, this.tmpColor.setRGB(col[0], col[1], col[2]));
    }
    mesh.count = items.length;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.castShadow = first.castShadow;
    mesh.receiveShadow = first.matKind !== "unlit";
    mesh.visible = true;
    g.seen = frame;
  }

  private placeDirectional(
    l: THREE.DirectionalLight,
    dir: Vec3,
    radiance: Vec3,
    camPos: Vec3,
    key: boolean,
  ): void {
    const peak = Math.max(radiance[0], radiance[1], radiance[2], 1e-6);
    l.color.setRGB(radiance[0] / peak, radiance[1] / peak, radiance[2] / peak);
    l.intensity = peak;
    // Shadow region: a sphere in front of the camera, like backend A.
    const radius = this.shadowRange * 0.5;
    const fwd = new THREE.Vector3(0, 0, -1).applyMatrix3(new THREE.Matrix3().setFromMatrix4(this.camera.matrixWorld)).normalize();
    const center = key
      ? new THREE.Vector3(camPos[0] + fwd.x * radius, camPos[1] + fwd.y * radius, camPos[2] + fwd.z * radius)
      : new THREE.Vector3(camPos[0], camPos[1], camPos[2]);
    const dist = radius * 2;
    l.position.set(center.x - dir[0] * dist, center.y - dir[1] * dist, center.z - dir[2] * dist);
    l.updateMatrix();
    l.matrixWorld.copy(l.matrix);
    l.target.position.copy(center);
    l.target.updateMatrix();
    l.target.matrixWorld.copy(l.target.matrix);
    if (key && l.castShadow) {
      const cam = l.shadow.camera as THREE.OrthographicCamera;
      cam.left = -radius; cam.right = radius; cam.top = radius; cam.bottom = -radius;
      cam.near = 0.05; cam.far = dist + radius * 2;
      cam.updateProjectionMatrix();
    }
  }

  private syncLocalLights(local: ReturnType<typeof collectSceneLights>["local"]): void {
    for (let i = 0; i < MAX_LOCAL_LIGHTS; i++) {
      const src = local[i];
      let l = this.localLights[i];
      const wantSpot = !!src?.spotDir;
      if (src && (!l || (l instanceof THREE.SpotLight) !== wantSpot)) {
        if (l) this.scene3.remove(l);
        l = wantSpot ? new THREE.SpotLight() : new THREE.PointLight();
        l.matrixAutoUpdate = false;
        this.scene3.add(l);
        if (l instanceof THREE.SpotLight) { l.target.matrixAutoUpdate = false; this.scene3.add(l.target); }
        this.localLights[i] = l;
      }
      if (!l) continue;
      l.visible = !!src;
      if (!src) continue;
      const peak = Math.max(src.radiance[0], src.radiance[1], src.radiance[2], 1e-6);
      l.color.setRGB(src.radiance[0] / peak, src.radiance[1] / peak, src.radiance[2] / peak);
      l.intensity = peak;
      l.distance = src.range;
      l.decay = 2;
      l.position.set(src.position[0], src.position[1], src.position[2]);
      l.updateMatrix();
      l.matrixWorld.copy(l.matrix);
      if (l instanceof THREE.SpotLight && src.spotDir) {
        l.angle = Math.acos(src.cosOuter ?? Math.cos(Math.PI / 4));
        const inner = Math.acos(src.cosInner ?? Math.cos(Math.PI / 5));
        l.penumbra = l.angle > 0 ? Math.min(1, Math.max(0, 1 - inner / l.angle)) : 0;
        l.target.position.set(src.position[0] + src.spotDir[0], src.position[1] + src.spotDir[1], src.position[2] + src.spotDir[2]);
        l.target.updateMatrix();
        l.target.matrixWorld.copy(l.target.matrix);
      }
    }
  }

  /** Camera from the scene's active camera entity, same convention as backend A. */
  private syncCamera(scene: Scene): Vec3 {
    const w = this.canvas.width;
    const h = Math.max(1, this.canvas.height);
    const cam = this.camera;
    cam.aspect = w / h;
    const camId = scene.activeCamera ?? null;
    const wt = scene.worldTransforms();
    const idx = camId ? wt.indexById.get(camId) : undefined;
    if (!camId || idx === undefined) {
      cam.fov = 60; cam.near = 0.1; cam.far = 100;
      cam.matrixWorld.identity();
    } else {
      const m = wt.worldMatrices.subarray(idx * 16, idx * 16 + 16);
      // Strip scale: rebuild an orthonormal basis from forward (−Z) and up (+Y).
      const eye = new THREE.Vector3(m[12]!, m[13]!, m[14]!);
      const fwd = new THREE.Vector3(-m[8]!, -m[9]!, -m[10]!).normalize();
      const up = new THREE.Vector3(m[4]!, m[5]!, m[6]!).normalize();
      this.tmpMat.lookAt(eye, eye.clone().add(fwd), up);
      cam.matrixWorld.copy(this.tmpMat).setPosition(eye);
      const comp = scene.entities.get(camId)?.components[Tag.Camera];
      cam.fov = THREE.MathUtils.radToDeg(comp?.fovY ?? Math.PI / 3);
      cam.near = comp?.near ?? 0.1;
      cam.far = comp?.far ?? 100;
    }
    cam.matrixWorldInverse.copy(cam.matrixWorld).invert();
    cam.updateProjectionMatrix();
    return [cam.matrixWorld.elements[12]!, cam.matrixWorld.elements[13]!, cam.matrixWorld.elements[14]!];
  }

  resize(width: number, height: number): void {
    this.canvas.width = width;
    this.canvas.height = height;
    this.renderer?.setSize(width, height, false);
  }

  private autosizeCanvas(): void {
    const c = this.canvas as HTMLCanvasElement;
    if (!c || typeof c.clientWidth !== "number") return;
    const dpr = Math.min((typeof window !== "undefined" && window.devicePixelRatio) || 1, 3);
    const w = Math.max(1, Math.round(c.clientWidth * dpr));
    const h = Math.max(1, Math.round(c.clientHeight * dpr));
    if (c.width === w && c.height === h) return;
    c.width = w;
    c.height = h;
  }

  destroy(): void {
    this.ready = false;
    for (const eo of this.entities.values()) this.scene3.remove(eo.obj);
    this.entities.clear();
    for (const g of this.groups.values()) { this.scene3.remove(g.mesh); g.mesh.dispose(); }
    this.groups.clear();
    for (const m of this.meshes.values()) {
      m.geometry?.dispose();
      m.textures.map?.dispose();
      m.textures.metalRough?.dispose();
      m.textures.normal?.dispose();
    }
    this.meshes.clear();
    for (const m of this.materials.values()) m.dispose();
    this.materials.clear();
    this.pipeline?.dispose();
    this.pipeline = null;
    void this.renderer?.dispose();
    this.renderer = null;
  }
}
