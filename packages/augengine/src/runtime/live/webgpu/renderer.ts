/**
 * WebGPU live renderer.
 *
 * Phase 1 shipped a single hardcoded pipeline. Phase 9 closeout swaps it for
 * the `PipelineFactory` so each entity draws with the WGSL pipeline matching
 * its material kind (unlit / pbr / cel / outline / vrm-mtoon). Pipelines are
 * compiled lazily and cached on (materialKind, vertexStride, sampleCount,
 * hasSkinning).
 */

import { cubeMesh, primitiveMesh, type MeshData, type PrimitiveShape } from "./primitives.js";
import { PipelineFactory, pipelineKeyString, type CompiledPipeline, type PipelineKey } from "./pipeline-factory.js";
import {
  DRAW_RECORD_FLOATS,
  INSTANCE_MIN_GROUP,
  instanceCapacity,
  packDrawRecord,
  planInstanceGroups,
} from "./instanced-draws.js";
import { SKINNING_COMPUTE_WGSL, SKINNING_COMPUTE_WGSL_UV, buildBoneMatricesGltf } from "./skinning.js";
import { PostProcessingRunner, type Tonemapper } from "./post-processing.js";
import { SkyPass, OVERCLOCK_SKY, type SkyGradient } from "./sky.js";
import { TemporalResolver, applyProjectionJitter, jitterToUv, scaledSize, taaJitter } from "./temporal.js";
import { packSkyAmbient, packSkySpecular } from "./ibl.js";
import { PROBE_RING_FLOATS, ProbeRing, type ProbeGridParams, type ProbeTileLike, type RingRange } from "./probe-ring.js";
import { MotionPass, type MotionDraw } from "./motion.js";
import {
  ParticlePass,
  FIRE_PRESET,
  type ParticleEmitterOptions,
  type ParticleEmitterSpec,
} from "./particle-pass.js";
import {
  WaterSurfacePass,
  OCEAN_WATER_LOOK,
  type WaterSurfaceOptions,
  type WaterSurfaceStats,
  type WaterLook,
} from "./water-surface.js";
import {
  POOL_PRESET,
  type ShallowWaterSpec,
  type ShallowWaterFieldStats,
} from "./shallow-water.js";
import {
  TimestampSession,
  PassTimestampSession,
  probeTimestampQuerySupport,
  type GpuMsSource,
  type PassTimingSample,
} from "../../perf/timestamp-queries.js";
import {
  BlendshapeUniformPool,
  type BlendshapeBufferLike,
} from "./blendshapes.js";
import {
  MorphDeltaStorage,
  uploadMorphTargets,
  type MorphDeltasHandle,
} from "./morph-deltas.js";
import {
  cascadeSpheres,
  collectSceneLights,
  fitDirectionalShadow,
  type ShadowUniform,
  FRAME_LAYOUT,
  packFrameLights,
  shadowSettingsFor,
  FRAME_UNIFORM_SIZE,
  type FrameLights,
  type ShadowQuality,
  type ShadowSettings,
} from "./lights.js";
import {
  m4Create,
  m4Identity,
  m4Invert,
  m4LookAt,
  m4Mul,
  m4Perspective,
  type MutVec3,
  v3,
} from "../../../scene/math.js";
import type { Mat4, Vec3, Quat } from "../../../scene/types.js";
import { Tag } from "../../../scene/types.js";
import type { Scene } from "../../../scene/scene.js";
import type { MaterialKind } from "../../../scene/material.js";
import type { LiveRenderer } from "../renderer-api.js";

interface MeshGpu {
  vbuf: GPUBuffer;
  ibuf: GPUBuffer;
  indexCount: number;
  vertexCount: number;
  /** Vertex stride matches one of the factory's vertexStride enum values. */
  vertexStride: 6 | 8 | 16;
  vbufSkinned?: GPUBuffer;
  vAttrsBuf?: GPUBuffer;
  vertexCountBuf?: GPUBuffer;
  vcbuf?: GPUBuffer;
  /** Last frame's skinned positions (motion vectors); copied after the motion pass. */
  vbufSkinnedPrev?: GPUBuffer;
  skinnedPrevValid?: boolean;
  morph?: MorphDeltasHandle;
  pbrFactors?: { baseColor: Vec3; metallic: number; roughness: number };
  baseColorTexture?: GPUTexture;
  metallicRoughnessTexture?: GPUTexture;
  normalTexture?: GPUTexture;
  pbrTextureBindGroups?: Map<string, GPUBindGroup>;
}

// Frame uniform size comes from lights.ts (FRAME_LAYOUT): camera, fog, key
// light + shadow matrix, extra directional lights and point/spot lights.
const DRAW_UNIFORM_SIZE = 16 * 4 + 4 * 4 + 4 * 4 + 4 * 4; // mat4 + color + params + emissive = 112 bytes

/** Depth-only pass into the key light's shadow map. Reuses each entity's draw
 *  bind group (group 1), so its layout must stay group-equivalent to the
 *  pipeline factory's draw layout. Morph-target offsets are not applied — a
 *  face's blendshape deltas are far below one shadow texel. */
const SHADOW_DEPTH_WGSL = /* wgsl */ `
struct ShadowFrame { lightViewProj: mat4x4<f32>, };
@group(0) @binding(0) var<uniform> u_shadow: ShadowFrame;
struct DrawUniforms { model: mat4x4<f32>, color: vec4<f32>, params: vec4<f32>, emissive: vec4<f32>, };
@group(1) @binding(0) var<uniform> u_draw: DrawUniforms;
@vertex
fn vs_main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
  return u_shadow.lightViewProj * (u_draw.model * vec4<f32>(position, 1.0));
}
`;

/** Shadow depth for instanced draws: the draw records are a storage array. */
const SHADOW_DEPTH_INSTANCED_WGSL = /* wgsl */ `
struct ShadowFrame { lightViewProj: mat4x4<f32>, };
@group(0) @binding(0) var<uniform> u_shadow: ShadowFrame;
struct DrawUniforms { model: mat4x4<f32>, color: vec4<f32>, params: vec4<f32>, emissive: vec4<f32>, };
@group(1) @binding(0) var<storage, read> u_draws: array<DrawUniforms>;
@vertex
fn vs_main(@location(0) position: vec3<f32>, @builtin(instance_index) inst: u32) -> @builtin(position) vec4<f32> {
  return u_shadow.lightViewProj * (u_draws[inst].model * vec4<f32>(position, 1.0));
}
`;

/** One visible mesh, prepared once per frame and drawn by both the shadow
 *  pass and the main pass. */
interface DrawRecord {
  mesh: MeshGpu;
  entry: PipelineCacheEntry;
  cacheKey: string;
  drawBindGroup: GPUBindGroup;
  hasSkinning: boolean;
  morphBindGroup: GPUBindGroup | null;
  textureBindGroup: GPUBindGroup | null;
  castsShadow: boolean;
  /** 1 for a per-entity draw; >1 draws that many records from a storage buffer. */
  instanceCount: number;
  instanced: boolean;
  /** Per-entity draws only: what the motion pass needs to redraw it. */
  entityId?: string;
  model?: Float32Array;
  hasMorph?: boolean;
}

/** An entity eligible for instancing, held until the frame's groups are known. */
interface InstanceCandidate {
  entityId: string;
  meshKey: string;
  mesh: MeshGpu;
  entry: PipelineCacheEntry;
  cacheKey: string;
  matKind: MaterialKind["kind"];
  worldMat: Float32Array;
  color: readonly number[];
  params: readonly number[];
  emissive: readonly number[];
}

interface InstanceGroupGpu {
  buf: GPUBuffer;
  bindGroup: GPUBindGroup;
  capacity: number;
  data: Float32Array;
  uploaded: Float32Array;
  uploadedCount: number;
}

/** Per-frame draw counts, for perf HUDs and captures. */
export interface RendererFrameStats {
  /** drawIndexed calls in the main pass. */
  drawCalls: number;
  /** Of those, instanced draws. */
  instancedDraws: number;
  /** Entities drawn through instanced draws. */
  instancedEntities: number;
  /** All entities drawn (instanced or not). */
  entitiesDrawn: number;
  /** Triangles submitted in the main pass. */
  triangles: number;
}

export interface LiveRendererOptions {
  /**
   * Sample count for MSAA — 1 (off), 4, or "auto". Default 4.
   * "auto" resolves during `init()` from the adapter: integrated / software
   * adapters get 1 (FXAA still runs), everything else 4. Measured on Intel UHD
   * gen-11 at 1440×900: 4× MSAA = 12.5 ms GPU/frame, 1× = 7.3 ms.
   */
  sampleCount?: 1 | 4 | "auto";
  /** Preferred swap-chain format. Defaults to navigator preferred. */
  format?: GPUTextureFormat;
}

interface PipelineCacheEntry {
  compiled: CompiledPipeline;
  /** One frame uniform per pipeline (layouts differ between cached entries). */
  frameBuf: GPUBuffer;
  frameBindGroup: GPUBindGroup;
  /** Shadow-map generation the frame bind group was built against. */
  shadowGen: number;
}

export class WebGpuLiveRenderer implements LiveRenderer {
  readonly backend = "webgpu" as const;
  private canvas: HTMLCanvasElement | OffscreenCanvas;
  private device: GPUDevice | null = null;
  private context: GPUCanvasContext | null = null;
  private format: GPUTextureFormat = "bgra8unorm";
  private factory: PipelineFactory | null = null;
  private pipelineCache = new Map<string, PipelineCacheEntry>();
  private depthTexture: GPUTexture | null = null;
  private msaaTexture: GPUTexture | null = null;
  private sampleCount: 1 | 4;
  /** Per-entity ARKit weights bound for morph-capable native mesh draws. */
  private blendshapePool: BlendshapeUniformPool | null = null;
  private morphDeltaStorage: MorphDeltaStorage | null = null;
  private pbrSampler: GPUSampler | null = null;
  private skinComputePipeline: GPUComputePipeline | null = null;
  private skinComputeUvPipeline: GPUComputePipeline | null = null;
  private skinCache = new Map<string, { meshName: string; boneBuf: GPUBuffer; bindGroup: GPUBindGroup; packed: Float32Array }>();
  private entityDrawCache = new Map<string, { drawBuf: GPUBuffer; bindGroup: GPUBindGroup; pipelineKey: string; uploaded: Float32Array }>();
  private entityMorphCache = new Map<string, { bindGroup: GPUBindGroup; pipelineKey: string; meshKey: string }>();
  private postRunner: PostProcessingRunner | null = null;
  /** Gradient-sky background pass — fills the upper-half void behind geometry.
   *  Created in init(); defaults to the OVERCLOCK neon-night gradient. */
  private skyPass: SkyPass | null = null;
  private skyEnabled = false;
  private skyGradient: SkyGradient = OVERCLOCK_SKY;
  /** GPU particle emitters (fire / smoke / embers). Created lazily on the
   *  first `addParticleEmitter` call so a scene with no VFX pays nothing —
   *  no pipelines compiled, no buffers allocated, no passes encoded. */
  private particlePass: ParticlePass | null = null;
  /** GPU shallow-water solver + surface draw. Created lazily on the first
   *  `addWaterSurface` call so a scene with no water pays nothing — no
   *  pipelines compiled, no cell buffer, no scene-colour copy, no second
   *  render pass. See water-surface.ts and shallow-water.ts. */
  private waterPass: WaterSurfacePass | null = null;
  /** Copy of the resolved opaque scene colour, taken between the opaque pass
   *  and the water pass so the water fragment shader can sample what is
   *  behind it for refraction. A texture cannot be sampled while it is the
   *  open pass's colour attachment, which is the whole reason the main pass
   *  gets split when water is on screen. Allocated only with `waterPass`. */
  private sceneColorTexture: GPUTexture | null = null;
  // ─── Directional shadow map (key light) ──────────────────────────
  // Off by default so existing callers render exactly as before; hosts opt in
  // with setShadowQuality("low" | "high"). When off, pipelines still bind a
  // 1×1 depth texture so one frame layout serves every setting.
  private shadowQuality: ShadowQuality = "off";
  private shadowSettings: ShadowSettings | null = null;
  private shadowTexture: GPUTexture | null = null;
  private shadowView: GPUTextureView | null = null;
  private dummyShadowTexture: GPUTexture | null = null;
  private dummyShadowView: GPUTextureView | null = null;
  private shadowSampler: GPUSampler | null = null;
  /** Bumped whenever the bound shadow view changes; frame bind groups rebuild. */
  private shadowGen = 0;
  private shadowFrameBuf: GPUBuffer | null = null;
  private shadowFrameBindGroup: GPUBindGroup | null = null;
  /** Far cascade's light view-projection (shadow pass, second viewport). */
  private shadowFrameBuf2: GPUBuffer | null = null;
  private shadowFrameBindGroup2: GPUBindGroup | null = null;
  private shadowScratch2 = new Float32Array(16);
  private shadowFitScratch2 = new Float32Array(16);
  /** 1 = one key-light map (default); 2 = near + far cascade atlas. */
  private shadowCascades: 1 | 2 = 1;
  /** Far cascade diameter as a multiple of `shadowRange`. */
  private shadowFarScale = 4;
  private shadowFrameLayout: GPUBindGroupLayout | null = null;
  private shadowDrawLayout: GPUBindGroupLayout | null = null;
  private shadowModule: GPUShaderModule | null = null;
  private shadowPipelines = new Map<number, GPURenderPipeline>();
  private shadowInstancedModule: GPUShaderModule | null = null;
  private shadowInstancedDrawLayout: GPUBindGroupLayout | null = null;
  private shadowInstancedPipelines = new Map<number, GPURenderPipeline>();
  /** Batch entities that share a mesh + pipeline into one instanced draw. */
  private instancingEnabled = true;
  private instanceGroups = new Map<string, InstanceGroupGpu>();
  private instanceCandidates: InstanceCandidate[] = [];
  private frameStats: RendererFrameStats = { drawCalls: 0, instancedDraws: 0, instancedEntities: 0, entitiesDrawn: 0, triangles: 0 };
  /** Diameter (world units) of the region in front of the camera that the
   *  shadow map covers. Smaller = sharper shadows, shorter reach. */
  private shadowRange = 12;
  /** Lights resolved for the last frame — exposed for HUD / tests. */
  private lastFrameLights: FrameLights | null = null;
  // Per-frame scratch reused across frames (render() used to allocate these).
  private frameScratch = new Float32Array(FRAME_UNIFORM_SIZE / 4);
  private drawScratch = new Float32Array(DRAW_UNIFORM_SIZE / 4);
  private shadowScratch = new Float32Array(16);
  private shadowFitScratch = new Float32Array(16);
  private camView = m4Create();
  private camProj = m4Create();
  private camVP = m4Create();
  private camPosScratch: MutVec3 = [0, 0, 0];
  private camEye: MutVec3 = [0, 0, 0];
  private camFwd: MutVec3 = [0, 0, 0];
  private camTarget: MutVec3 = [0, 0, 0];
  private camUp: MutVec3 = [0, 1, 0];
  /** Pipelines whose frame uniform was considered this frame (the bytes may already match). */
  private frameWritten = new Set<string>();
  /** Last frame-uniform bytes uploaded per pipeline key. Skips a queue write when the camera is still. */
  private frameUploaded = new Map<string, Float32Array>();
  private drawList: DrawRecord[] = [];

  // ─── Temporal AA / upscaling + SSAO (see temporal.ts) ──────────────
  // Off by default: callers opt in with setImageQuality(). TAA needs a
  // sampleable single-sample depth buffer, so it only runs at 1× MSAA.
  private taaEnabled = false;
  private renderScale = 1;
  private temporal: TemporalResolver | null = null;
  private taaFrame = 0;
  private taaHistoryValid = false;
  private depthViewCached: GPUTextureView | null = null;
  private projUnjittered = m4Create();
  private vpUnjittered = m4Create();
  private prevVP = m4Create();
  private invVP = m4Create();
  private invProj = m4Create();
  private pendingQuality: ImageQualityOptions | null = null;
  /** Per-object motion vectors for TAA (motion.ts). On by default with TAA. */
  private motionVectors = true;
  private motionPass: MotionPass | null = null;
  private motionDraws: MotionDraw[] = [];
  /** Each per-entity draw's model last frame — a changed matrix = moving. */
  private lastModels = new Map<string, Float32Array>();
  /** Blendshape weights drawn last frame, per entity. */
  private prevMorphWeights = new Map<string, Float32Array>();
  /** Sky-shaped ambient (ibl.ts); off = the flat ambient, exactly as before. */
  private skyAmbient = false;
  /** Split-sum sky specular on PBR (ibl.ts); off = no environment reflection. */
  private skySpecular = false;
  private groundBounce = 0.35;
  /** Streamed irradiance probes (probe-ring.ts); off until a host enables them. */
  private probeRing = new ProbeRing();
  private probeBuffer: GPUBuffer | null = null;
  private probesEnabled = false;
  private probeGrid: ProbeGridParams = { spacing: 16, heightAboveGround: 1.5, layerSpacing: 12 };

  /** Image-based lighting proxy — hemisphere irradiance used as ambient. */
  private iblEnabled = false;
  private iblIrradiance: Vec3 = [0.22, 0.24, 0.28];
  private iblIntensity = 1;

  // Track 6 — WebGPU timestamp queries around the main render pass.
  // Lazy: only created when the adapter supports `timestamp-query` AND
  // a caller has armed instrumentation via `enableGpuTimestamps()`.
  // Falls back to a CPU wall-clock around `device.queue.submit()` when
  // the feature is unavailable. Listener fires per resolved sample so
  // the smoke-game can push values onto `window.__augengineFrameBudget`
  // without coupling renderer → smoke-game.
  private timestampSession: TimestampSession | null = null;
  private timestampSupport: { supported: boolean; reason: string } = {
    supported: false,
    reason: "not probed",
  };
  private gpuMsListener:
    | ((sample: { gpu_ms: number; source: GpuMsSource }) => void)
    | null = null;
  /** Per-pass GPU timer (shadow / main / water / post / skin). Replaces the
   *  frame-spanning session while armed: one pass carries one query set. */
  private passTimer: PassTimestampSession | null = null;
  private passListener: ((sample: PassTimingSample) => void) | null = null;
  /** Look set before init(); replayed onto the post runner. */
  private pendingLook: LookOptions | null = null;

  private meshes = new Map<string, MeshGpu>();

  /**
   * Resolve which mesh key to draw for an entity. Default: dispatch on the
   * entity's MeshComponent.geometry — primitives become "primitive:<shape>",
   * everything else uses an explicit string key the caller registered via
   * `uploadMesh(name, data)`. Override to swap meshes per-entity.
   */
  meshKeyForEntity = (entityId: string, scene?: Scene): string => {
    if (!scene) return "cube";
    const e = scene.entities.get(entityId as never);
    const m = e?.components[Tag.Mesh];
    if (!m) return "cube";
    const g = m.geometry;
    if (g.kind === "primitive") {
      const mat = m.material;
      if (mat && mat.kind === "pbr" && mat.baseColorTexture) {
        return `primitive:${g.shape}:${mat.baseColorTexture}`;
      }
      return `primitive:${g.shape}`;
    }
    if (g.kind === "gltf") return `gltf:${g.url}#${g.primitiveIndex ?? 0}`;
    if (g.kind === "procedural") return `proc:${g.generator}:${g.seed}`;
    return "cube";
  };

  ready = false;

  /** Set by destroy(); stops late ResizeObserver callbacks from touching freed GPU objects. */
  private destroyed = false;
  private resizeObserver: ResizeObserver | null = null;

  private autoSampleCount = false;

  constructor(canvas: HTMLCanvasElement | OffscreenCanvas, opts: LiveRendererOptions = {}) {
    this.canvas = canvas;
    const requested = opts.sampleCount ?? 4;
    this.autoSampleCount = requested === "auto";
    this.sampleCount = requested === "auto" ? 4 : requested;
    if (opts.format) this.format = opts.format;
  }

  /** Resolved MSAA sample count (after `init()` when "auto" was requested). */
  getSampleCount(): 1 | 4 {
    return this.sampleCount;
  }

  async init(): Promise<void> {
    if (typeof navigator === "undefined" || !navigator.gpu) {
      throw new Error("WebGPU unavailable in this environment");
    }
    this.destroyed = false; // destroy() documents re-init as supported
    // Match the backing buffer to the canvas's laid-out CSS size × DPR.
    // Without this the buffer stays at the HTML default (300×150) regardless
    // of how big the canvas looks on screen — the swapchain configures to
    // 300×150, the engine draws into 300×150, the browser then CSS-stretches
    // that tiny buffer up to clientWidth×clientHeight which is what produces
    // the "grey screen" with everything off-camera. (Caught 2026-05-15.)
    this.autosizeCanvas();
    // Wait for the canvas to have a non-zero CSS-laid-out size BEFORE we
    // configure the swapchain + allocate MSAA/depth targets. If init runs
    // synchronously from <script type="module"> before the first layout
    // pass, `clientWidth/clientHeight` can both be 0 → `autosizeCanvas`
    // clamps to 1×1, `createTargets` allocates 1×1 depth+MSAA, and the
    // first frame renders into a 1-pixel buffer that's then stretched.
    // This is the second-order "grey screen" suspect from the
    // 2026-05-18 investigation. One rAF tick + a non-zero-size check is
    // cheap and silent on the happy path. (Added 2026-05-19.)
    await this.waitForNonZeroCanvasSize();
    // Re-autosize whenever the page reflows. ResizeObserver only exists on
    // HTMLCanvasElement, not OffscreenCanvas — guard accordingly.
    if (typeof ResizeObserver !== "undefined" && "clientWidth" in this.canvas) {
      // Kept so destroy() can disconnect it: hosts reuse one canvas across
      // renderers (React remounts, play/edit toggles), and a stale observer
      // would resize a destroyed renderer's torn-down post chain.
      this.resizeObserver?.disconnect();
      this.resizeObserver = new ResizeObserver(() => this.autosizeCanvas());
      try { this.resizeObserver.observe(this.canvas as HTMLCanvasElement); } catch { /* OffscreenCanvas in worker */ }
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("WebGPU: no adapter");
    // Track 6 — opportunistically request `timestamp-query` so the
    // perf-tooling can read GPU pass duration with nanosecond precision.
    // If the adapter doesn't advertise it, we fall back to a CPU-side
    // wall-clock measurement around `queue.submit()` later in render().
    // Failure to request the feature must not block init — pre-2024 driver
    // builds drop the entire requestDevice() call when an unsupported
    // feature is required, so we probe + only ask if confirmed available.
    this.timestampSupport = probeTimestampQuerySupport(adapter);
    const requiredFeatures: GPUFeatureName[] = this.timestampSupport.supported
      ? (["timestamp-query"] as GPUFeatureName[])
      : [];
    this.device = await adapter.requestDevice(
      requiredFeatures.length > 0 ? { requiredFeatures } : undefined,
    );
    if (this.autoSampleCount) {
      const info = (adapter as unknown as { info?: GPUAdapterInfo }).info;
      this.sampleCount = autoSampleCountFor({
        vendor: info?.vendor ?? "",
        description: info?.description ?? "",
        architecture: info?.architecture ?? "",
        isFallbackAdapter: (adapter as unknown as { isFallbackAdapter?: boolean }).isFallbackAdapter === true,
      });
    }
    // Adapter provenance — proves a real hardware adapter is in use, not a
    // software fallback. Surfaced so the Track 1 WebGPU capture spec, perf
    // baselines, and any future user-facing diagnostics can assert against a
    // known-good baseline rather than guessing whether SwiftShader silently
    // backed the run. (Added 2026-05-28 per AUGENGINE-9-TRACK-PLAN T1#6.)
    try {
      const info = (adapter as unknown as { info?: GPUAdapterInfo }).info;
      const isFallback = (adapter as unknown as { isFallbackAdapter?: boolean }).isFallbackAdapter === true;
      const provenance = {
        vendor: info?.vendor ?? "unknown",
        architecture: info?.architecture ?? "unknown",
        device: info?.device ?? "unknown",
        description: info?.description ?? "unknown",
        isFallbackAdapter: isFallback,
        maxComputeWorkgroupSizeX: this.device.limits.maxComputeWorkgroupSizeX,
        maxBufferSize: this.device.limits.maxBufferSize,
        maxStorageBufferBindingSize: this.device.limits.maxStorageBufferBindingSize,
        ts: Date.now(),
      };
      // eslint-disable-next-line no-console
      console.info("[augengine/webgpu] adapter:", JSON.stringify(provenance));
      if (typeof window !== "undefined") {
        (window as unknown as { __augengineWebGpu?: typeof provenance }).__augengineWebGpu = provenance;
        window.dispatchEvent(new CustomEvent("augengine:webgpu-adapter", { detail: provenance }));
      }
    } catch {
      /* GPUAdapter.info is still gated behind feature in some browsers — skip silently. */
    }
    // Surface any silent WGPU validation failures. The prior version only
    // `console.error`'d these — which is invisible to anyone not actively
    // scrolling devtools, so a silent pipeline-validation failure paints
    // the clear color forever and looks like a "grey screen". We now also
    // (a) paint a banner overlay on a 2D canvas above the WebGPU canvas
    // so the failure is visible without devtools, and
    // (b) dispatch a `augengine:webgpu-error` CustomEvent on `window` so tests
    //     + downstream UI can react.
    // Updated 2026-05-19 per AUGENGINE-GREY-SCREEN-INVESTIGATION-2026-05-18.
    this.device.addEventListener("uncapturederror", (event) => {
      const e = (event as GPUUncapturedErrorEvent).error;
      const msg = e?.message ?? String(e);
      // eslint-disable-next-line no-console
      console.error("[augengine/webgpu] uncaptured:", msg);
      this.surfaceWebGpuError(msg);
    });
    this.format = navigator.gpu.getPreferredCanvasFormat();
    const ctx = (this.canvas as HTMLCanvasElement).getContext("webgpu") as GPUCanvasContext | null;
    if (!ctx) throw new Error("WebGPU: getContext('webgpu') returned null");
    this.context = ctx;
    this.context.configure({
      device: this.device,
      format: this.format,
      alphaMode: "premultiplied",
    });

    this.factory = new PipelineFactory(this.device, "rgba16float");
    // Fixed-size probe ring: bound in every frame bind group whether or not
    // probes are on, so enabling them never rebuilds a bind group.
    this.probeBuffer = this.device.createBuffer({
      label: "probes/ring",
      size: PROBE_RING_FLOATS * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.probeBuffer, 0, this.probeRing.data as BufferSource);
    const computeModule = this.device.createShaderModule({ code: SKINNING_COMPUTE_WGSL });
    this.skinComputePipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: { module: computeModule, entryPoint: "main" },
    });
    const computeUvModule = this.device.createShaderModule({ code: SKINNING_COMPUTE_WGSL_UV });
    this.skinComputeUvPipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: { module: computeUvModule, entryPoint: "main" },
    });
    this.postRunner = new PostProcessingRunner(this.device, this.format);
    if (this.pendingLook) this.postRunner.setLook(this.pendingLook);
    // Sky renders into the SAME target the main scene pass writes to (the HDR
    // rgba16float texture when post is on, MSAA-resolved). Match the sample
    // count so the pipeline is compatible with the live render pass.
    this.skyPass = new SkyPass(this.device, "rgba16float", this.sampleCount, this.skyGradient);
    this.skyPass.enabled = this.skyEnabled;
    if (this.pendingQuality) this.applyQuality(this.pendingQuality);
    this.createTargets();
    this.uploadDefaultMeshes();

    // Phase 6 — initialise the blendshape uniform pool so any entity
    // can receive per-frame weights from augengine-audio2face. The actual
    // bind-into-draw step ships with the morph-target importer; until
    // then the pool serves as the staging surface.
    //
    // Cast through `unknown` because GPUDevice.queue.writeBuffer's
    // signature is stricter than our interface (variant ArrayBufferView).
    // Runtime-compatible; tsc just won't infer the variance.
    this.blendshapePool = new BlendshapeUniformPool(
      this.device as unknown as ConstructorParameters<typeof BlendshapeUniformPool>[0],
    );
    this.morphDeltaStorage = new MorphDeltaStorage(
      this.device as unknown as ConstructorParameters<typeof MorphDeltaStorage>[0],
    );
    this.pbrSampler = this.device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      mipmapFilter: "linear",
      addressModeU: "repeat",
      addressModeV: "repeat",
    });
    this.initShadowResources();

    this.ready = true;
  }

  // ─── Shadows ───────────────────────────────────────────────────────
  private initShadowResources(): void {
    if (!this.device) return;
    // Linear filtering on a comparison sampler = hardware 2×2 PCF per tap.
    this.shadowSampler = this.device.createSampler({
      label: "shadow/compare",
      compare: "less",
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    });
    this.dummyShadowTexture = this.device.createTexture({
      label: "shadow/off",
      size: { width: 1, height: 1 },
      format: "depth32float",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.dummyShadowView = this.dummyShadowTexture.createView();
    this.shadowFrameLayout = this.device.createBindGroupLayout({
      label: "shadow/frame",
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } }],
    });
    // Must stay group-equivalent to PipelineFactory's draw layout so the
    // per-entity draw bind groups can be reused in the shadow pass.
    this.shadowDrawLayout = this.device.createBindGroupLayout({
      label: "shadow/draw",
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } }],
    });
    this.shadowFrameBuf = this.device.createBuffer({
      label: "shadow/frame-uniform",
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.shadowFrameBindGroup = this.device.createBindGroup({
      layout: this.shadowFrameLayout,
      entries: [{ binding: 0, resource: { buffer: this.shadowFrameBuf } }],
    });
    this.shadowFrameBuf2 = this.device.createBuffer({
      label: "shadow/frame-uniform-far",
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.shadowFrameBindGroup2 = this.device.createBindGroup({
      layout: this.shadowFrameLayout,
      entries: [{ binding: 0, resource: { buffer: this.shadowFrameBuf2 } }],
    });
    this.shadowModule = this.device.createShaderModule({ label: "shadow/depth", code: SHADOW_DEPTH_WGSL });
    // Group-equivalent to the factory's instanced draw layout (storage records).
    this.shadowInstancedDrawLayout = this.device.createBindGroupLayout({
      label: "shadow/draw-instanced",
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } }],
    });
    this.shadowInstancedModule = this.device.createShaderModule({ label: "shadow/depth-instanced", code: SHADOW_DEPTH_INSTANCED_WGSL });
    this.applyShadowQuality();
  }

  /** Turn per-mesh instancing on/off (on by default). Off = one draw per entity, as before. */
  setInstancing(enabled: boolean): void {
    this.instancingEnabled = enabled;
  }

  /** Draw counts from the last rendered frame. */
  getFrameStats(): Readonly<RendererFrameStats> {
    return this.frameStats;
  }

  /**
   * Key-light shadow quality. "off" (default) | "low" (1024², hardware 2×2
   * PCF) | "high" (2048², 3×3 PCF over hardware-filtered taps). Safe before
   * or after init().
   */
  /** "full" keeps bloom and FXAA. "tonemap" is one ACES pass to the canvas. */
  setPostChain(chain: "full" | "tonemap"): void {
    this.postRunner?.setChain(chain);
  }

  /**
   * Tonemapper (ACES / Khronos PBR Neutral / AgX), exposure, bloom and a
   * display grade (contrast, vibrance, lift). Applied to the post runner; a
   * call before init() is replayed once the runner exists.
   */
  setLook(look: LookOptions): void {
    this.pendingLook = { ...this.pendingLook, ...look };
    this.postRunner?.setLook(look);
  }

  /**
   * Temporal AA / upscaling, sharpen and SSAO. `renderScale` < 1 renders the
   * main pass smaller and lets TAA reconstruct full resolution; it only
   * applies while `taa` is on. TAA requires 1× MSAA (ignored at 4×). Safe
   * before or after init().
   */
  setImageQuality(q: ImageQualityOptions): void {
    this.pendingQuality = { ...this.pendingQuality, ...q };
    if (this.device) {
      this.applyQuality(q);
      this.createTargets();
    }
  }

  /** Effective settings after init() (TAA is reported off at 4× MSAA). */
  getImageQuality(): { taa: boolean; renderScale: number; ao: boolean; internalWidth: number; internalHeight: number } {
    const { width, height } = this.internalSize();
    const on = this.temporalActive();
    return { taa: on, renderScale: on ? this.renderScale : 1, ao: on && !!this.temporal?.aoEnabled, internalWidth: width, internalHeight: height };
  }

  private applyQuality(q: ImageQualityOptions): void {
    if (q.taa !== undefined) this.taaEnabled = q.taa;
    if (q.renderScale !== undefined && Number.isFinite(q.renderScale)) this.renderScale = Math.min(1, Math.max(0.25, q.renderScale));
    if (this.device && !this.temporal) this.temporal = new TemporalResolver(this.device, this.format);
    const t = this.temporal;
    if (t) {
      if (q.ao !== undefined) t.aoEnabled = q.ao;
      if (q.aoSamples !== undefined) t.aoSamples = Math.max(1, Math.min(16, Math.round(q.aoSamples)));
      if (q.aoRadius !== undefined) t.aoRadius = q.aoRadius;
      if (q.aoIntensity !== undefined) t.aoIntensity = q.aoIntensity;
      if (q.feedback !== undefined) t.feedback = Math.min(1, Math.max(0.02, q.feedback));
    }
    if (q.sharpen !== undefined && t) t.sharpen = Math.max(0, q.sharpen);
    if (q.motionVectors !== undefined) this.motionVectors = q.motionVectors;
    if (q.skyAmbient !== undefined) this.skyAmbient = q.skyAmbient;
    if (q.skySpecular !== undefined) this.skySpecular = q.skySpecular;
    if (q.groundBounce !== undefined && Number.isFinite(q.groundBounce)) this.groundBounce = Math.max(0, q.groundBounce);
    this.taaHistoryValid = false;
  }

  private temporalActive(): boolean {
    return this.taaEnabled && this.sampleCount === 1 && !!this.temporal;
  }

  /** Main-pass render size: the canvas, or canvas × renderScale under TAA. */
  private internalSize(): { width: number; height: number } {
    const w = this.canvas.width, h = this.canvas.height;
    return this.temporalActive() ? scaledSize(w, h, this.renderScale) : { width: w, height: h };
  }

  setShadowQuality(quality: ShadowQuality): void {
    if (quality === this.shadowQuality && (quality === "off" || this.shadowTexture)) return;
    this.shadowQuality = quality;
    this.applyShadowQuality();
  }

  getShadowQuality(): ShadowQuality {
    return this.shadowQuality;
  }

  /** Diameter in world units of the shadowed region in front of the camera. */
  setShadowRange(range: number): void {
    if (Number.isFinite(range) && range > 0.5) this.shadowRange = range;
  }

  /**
   * Key-light cascades. 1 (default) = one map over `shadowRange`. 2 = that
   * map plus a far cascade `farScale`× wider (default 4), side by side in one
   * 2:1 depth atlas, blended across the near cascade's outer 15%.
   */
  setShadowCascades(count: 1 | 2, farScale?: number): void {
    const next: 1 | 2 = count === 2 ? 2 : 1;
    if (farScale !== undefined && Number.isFinite(farScale) && farScale > 1) this.shadowFarScale = farScale;
    if (next === this.shadowCascades) return;
    this.shadowCascades = next;
    this.applyShadowQuality();
  }

  /** Lights the last frame resolved (key, extra directional, local, ambient). */
  getFrameLights(): FrameLights | null {
    return this.lastFrameLights;
  }

  private applyShadowQuality(): void {
    if (!this.device) return;
    this.shadowSettings = shadowSettingsFor(this.shadowQuality, this.shadowCascades);
    this.shadowTexture?.destroy();
    this.shadowTexture = null;
    this.shadowView = null;
    if (this.shadowSettings) {
      this.shadowTexture = this.device.createTexture({
        label: `shadow/map-${this.shadowSettings.mapSize}x${this.shadowSettings.cascades}`,
        size: { width: this.shadowSettings.mapSize * this.shadowSettings.cascades, height: this.shadowSettings.mapSize },
        format: "depth32float",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
      this.shadowView = this.shadowTexture.createView();
    }
    this.shadowGen++;
  }

  private shadowPipelineFor(vertexStride: number, instanced = false): GPURenderPipeline | null {
    if (!this.device || !this.shadowFrameLayout) return null;
    const module = instanced ? this.shadowInstancedModule : this.shadowModule;
    const drawLayout = instanced ? this.shadowInstancedDrawLayout : this.shadowDrawLayout;
    if (!module || !drawLayout) return null;
    const cache = instanced ? this.shadowInstancedPipelines : this.shadowPipelines;
    let p = cache.get(vertexStride);
    if (p) return p;
    p = this.device.createRenderPipeline({
      label: `shadow/depth-stride${vertexStride}${instanced ? "-instanced" : ""}`,
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.shadowFrameLayout, drawLayout] }),
      vertex: {
        module,
        entryPoint: "vs_main",
        buffers: [{ arrayStride: vertexStride * 4, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }] }],
      },
      // No cull: thin geometry (cloth planes, panels) still casts.
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: {
        format: "depth32float",
        depthWriteEnabled: true,
        depthCompare: "less",
        depthBias: 2,
        depthBiasSlopeScale: 2.5,
        depthBiasClamp: 0,
      },
    });
    cache.set(vertexStride, p);
    return p;
  }

  /** Upload this frame's frame uniform once per pipeline, skipping unchanged bytes. */
  private writeFrameUniform(entry: PipelineCacheEntry, cacheKey: string): void {
    if (this.frameWritten.has(cacheKey)) return;
    const frameScratch = this.frameScratch;
    let prev = this.frameUploaded.get(cacheKey);
    if (!prev || prev.length !== frameScratch.length) {
      prev = new Float32Array(frameScratch.length);
      this.frameUploaded.set(cacheKey, prev);
    }
    // A still camera writes the same view-projection every frame. The
    // GPU process was spending more time copying those bytes than the
    // proof room spends skinning.
    if (!sameFloats(prev, frameScratch, frameScratch.length)) {
      prev.set(frameScratch);
      this.device!.queue.writeBuffer(entry.frameBuf, 0, frameScratch);
    }
    this.frameWritten.add(cacheKey);
  }

  /** An instancing candidate whose group was too small: draw it on its own. */
  private pushSingleDraw(c: InstanceCandidate, shadowOn: boolean, draws: DrawRecord[]): void {
    this.writeFrameUniform(c.entry, c.cacheKey);
    let entityDraw = this.entityDrawCache.get(c.entityId);
    if (!entityDraw || entityDraw.pipelineKey !== c.cacheKey) {
      if (entityDraw) entityDraw.drawBuf.destroy();
      const drawBuf = this.device!.createBuffer({
        size: DRAW_UNIFORM_SIZE,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      const bindGroup = this.device!.createBindGroup({
        layout: c.entry.compiled.drawBindGroupLayout,
        entries: [{ binding: 0, resource: { buffer: drawBuf } }],
      });
      entityDraw = { drawBuf, bindGroup, pipelineKey: c.cacheKey, uploaded: new Float32Array(this.drawScratch.length) };
      this.entityDrawCache.set(c.entityId, entityDraw);
    }
    const drawScratch = this.drawScratch;
    packDrawRecord(drawScratch, 0, c.worldMat, c.color, c.params, c.emissive);
    if (!sameFloats(entityDraw.uploaded, drawScratch, drawScratch.length)) {
      entityDraw.uploaded.set(drawScratch);
      this.device!.queue.writeBuffer(entityDraw.drawBuf, 0, drawScratch);
    }
    draws.push({
      mesh: c.mesh,
      entry: c.entry,
      cacheKey: c.cacheKey,
      drawBindGroup: entityDraw.bindGroup,
      hasSkinning: false,
      morphBindGroup: null,
      textureBindGroup: null,
      castsShadow: shadowOn && !c.entityId.endsWith(".shadow") && c.matKind !== "unlit",
      instanceCount: 1,
      instanced: false,
      entityId: c.entityId,
      model: c.worldMat,
    });
  }

  /** One instanced draw for every member of a group (same mesh + pipeline variant). */
  private pushInstancedDraw(groupKey: string, items: InstanceCandidate[], shadowOn: boolean, draws: DrawRecord[]): void {
    const first = items[0]!;
    const entry = this.getPipelineFor(first.matKind, first.mesh.vertexStride, !!first.mesh.vcbuf, false, false, false, false, true);
    if (!entry) {
      for (const c of items) this.pushSingleDraw(c, shadowOn, draws);
      return;
    }
    const instKey = pipelineKeyString(entry.compiled.key);
    this.writeFrameUniform(entry, instKey);
    let g = this.instanceGroups.get(groupKey);
    if (!g || g.capacity < items.length) {
      g?.buf.destroy();
      const capacity = instanceCapacity(items.length);
      const buf = this.device!.createBuffer({
        label: `instances/${groupKey}`,
        size: capacity * DRAW_RECORD_FLOATS * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      const bindGroup = this.device!.createBindGroup({
        layout: entry.compiled.drawBindGroupLayout,
        entries: [{ binding: 0, resource: { buffer: buf } }],
      });
      g = {
        buf, bindGroup, capacity,
        data: new Float32Array(capacity * DRAW_RECORD_FLOATS),
        uploaded: new Float32Array(capacity * DRAW_RECORD_FLOATS),
        uploadedCount: -1,
      };
      this.instanceGroups.set(groupKey, g);
    }
    for (let i = 0; i < items.length; i++) {
      const c = items[i]!;
      packDrawRecord(g.data, i, c.worldMat, c.color, c.params, c.emissive);
    }
    const n = items.length * DRAW_RECORD_FLOATS;
    // A still camera over an unchanged set uploads nothing.
    if (g.uploadedCount !== items.length || !sameFloats(g.uploaded, g.data, n)) {
      g.uploaded.set(g.data.subarray(0, n));
      g.uploadedCount = items.length;
      this.device!.queue.writeBuffer(g.buf, 0, g.data as unknown as BufferSource, 0, n);
    }
    draws.push({
      mesh: first.mesh,
      entry,
      cacheKey: instKey,
      drawBindGroup: g.bindGroup,
      hasSkinning: false,
      morphBindGroup: null,
      textureBindGroup: null,
      castsShadow: shadowOn && first.matKind !== "unlit",
      instanceCount: items.length,
      instanced: true,
    });
  }

  private frameBindGroupFor(entry: PipelineCacheEntry): GPUBindGroup {
    if (entry.shadowGen === this.shadowGen) return entry.frameBindGroup;
    entry.frameBindGroup = this.device!.createBindGroup({
      layout: entry.compiled.frameBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: entry.frameBuf } },
        { binding: 1, resource: this.shadowView ?? this.dummyShadowView! },
        { binding: 2, resource: this.shadowSampler! },
        { binding: 3, resource: { buffer: this.probeBuffer! } },
      ],
    });
    entry.shadowGen = this.shadowGen;
    return entry.frameBindGroup;
  }

  // ─── Phase 6 — blendshape weights ─────────────────────────────────
  /**
   * Push a 52-entry ARKit weight buffer for the given entity. Lazy-
   * allocates the GPU uniform on first call per entity. Safe to call
   * before `init()` finishes — drops the write silently in that case.
   *
   * Wire this to `BlendshapeBuffer` / `BlendshapePlayer` from
   * `@mnemopay/augengine-audio2face` to drive a character's mouth shape
   * from inferred provider frames.
   */
  setBlendshapeWeights(entityId: string, weights: Float32Array): void {
    if (!this.blendshapePool) return;
    this.blendshapePool.setWeights(entityId, weights);
  }

  /** Read back the last weights pushed for an entity. Snapshot, safe
   *  to mutate. Returns null when nothing has been pushed. */
  peekBlendshapeWeights(entityId: string): Float32Array | null {
    return this.blendshapePool?.peekWeights(entityId) ?? null;
  }

  /** Get the GPU uniform buffer for an entity — for binding into a
   *  pipeline that consumes morph weights. Lazy-allocates when absent. */
  getBlendshapeBuffer(entityId: string): BlendshapeBufferLike | null {
    if (!this.blendshapePool) return null;
    return this.blendshapePool.ensureBuffer(entityId);
  }

  /** Free the blendshape uniform for an entity (e.g. on entity remove). */
  clearBlendshapes(entityId: string): void {
    this.blendshapePool?.clear(entityId);
    this.entityMorphCache.delete(entityId);
  }

  // ─── Track 6 — GPU timestamp queries ──────────────────────────────
  /**
   * Arm WebGPU timestamp-query capture around the main render pass.
   * If the adapter advertised `timestamp-query` we instrument the pass
   * with `timestampWrites` and resolve into a 16-byte readback buffer
   * after each submit. Otherwise we wrap `device.queue.submit()` in a
   * wall-clock measurement (worse precision, always works) and report
   * `gpu_ms_source: 'cpu-fallback'`.
   *
   * Callback fires once per frame with the latest sample. Pass `null`
   * to disable. Safe to call any number of times — only the latest
   * listener is invoked.
   *
   * The smoke-game wires this so the published number lands on
   * `window.__augengineFrameBudget.latest.gpu_ms` without coupling the
   * renderer to the smoke-game's globals.
   */
  enableGpuTimestamps(
    listener: ((sample: { gpu_ms: number; source: GpuMsSource }) => void) | null,
  ): void {
    this.gpuMsListener = listener;
    if (listener === null) {
      this.timestampSession?.destroy();
      this.timestampSession = null;
      return;
    }
    if (!this.device) return;
    if (this.timestampSupport.supported && !this.timestampSession) {
      try {
        this.timestampSession = new TimestampSession(this.device);
      } catch (e) {
        // Driver rejected the query set — fall back to CPU clock path.
        // eslint-disable-next-line no-console
        console.warn("[augengine/perf] timestamp-query session failed:", e);
        this.timestampSession = null;
      }
    }
  }

  /** Public accessor — useful for the HUD / diagnostics. */
  getTimestampSupport(): { supported: boolean; reason: string } {
    return this.timestampSupport;
  }

  /**
   * Arm per-pass GPU timing. Each frame's skinning compute, shadow, main
   * (opaque + sky + particles), water and post passes get their own
   * timestamp pair; the listener receives the most recently resolved frame.
   * While armed, the frame-spanning `enableGpuTimestamps` listener is fed the
   * pass timer's frame span instead (a pass can carry only one query set).
   * Returns false when the adapter has no `timestamp-query` — there is no
   * meaningful per-pass CPU fallback. Pass `null` to disarm.
   */
  enablePassTimings(listener: ((sample: PassTimingSample) => void) | null): boolean {
    this.passListener = listener;
    if (listener === null) {
      this.passTimer?.destroy();
      this.passTimer = null;
      return false;
    }
    if (!this.device || !this.timestampSupport.supported) return false;
    if (!this.passTimer) {
      try {
        this.passTimer = new PassTimestampSession(this.device);
      } catch (e) {
        // eslint-disable-next-line no-console
        console.warn("[augengine/perf] per-pass timestamp session failed:", e);
        this.passTimer = null;
        return false;
      }
    }
    return true;
  }

  // ─── Gradient sky background ──────────────────────────────────────
  /**
   * Enable/disable the gradient-sky background pass and optionally override
   * the gradient. The sky is drawn after opaque geometry, on the far plane,
   * so it fills only pixels the scene left empty. Disabled by default.
   * Safe to call before or after `init()`.
   */
  setSky(enabled: boolean, gradient?: Partial<SkyGradient>): void {
    this.skyEnabled = enabled;
    if (gradient) {
      this.skyGradient = { ...this.skyGradient, ...gradient };
      this.skyPass?.setGradient(gradient);
    }
    if (this.skyPass) this.skyPass.enabled = enabled;
  }

  /**
   * Hemisphere IBL proxy. When enabled and the caller does not pass
   * `opts.ambient`, the render pass uses `irradiance * intensity` as the
   * frame ambient. Cubemap convolution is the native Substrate path.
   */
  setIbl(enabled: boolean, opts?: { irradiance?: Vec3; intensity?: number }): void {
    this.iblEnabled = enabled;
    if (opts?.irradiance) this.iblIrradiance = opts.irradiance;
    if (opts?.intensity !== undefined) this.iblIntensity = opts.intensity;
  }

  // ─── Streamed irradiance probes ───────────────────────────────────
  /**
   * Per-tile irradiance probes (world/irradiance-probes.ts): when on, PBR /
   * cel / toon ambient comes from the probes around each pixel (relative to
   * the frame ambient), fading back to the sky/flat ambient where no probes
   * are resident. Safe before or after init().
   */
  setIrradianceProbes(enabled: boolean, grid?: Partial<ProbeGridParams>): void {
    this.probesEnabled = enabled;
    if (grid) this.probeGrid = { ...this.probeGrid, ...grid };
  }

  /** Store a tile's baked probes in the ring (replaces whatever aliased there). */
  setProbeTile(tile: ProbeTileLike): void {
    this.uploadProbeRanges(this.probeRing.setTile(tile));
  }

  /** Invalidate a tile's probes (its slots, if they still hold that tile). */
  clearProbeTile(gi0: number, gj0: number, perAxis: number, layers: number): void {
    this.uploadProbeRanges(this.probeRing.clearTile(gi0, gj0, perAxis, layers));
  }

  private uploadProbeRanges(ranges: RingRange[]): void {
    if (!this.device || !this.probeBuffer) return; // init() uploads the whole mirror
    for (const [start, count] of ranges) {
      this.device.queue.writeBuffer(this.probeBuffer, start * 4, this.probeRing.data as BufferSource, start, count);
    }
  }

  // ─── GPU particles (fire / smoke / embers) ────────────────────────
  //
  // Emitters are renderer-owned, not scene entities: a 4096-particle flame
  // as scene entities would blow the entity budget and the draw-call count.
  // Simulation runs in a compute pass and the draw is encoded into the main
  // render pass, after opaque geometry. See particle-pass.ts.
  //
  // Particles only advance when the caller passes `dt` to `render()`. Without
  // it they are drawn frozen — that is deliberate, so a host with its own
  // fixed-step clock stays in control of the simulation.

  /**
   * Create or replace a particle emitter. Returns the resolved (clamped) spec,
   * or null if the renderer has not finished `init()` yet.
   *
   * Defaults come from `FIRE_PRESET`; pass `SMOKE_PRESET` / `EMBER_PRESET` as
   * `base` for those looks.
   */
  addParticleEmitter(
    id: string,
    options: ParticleEmitterOptions = {},
    base: Readonly<ParticleEmitterSpec> = FIRE_PRESET,
  ): ParticleEmitterSpec | null {
    if (!this.device) return null;
    this.particlePass ??= new ParticlePass(this.device, {
      // Must match the main render pass: the scene draws into the HDR
      // rgba16float target so bloom can pick up an intensity > 1 flame.
      targetFormat: "rgba16float",
      sampleCount: this.sampleCount,
      depthFormat: "depth24plus",
    });
    return this.particlePass.attach(id, options, base);
  }

  /** Patch emitter knobs. A changed `capacity` restarts that emitter's pool. */
  updateParticleEmitter(id: string, patch: ParticleEmitterOptions): ParticleEmitterSpec | null {
    return this.particlePass?.update(id, patch) ?? null;
  }

  /** Release an emitter's GPU buffers. Safe on an unknown id. */
  removeParticleEmitter(id: string): void {
    this.particlePass?.detach(id);
  }

  /** Resolved spec for an emitter, or null if it does not exist. */
  getParticleEmitter(id: string): ParticleEmitterSpec | null {
    return this.particlePass?.getSpec(id) ?? null;
  }

  /**
   * Emitter count, pool bytes, and a DERIVED alive estimate (rate × lifetime).
   * `estimatedAlive` is not read back from the GPU — do not present it as a
   * measured particle count.
   */
  getParticleStats(): { emitters: number; gpuBytes: number; estimatedAlive: number } {
    return {
      emitters: this.particlePass?.emitterCount ?? 0,
      gpuBytes: this.particlePass?.gpuBytes() ?? 0,
      estimatedAlive: this.particlePass?.estimatedAliveCount() ?? 0,
    };
  }

  // ─── Water (shallow-water solver + surface draw) ──────────────────
  //
  // Water is renderer-owned, not a scene entity: the surface is a 128×128
  // simulation grid drawn as instanced quads with no vertex buffer, which is
  // neither a mesh the scene graph could hold nor an entity count it could
  // afford.
  //
  // Like particles, the solver only advances when the caller passes `dt` to
  // `render()`. Without it the field is drawn frozen — deliberate, so a host
  // with its own fixed-step clock stays in control of the simulation. The
  // solver substeps that `dt` internally to respect the Courant limit and
  // reports any seconds it had to drop (`getWaterFieldStats().droppedSeconds`).
  //
  // Water requires the post chain, because refraction samples a copy of the
  // HDR scene-colour target. The post chain is always on in this renderer, so
  // that is not a configuration the caller has to make.

  /**
   * Create or replace a water surface. Returns the resolved (clamped) solver
   * spec and look, or null if the renderer has not finished `init()` yet.
   *
   * Defaults come from `POOL_PRESET` + `OCEAN_WATER_LOOK`. Pass a `bed`
   * callback for a sloping floor — anywhere the bed rises above
   * `surfaceLevel` starts dry, which is how a shoreline exists.
   */
  addWaterSurface(
    id: string,
    options: WaterSurfaceOptions = {},
    base: Readonly<ShallowWaterSpec> = POOL_PRESET,
    lookBase: Readonly<WaterLook> = OCEAN_WATER_LOOK,
  ): { spec: ShallowWaterSpec; look: WaterLook } | null {
    if (!this.device) return null;
    const firstSurface = this.waterPass === null;
    this.waterPass ??= new WaterSurfacePass(this.device, {
      // Must match the main render pass: the scene draws into the HDR
      // rgba16float target so bloom can pick up the specular glint.
      targetFormat: "rgba16float",
      sampleCount: this.sampleCount,
      depthFormat: "depth24plus",
    });
    const result = this.waterPass.attach(id, options, base, lookBase);
    // The scene-colour copy is only allocated once water exists, so the first
    // surface has to trigger a target reallocation to get one.
    if (firstSurface) this.createTargets();
    return result;
  }

  /** Patch solver and/or look knobs. Grid geometry changes restart the field. */
  updateWaterSurface(
    id: string,
    patch: WaterSurfaceOptions,
  ): { spec: ShallowWaterSpec; look: WaterLook } | null {
    return this.waterPass?.update(id, patch) ?? null;
  }

  /** Release a water surface's GPU resources. Safe on an unknown id. */
  removeWaterSurface(id: string): void {
    this.waterPass?.detach(id);
  }

  /** Resolved solver spec for a surface, or null if it does not exist. */
  getWaterSurface(id: string): ShallowWaterSpec | null {
    return this.waterPass?.getSpec(id) ?? null;
  }

  /** Resolved look for a surface, or null if it does not exist. */
  getWaterLook(id: string): WaterLook | null {
    return this.waterPass?.getLook(id) ?? null;
  }

  /**
   * Drop a radial disturbance into a field at a world X/Z. It propagates,
   * reflects off the domain walls and interferes like any other wave, because
   * it is an initial condition for the solver rather than an animation.
   *
   * A splash adds water by design (a raindrop does). Returns false for an
   * unknown id or a non-finite argument.
   */
  splashWater(id: string, worldX: number, worldZ: number, radius: number, amplitude: number): boolean {
    return this.waterPass?.splash(id, worldX, worldZ, radius, amplitude) ?? false;
  }

  /**
   * Surface count, cell-buffer bytes, quads drawn per frame, cell count, and
   * whether a real scene-colour copy is bound. All measured, none estimated.
   */
  getWaterStats(): WaterSurfaceStats {
    return (
      this.waterPass?.getStats() ?? {
        surfaces: 0,
        gpuBytes: 0,
        quads: 0,
        cells: 0,
        sceneColorBound: false,
      }
    );
  }

  /**
   * Per-field solver stats: substeps taken last frame, the CFL substep size,
   * and the cumulative seconds the substep budget refused. A non-zero
   * `droppedSeconds` means the field is running slower than wall-clock and
   * should be reported, not hidden.
   */
  getWaterFieldStats(id: string): ShallowWaterFieldStats | null {
    return this.waterPass?.getFieldStats(id) ?? null;
  }

  /**
   * Read a water field's cells back to the CPU. Tests and tooling ONLY — it
   * awaits `mapAsync` and stalls until the GPU drains.
   */
  readWaterFieldForDebug(id: string): Promise<Float32Array | null> {
    return this.waterPass?.readFieldForDebug(id) ?? Promise.resolve(null);
  }

  private getPipelineFor(
    materialKind: MaterialKind["kind"],
    vertexStride: 6 | 8 | 16,
    hasVertexColors = false,
    hasMorphTargets = false,
    hasPbrBaseColorTexture = false,
    hasPbrMetallicRoughnessTexture = false,
    hasPbrNormalTexture = false,
    instanced = false,
  ): PipelineCacheEntry | null {
    if (!this.device || !this.factory) return null;
    // Splat materials use a separate point pipeline that doesn't run through
    // pipeline-factory; phase 9 falls back to unlit so the entity still draws.
    const safeKind = materialKind === "splat" ? "unlit" : materialKind;
    const key: PipelineKey = {
      materialKind: safeKind,
      vertexStride,
      sampleCount: this.sampleCount,
      hasVertexColors,
      hasMorphTargets,
      hasPbrBaseColorTexture,
      hasPbrMetallicRoughnessTexture,
      hasPbrNormalTexture,
      ...(instanced ? { instanced: true } : {}),
    };
    const cacheKey = pipelineKeyString(key);
    let entry = this.pipelineCache.get(cacheKey);
    if (entry) return entry;

    const compiled = this.factory.get(key);
    const frameBuf = this.device.createBuffer({
      size: FRAME_UNIFORM_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const frameBindGroup = this.device.createBindGroup({
      layout: compiled.frameBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: frameBuf } },
        { binding: 1, resource: this.shadowView ?? this.dummyShadowView! },
        { binding: 2, resource: this.shadowSampler! },
        { binding: 3, resource: { buffer: this.probeBuffer! } },
      ],
    });
    entry = { compiled, frameBuf, frameBindGroup, shadowGen: this.shadowGen };
    this.pipelineCache.set(cacheKey, entry);
    return entry;
  }

  private createTargets(): void {
    if (!this.device) return;
    const w = this.canvas.width, h = this.canvas.height;
    // The main pass (and everything it draws into) runs at the internal size.
    const { width: iw, height: ih } = this.internalSize();
    this.depthTexture?.destroy();
    this.depthTexture = this.device.createTexture({
      size: { width: iw, height: ih },
      format: "depth24plus",
      sampleCount: this.sampleCount,
      // Single-sample depth is sampled by TAA reprojection and SSAO.
      usage: this.sampleCount === 1
        ? GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
        : GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.depthViewCached = this.depthTexture.createView();
    if (this.sampleCount === 4) {
      this.msaaTexture?.destroy();
      this.msaaTexture = this.device.createTexture({
        size: { width: iw, height: ih },
        format: "rgba16float",
        sampleCount: 4,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
    }
    this.postRunner?.resize(w, h, iw, ih);
    if (this.temporalActive()) this.temporal!.resize(w, h, iw, ih);
    this.taaHistoryValid = false;
    // Scene-colour copy for water refraction. Only allocated while a water
    // pass exists, so a scene without water carries no extra full-resolution
    // texture. Must match the HDR target's format: it is a straight
    // copyTextureToTexture of it.
    if (this.waterPass) {
      this.sceneColorTexture?.destroy();
      this.sceneColorTexture = this.device.createTexture({
        label: "water/scene-color",
        size: { width: iw, height: ih },
        format: "rgba16float",
        usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
      });
      this.waterPass.setSceneColor(this.sceneColorTexture);
    }
  }

  private uploadDefaultMeshes(): void {
    if (!this.device) return;
    this.uploadMesh("cube", cubeMesh(), 8);
    this.uploadMesh("primitive:box", cubeMesh(), 8);
    this.uploadMesh("primitive:plane", primitiveMesh("plane"), 8);
    this.uploadMesh("primitive:sphere", primitiveMesh("sphere"), 8);
    this.uploadMesh("primitive:cylinder", primitiveMesh("cylinder"), 8);
  }

  /** Ensure a primitive shape is uploaded to the GPU — caller hits this on demand. */
  ensurePrimitive(shape: PrimitiveShape): void {
    const key = `primitive:${shape}`;
    if (this.meshes.has(key)) return;
    this.uploadMesh(key, primitiveMesh(shape), 8);
  }

  /**
   * Free a mesh's GPU buffers (streamed meshes that leave with their tile).
   * Only call once nothing visible draws it: draws look a mesh up per frame,
   * and a hidden entity is skipped before the lookup.
   */
  releaseMesh(name: string): void {
    const m = this.meshes.get(name);
    if (!m) return;
    this.meshes.delete(name);
    m.vbuf.destroy();
    m.ibuf.destroy();
    m.vbufSkinned?.destroy();
    m.vbufSkinnedPrev?.destroy();
    m.vAttrsBuf?.destroy();
    m.vertexCountBuf?.destroy();
    m.vcbuf?.destroy();
    m.baseColorTexture?.destroy();
    m.metallicRoughnessTexture?.destroy();
    m.normalTexture?.destroy();
  }

  /** How many meshes are resident on the GPU (diagnostics). */
  get meshCount(): number {
    return this.meshes.size;
  }

  uploadMesh(name: string, data: MeshData, vertexStride: 6 | 8 | 16 = 6): void {
    if (!this.device) return;
    if (this.meshes.has(name)) return;
    const vbuf = this.device.createBuffer({
      size: data.vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
      mappedAtCreation: true,
    });
    new Float32Array(vbuf.getMappedRange()).set(data.vertices);
    vbuf.unmap();
    const ibuf = this.device.createBuffer({
      size: data.indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint32Array(ibuf.getMappedRange()).set(data.indices);
    ibuf.unmap();

    let vbufSkinned: GPUBuffer | undefined;
    let vAttrsBuf: GPUBuffer | undefined;
    let vertexCountBuf: GPUBuffer | undefined;
    
    if (data.joints && data.weights) {
      vbufSkinned = this.device.createBuffer({
        size: data.vertices.byteLength,
        // COPY_SRC: the motion pass keeps last frame's skinned positions.
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      const attrs = new Float32Array(data.vertexCount * 8);
      const attrsU32 = new Uint32Array(attrs.buffer);
      for (let i = 0; i < data.vertexCount; i++) {
        attrsU32[i * 8 + 0] = data.joints[i * 4 + 0]!;
        attrsU32[i * 8 + 1] = data.joints[i * 4 + 1]!;
        attrsU32[i * 8 + 2] = data.joints[i * 4 + 2]!;
        attrsU32[i * 8 + 3] = data.joints[i * 4 + 3]!;
        attrs[i * 8 + 4] = data.weights[i * 4 + 0]!;
        attrs[i * 8 + 5] = data.weights[i * 4 + 1]!;
        attrs[i * 8 + 6] = data.weights[i * 4 + 2]!;
        attrs[i * 8 + 7] = data.weights[i * 4 + 3]!;
      }
      vAttrsBuf = this.device.createBuffer({
        size: attrs.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      });
      new Uint8Array(vAttrsBuf.getMappedRange()).set(new Uint8Array(attrs.buffer));
      vAttrsBuf.unmap();

      vertexCountBuf = this.device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      });
      new Uint32Array(vertexCountBuf.getMappedRange())[0] = data.vertexCount;
      vertexCountBuf.unmap();
    }
    
    let vcbuf: GPUBuffer | undefined;
    if (data.colors) {
      vcbuf = this.device.createBuffer({
        size: data.colors.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      });
      new Float32Array(vcbuf.getMappedRange()).set(data.colors);
      vcbuf.unmap();
    }
    
    const meshGpu: MeshGpu = { vbuf, ibuf, indexCount: data.indexCount, vertexCount: data.vertexCount, vertexStride };
    if (vbufSkinned) meshGpu.vbufSkinned = vbufSkinned;
    if (vAttrsBuf) meshGpu.vAttrsBuf = vAttrsBuf;
    if (vertexCountBuf) meshGpu.vertexCountBuf = vertexCountBuf;
    if (vcbuf) meshGpu.vcbuf = vcbuf;
    if (data.morphTargets && data.morphTargets.length > 0 && this.morphDeltaStorage) {
      uploadMorphTargets(this.morphDeltaStorage, name, data.morphTargets);
      const morph = this.morphDeltaStorage.get(name);
      if (morph) meshGpu.morph = morph;
    }
    if (data.pbrMaterial) {
      meshGpu.pbrFactors = {
        baseColor: data.pbrMaterial.baseColor,
        metallic: data.pbrMaterial.metallic,
        roughness: data.pbrMaterial.roughness,
      };
    }
    this.meshes.set(name, meshGpu);
  }

  /**
   * Decode and upload a glTF base-color texture for an already uploaded mesh.
   * Until decode completes, the mesh continues rendering with scalar PBR
   * factors; the next frame selects the sampled-texture pipeline variant.
   */
  async uploadPbrBaseColorTexture(
    meshName: string,
    source: { bytes?: Uint8Array; mimeType?: string; uri?: string },
  ): Promise<boolean> {
    return this.uploadPbrTextures(meshName, { baseColor: source });
  }

  /** Upload any native glTF PBR texture channels currently available. */
  async uploadPbrTextures(
    meshName: string,
    sources: {
      baseColor?: { bytes?: Uint8Array; mimeType?: string; uri?: string };
      metallicRoughness?: { bytes?: Uint8Array; mimeType?: string; uri?: string };
      normal?: { bytes?: Uint8Array; mimeType?: string; uri?: string };
    },
  ): Promise<boolean> {
    if (!this.device || !this.pbrSampler) return false;
    const device = this.device;
    const mesh = this.meshes.get(meshName);
    if (!mesh || mesh.vertexStride < 8) return false;
    const upload = async (
      source: { bytes?: Uint8Array; mimeType?: string; uri?: string },
      format: GPUTextureFormat,
    ): Promise<GPUTexture | null> => {
      let blob: Blob;
      if (source.bytes) {
        const bytes = source.bytes.slice();
        blob = new Blob([bytes.buffer], { type: source.mimeType ?? "image/png" });
      } else if (source.uri) {
        const response = await fetch(source.uri);
        if (!response.ok) throw new Error(`texture fetch failed: ${response.status} ${source.uri}`);
        blob = await response.blob();
      } else {
        return null;
      }
      const bitmap = await createImageBitmap(blob);
      const texture = device.createTexture({
        size: { width: bitmap.width, height: bitmap.height },
        format,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      });
      device.queue.copyExternalImageToTexture(
        { source: bitmap },
        { texture },
        { width: bitmap.width, height: bitmap.height },
      );
      bitmap.close();
      return texture;
    };
    let uploaded = false;
    if (sources.baseColor) {
      const texture = await upload(sources.baseColor, "rgba8unorm-srgb");
      if (texture) {
        mesh.baseColorTexture?.destroy();
        mesh.baseColorTexture = texture;
        uploaded = true;
      }
    }
    if (sources.metallicRoughness) {
      const texture = await upload(sources.metallicRoughness, "rgba8unorm");
      if (texture) {
        mesh.metallicRoughnessTexture?.destroy();
        mesh.metallicRoughnessTexture = texture;
        uploaded = true;
      }
    }
    if (sources.normal) {
      const texture = await upload(sources.normal, "rgba8unorm");
      if (texture) {
        mesh.normalTexture?.destroy();
        mesh.normalTexture = texture;
        uploaded = true;
      }
    }
    mesh.pbrTextureBindGroups?.clear();
    return uploaded;
  }

  /** Resize the canvas and recreate depth/MSAA targets. */
  resize(width: number, height: number): void {
    this.canvas.width = width;
    this.canvas.height = height;
    if (this.device) this.createTargets();
  }

  /**
   * Resolve once the canvas has a non-zero CSS-laid-out size, or after at
   * most ~10 rAF ticks (≈ 165 ms @ 60 Hz). Used by `init()` to dodge the
   * race where `init` runs synchronously from a `<script type="module">`
   * BEFORE the browser has done its first layout pass — in that window
   * `clientWidth/clientHeight` are both 0, and the renderer would size
   * its swapchain + depth + MSAA targets to 1×1.
   *
   * Resolves immediately for `OffscreenCanvas` (worker path — no DOM
   * layout to wait for) or in non-DOM environments. Caps the rAF budget
   * so a never-laid-out canvas (display:none, detached, etc.) doesn't
   * hang init forever — the renderer will still come up at the
   * `autosizeCanvas`-clamped 1×1, just like before this guard existed.
   *
   * Added 2026-05-19 per `AUGENGINE-GREY-SCREEN-INVESTIGATION-2026-05-18.md`
   * H3 (autosizeCanvas race).
   */
  private waitForNonZeroCanvasSize(): Promise<void> {
    const c = this.canvas as HTMLCanvasElement;
    // OffscreenCanvas / non-DOM env — nothing to wait for.
    if (!c || typeof c.clientWidth !== "number") return Promise.resolve();
    if (typeof requestAnimationFrame !== "function") return Promise.resolve();
    if (c.clientWidth > 0 && c.clientHeight > 0) return Promise.resolve();
    return new Promise<void>((resolve) => {
      let ticks = 0;
      const MAX_TICKS = 10;
      const check = () => {
        if (c.clientWidth > 0 && c.clientHeight > 0) {
          this.autosizeCanvas();
          resolve();
          return;
        }
        if (++ticks >= MAX_TICKS) {
          // Give up — fall through with whatever size we have.
          resolve();
          return;
        }
        requestAnimationFrame(check);
      };
      requestAnimationFrame(check);
    });
  }

  /**
   * Paint a banner overlay on a 2D canvas stacked above the WebGPU
   * canvas so an `uncapturederror` failure is visible without devtools.
   * Also dispatches `augengine:webgpu-error` on `window` so tests +
   * downstream UI can react. Idempotent — keeps the first error and
   * updates the message if more arrive (most validation cascades emit
   * a flood per frame; the first is the load-bearing one).
   *
   * No-op in non-DOM environments (worker / Node).
   *
   * Added 2026-05-19 per `AUGENGINE-GREY-SCREEN-INVESTIGATION-2026-05-18.md`
   * "Confirmed safe to fix RIGHT NOW" section.
   */
  private webgpuErrorOverlay: HTMLDivElement | null = null;
  private webgpuFirstErrorMessage: string | null = null;
  private surfaceWebGpuError(message: string): void {
    // Always emit the event, even in headless / non-DOM environments
    // where the overlay would no-op.
    if (typeof window !== "undefined" && typeof CustomEvent === "function") {
      try {
        window.dispatchEvent(
          new CustomEvent("augengine:webgpu-error", {
            detail: { message, first: this.webgpuFirstErrorMessage === null },
          }),
        );
      } catch {
        /* CustomEvent unsupported — non-fatal. */
      }
    }
    if (this.webgpuFirstErrorMessage === null) {
      this.webgpuFirstErrorMessage = message;
    }
    if (typeof document === "undefined") return;
    const c = this.canvas as HTMLCanvasElement;
    if (!c || typeof c.parentElement === "undefined") return;
    const parent = c.parentElement ?? document.body;
    if (!parent) return;
    if (!this.webgpuErrorOverlay) {
      const overlay = document.createElement("div");
      overlay.id = "augengine-webgpu-error-banner";
      overlay.setAttribute(
        "style",
        [
          "position:fixed",
          "top:0",
          "left:0",
          "right:0",
          "z-index:99999",
          "padding:12px 16px",
          "background:#3a0d12",
          "color:#ffd0d0",
          "font:600 13px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace",
          "border-bottom:2px solid #ff4d5e",
          "box-shadow:0 4px 16px rgba(0,0,0,0.45)",
          "max-height:40vh",
          "overflow:auto",
          "white-space:pre-wrap",
          "pointer-events:auto",
        ].join(";"),
      );
      overlay.innerHTML =
        '<div style="font-size:11px;letter-spacing:0.12em;text-transform:uppercase;color:#ff8a96;margin-bottom:4px">AUGENGINE · WEBGPU PIPELINE ERROR</div>' +
        '<div id="augengine-webgpu-error-msg"></div>';
      parent.appendChild(overlay);
      this.webgpuErrorOverlay = overlay;
    }
    const slot = this.webgpuErrorOverlay.querySelector("#augengine-webgpu-error-msg");
    if (slot) {
      // Show the first (root-cause) error verbatim; subsequent errors get
      // a small counter so the operator can tell the cascade is ongoing.
      const first = this.webgpuFirstErrorMessage ?? message;
      slot.textContent = first === message ? first : `${first}\n\n(plus subsequent: ${message})`;
    }
  }

  /** Public accessor — first uncaptured WebGPU error message, if any.
   *  Useful for tests + diagnostics overlays that want to read state
   *  without listening for the `augengine:webgpu-error` event. */
  getFirstWebGpuError(): string | null {
    return this.webgpuFirstErrorMessage;
  }

  /**
   * Resize the backing buffer to match the CSS-laid-out client size × DPR.
   * Safe to call on every frame — only invokes `createTargets()` (cheap
   * texture re-allocation) when the size actually changed.
   */
  private autosizeCanvas(): void {
    if (this.destroyed) return;
    const c = this.canvas as HTMLCanvasElement;
    if (!c || typeof c.clientWidth !== "number") return;
    const dpr = (typeof window !== "undefined" && window.devicePixelRatio) || 1;
    // Clamp DPR for very-high-DPI displays — past ~3 you hit pipeline-cost
    // cliffs without proportional visual gain. Two-decimal cap is plenty.
    const dprC = Math.min(dpr, 3);
    const w = Math.max(1, Math.round(c.clientWidth * dprC));
    const h = Math.max(1, Math.round(c.clientHeight * dprC));
    if (c.width === w && c.height === h) return;
    c.width = w;
    c.height = h;
    // Don't create targets until the device exists — init() will call
    // createTargets() once it's done. After that this branch keeps targets
    // in sync with the canvas through window resizes / DPR changes.
    if (this.device && this.format) this.createTargets();
  }

  /** Render one frame from the given Scene. */
  render(scene: Scene, opts: RenderFrameOptions = {}): void {
    if (!this.ready || !this.device || !this.context || !this.factory) return;
    // Auto-resize the offscreen targets if the canvas backing store has been
    // resized externally (CSS reflow, devicePixelRatio change, window resize
    // handlers that forgot to call renderer.resize()). The swapchain texture
    // tracks canvas.width × canvas.height automatically; MSAA + depth do not.
    // Without this guard the resolve-target size diverges from the swapchain
    // size on every resize and WebGPU rejects every frame as invalid.
    let internal = this.internalSize();
    if (
      this.depthTexture &&
      (this.depthTexture.width !== internal.width ||
        this.depthTexture.height !== internal.height ||
        (this.postRunner !== null && this.postRunner.width !== this.canvas.width))
    ) {
      this.createTargets();
      internal = this.internalSize();
    }
    const temporalOn = this.temporalActive() && !!this.postRunner?.hdrView;

    const camView = this.camView;
    const camProj = this.camProj;
    const camVP = this.camVP;
    const camPos = this.camPosScratch;
    this.computeCamera(scene, camView, camProj, camPos, opts);
    let jitterUv: [number, number] = [0, 0];
    if (temporalOn) {
      // Keep the unjittered matrices for reprojection, then jitter the
      // projection the frame actually renders with by a sub-pixel offset.
      this.projUnjittered.set(camProj);
      m4Mul(this.vpUnjittered, camProj, camView);
      const [jx, jy] = taaJitter(this.taaFrame);
      applyProjectionJitter(camProj as unknown as Float32Array, jx, jy, internal.width, internal.height);
      jitterUv = jitterToUv(jx, jy, internal.width, internal.height);
    }
    m4Mul(camVP, camProj, camView);

    const lightDir = opts.lightDir ?? [-0.4, -1.0, -0.2];
    const ambient = opts.ambient ?? (this.iblEnabled
      ? [
          this.iblIrradiance[0] * this.iblIntensity,
          this.iblIrradiance[1] * this.iblIntensity,
          this.iblIrradiance[2] * this.iblIntensity,
        ]
      : [0.18, 0.18, 0.22]);
    const fogColor = opts.fogColor ?? opts.clearColor ?? [0, 0, 0];
    const fogDensity = opts.fogDensity ?? 0;

    const encoder = this.device.createCommandEncoder();
    let timestampBegun = false;
    // Per-pass timer wins over the frame-spanning session when both are armed.
    const passTimer = this.passTimer && this.passListener ? this.passTimer : null;
    passTimer?.beginFrame();

    if (this.skinComputePipeline) {
      const computePass = encoder.beginComputePass(
        passTimer ? { timestampWrites: passTimer.writes("skin") } : undefined,
      );
      let activePipeline: GPUComputePipeline | null = null;

      for (const e of scene.entities.values()) {
        const skinComp = e.components[Tag.Skin];
        if (!skinComp) continue;
        const meshComp = e.components[Tag.Mesh];
        if (!meshComp || !meshComp.visible) continue;
        const meshName = this.meshKeyForEntity(e.id, scene);
        const mesh = this.meshes.get(meshName);
        if (!mesh || !mesh.vbufSkinned || !mesh.vAttrsBuf || !mesh.vertexCountBuf) continue;
        if (mesh.vertexStride !== 6 && mesh.vertexStride !== 8) continue;
        const computePipeline = mesh.vertexStride === 8
          ? this.skinComputeUvPipeline
          : this.skinComputePipeline;
        if (!computePipeline) continue;
        if (computePipeline !== activePipeline) {
          computePass.setPipeline(computePipeline);
          activePipeline = computePipeline;
        }

        let entry = this.skinCache.get(e.id);
        if (!entry || entry.meshName !== meshName) {
          if (entry) entry.boneBuf.destroy();
          const boneBuf = this.device.createBuffer({
            size: skinComp.jointEntities.length * 64,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          });
          const bindGroup = this.device.createBindGroup({
            layout: computePipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: mesh.vbuf } },
              { binding: 1, resource: { buffer: mesh.vAttrsBuf } },
              { binding: 2, resource: { buffer: boneBuf } },
              { binding: 3, resource: { buffer: mesh.vbufSkinned } },
              { binding: 4, resource: { buffer: mesh.vertexCountBuf } },
            ]
          });
          const packed = new Float32Array(skinComp.jointEntities.length * 16);
          entry = { meshName, boneBuf, bindGroup, packed };
          this.skinCache.set(e.id, entry);
        }

        // Exact-length view. writeBuffer's optional byte count is rejected
        // when it is larger than the typed-array length, so the packed
        // array itself is the upload.
        buildBoneMatricesGltf({
          scene,
          jointEntityIds: skinComp.jointEntities,
          inverseBindMatrices: skinComp.inverseBindMatrices,
          out: entry.packed,
        });
        this.device.queue.writeBuffer(entry.boneBuf, 0, entry.packed as BufferSource);

        computePass.setBindGroup(0, entry.bindGroup);
        computePass.dispatchWorkgroups(Math.ceil(mesh.vertexCount / 64));
      }
      computePass.end();
    }

    // Particle simulation — a separate compute pass on the SAME encoder, so
    // WebGPU orders it before the render pass below reads the pool as a
    // read-only storage buffer in the vertex stage.
    if (this.particlePass && opts.dt !== undefined) {
      this.particlePass.simulate(encoder, opts.dt);
    }

    // Shallow-water solver — its own compute pass on the SAME encoder, so the
    // CFL substeps all land before the render pass below reads the cell buffer
    // as read-only storage in the vertex stage.
    if (this.waterPass && opts.dt !== undefined) {
      this.waterPass.simulate(encoder, opts.dt);
    }

    // ─── Lights + per-entity draw prep (shared by shadow + main pass) ───
    const lights = collectSceneLights(scene, lightDir, camPos as Vec3);
    this.lastFrameLights = lights;
    const frameAmbient = lights.ambient ?? ambient;
    const frameScratch = this.frameScratch;
    frameScratch.set(camVP, 0);
    // ambient.a carries the fog height falloff (0 = uniform fog); see augengine_fog_factor.
    frameScratch[20] = frameAmbient[0]; frameScratch[21] = frameAmbient[1]; frameScratch[22] = frameAmbient[2]; frameScratch[23] = Math.max(0, opts.fogHeightFalloff ?? 0);
    frameScratch[24] = camPos[0]; frameScratch[25] = camPos[1]; frameScratch[26] = camPos[2]; frameScratch[27] = 1;
    frameScratch[28] = fogColor[0]; frameScratch[29] = fogColor[1]; frameScratch[30] = fogColor[2]; frameScratch[31] = fogDensity;
    if (this.skyAmbient) {
      // Shape from the visible sky gradient (a neutral dome when the sky is
      // off), level from the frame ambient, so the scene's exposure holds.
      packSkyAmbient(frameScratch, FRAME_LAYOUT.ambientSH, this.skyEnabled ? this.skyGradient : null, frameAmbient as Vec3, this.groundBounce);
    } else {
      frameScratch.fill(0, FRAME_LAYOUT.ambientSH, FRAME_LAYOUT.ambientSH + 12);
    }
    if (this.skySpecular) {
      packSkySpecular(frameScratch, FRAME_LAYOUT.skySpecSH, this.skyEnabled ? this.skyGradient : null, frameAmbient as Vec3, this.groundBounce);
    } else {
      frameScratch.fill(0, FRAME_LAYOUT.skySpecSH, FRAME_LAYOUT.skySpecSH + 12);
    }
    frameScratch[FRAME_LAYOUT.probeParams] = this.probesEnabled ? 1 : 0;
    frameScratch[FRAME_LAYOUT.probeParams + 1] = this.probeGrid.spacing;
    frameScratch[FRAME_LAYOUT.probeParams + 2] = this.probeGrid.heightAboveGround;
    frameScratch[FRAME_LAYOUT.probeParams + 3] = this.probeGrid.layerSpacing;
    const shadowOn = !!(this.shadowSettings && this.shadowView);
    let shadowVP: Float32Array | null = null;
    let shadowVP2: Float32Array | null = null;
    if (shadowOn) {
      // Fit each cascade to a sphere in front of the camera. camView row 2 is
      // the camera's +Z (backward), so forward = −(m[2], m[6], m[10]).
      const mapSize = this.shadowSettings!.mapSize;
      const spheres = cascadeSpheres(this.shadowRange, this.shadowFarScale);
      const fit = (i: number, out: Float32Array): Float32Array => {
        const s = spheres[i]!;
        const center: Vec3 = [
          camPos[0] - camView[2]! * s.centerDist,
          camPos[1] - camView[6]! * s.centerDist,
          camPos[2] - camView[10]! * s.centerDist,
        ];
        return fitDirectionalShadow(lights.key.dir, center, s.radius, mapSize, out);
      };
      shadowVP = fit(0, this.shadowFitScratch);
      const texelWorld = (2 * spheres[0]!.radius) / mapSize;
      let far: ShadowUniform["far"];
      if (this.shadowSettings!.cascades === 2) {
        shadowVP2 = fit(1, this.shadowFitScratch2);
        const texelWorld2 = (2 * spheres[1]!.radius) / mapSize;
        far = { viewProj: shadowVP2, texelSize: 1 / mapSize, depthBias: 0.0006, normalOffset: texelWorld2 * 1.5, blend: 0.15 };
      }
      packFrameLights(frameScratch, lights, {
        enabled: true,
        viewProj: shadowVP,
        texelSize: 1 / mapSize,
        depthBias: 0.0006,
        normalOffset: texelWorld * 1.5,
        pcfTaps: this.shadowSettings!.pcfTaps,
        ...(far ? { far } : {}),
      });
    } else {
      packFrameLights(frameScratch, lights, null);
    }

    const wt = scene.worldTransforms();
    const drawScratch = this.drawScratch;
    // Track which pipelines we've already populated this frame so we only
    // write the frame uniform once per active pipeline.
    const frameWritten = this.frameWritten;
    frameWritten.clear();
    const draws = this.drawList;
    draws.length = 0;

    for (const e of scene.entities.values()) {
      const meshComp = e.components[Tag.Mesh];
      if (!meshComp || !meshComp.visible) continue;
      // Contact discs (ids ending `.shadow`) stand in for a shadow map.
      // Once the key light writes a real map they would draw a second dark
      // blob under the character and cast a shadow of that blob.
      if (shadowOn && e.id.endsWith(".shadow")) continue;
      const meshKey = this.meshKeyForEntity(e.id, scene);
      let mesh = this.meshes.get(meshKey);
      if (!mesh) {
        if (meshKey.startsWith("primitive:") && meshKey.includes(":")) {
          const parts = meshKey.split(":");
          const shape = parts[1] as PrimitiveShape;
          this.uploadMesh(meshKey, primitiveMesh(shape), 8);
          mesh = this.meshes.get(meshKey);
          if (mesh) {
            const mat = meshComp.material;
            if (mat && mat.kind === "pbr") {
              const textures: {
                baseColor?: { uri: string };
                metallicRoughness?: { uri: string };
                normal?: { uri: string };
              } = {};
              if (mat.baseColorTexture) textures.baseColor = { uri: mat.baseColorTexture };
              if (mat.metallicRoughnessTexture) textures.metallicRoughness = { uri: mat.metallicRoughnessTexture };
              if (mat.normalTexture) textures.normal = { uri: mat.normalTexture };
              if (Object.keys(textures).length > 0) {
                this.uploadPbrTextures(meshKey, textures).catch((err) => {
                  console.error(`Failed to upload PBR textures for ${meshKey}:`, err);
                });
              }
            }
          }
        }
      }
      if (!mesh) continue;

      const worldIdx = wt.indexById.get(e.id);
      if (worldIdx === undefined) continue;
      const worldMat = wt.worldMatrices.subarray(worldIdx * 16, worldIdx * 16 + 16);

      // Skinned meshes already bake the instance's world placement into their
      // vertices: buildBoneMatricesGltf multiplies each joint's *world* matrix
      // (which includes the gltf root transform) by the inverse-bind matrix, so
      // the compute pass emits vertices in absolute world space. The model
      // matrix for the draw must therefore be identity — applying the entity's
      // world matrix on top would transform the mesh a second time, pushing it
      // off-screen. (Unskinned meshes still use their world matrix below.)
      const hasSkinning = !!e.components[Tag.Skin] && !!mesh.vbufSkinned;

      // Resolve pipeline for this entity's material.
      const matKind = meshComp.material.kind as MaterialKind["kind"];
      const hasColors = !!mesh.vcbuf;
      const hasMorphTargets = !!mesh.morph && !!this.blendshapePool;
      const hasPbrBaseColorTexture = matKind === "pbr" && !!mesh.baseColorTexture && mesh.vertexStride >= 8;
      const hasPbrMetallicRoughnessTexture = matKind === "pbr" && !!mesh.metallicRoughnessTexture && mesh.vertexStride >= 8;
      const hasPbrNormalTexture = matKind === "pbr" && !!mesh.normalTexture && mesh.vertexStride >= 8;
      const entry = this.getPipelineFor(
        matKind,
        mesh.vertexStride,
        hasColors,
        hasMorphTargets,
        hasPbrBaseColorTexture,
        hasPbrMetallicRoughnessTexture,
        hasPbrNormalTexture,
      );
      if (!entry) continue;

      const pipelineKind = entry.compiled.key.materialKind;
      const cacheKey = `${pipelineKind}|${mesh.vertexStride}|${this.sampleCount}|static|${hasMorphTargets ? "morph" : "rigid"}|${hasColors ? "color" : "nocolor"}|${hasPbrBaseColorTexture ? "basecolor" : "flat"}|${hasPbrMetallicRoughnessTexture ? "mr" : "nomr"}|${hasPbrNormalTexture ? "normal" : "nonormal"}`;

      // Rigid, untextured draws can share an instanced draw with every other
      // visible entity on the same mesh + pipeline (vertex colours ride along:
      // they belong to the mesh, not the entity). Decide after the loop, once
      // the group sizes are known.
      if (
        this.instancingEnabled &&
        !hasSkinning && !hasMorphTargets &&
        !hasPbrBaseColorTexture && !hasPbrMetallicRoughnessTexture && !hasPbrNormalTexture &&
        (pipelineKind === "pbr" || pipelineKind === "unlit" || pipelineKind === "cel") &&
        mesh.vertexStride !== 16
      ) {
        const materialForDraw = mesh.pbrFactors && matKind === "pbr"
          ? { kind: "pbr", ...mesh.pbrFactors }
          : meshComp.material;
        this.instanceCandidates.push({
          entityId: e.id,
          meshKey,
          mesh,
          entry,
          cacheKey,
          matKind,
          worldMat,
          color: colorOfMaterial(materialForDraw),
          params: paramsOfMaterial(materialForDraw),
          emissive: emissiveOfMaterial(materialForDraw),
        });
        continue;
      }

      this.writeFrameUniform(entry, cacheKey);

      let entityDraw = this.entityDrawCache.get(e.id);
      if (!entityDraw || entityDraw.pipelineKey !== cacheKey) {
        if (entityDraw) entityDraw.drawBuf.destroy();
        const drawBuf = this.device.createBuffer({
          size: DRAW_UNIFORM_SIZE,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        const bindGroup = this.device.createBindGroup({
          layout: entry.compiled.drawBindGroupLayout,
          entries: [{ binding: 0, resource: { buffer: drawBuf } }],
        });
        entityDraw = { drawBuf, bindGroup, pipelineKey: cacheKey, uploaded: new Float32Array(drawScratch.length) };
        this.entityDrawCache.set(e.id, entityDraw);
      }

      if (hasSkinning) m4Identity(drawScratch.subarray(0, 16) as unknown as Mat4);
      else drawScratch.set(worldMat, 0);
      const materialForDraw = mesh.pbrFactors && matKind === "pbr"
        ? { kind: "pbr", ...mesh.pbrFactors }
        : meshComp.material;
      const color = colorOfMaterial(materialForDraw);
      drawScratch[16] = color[0]; drawScratch[17] = color[1]; drawScratch[18] = color[2]; drawScratch[19] = 1;
      const params = paramsOfMaterial(materialForDraw);
      drawScratch[20] = params[0]; drawScratch[21] = params[1]; drawScratch[22] = params[2]; drawScratch[23] = params[3];
      const emissive = emissiveOfMaterial(materialForDraw);
      drawScratch[24] = emissive[0]; drawScratch[25] = emissive[1]; drawScratch[26] = emissive[2]; drawScratch[27] = 0;
      if (!sameFloats(entityDraw.uploaded, drawScratch, drawScratch.length)) {
        entityDraw.uploaded.set(drawScratch);
        this.device.queue.writeBuffer(entityDraw.drawBuf, 0, drawScratch);
      }

      let morphBindGroup: GPUBindGroup | null = null;
      if (hasMorphTargets && mesh.morph && entry.compiled.morphBindGroupLayout) {
        let entityMorph = this.entityMorphCache.get(e.id);
        if (
          !entityMorph ||
          entityMorph.pipelineKey !== cacheKey ||
          entityMorph.meshKey !== meshKey
        ) {
          const weights = this.getBlendshapeBuffer(e.id);
          if (!weights) continue;
          const bindGroup = this.device.createBindGroup({
            layout: entry.compiled.morphBindGroupLayout,
            entries: [
              { binding: 0, resource: { buffer: weights as unknown as GPUBuffer } },
              { binding: 1, resource: { buffer: mesh.morph.positionBuffer as unknown as GPUBuffer } },
              { binding: 2, resource: { buffer: mesh.morph.normalBuffer as unknown as GPUBuffer } },
            ],
          });
          entityMorph = { bindGroup, pipelineKey: cacheKey, meshKey };
          this.entityMorphCache.set(e.id, entityMorph);
        }
        morphBindGroup = entityMorph.bindGroup;
      }
      let textureBindGroup: GPUBindGroup | null = null;
      if (
        (hasPbrBaseColorTexture || hasPbrMetallicRoughnessTexture || hasPbrNormalTexture) &&
        this.pbrSampler &&
        entry.compiled.pbrTextureBindGroupLayout
      ) {
        mesh.pbrTextureBindGroups ??= new Map<string, GPUBindGroup>();
        let tbg = mesh.pbrTextureBindGroups.get(cacheKey);
        if (!tbg) {
          tbg = this.device.createBindGroup({
            layout: entry.compiled.pbrTextureBindGroupLayout,
            entries: [
              ...(hasPbrBaseColorTexture && mesh.baseColorTexture
                ? [{ binding: 0, resource: mesh.baseColorTexture.createView() }]
                : []),
              { binding: 1, resource: this.pbrSampler },
              ...(hasPbrMetallicRoughnessTexture && mesh.metallicRoughnessTexture
                ? [{ binding: 2, resource: mesh.metallicRoughnessTexture.createView() }]
                : []),
              ...(hasPbrNormalTexture && mesh.normalTexture
                ? [{ binding: 3, resource: mesh.normalTexture.createView() }]
                : []),
            ],
          });
          mesh.pbrTextureBindGroups.set(cacheKey, tbg);
        }
        textureBindGroup = tbg;
      }
      draws.push({
        mesh,
        entry,
        cacheKey,
        drawBindGroup: entityDraw.bindGroup,
        hasSkinning,
        morphBindGroup,
        textureBindGroup,
        // Outlines, unlit cards, and splats do not occlude the key light.
        // Contact discs are already skipped above while shadows are on.
        castsShadow:
          shadowOn &&
          !e.id.endsWith(".shadow") &&
          matKind !== "unlit" &&
          matKind !== "outline" &&
          matKind !== "splat",
        instanceCount: 1,
        instanced: false,
        entityId: e.id,
        // Skinned vertices are already in world space (identity model).
        model: hasSkinning ? IDENTITY_M4 : worldMat,
        hasMorph: !!morphBindGroup,
      });
    }

    // ─── Instancing: big groups → one draw each, small ones per entity ──
    const candidates = this.instanceCandidates;
    if (candidates.length > 0) {
      const plan = planInstanceGroups(
        candidates.map((c) => ({ key: `${c.cacheKey}|${c.meshKey}`, item: c })),
        INSTANCE_MIN_GROUP,
      );
      for (const c of plan.singles) this.pushSingleDraw(c, shadowOn, draws);
      for (const g of plan.groups) this.pushInstancedDraw(g.key, g.items, shadowOn, draws);
      candidates.length = 0;
    }

    // ─── Shadow pass: key light depth into the shadow map ──────────────
    const timing = !passTimer && this.timestampSession && this.gpuMsListener ? this.timestampSession : null;
    if (shadowOn && shadowVP && this.shadowFrameBuf && this.shadowFrameBindGroup) {
      if (!sameFloats(this.shadowScratch, shadowVP, 16)) {
        this.shadowScratch.set(shadowVP);
        this.device.queue.writeBuffer(this.shadowFrameBuf, 0, this.shadowScratch);
      }
      const shadowDesc: GPURenderPassDescriptor = {
        label: "shadow-pass",
        colorAttachments: [],
        depthStencilAttachment: {
          view: this.shadowView!,
          depthClearValue: 1,
          depthLoadOp: "clear",
          depthStoreOp: "store",
        },
      };
      if (passTimer) {
        shadowDesc.timestampWrites = passTimer.writes("shadow");
      } else if (timing) {
        shadowDesc.timestampWrites = timing.frameBeginWrites();
        timestampBegun = true;
      }
      if (shadowVP2 && this.shadowFrameBuf2 && !sameFloats(this.shadowScratch2, shadowVP2, 16)) {
        this.shadowScratch2.set(shadowVP2);
        this.device.queue.writeBuffer(this.shadowFrameBuf2, 0, this.shadowScratch2);
      }
      const sp = encoder.beginRenderPass(shadowDesc);
      const mapSize = this.shadowSettings!.mapSize;
      // Near cascade in the left half (or the whole map), far in the right.
      const cascadesToDraw: Array<[GPUBindGroup, number]> = [[this.shadowFrameBindGroup, 0]];
      if (shadowVP2 && this.shadowFrameBindGroup2) cascadesToDraw.push([this.shadowFrameBindGroup2, mapSize]);
      for (const [frameBg, x] of cascadesToDraw) {
        sp.setViewport(x, 0, mapSize, mapSize, 0, 1);
        sp.setBindGroup(0, frameBg);
        let boundShadow = "";
        for (const d of draws) {
          if (!d.castsShadow) continue;
          const shadowKey = `${d.mesh.vertexStride}|${d.instanced ? "i" : "s"}`;
          if (shadowKey !== boundShadow) {
            const pipeline = this.shadowPipelineFor(d.mesh.vertexStride, d.instanced);
            if (!pipeline) continue;
            sp.setPipeline(pipeline);
            boundShadow = shadowKey;
          }
          sp.setBindGroup(1, d.drawBindGroup);
          sp.setVertexBuffer(0, d.hasSkinning ? d.mesh.vbufSkinned! : d.mesh.vbuf);
          sp.setIndexBuffer(d.mesh.ibuf, "uint32");
          sp.drawIndexed(d.mesh.indexCount, d.instanceCount);
        }
      }
      sp.end();
    }

    const view = (this.context.getCurrentTexture()).createView();
    const hasPost = this.postRunner && this.postRunner.hdrTexture;
    const hdrView = hasPost ? (this.postRunner!.hdrView ?? this.postRunner!.hdrTexture!.createView()) : view;
    const colorView =
      this.sampleCount === 4 ? this.msaaTexture!.createView() : hdrView;
    const clear: GPUColor = opts.clearColor
      ? { r: opts.clearColor[0], g: opts.clearColor[1], b: opts.clearColor[2], a: 1 }
      : { r: 0.05, g: 0.06, b: 0.09, a: 1 };

    const colorAttachment: GPURenderPassColorAttachment = {
      view: colorView,
      clearValue: clear,
      loadOp: "clear",
      storeOp: "store",
    };
    if (this.sampleCount === 4) colorAttachment.resolveTarget = hdrView;

    // Track 6 — attach `timestampWrites` to the main render pass only
    // when a TimestampSession is live. The 2-slot query set captures
    // begin/end of pass; resolveAfterPass + readFrameMs handle readback
    // post-submit.
    // One depth view shared by the main pass and the water pass, so the water
    // draw depth-tests against exactly the depth the opaque geometry wrote.
    const depthView = this.depthViewCached ?? this.depthTexture!.createView();
    // Water needs the opaque scene colour as a SAMPLED texture, which is
    // impossible while that texture is the open pass's colour attachment. So
    // when water is on screen the main pass is split in two with a copy in
    // between. `hasPost` is the gate because the copy source is the HDR
    // target; the post chain is always on in this renderer.
    const waterActive =
      this.waterPass !== null &&
      this.waterPass.surfaceCount > 0 &&
      !!hasPost &&
      this.sceneColorTexture !== null;
    const passDescriptor: GPURenderPassDescriptor = {
      colorAttachments: [colorAttachment],
      depthStencilAttachment: {
        view: depthView,
        depthClearValue: 1,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    };
    // GPU timer spans the whole frame when the post chain runs: begin on the
    // first pass (shadow pass if one ran, else this one), end on the last post
    // pass. Without post there is only this pass, so it carries both writes.
    if (passTimer) {
      passDescriptor.timestampWrites = passTimer.writes("main");
    } else if (timing) {
      if (!timestampBegun) {
        passDescriptor.timestampWrites = hasPost ? timing.frameBeginWrites() : timing.passTimestampWrites();
        timestampBegun = true;
      } else if (!hasPost) {
        // Shadow pass opened the timer; without post this is the last pass.
        passDescriptor.timestampWrites = timing.frameEndWrites();
      }
    }
    const pass = encoder.beginRenderPass(passDescriptor);

    let lastBoundPipelineKey = "";
    const stats: RendererFrameStats = { drawCalls: 0, instancedDraws: 0, instancedEntities: 0, entitiesDrawn: 0, triangles: 0 };
    for (const d of draws) {
      stats.drawCalls += 1;
      stats.entitiesDrawn += d.instanceCount;
      stats.triangles += (d.mesh.indexCount / 3) * d.instanceCount;
      if (d.instanced) { stats.instancedDraws += 1; stats.instancedEntities += d.instanceCount; }
      if (d.cacheKey !== lastBoundPipelineKey) {
        pass.setPipeline(d.entry.compiled.pipeline);
        pass.setBindGroup(0, this.frameBindGroupFor(d.entry));
        lastBoundPipelineKey = d.cacheKey;
      }
      pass.setBindGroup(1, d.drawBindGroup);
      if (d.morphBindGroup) pass.setBindGroup(2, d.morphBindGroup);
      if (d.textureBindGroup) pass.setBindGroup(3, d.textureBindGroup);
      pass.setVertexBuffer(0, d.hasSkinning ? d.mesh.vbufSkinned! : d.mesh.vbuf);
      if (d.entry.compiled.key.hasVertexColors && d.mesh.vcbuf) {
        pass.setVertexBuffer(1, d.mesh.vcbuf);
      }
      pass.setIndexBuffer(d.mesh.ibuf, "uint32");
      pass.drawIndexed(d.mesh.indexCount, d.instanceCount);
    }
    this.frameStats = stats;

    // Sky after opaques. The triangle is on the far plane and tests equal to
    // the clear depth, so it fills only pixels the room did not cover.
    if (this.skyPass && this.skyEnabled) {
      this.skyPass.encode(pass, camVP, camPos as Vec3);
    }

    if (!waterActive) {
      // No water: the pass structure is exactly what it was before water
      // existed. Particles draw LAST in the main pass — depth-tested against
      // the opaque geometry above (so the world occludes them) but writing no
      // depth, and blended into the HDR target so bloom catches the hot core.
      if (this.particlePass) {
        this.particlePass.encode(pass, camView, camVP);
      }
      pass.end();
    } else {
      // Water on screen: end the opaque pass, snapshot the resolved colour,
      // then reopen a pass that loads both colour and depth. Water draws into
      // it (depth-tested against the opaque geometry, and writing depth
      // itself), and particles draw after water so spray above the surface is
      // not hidden by it.
      pass.end();
      encoder.copyTextureToTexture(
        { texture: this.postRunner!.hdrTexture! },
        { texture: this.sceneColorTexture! },
        { width: internal.width, height: internal.height },
      );
      const waterColorAttachment: GPURenderPassColorAttachment = {
        view: colorView,
        loadOp: "load",
        storeOp: "store",
      };
      if (this.sampleCount === 4) waterColorAttachment.resolveTarget = hdrView;
      const waterRenderPass = encoder.beginRenderPass({
        label: "water-pass",
        colorAttachments: [waterColorAttachment],
        depthStencilAttachment: {
          view: depthView,
          depthLoadOp: "load",
          depthStoreOp: "store",
        },
        ...(passTimer ? { timestampWrites: passTimer.writes("water") } : {}),
      });
      // Travel direction, the same convention as the frame uniform. The water
      // shader turns it into a toward-light vector with a negation.
      this.waterPass!.encode(waterRenderPass, camVP, camPos as Vec3, lights.key.dir);
      if (this.particlePass) {
        this.particlePass.encode(waterRenderPass, camView, camVP);
      }
      waterRenderPass.end();
      // NOTE on instrumentation: `timestampWrites` is attached to the opaque
      // pass only (WebGPU scopes it per pass), so while water is active the
      // reported `gpu_ms` EXCLUDES the water pass. Do not publish that number
      // as a whole-frame cost in a scene with water.
    }
    let resolved = false;
    if (hasPost && temporalOn) {
      m4Invert(this.invVP, this.vpUnjittered);
      m4Invert(this.invProj, this.projUnjittered);
      const velocityView = this.encodeMotion(encoder, draws, depthView, camVP, internal, passTimer);
      // Bloom from the internal-resolution frame (blurry by design), then the
      // fused TAA + composite + tonemap pass writes the canvas directly.
      const timeBloom = !!passTimer && this.postRunner!.chain === "full";
      const bloomRan = this.postRunner!.runBloom(
        encoder,
        timeBloom ? passTimer!.spanBegin("post") : undefined,
        timeBloom ? passTimer!.spanEnd("post") : undefined,
      );
      resolved = this.temporal!.resolve(
        encoder,
        hdrView,
        depthView,
        {
          invViewProj: this.invVP as unknown as Float32Array,
          prevViewProj: this.prevVP as unknown as Float32Array,
          proj: this.projUnjittered as unknown as Float32Array,
          invProj: this.invProj as unknown as Float32Array,
          jitterUv,
          historyValid: this.taaHistoryValid,
          frameIndex: this.taaFrame,
        },
        view,
        bloomRan ? this.postRunner!.bloomView : null,
        this.postRunner!.lookUniforms(),
        passTimer ? { ao: passTimer.writes("ao"), taa: passTimer.writes("taa") } : undefined,
        velocityView,
      );
      this.prevVP.set(this.vpUnjittered);
      this.taaHistoryValid = true;
      this.taaFrame++;
    } else {
      this.taaHistoryValid = false;
    }
    if (hasPost && !resolved) {
      if (passTimer) {
        this.postRunner!.run(encoder, view, passTimer.spanEnd("post"), passTimer.spanBegin("post"));
      } else {
        this.postRunner!.run(encoder, view, timing ? timing.frameEndWrites() : undefined);
      }
    }
    if (passTimer) passTimer.resolve(encoder);
    // Track 6 — resolve timestamp queries into the readback buffer
    // before we hand the encoder to the queue. Safe to call even when
    // post-processing ran first; it's a separate region of the encoder.
    if (timing && this.timestampSession && this.gpuMsListener) {
      try {
        this.timestampSession.resolveAfterPass(encoder);
      } catch (e) {
        // Driver disagreed on timestampWrites — disable this frame's
        // capture and continue. A subsequent enableGpuTimestamps(null)
        // will fully unwire if the failure persists.
        // eslint-disable-next-line no-console
        console.warn("[augengine/perf] resolveAfterPass failed:", e);
      }
    }
    const cpuSubmitStart = this.gpuMsListener && !this.timestampSession && !passTimer
      ? performance.now()
      : 0;
    this.device.queue.submit([encoder.finish()]);
    if (passTimer) {
      const passListener = this.passListener!;
      const frameListener = this.gpuMsListener;
      void passTimer.read().then((sample) => {
        if (!sample) return;
        passListener(sample);
        frameListener?.({ gpu_ms: sample.frame_ms, source: "timestamp-query" });
      });
      return;
    }
    // Track 6 — feed the listener. Two paths:
    //  (a) timestamp-query: kick off async readback; previous frame's
    //      sample is what the listener actually gets so we never await
    //      on the render hot path.
    //  (b) cpu-fallback: queue.submit returns as soon as the command
    //      buffer is queued, so this is coarse (driver buffers behind),
    //      but it's a stable baseline that's always available.
    if (this.gpuMsListener) {
      if (this.timestampSession) {
        const session = this.timestampSession;
        const listener = this.gpuMsListener;
        // Fire-and-forget — the resolved sample is from a prior frame
        // because mapAsync drains the readback queue. Treat as a
        // rolling 1-frame-lag value.
        void session.readFrameMs().then((ms) => {
          if (ms === null) return;
          listener({ gpu_ms: ms, source: "timestamp-query" });
        });
      } else {
        const ms = performance.now() - cpuSubmitStart;
        this.gpuMsListener({ gpu_ms: ms, source: "cpu-fallback" });
      }
    }
  }

  /**
   * Velocity for moving per-entity draws (skinned, morph targets, or a model
   * matrix that changed since last frame), then snapshot this frame's skinned
   * positions, weights and models for the next one. Returns the velocity view
   * for the TAA resolve, or null when nothing moved (the pass is skipped).
   */
  private encodeMotion(
    encoder: GPUCommandEncoder,
    draws: readonly DrawRecord[],
    depthView: GPUTextureView,
    jitteredVP: Mat4,
    internal: { width: number; height: number },
    passTimer: PassTimestampSession | null,
  ): GPUTextureView | null {
    if (!this.motionVectors || !this.device) return null;
    const motion = this.motionDraws;
    motion.length = 0;
    const skinned: MeshGpu[] = [];
    for (const d of draws) {
      if (d.instanced || !d.entityId || !d.model) continue;
      const last = this.lastModels.get(d.entityId);
      const moved = !last || !sameFloats(last, d.model, 16);
      if (d.hasSkinning && d.mesh.vbufSkinned) {
        skinned.push(d.mesh);
        motion.push({
          entityId: d.entityId,
          vbuf: d.mesh.vbufSkinned!,
          prevVbuf: d.mesh.skinnedPrevValid ? d.mesh.vbufSkinnedPrev! : null,
          ibuf: d.mesh.ibuf,
          indexCount: d.mesh.indexCount,
          vertexStride: d.mesh.vertexStride,
          model: d.model,
          prevModel: d.model,
          morph: null,
        });
      } else if (d.hasMorph && d.mesh.morph && this.blendshapePool) {
        const cur = this.blendshapePool.peekWeights(d.entityId) ?? new Float32Array(52);
        motion.push({
          entityId: d.entityId,
          vbuf: d.mesh.vbuf,
          prevVbuf: null,
          ibuf: d.mesh.ibuf,
          indexCount: d.mesh.indexCount,
          vertexStride: d.mesh.vertexStride,
          model: d.model,
          prevModel: last ?? d.model,
          morph: {
            weights: this.blendshapePool.ensureBuffer(d.entityId) as unknown as GPUBuffer,
            prevWeights: this.prevMorphWeights.get(d.entityId) ?? cur,
            deltas: d.mesh.morph.positionBuffer as unknown as GPUBuffer,
          },
        });
        this.prevMorphWeights.set(d.entityId, cur);
      } else if (moved && last) {
        motion.push({
          entityId: d.entityId,
          vbuf: d.mesh.vbuf,
          prevVbuf: null,
          ibuf: d.mesh.ibuf,
          indexCount: d.mesh.indexCount,
          vertexStride: d.mesh.vertexStride,
          model: d.model,
          prevModel: last,
          morph: null,
        });
      }
      if (moved) {
        if (last) last.set(d.model);
        else this.lastModels.set(d.entityId, Float32Array.from(d.model));
      }
    }
    if (motion.length === 0) return null;
    this.motionPass ??= new MotionPass(this.device);
    this.motionPass.resize(internal.width, internal.height);
    const drew = this.motionPass.encode(
      encoder,
      depthView,
      motion,
      jitteredVP,
      this.vpUnjittered,
      this.prevVP,
      passTimer ? passTimer.writes("motion") : undefined,
    );
    // Snapshot skinned positions AFTER the motion pass read them as "now".
    for (const mesh of skinned) {
      const src = mesh.vbufSkinned!;
      if (!mesh.vbufSkinnedPrev) {
        mesh.vbufSkinnedPrev = this.device.createBuffer({
          label: "skin/prev",
          size: src.size,
          usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
      }
      encoder.copyBufferToBuffer(src, 0, mesh.vbufSkinnedPrev, 0, src.size);
      mesh.skinnedPrevValid = true;
    }
    return drew ? this.motionPass.view : null;
  }

  private computeCamera(
    scene: Scene,
    outView: Mat4,
    outProj: Mat4,
    outEye: MutVec3,
    opts: RenderFrameOptions,
  ): void {
    const w = this.canvas.width;
    const h = Math.max(1, this.canvas.height);
    const aspect = w / h;
    const camId = scene.activeCamera ?? null;
    if (!camId) {
      m4Identity(outView);
      m4Perspective(outProj, Math.PI / 3, aspect, 0.1, 100);
      outEye[0] = 0; outEye[1] = 0; outEye[2] = 0;
      return;
    }
    const wt = scene.worldTransforms();
    const camIdx = wt.indexById.get(camId);
    if (camIdx === undefined) {
      m4Identity(outView);
      m4Perspective(outProj, Math.PI / 3, aspect, 0.1, 100);
      outEye[0] = 0; outEye[1] = 0; outEye[2] = 0;
      return;
    }
    const camWorld = wt.worldMatrices.subarray(camIdx * 16, camIdx * 16 + 16);
    const eye = this.camEye;
    eye[0] = camWorld[12]!; eye[1] = camWorld[13]!; eye[2] = camWorld[14]!;
    const fwd = this.camFwd;
    fwd[0] = -camWorld[8]!; fwd[1] = -camWorld[9]!; fwd[2] = -camWorld[10]!;
    const target = this.camTarget;
    target[0] = eye[0] + fwd[0]; target[1] = eye[1] + fwd[1]; target[2] = eye[2] + fwd[2];
    const up = this.camUp;
    up[0] = camWorld[4]!; up[1] = camWorld[5]!; up[2] = camWorld[6]!;
    m4LookAt(outView, eye, target, up);
    outEye[0] = eye[0]; outEye[1] = eye[1]; outEye[2] = eye[2];
    const camComp = scene.entities.get(camId)?.components[Tag.Camera];
    const fov = camComp?.fovY ?? Math.PI / 3;
    const near = camComp?.near ?? 0.1;
    const far = camComp?.far ?? 100;
    m4Perspective(outProj, fov, aspect, near, far);
    void opts;
  }

  destroy(): void {
    this.destroyed = true;
    this.resizeObserver?.disconnect();
    this.resizeObserver = null;
    this.depthTexture?.destroy();
    this.msaaTexture?.destroy();
    this.postRunner?.destroy();
    this.temporal?.destroy();
    this.temporal = null;
    this.depthViewCached = null;
    this.skyPass?.destroy();
    this.skyPass = null;
    this.particlePass?.destroy();
    this.particlePass = null;
    this.waterPass?.destroy();
    this.waterPass = null;
    this.sceneColorTexture?.destroy();
    this.sceneColorTexture = null;
    this.shadowTexture?.destroy();
    this.shadowTexture = null;
    this.shadowView = null;
    this.dummyShadowTexture?.destroy();
    this.dummyShadowTexture = null;
    this.dummyShadowView = null;
    this.probeBuffer?.destroy();
    this.probeBuffer = null;
    this.shadowFrameBuf?.destroy();
    this.shadowFrameBuf = null;
    this.shadowFrameBindGroup = null;
    this.shadowFrameBuf2?.destroy();
    this.shadowFrameBuf2 = null;
    this.shadowFrameBindGroup2 = null;
    this.shadowScratch2.fill(Number.NaN);
    // The shadow VP upload is skipped when it matches the last bytes sent.
    // A re-init creates a fresh (zeroed) buffer, so forget those bytes.
    this.shadowScratch.fill(Number.NaN);
    this.shadowPipelines.clear();
    this.shadowInstancedPipelines.clear();
    for (const g of this.instanceGroups.values()) g.buf.destroy();
    this.instanceGroups.clear();
    this.instanceCandidates.length = 0;
    this.drawList.length = 0;
    for (const entry of this.pipelineCache.values()) {
      entry.frameBuf.destroy();
    }
    this.pipelineCache.clear();
    this.frameUploaded.clear();
    this.frameWritten.clear();
    for (const entry of this.entityDrawCache.values()) {
      entry.drawBuf.destroy();
    }
    this.entityDrawCache.clear();
    this.entityMorphCache.clear();
    for (const m of this.meshes.values()) {
      m.vbuf.destroy();
      m.ibuf.destroy();
      m.vbufSkinned?.destroy();
      m.vbufSkinnedPrev?.destroy();
      m.vAttrsBuf?.destroy();
      m.vertexCountBuf?.destroy();
      m.vcbuf?.destroy();
      m.baseColorTexture?.destroy();
      m.metallicRoughnessTexture?.destroy();
      m.normalTexture?.destroy();
    }
    this.meshes.clear();
    this.motionPass?.destroy();
    this.motionPass = null;
    this.lastModels.clear();
    this.prevMorphWeights.clear();
    this.morphDeltaStorage?.clearAll();
    this.morphDeltaStorage = null;
    this.blendshapePool?.clearAll();
    this.blendshapePool = null;
    this.pbrSampler = null;
    for (const entry of this.skinCache.values()) {
      entry.boneBuf.destroy();
    }
    this.skinCache.clear();
    this.factory?.invalidate();
    this.timestampSession?.destroy();
    this.timestampSession = null;
    this.gpuMsListener = null;
    this.passTimer?.destroy();
    this.passTimer = null;
    this.passListener = null;
    this.ready = false;
  }
}

/** Temporal AA / upscaling, sharpen and SSAO (backend A). */
export interface ImageQualityOptions {
  /** Temporal anti-aliasing (replaces FXAA). Needs 1× MSAA. */
  taa?: boolean;
  /** Main-pass resolution scale (0.25–1) while TAA is on; TAA upsamples. */
  renderScale?: number;
  /** CAS sharpen strength after TAA (0 = off, default 0.35). */
  sharpen?: number;
  /** Current-frame weight in the TAA blend (default 0.1). */
  feedback?: number;
  /**
   * Per-object motion vectors for moving, skinned and morphing meshes, so TAA
   * reprojects them along their own motion instead of the camera's (default
   * true). Static geometry always uses depth reprojection.
   */
  motionVectors?: boolean;
  /** Half-resolution depth-only SSAO, denoised by the TAA history. */
  ao?: boolean;
  aoSamples?: number;
  /** World-space sample radius in metres (default 0.6). */
  aoRadius?: number;
  aoIntensity?: number;
  /**
   * Sky-shaped ambient (diffuse IBL from the sky gradient, zonal SH L0-L2),
   * normalised to the frame's flat ambient level. Off = flat ambient.
   */
  skyAmbient?: boolean;
  /** Fraction of the horizon colour the ground reflects upward (default 0.35). */
  groundBounce?: number;
  /**
   * Split-sum sky reflections on PBR materials (ibl.ts): the sky's zonal SH
   * prefiltered per roughness + an analytic environment BRDF. Off by default.
   */
  skySpecular?: boolean;
}

/** Post look knobs shared by both backends (B maps what three supports). */
export interface LookOptions {
  tonemapper?: Tonemapper;
  exposure?: number;
  bloomIntensity?: number;
  bloomThreshold?: number;
  contrast?: number;
  vibrance?: number;
  lift?: number;
}

export interface RenderFrameOptions {
  clearColor?: Vec3;
  lightDir?: Vec3;
  ambient?: Vec3;
  /**
   * Linear RGB color geometry fades toward at distance — typically the
   * horizon color. Defaults to `clearColor` if omitted so callers that
   * already match clear-to-horizon get a sensible blend.
   */
  fogColor?: Vec3;
  /**
   * Exponential fog density. 0 (default) = fog disabled. Typical outdoor
   * values 0.005–0.04. The smoke-game ships with ~0.022 for a gentle haze.
   */
  fogDensity?: number;
  /**
   * Height falloff k (1/m) for exponential height fog: density scales by
   * e^(-k*y), so fog pools near y = 0 and thins with altitude. 0 (default)
   * keeps uniform distance fog.
   */
  fogHeightFalloff?: number;
  /**
   * Seconds elapsed since the previous frame. Required for GPU particle
   * emitters to advance; omit it and particles are drawn frozen. The host
   * owns this value so a fixed-step loop keeps control of the simulation.
   * Internally clamped to 0.1 s so a stall cannot teleport particles.
   */
  dt?: number;
}

/**
 * MSAA choice for `sampleCount: "auto"`. Integrated and software adapters are
 * bandwidth-bound, and 4× rgba16float colour + depth is the largest cost in the
 * opaque pass, so they get 1× (FXAA still runs). Apple GPUs are integrated but
 * tile-based — MSAA resolves on-chip there, so they keep 4×.
 */
export function autoSampleCountFor(info: {
  vendor: string;
  description: string;
  architecture: string;
  isFallbackAdapter: boolean;
}): 1 | 4 {
  if (info.isFallbackAdapter) return 1;
  const s = `${info.vendor} ${info.description} ${info.architecture}`.toLowerCase();
  if (/apple/.test(s)) return 4;
  if (/intel|uhd|iris|hd graphics|swiftshader|llvmpipe|lavapipe|basic render|mali|adreno|powervr/.test(s)) return 1;
  return 4;
}

const IDENTITY_M4 = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

function sameFloats(a: ArrayLike<number>, b: ArrayLike<number>, n: number): boolean {
  for (let i = 0; i < n; i++) if (a[i] !== b[i]) return false;
  return true;
}

function colorOfMaterial(m: { kind: string; baseColor?: Vec3; color?: Vec3 } | unknown): Vec3 {
  const mat = m as { kind: string; baseColor?: Vec3; color?: Vec3 };
  if (mat.baseColor) return mat.baseColor;
  if (mat.color) return mat.color;
  return [0.7, 0.7, 0.75];
}

/**
 * Self-illumination color packed into the draw uniform's emissive slot.
 * HDR linear RGB — values >1 survive into the bloom-extract pass, so this is
 * the intentional "glow" channel that's independent of scene lighting. Only
 * the lit material kinds carry emissive; everything else emits nothing.
 */
function emissiveOfMaterial(m: unknown): Vec3 {
  const mat = m as { emissive?: Vec3 };
  return mat.emissive ?? [0, 0, 0];
}

/**
 * Per-material params packed into vec4 for the WGSL shader (see pipeline-factory):
 *   - pbr:       (metallic, roughness, 0, 0)
 *   - cel:       (steps, 0, 0, 0)
 *   - outline:   (width, 0, 0, 0)
 *   - vrm-mtoon: (shadingShift, shadingToony, 0, 0)
 *   - unlit:     (0, 0, 0, 0)
 */
function paramsOfMaterial(m: unknown): [number, number, number, number] {
  const mat = m as Record<string, unknown>;
  switch (mat.kind) {
    case "pbr": return [Number(mat.metallic ?? 0), Number(mat.roughness ?? 0.5), 0, 0];
    case "cel": return [Number(mat.steps ?? 3), 0, 0, 0];
    case "outline": return [Number(mat.width ?? 0.02), 0, 0, 0];
    case "vrm-mtoon": return [Number(mat.shadingShift ?? 0), Number(mat.shadingToony ?? 0.5), 0, 0];
    default: return [0, 0, 0, 0];
  }
}

/** Helper for placing a camera entity from eye+target+up at scene-build time. */
export function lookAtTransform(eye: Vec3, target: Vec3, up: Vec3): {
  position: Vec3;
  rotation: Quat;
  scale: Vec3;
} {
  const forward = v3(0, 0, 0);
  forward[0] = target[0] - eye[0];
  forward[1] = target[1] - eye[1];
  forward[2] = target[2] - eye[2];
  const len = Math.hypot(forward[0], forward[1], forward[2]);
  if (len > 0) {
    forward[0] /= len;
    forward[1] /= len;
    forward[2] /= len;
  }
  const z: Vec3 = [-forward[0], -forward[1], -forward[2]];
  const xRaw = v3(0, 0, 0);
  xRaw[0] = up[1] * z[2] - up[2] * z[1];
  xRaw[1] = up[2] * z[0] - up[0] * z[2];
  xRaw[2] = up[0] * z[1] - up[1] * z[0];
  const xLen = Math.hypot(xRaw[0], xRaw[1], xRaw[2]);
  const x: Vec3 =
    xLen > 0 ? [xRaw[0] / xLen, xRaw[1] / xLen, xRaw[2] / xLen] : [1, 0, 0];
  const y: Vec3 = [
    z[1] * x[2] - z[2] * x[1],
    z[2] * x[0] - z[0] * x[2],
    z[0] * x[1] - z[1] * x[0],
  ];
  const r00 = x[0], r10 = x[1], r20 = x[2];
  const r01 = y[0], r11 = y[1], r21 = y[2];
  const r02 = z[0], r12 = z[1], r22 = z[2];
  const trace = r00 + r11 + r22;
  let qx = 0, qy = 0, qz = 0, qw = 1;
  if (trace > 0) {
    const s = Math.sqrt(trace + 1) * 2;
    qw = 0.25 * s;
    qx = (r21 - r12) / s;
    qy = (r02 - r20) / s;
    qz = (r10 - r01) / s;
  } else if (r00 > r11 && r00 > r22) {
    const s = Math.sqrt(1 + r00 - r11 - r22) * 2;
    qw = (r21 - r12) / s;
    qx = 0.25 * s;
    qy = (r01 + r10) / s;
    qz = (r02 + r20) / s;
  } else if (r11 > r22) {
    const s = Math.sqrt(1 + r11 - r00 - r22) * 2;
    qw = (r02 - r20) / s;
    qx = (r01 + r10) / s;
    qy = 0.25 * s;
    qz = (r12 + r21) / s;
  } else {
    const s = Math.sqrt(1 + r22 - r00 - r11) * 2;
    qw = (r10 - r01) / s;
    qx = (r02 + r20) / s;
    qy = (r12 + r21) / s;
    qz = 0.25 * s;
  }
  return {
    position: eye,
    rotation: [qx, qy, qz, qw] as Quat,
    scale: [1, 1, 1],
  };
}
