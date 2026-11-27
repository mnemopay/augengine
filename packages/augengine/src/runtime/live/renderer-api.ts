/**
 * Renderer adapter boundary.
 *
 * Everything a playable shell (smoke-game, EngineRunner, benchmarks) asks of a
 * renderer goes through this interface, so the scene graph, ECS, physics and
 * persona layers do not care which backend draws the pixels:
 *
 *   - backend "webgpu" — WebGpuLiveRenderer, AugEngine's own WebGPU pipeline.
 *   - backend "three"  — ThreeLiveRenderer (`@mnemopay/augengine/runtime/live/three`),
 *     three.js WebGPURenderer with its automatic WebGL2 fallback.
 *
 * The scene format stays the single source of truth: backends read
 * `Scene.entities` + `Scene.worldTransforms()` every frame and own nothing
 * the game state depends on. Renderer-only features that a backend does not
 * implement are no-ops there and documented on the backend class.
 */

import type { Scene } from "../../scene/scene.js";
import type { Vec3 } from "../../scene/types.js";
import type { MeshData } from "./webgpu/primitives.js";
import type { SkyGradient } from "./webgpu/sky.js";
import type { ShadowQuality } from "./webgpu/lights.js";
import type { ProbeGridParams, ProbeTileLike } from "./webgpu/probe-ring.js";
import type { ImageQualityOptions, LookOptions, RenderFrameOptions, RendererFrameStats } from "./webgpu/renderer.js";
import type { GpuMsSource, PassTimingSample } from "../perf/timestamp-queries.js";

export type LiveRendererBackend = "webgpu" | "three";

export interface LiveRenderer {
  /** Which backend this is — recorded in every benchmark capture. */
  readonly backend: LiveRendererBackend;
  ready: boolean;
  init(): Promise<void>;
  render(scene: Scene, opts?: RenderFrameOptions): void;
  resize(width: number, height: number): void;
  destroy(): void;

  /** Resolved MSAA sample count (1 when the backend uses post AA instead). */
  getSampleCount(): 1 | 4;
  /** Register mesh data under a key the scene's mesh components reference. */
  uploadMesh(name: string, data: MeshData, vertexStride?: 6 | 8 | 16): void;
  /** Free a mesh registered with uploadMesh (streamed meshes leaving with their tile). No-op for unknown keys. */
  releaseMesh(name: string): void;
  /** Decode + attach glTF PBR textures to a mesh already registered with uploadMesh. */
  uploadPbrTextures(
    meshName: string,
    sources: {
      baseColor?: { bytes?: Uint8Array; mimeType?: string; uri?: string };
      metallicRoughness?: { bytes?: Uint8Array; mimeType?: string; uri?: string };
      normal?: { bytes?: Uint8Array; mimeType?: string; uri?: string };
    },
  ): Promise<boolean>;
  /** 52 ARKit weights for an entity whose mesh carries morph targets. */
  setBlendshapeWeights(entityId: string, weights: Float32Array): void;

  setSky(enabled: boolean, gradient?: Partial<SkyGradient>): void;
  setIbl(enabled: boolean, opts?: { irradiance?: Vec3; intensity?: number }): void;
  setShadowQuality(quality: ShadowQuality): void;
  setShadowRange(range: number): void;
  setPostChain(chain: "full" | "tonemap"): void;
  /** Tonemapper, exposure, bloom and display grade. */
  setLook(look: LookOptions): void;
  setInstancing(enabled: boolean): void;
  /** TAA / render scale / sharpen / SSAO. Optional: backend B does not implement it. */
  setImageQuality?(q: ImageQualityOptions): void;
  /** Streamed irradiance probes (backend A only; probe-ring.ts). */
  setIrradianceProbes?(enabled: boolean, grid?: Partial<ProbeGridParams>): void;
  setProbeTile?(tile: ProbeTileLike): void;
  clearProbeTile?(gi0: number, gj0: number, perAxis: number, layers: number): void;
  getFrameStats(): Readonly<RendererFrameStats>;

  /** Frame-level GPU time listener (HUD). */
  enableGpuTimestamps(listener: ((sample: { gpu_ms: number; source: GpuMsSource }) => void) | null): void;
  getTimestampSupport(): { supported: boolean; reason: string };
  /** Per-pass GPU timing for benchmarks. False when the backend cannot time passes. */
  enablePassTimings(listener: ((sample: PassTimingSample) => void) | null): boolean;
}
