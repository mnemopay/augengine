/**
 * 3D spatial audio — Phase 10.
 *
 * Wraps Web Audio's PannerNode + ConvolverNode for a turnkey spatial mixer.
 * Each spatial source plugs into a per-bus chain: source → panner →
 * (optional reverb send) → bus → master. Listener position + orientation
 * sync from the active camera every frame.
 *
 * Web Audio's PannerNode handles distance attenuation + HRTF panning natively,
 * so the engine's job is just to keep listener + source positions in sync.
 */

import type { Vec3 } from "../scene/types.js";
import type { LiveMixer, OfflineMixer, BusName } from "./mixer.js";

export interface SpatialSourceOptions {
  buffer: AudioBuffer;
  /** Initial 3D position. Updated per-frame by the engine. */
  position: Vec3;
  /** Bus to route through. Default: "sfx". */
  bus?: BusName;
  /** Loop. */
  loop?: boolean;
  /** Inner cone half-angle in degrees. Audible at full gain inside this. */
  coneInnerAngle?: number;
  /** Outer cone half-angle. Linearly attenuates between inner and outer. */
  coneOuterAngle?: number;
  /** Gain factor outside the outer cone. */
  coneOuterGain?: number;
  /** Distance model. "inverse" (most realistic), "linear", "exponential". */
  distanceModel?: DistanceModelType;
  /** Reference distance — closer than this is full volume. */
  refDistance?: number;
  /** Max distance — beyond this is min volume (or silent). */
  maxDistance?: number;
  /** Roll-off factor for inverse / exponential models. */
  rolloffFactor?: number;
  /** Optional reverb send wet level 0..1. */
  reverbSend?: number;
}

export interface SpatialSourceHandle {
  /** Update the source's world position. Call per-frame from the engine. */
  setPosition(p: Vec3): void;
  /** Update the source's facing direction. */
  setOrientation(dir: Vec3): void;
  /** Stop and disconnect. Idempotent. */
  stop(): void;
}

export class SpatialMixer {
  private ctx: BaseAudioContext;
  private listener: AudioListener;
  private buses: { master: AudioNode; music: AudioNode; sfx: AudioNode; voice: AudioNode; ui: AudioNode };
  private reverb: ConvolverNode | null = null;
  private reverbBus: GainNode | null = null;

  constructor(mixer: LiveMixer | OfflineMixer) {
    this.ctx = mixer.ctx;
    this.listener = mixer.ctx.listener;
    this.buses = mixer.buses;
  }

  /** Update listener position + orientation. Call every frame from the camera. */
  setListener(opts: { position: Vec3; forward: Vec3; up: Vec3 }): void {
    const { position, forward, up } = opts;
    if (this.listener.positionX) {
      // Modern AudioParam path (Chromium, Firefox 70+, Safari 14+).
      const t = this.ctx.currentTime;
      this.listener.positionX.setValueAtTime(position[0], t);
      this.listener.positionY.setValueAtTime(position[1], t);
      this.listener.positionZ.setValueAtTime(position[2], t);
      this.listener.forwardX.setValueAtTime(forward[0], t);
      this.listener.forwardY.setValueAtTime(forward[1], t);
      this.listener.forwardZ.setValueAtTime(forward[2], t);
      this.listener.upX.setValueAtTime(up[0], t);
      this.listener.upY.setValueAtTime(up[1], t);
      this.listener.upZ.setValueAtTime(up[2], t);
    } else {
      // Legacy path (older Safari).
      const legacy = this.listener as unknown as {
        setPosition(x: number, y: number, z: number): void;
        setOrientation(fx: number, fy: number, fz: number, ux: number, uy: number, uz: number): void;
      };
      legacy.setPosition(position[0], position[1], position[2]);
      legacy.setOrientation(forward[0], forward[1], forward[2], up[0], up[1], up[2]);
    }
  }

  /** Configure a convolution reverb shared across all spatial sources. */
  setReverbImpulse(impulse: AudioBuffer): void {
    if (!this.reverb) {
      this.reverb = this.ctx.createConvolver();
      this.reverbBus = this.ctx.createGain();
      this.reverbBus.gain.value = 0.6;
      this.reverb.connect(this.reverbBus);
      this.reverbBus.connect(this.buses.master);
    }
    this.reverb.buffer = impulse;
  }

  /** Spawn a positioned audio source that follows the engine's transform updates. */
  spawn(opts: SpatialSourceOptions): SpatialSourceHandle {
    const ctx = this.ctx;
    const src = ctx.createBufferSource();
    src.buffer = opts.buffer;
    src.loop = opts.loop ?? false;

    const panner = ctx.createPanner();
    panner.panningModel = "HRTF";
    panner.distanceModel = opts.distanceModel ?? "inverse";
    panner.refDistance = opts.refDistance ?? 1;
    panner.maxDistance = opts.maxDistance ?? 100;
    panner.rolloffFactor = opts.rolloffFactor ?? 1;
    panner.coneInnerAngle = opts.coneInnerAngle ?? 360;
    panner.coneOuterAngle = opts.coneOuterAngle ?? 0;
    panner.coneOuterGain = opts.coneOuterGain ?? 0;

    src.connect(panner);
    const bus = this.buses[opts.bus ?? "sfx"];
    panner.connect(bus);

    // Optional reverb send.
    if (opts.reverbSend && this.reverb) {
      const sendGain = ctx.createGain();
      sendGain.gain.value = opts.reverbSend;
      panner.connect(sendGain);
      sendGain.connect(this.reverb);
    }

    // Initial position.
    setPannerPosition(panner, ctx, opts.position);

    src.start(ctx.currentTime);

    let stopped = false;
    return {
      setPosition: (p) => {
        if (stopped) return;
        setPannerPosition(panner, ctx, p);
      },
      setOrientation: (dir) => {
        if (stopped) return;
        const t = ctx.currentTime;
        if (panner.orientationX) {
          panner.orientationX.setValueAtTime(dir[0], t);
          panner.orientationY.setValueAtTime(dir[1], t);
          panner.orientationZ.setValueAtTime(dir[2], t);
        } else {
          (panner as unknown as { setOrientation(x: number, y: number, z: number): void })
            .setOrientation(dir[0], dir[1], dir[2]);
        }
      },
      stop: () => {
        if (stopped) return;
        stopped = true;
        try { src.stop(); } catch { /* may already have stopped */ }
        try { src.disconnect(); } catch { /* idempotent */ }
        try { panner.disconnect(); } catch { /* idempotent */ }
      },
    };
  }
}

function setPannerPosition(panner: PannerNode, ctx: BaseAudioContext, p: Vec3): void {
  const t = ctx.currentTime;
  if (panner.positionX) {
    panner.positionX.setValueAtTime(p[0], t);
    panner.positionY.setValueAtTime(p[1], t);
    panner.positionZ.setValueAtTime(p[2], t);
  } else {
    (panner as unknown as { setPosition(x: number, y: number, z: number): void })
      .setPosition(p[0], p[1], p[2]);
  }
}
