/**
 * WebCodecsRecorder — Track 9 (AUGENGINE-9-TRACK-PLAN.md, 2026-05-28).
 *
 * Higher-fidelity, frame-exact MP4 capture path. Sits alongside the Track 1
 * Playwright `recordVideo` pipeline (browser-native WebM, ~6s, opaque
 * timing) as the explicit-control cousin: callers drive the timestamps,
 * the bitrate, the GOP, and the muxer — everything that matters for
 * marketing-grade artifacts.
 *
 * Three-call surface deliberately mirrors `MediaRecorder` so the call site
 * reads like one:
 *
 *   const rec = new WebCodecsRecorder({ width, height, fps, codec, ... });
 *   await rec.start();
 *   for (let i = 0; i < totalFrames; i++) {
 *     // render canvas...
 *     rec.captureFrame(canvas, Math.round(i / fps * 1_000_000));
 *   }
 *   const blob = await rec.stop();
 *
 * Spec defaults (per AUGENGINE-9-TRACK-PLAN.md Track 9):
 *   codec: avc1.640028 (H.264 High @ L4.2)
 *   bitrate: 12 Mbps (variable)
 *   GOP: 60 frames (1 keyframe / second @ 60fps)
 *   fastStart: in-memory (metadata at front of file — plays before fully loaded)
 *
 * Fallback: if `VideoEncoder` is undefined (Firefox < 130, older Safari),
 * `start()` throws a clear error pointing at the Track 1 Playwright path.
 */

import {
  WebCodecsEncoder,
  muxToMp4,
  type EncodedFrame,
} from "./encoder.js";
import type { EncoderOptions } from "../types.js";

export interface WebCodecsRecorderOptions {
  /** Output width in pixels. Must match the source canvas. */
  width: number;
  /** Output height in pixels. Must match the source canvas. */
  height: number;
  /** Target frames-per-second. Default 60. */
  fps?: number;
  /** Codec family. Default "h264" (only family verified end-to-end). */
  codec?: EncoderOptions["codec"];
  /** Encoded bitrate in bits/second. Default 12_000_000 (12 Mbps). */
  bitrate?: number;
  /** Constant vs variable bitrate. Default "variable". */
  bitrateMode?: EncoderOptions["bitrateMode"];
  /** Frames per GOP (keyframe interval). Default = fps (1 / sec). */
  keyframeInterval?: number;
  /** MIME type for the returned Blob. Default "video/mp4". */
  mimeType?: string;
}

export interface WebCodecsRecorderResult {
  /** The muxed MP4 as a Blob. */
  blob: Blob;
  /** Raw byte count (= blob.size, here for ergonomics in tests). */
  bytes: number;
  /** Number of frames the encoder accepted. */
  framesEncoded: number;
  /** Encoder options actually used (after defaults applied). */
  encoderOptions: EncoderOptions;
}

const DEFAULT_FPS = 60;
const DEFAULT_BITRATE = 12_000_000;
const DEFAULT_CODEC: EncoderOptions["codec"] = "h264";
const DEFAULT_BITRATE_MODE: EncoderOptions["bitrateMode"] = "variable";
const DEFAULT_MIME = "video/mp4";

/**
 * Build the full `EncoderOptions` object from the recorder-level shape.
 * Exported so the unit tests can assert exact defaults without standing
 * up a real encoder.
 */
export function buildEncoderOptions(
  opts: WebCodecsRecorderOptions,
): EncoderOptions {
  const fps = opts.fps ?? DEFAULT_FPS;
  return {
    codec: opts.codec ?? DEFAULT_CODEC,
    width: opts.width,
    height: opts.height,
    fps,
    bitrate: opts.bitrate ?? DEFAULT_BITRATE,
    bitrateMode: opts.bitrateMode ?? DEFAULT_BITRATE_MODE,
    keyframeInterval: opts.keyframeInterval ?? fps,
  };
}

/**
 * Frame-exact WebGPU canvas → MP4 recorder. See file header for usage.
 *
 * State machine: `idle` → (start) → `recording` → (stop) → `stopped`.
 * Calling `captureFrame` outside `recording` throws. `start` and `stop`
 * may each be called only once per instance — make a fresh recorder
 * for the next clip.
 */
export class WebCodecsRecorder {
  private readonly opts: WebCodecsRecorderOptions;
  private readonly encoderOptions: EncoderOptions;
  private readonly mimeType: string;
  private encoder: WebCodecsEncoder | null = null;
  private state: "idle" | "recording" | "stopped" = "idle";

  constructor(opts: WebCodecsRecorderOptions) {
    if (!Number.isFinite(opts.width) || !Number.isFinite(opts.height)) {
      throw new Error("WebCodecsRecorder: width/height required");
    }
    if (opts.width <= 0 || opts.height <= 0) {
      throw new Error("WebCodecsRecorder: width/height must be positive");
    }
    // VideoEncoder rejects odd dimensions for AVC — surface the constraint up-front.
    if ((opts.width % 2) !== 0 || (opts.height % 2) !== 0) {
      throw new Error(
        `WebCodecsRecorder: width/height must be even (got ${opts.width}x${opts.height})`,
      );
    }
    this.opts = opts;
    this.encoderOptions = buildEncoderOptions(opts);
    this.mimeType = opts.mimeType ?? DEFAULT_MIME;
  }

  /** Provision and configure the underlying VideoEncoder. */
  async start(): Promise<void> {
    if (this.state !== "idle") {
      throw new Error(`WebCodecsRecorder: cannot start from state '${this.state}'`);
    }
    if (typeof VideoEncoder === "undefined") {
      throw new Error(
        "WebCodecsRecorder: VideoEncoder unavailable in this browser. " +
          "Use the Track 1 Playwright recording path " +
          "(e2e/proof-room-capture.spec.ts) for browsers without WebCodecs.",
      );
    }
    this.encoder = new WebCodecsEncoder(this.encoderOptions);
    await this.encoder.configure();
    this.state = "recording";
  }

  /**
   * Capture one canvas frame. `timestampUs` is the presentation time in
   * microseconds. When omitted, the encoder auto-numbers from frame index
   * 0 at the configured fps (deterministic-render convention).
   */
  captureFrame(
    source: HTMLCanvasElement | OffscreenCanvas | VideoFrame | ImageBitmap,
    timestampUs?: number,
  ): void {
    if (this.state !== "recording" || !this.encoder) {
      throw new Error(`WebCodecsRecorder: cannot captureFrame in state '${this.state}'`);
    }
    this.encoder.encodeFrame(source, timestampUs);
  }

  /**
   * Flush the encoder, mux into MP4, return a Blob. Throws if no frames
   * were captured.
   */
  async stop(): Promise<Blob> {
    if (this.state !== "recording" || !this.encoder) {
      throw new Error(`WebCodecsRecorder: cannot stop in state '${this.state}'`);
    }
    const frames: EncodedFrame[] = await this.encoder.close();
    this.state = "stopped";
    if (frames.length === 0) {
      throw new Error("WebCodecsRecorder.stop: no frames were captured");
    }
    const mp4 = muxToMp4(frames, this.encoderOptions);
    // Wrap in a fresh ArrayBuffer copy so the Blob owns disjoint memory
    // (avoids the rare GC tug-of-war when callers `.arrayBuffer()` the
    // blob after we've already let the encoder buffer fall out of scope).
    // Cast away the `ArrayBufferLike` (= ArrayBuffer | SharedArrayBuffer)
    // union the DOM lib 2025+ surfaces — the encoder always returns a
    // plain ArrayBuffer-backed Uint8Array.
    const owned = new ArrayBuffer(mp4.byteLength);
    new Uint8Array(owned).set(mp4);
    return new Blob([owned], { type: this.mimeType });
  }

  /**
   * Like `stop()` but returns the additional bookkeeping a test or CLI
   * usually wants. Equivalent to `stop()` for state-machine purposes.
   */
  async stopWithResult(): Promise<WebCodecsRecorderResult> {
    const blob = await this.stop();
    return {
      blob,
      bytes: blob.size,
      framesEncoded: this.encoderOptions.keyframeInterval > 0 ? (this.encoder?.framesEncoded() ?? 0) : 0,
      encoderOptions: this.encoderOptions,
    };
  }

  /** Current recorder state. */
  getState(): "idle" | "recording" | "stopped" {
    return this.state;
  }

  /** The resolved encoder options actually in use. */
  getEncoderOptions(): EncoderOptions {
    return this.encoderOptions;
  }
}
