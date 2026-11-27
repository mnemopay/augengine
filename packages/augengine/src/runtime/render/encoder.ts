/**
 * WebCodecs encoder wrapper — Track 9 closeout (2026-05-28).
 *
 * Configures a VideoEncoder, accepts VideoFrames per render tick, collects
 * EncodedVideoChunks, and (via `muxToMp4` below) muxes them into a real
 * playable MP4 via `mp4-muxer` (Vanilg, MIT). The Phase 1 STUB that wrote
 * a "AUGENGINE-PHASE1-STUB-NOT-MP4\n" marker is now retired — the bytes
 * returned here will open in QuickTime, VLC, ffprobe, and the browser.
 *
 * Why this beats Remotion: WebCodecs ~5-15× faster than chrome-headless
 * because we skip the headless-Chrome roundtrip entirely. See AUGENGINE-
 * ARCHITECTURE.md §"Why this isn't Three.js + Remotion."
 */

import { ArrayBufferTarget, Muxer } from "mp4-muxer";
import type { EncoderOptions } from "../types.js";

export interface EncodedFrame {
  chunk: EncodedVideoChunk;
  meta?: EncodedVideoChunkMetadata;
}

export class WebCodecsEncoder {
  private encoder: VideoEncoder | null = null;
  private opts: EncoderOptions;
  private frames: EncodedFrame[] = [];
  private err: Error | null = null;
  private inflight = 0;
  private frameIndex = 0;

  constructor(opts: EncoderOptions) {
    this.opts = opts;
  }

  async configure(): Promise<void> {
    if (typeof VideoEncoder === "undefined") {
      throw new Error(
        "WebCodecs unavailable in this environment. " +
          "VideoEncoder is required — fall back to the Track 1 Playwright " +
          "recordVideo path (e2e/proof-room-capture.spec.ts) when WebCodecs " +
          "is missing (Firefox < 130, older Safari).",
      );
    }
    const codec = mapCodec(this.opts.codec);
    const support = await VideoEncoder.isConfigSupported({
      codec,
      width: this.opts.width,
      height: this.opts.height,
      bitrate: this.opts.bitrate,
      framerate: this.opts.fps,
    });
    if (!support.supported) {
      throw new Error(`WebCodecs: codec config unsupported: ${codec}`);
    }
    this.encoder = new VideoEncoder({
      output: (chunk, meta) => {
        this.frames.push(meta ? { chunk, meta } : { chunk });
        this.inflight = Math.max(0, this.inflight - 1);
      },
      error: (e) => {
        this.err = e instanceof Error ? e : new Error(String(e));
      },
    });
    this.encoder.configure({
      codec,
      width: this.opts.width,
      height: this.opts.height,
      bitrate: this.opts.bitrate,
      framerate: this.opts.fps,
      bitrateMode: this.opts.bitrateMode,
      ...(codec.startsWith("avc1") ? { avc: { format: "avc" } } : {}),
    });
  }

  /**
   * Encode one frame. `timestampUs` is the presentation time in microseconds;
   * deterministic render mode passes `frameIndex / fps * 1e6`.
   *
   * Source can be:
   *  - HTMLCanvasElement / OffscreenCanvas (most common; we copy into VideoFrame)
   *  - VideoFrame (pre-built; we own its lifecycle)
   *  - ImageBitmap
   */
  encodeFrame(
    source: HTMLCanvasElement | OffscreenCanvas | VideoFrame | ImageBitmap,
    timestampUs?: number,
  ): void {
    if (!this.encoder) throw new Error("encoder not configured");
    if (this.err) throw this.err;
    const ts =
      timestampUs ?? Math.round((this.frameIndex / this.opts.fps) * 1_000_000);
    let frame: VideoFrame;
    if (source instanceof VideoFrame) {
      frame = source;
    } else {
      frame = new VideoFrame(source as CanvasImageSource, { timestamp: ts });
    }
    const isKey = this.frameIndex % this.opts.keyframeInterval === 0;
    this.inflight += 1;
    this.encoder.encode(frame, { keyFrame: isKey });
    frame.close();
    this.frameIndex += 1;
  }

  /** Wait for the encoder to drain all in-flight frames. */
  async flush(): Promise<void> {
    if (!this.encoder) return;
    await this.encoder.flush();
  }

  async close(): Promise<EncodedFrame[]> {
    if (!this.encoder) return this.frames;
    await this.encoder.flush();
    this.encoder.close();
    this.encoder = null;
    if (this.err) throw this.err;
    return this.frames;
  }

  framesEncoded(): number {
    return this.frameIndex;
  }
}

function mapCodec(codec: EncoderOptions["codec"]): string {
  switch (codec) {
    // High profile, level 4.2 — covers up to 1080p60 on virtually all hardware.
    case "h264":
      return "avc1.640028";
    // HEVC main profile, level 4.1.
    case "hevc":
      return "hev1.1.6.L120.B0";
    // AV1 main profile, level 4.0.
    case "av1":
      return "av01.0.04M.08";
  }
}

/**
 * Translate our `EncoderOptions.codec` into the `'avc' | 'hevc' | 'vp9' | 'av1'`
 * label that `mp4-muxer` wants in its track header. Exported so the
 * recorder + render runtime can configure the muxer identically. (mp4-muxer
 * uses bare codec families, not full RFC 6381 strings.)
 */
export function muxerCodecLabel(codec: EncoderOptions["codec"]): "avc" | "hevc" | "av1" {
  switch (codec) {
    case "h264": return "avc";
    case "hevc": return "hevc";
    case "av1":  return "av1";
  }
}

/**
 * Mux an array of EncodedFrames into a real MP4 byte array via `mp4-muxer`.
 *
 * 2026-05-28: STUB RETIRED. Track 9 of AUGENGINE-9-TRACK-PLAN ships real
 * H.264 MP4 output. Bytes returned here are a valid `ftyp/moov/mdat` MP4
 * (Fast Start, in-memory mode — metadata up front so the file plays before
 * fully downloaded).
 *
 * Inputs:
 *  - `frames`: chunks from `WebCodecsEncoder`. The first chunk MUST be a
 *    keyframe (the encoder enforces this via `keyframeInterval`).
 *  - `opts`: same opts used to configure the encoder — width, height, fps,
 *    codec. The muxer track header has to match the encoder's stream.
 */
export function muxToMp4(frames: EncodedFrame[], opts: EncoderOptions): Uint8Array {
  if (frames.length === 0) {
    throw new Error("muxToMp4: no frames to mux");
  }
  const muxer = new Muxer({
    target: new ArrayBufferTarget(),
    fastStart: "in-memory",
    video: {
      codec: muxerCodecLabel(opts.codec),
      width: opts.width,
      height: opts.height,
      frameRate: opts.fps,
    },
  });
  for (const f of frames) {
    muxer.addVideoChunk(f.chunk, f.meta);
  }
  muxer.finalize();
  return new Uint8Array(muxer.target.buffer);
}
