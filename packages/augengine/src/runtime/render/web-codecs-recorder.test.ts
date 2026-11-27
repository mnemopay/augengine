/**
 * WebCodecsRecorder + encoder unit tests.
 *
 * Pins the deterministic encoder/muxer config the spec calls out
 * (AUGENGINE-9-TRACK-PLAN.md Track 9): codec=avc1.640028 → muxer label "avc",
 * bitrate=12 Mbps, framerate=60, GOP=60 (1 keyframe / sec). These are the
 * "marketing-grade artifact" defaults; if they drift, outputs stop being
 * comparable across runs.
 *
 * Render-loop behaviour belongs in the Playwright e2e (track9-export.spec.ts) —
 * VideoEncoder and the mp4-muxer Muxer aren't available in jsdom.
 */
import { describe, it, expect } from "vitest";
import {
  buildEncoderOptions,
  WebCodecsRecorder,
} from "./web-codecs-recorder.js";
import { muxerCodecLabel } from "./encoder.js";

describe("buildEncoderOptions — spec defaults pinned", () => {
  it("matches AUGENGINE-9-TRACK-PLAN.md Track 9 acceptance spec exactly", () => {
    const opts = buildEncoderOptions({ width: 1440, height: 900 });
    expect(opts).toEqual({
      codec: "h264",
      width: 1440,
      height: 900,
      fps: 60,
      bitrate: 12_000_000,
      bitrateMode: "variable",
      keyframeInterval: 60, // GOP = fps so 1 keyframe per second
    });
  });

  it("derives keyframeInterval from fps when omitted (1 keyframe / second)", () => {
    expect(buildEncoderOptions({ width: 640, height: 360, fps: 30 }).keyframeInterval).toBe(30);
    expect(buildEncoderOptions({ width: 640, height: 360, fps: 24 }).keyframeInterval).toBe(24);
    expect(buildEncoderOptions({ width: 640, height: 360 }).keyframeInterval).toBe(60);
  });

  it("respects explicit overrides", () => {
    const opts = buildEncoderOptions({
      width: 1920,
      height: 1080,
      fps: 30,
      codec: "av1",
      bitrate: 6_000_000,
      bitrateMode: "constant",
      keyframeInterval: 90,
    });
    expect(opts.codec).toBe("av1");
    expect(opts.bitrate).toBe(6_000_000);
    expect(opts.bitrateMode).toBe("constant");
    expect(opts.keyframeInterval).toBe(90);
    expect(opts.fps).toBe(30);
  });
});

describe("muxerCodecLabel — mp4-muxer track header families", () => {
  it("maps h264→avc, hevc→hevc, av1→av1 (no surprises)", () => {
    expect(muxerCodecLabel("h264")).toBe("avc");
    expect(muxerCodecLabel("hevc")).toBe("hevc");
    expect(muxerCodecLabel("av1")).toBe("av1");
  });
});

describe("WebCodecsRecorder — constructor validation", () => {
  it("rejects missing dimensions", () => {
    expect(() => new WebCodecsRecorder({ width: NaN as number, height: 720 })).toThrow(/width\/height/);
    expect(() => new WebCodecsRecorder({ width: 1280, height: 0 })).toThrow(/must be positive/);
  });

  it("rejects odd dimensions (AVC requires even — surface up-front)", () => {
    expect(() => new WebCodecsRecorder({ width: 1281, height: 720 })).toThrow(/must be even/);
    expect(() => new WebCodecsRecorder({ width: 1280, height: 721 })).toThrow(/must be even/);
  });

  it("starts in 'idle' state and exposes the resolved encoder options", () => {
    const rec = new WebCodecsRecorder({ width: 1280, height: 720 });
    expect(rec.getState()).toBe("idle");
    const opts = rec.getEncoderOptions();
    expect(opts.codec).toBe("h264");
    expect(opts.fps).toBe(60);
    expect(opts.bitrate).toBe(12_000_000);
    expect(opts.keyframeInterval).toBe(60);
  });
});

describe("WebCodecsRecorder — state-machine errors are loud", () => {
  it("captureFrame from idle state throws", () => {
    const rec = new WebCodecsRecorder({ width: 1280, height: 720 });
    expect(() => rec.captureFrame({} as HTMLCanvasElement)).toThrow(/state 'idle'/);
  });

  it("stop from idle state throws", async () => {
    const rec = new WebCodecsRecorder({ width: 1280, height: 720 });
    await expect(rec.stop()).rejects.toThrow(/state 'idle'/);
  });
});
