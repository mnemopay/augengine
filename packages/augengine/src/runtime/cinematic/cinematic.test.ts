/**
 * Phase 10 tests — camera track sampling, sequence cue draining.
 */

import { describe, it, expect } from "vitest";
import {
  CameraTrack,
  CinematicSequence,
  type CameraKeyframe,
  type CameraSample,
} from "./index.js";

function blankSample(): CameraSample {
  return { position: [0, 0, 0], rotation: [0, 0, 0, 1], fovY: 0 };
}

describe("augengine / phase 10 — CameraTrack", () => {
  it("returns first kf at t=0", () => {
    const kfs: CameraKeyframe[] = [
      { t: 0, position: [0, 0, 5], lookAt: [0, 0, 0], fovY: Math.PI / 3 },
      { t: 2, position: [5, 0, 5], lookAt: [0, 0, 0], fovY: Math.PI / 3 },
    ];
    const track = new CameraTrack(kfs);
    const out = track.sample(0, blankSample());
    expect(out.position[0]).toBeCloseTo(0);
    expect(out.position[2]).toBeCloseTo(5);
  });

  it("clamps to last kf beyond duration", () => {
    const kfs: CameraKeyframe[] = [
      { t: 0, position: [0, 0, 5], lookAt: [0, 0, 0], fovY: 1 },
      { t: 2, position: [10, 0, 5], lookAt: [0, 0, 0], fovY: 2 },
    ];
    const track = new CameraTrack(kfs);
    const out = track.sample(99, blankSample());
    expect(out.position[0]).toBeCloseTo(10);
    expect(out.fovY).toBeCloseTo(2);
  });

  it("interpolates linearly between two kfs", () => {
    const kfs: CameraKeyframe[] = [
      { t: 0, position: [0, 0, 0], lookAt: [0, 0, -1], fovY: 1.0 },
      { t: 2, position: [10, 0, 0], lookAt: [0, 0, -1], fovY: 2.0, easing: "linear" },
    ];
    const track = new CameraTrack(kfs);
    const out = track.sample(1, blankSample());
    expect(out.position[0]).toBeCloseTo(5, 4);
    expect(out.fovY).toBeCloseTo(1.5, 4);
  });

  it("smoothstep eases", () => {
    const kfs: CameraKeyframe[] = [
      { t: 0, position: [0, 0, 0], lookAt: [0, 0, -1], fovY: 1, easing: "smoothstep" },
      { t: 2, position: [10, 0, 0], lookAt: [0, 0, -1], fovY: 2 },
    ];
    const track = new CameraTrack(kfs);
    // smoothstep at midpoint = 0.5; at 25% = 0.15625; at 75% = 0.84375
    const quarter = track.sample(0.5, blankSample());
    expect(quarter.position[0]).toBeCloseTo(10 * 0.15625, 3);
    const threeq = track.sample(1.5, blankSample());
    expect(threeq.position[0]).toBeCloseTo(10 * 0.84375, 3);
  });

  it("sorts unsorted keyframes on construction", () => {
    const kfs: CameraKeyframe[] = [
      { t: 2, position: [10, 0, 0], lookAt: [0, 0, -1], fovY: 1 },
      { t: 0, position: [0, 0, 0], lookAt: [0, 0, -1], fovY: 1 },
    ];
    const track = new CameraTrack(kfs);
    expect(track.keyframes[0]!.t).toBe(0);
    expect(track.keyframes[1]!.t).toBe(2);
  });
});

describe("augengine / phase 10 — CinematicSequence", () => {
  function makeSeq(): CinematicSequence {
    return new CinematicSequence({
      cameraTracks: [
        new CameraTrack([
          { t: 0, position: [0, 0, 5], lookAt: [0, 0, 0], fovY: 1 },
          { t: 3, position: [5, 0, 5], lookAt: [0, 0, 0], fovY: 1 },
        ]),
      ],
      audioCues: [
        { t: 0.5, buffer: {} as AudioBuffer, bus: "sfx" },
        { t: 1.5, buffer: {} as AudioBuffer, bus: "music" },
      ],
      events: [
        { t: 1.0, type: "shake", payload: 0.3 },
        { t: 2.5, type: "flash" },
      ],
    });
  }

  it("drainAudioCues fires cues exactly once in window", () => {
    const seq = makeSeq();
    const a = seq.drainAudioCues(0, 1);
    expect(a).toHaveLength(1);
    const b = seq.drainAudioCues(1, 1.6);
    expect(b).toHaveLength(1);
    expect(b[0]!.bus).toBe("music");
    const c = seq.drainAudioCues(0, 5);
    expect(c).toHaveLength(0); // both already fired
  });

  it("drainEvents fires events once", () => {
    const seq = makeSeq();
    const a = seq.drainEvents(0, 2);
    expect(a).toHaveLength(1);
    expect(a[0]!.type).toBe("shake");
    const b = seq.drainEvents(2, 5);
    expect(b).toHaveLength(1);
    expect(b[0]!.type).toBe("flash");
  });

  it("reset re-arms cues + events for replay", () => {
    const seq = makeSeq();
    seq.drainAudioCues(0, 5);
    seq.drainEvents(0, 5);
    seq.reset();
    const cues = seq.drainAudioCues(0, 5);
    expect(cues).toHaveLength(2);
  });

  it("sampleCamera picks the right track", () => {
    const seq = makeSeq();
    const out = seq.sampleCamera(1.5, { position: [0, 0, 0], rotation: [0, 0, 0, 1], fovY: 0 });
    expect(out.position[0]).toBeCloseTo(2.5, 4);
  });
});
