/**
 * Cinematic Sequence — composes multiple camera tracks + audio cues + scene
 * events into a single timeline. The sequence is the unit of "cutscene" or
 * "film clip" that the engine plays back deterministically.
 *
 * Phase 10. The same Sequence type drives:
 *   - Live cutscenes (RAF clock, sample(now - startTime))
 *   - Film export (frame-stepped, sample(frameIndex / fps))
 */

import type { CameraSample } from "./camera-track.js";
import { CameraTrack } from "./camera-track.js";

export interface AudioCue {
  /** Time in seconds when the cue fires. */
  t: number;
  /** AudioBuffer to play. Caller-supplied. */
  buffer: AudioBuffer;
  /** Bus to route through. */
  bus?: "music" | "sfx" | "voice" | "ui";
  /** Optional 3D position. If present, plays through the spatial mixer. */
  position?: [number, number, number];
}

export interface SceneEvent {
  /** Time in seconds. */
  t: number;
  /** Application-defined event tag. */
  type: string;
  /** Application-defined payload. */
  payload?: unknown;
}

export interface SequenceOptions {
  cameraTracks: CameraTrack[];
  audioCues?: AudioCue[];
  events?: SceneEvent[];
  /** Total duration; defaults to max(camera track durations). */
  duration?: number;
}

export class CinematicSequence {
  cameraTracks: CameraTrack[];
  audioCues: AudioCue[];
  events: SceneEvent[];
  duration: number;

  private firedCueIdxs = new Set<number>();
  private firedEventIdxs = new Set<number>();

  constructor(opts: SequenceOptions) {
    this.cameraTracks = opts.cameraTracks;
    this.audioCues = (opts.audioCues ?? []).slice().sort((a, b) => a.t - b.t);
    this.events = (opts.events ?? []).slice().sort((a, b) => a.t - b.t);
    this.duration = opts.duration ??
      Math.max(0, ...this.cameraTracks.map((t) => t.duration));
  }

  /** Sample the active camera track at time t. Picks the first track whose
   *  duration covers t; if none cover t, returns the last track's tail sample. */
  sampleCamera(t: number, out: CameraSample): CameraSample {
    const track = this.cameraTracks.find((tr) => t <= tr.duration) ??
      this.cameraTracks[this.cameraTracks.length - 1]!;
    return track.sample(Math.min(t, track.duration), out);
  }

  /** Drain audio cues whose `t` is in [prevT, currentT]. Caller plays them
   *  through their mixer + spatial mixer. Returns once-per-cue. */
  drainAudioCues(prevT: number, currentT: number): AudioCue[] {
    const out: AudioCue[] = [];
    for (let i = 0; i < this.audioCues.length; i++) {
      if (this.firedCueIdxs.has(i)) continue;
      const cue = this.audioCues[i]!;
      if (cue.t > currentT) break;
      if (cue.t < prevT - 0.001) continue; // late, but missed window
      out.push(cue);
      this.firedCueIdxs.add(i);
    }
    return out;
  }

  /** Drain scene events similarly. */
  drainEvents(prevT: number, currentT: number): SceneEvent[] {
    const out: SceneEvent[] = [];
    for (let i = 0; i < this.events.length; i++) {
      if (this.firedEventIdxs.has(i)) continue;
      const ev = this.events[i]!;
      if (ev.t > currentT) break;
      if (ev.t < prevT - 0.001) continue;
      out.push(ev);
      this.firedEventIdxs.add(i);
    }
    return out;
  }

  /** Reset cue + event firing state for replay. */
  reset(): void {
    this.firedCueIdxs.clear();
    this.firedEventIdxs.clear();
  }
}
