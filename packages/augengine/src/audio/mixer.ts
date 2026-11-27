/**
 * Audio mixer — live + offline (deterministic) modes share the same graph
 * shape so a film export at 60fps produces the same audio as the live
 * playback at the same wall-clock time.
 *
 * Bus structure: master ← {music, sfx, voice, ui}. Phase 9 expands to
 * per-bus compression + EQ; phase 1 ships clean gain control + spatial
 * panning hooks.
 *
 * Live: AudioContext, currentTime drives playback.
 * Render: OfflineAudioContext, sample-accurate, deterministic.
 */

export type BusName = "master" | "music" | "sfx" | "voice" | "ui";

export interface MixerNodes {
  master: GainNode;
  music: GainNode;
  sfx: GainNode;
  voice: GainNode;
  ui: GainNode;
}

export interface MixerOptions {
  /** Per-bus initial gain. Default master=0.9, others=1.0. */
  initialGains?: Partial<Record<BusName, number>>;
}

/** Build the bus graph in any AudioContext-shaped target. */
export function buildBusGraph(
  ctx: BaseAudioContext,
  destination: AudioNode,
  opts: MixerOptions = {},
): MixerNodes {
  const master = ctx.createGain();
  master.gain.value = opts.initialGains?.master ?? 0.9;
  master.connect(destination);

  const make = (name: Exclude<BusName, "master">): GainNode => {
    const g = ctx.createGain();
    g.gain.value = opts.initialGains?.[name] ?? 1.0;
    g.connect(master);
    return g;
  };
  return {
    master,
    music: make("music"),
    sfx: make("sfx"),
    voice: make("voice"),
    ui: make("ui"),
  };
}

/** Live mixer — wraps AudioContext + bus graph. */
export class LiveMixer {
  readonly ctx: AudioContext;
  readonly buses: MixerNodes;

  constructor(opts: MixerOptions & { latencyHint?: AudioContextLatencyCategory } = {}) {
    if (typeof AudioContext === "undefined") {
      throw new Error("AudioContext unavailable in this environment");
    }
    this.ctx = new AudioContext({ latencyHint: opts.latencyHint ?? "interactive" });
    this.buses = buildBusGraph(this.ctx, this.ctx.destination, opts);
  }

  /** Resume after user-gesture (autoplay policy). Call from a click handler. */
  async resume(): Promise<void> {
    if (this.ctx.state === "suspended") await this.ctx.resume();
  }

  /** Stop audio + release context. */
  async destroy(): Promise<void> {
    await this.ctx.close();
  }

  /** Setter for per-bus gain (animatable via setValueAtTime). */
  setBusGain(bus: BusName, value: number, atTime?: number): void {
    const node = this.buses[bus];
    if (atTime !== undefined) {
      node.gain.setValueAtTime(value, atTime);
    } else {
      node.gain.value = value;
    }
  }

  /** Play a one-shot synth tone for the smoke test. */
  playTone(opts: {
    frequency: number;
    durationSec: number;
    waveform?: OscillatorType;
    bus?: BusName;
    startAt?: number;
    gain?: number;
  }): void {
    const ctx = this.ctx;
    const osc = ctx.createOscillator();
    osc.type = opts.waveform ?? "sine";
    osc.frequency.value = opts.frequency;
    const g = ctx.createGain();
    g.gain.value = opts.gain ?? 0.3;
    osc.connect(g);
    g.connect(this.buses[opts.bus ?? "sfx"]);
    const t0 = opts.startAt ?? ctx.currentTime;
    // Tiny attack/release to avoid clicks.
    g.gain.setValueAtTime(0, t0);
    g.gain.linearRampToValueAtTime(opts.gain ?? 0.3, t0 + 0.005);
    g.gain.linearRampToValueAtTime(0, t0 + opts.durationSec);
    osc.start(t0);
    osc.stop(t0 + opts.durationSec + 0.05);
  }

  /** Play an AudioBuffer through a chosen bus. Returns the underlying source. */
  playBuffer(buffer: AudioBuffer, bus: BusName = "sfx", startAt?: number): AudioBufferSourceNode {
    const src = this.ctx.createBufferSource();
    src.buffer = buffer;
    src.connect(this.buses[bus]);
    src.start(startAt ?? this.ctx.currentTime);
    return src;
  }
}

/**
 * Offline / deterministic mixer for film export. Builds an
 * OfflineAudioContext, lets the caller schedule events at exact times,
 * then renders to an AudioBuffer ready to be muxed alongside the encoded
 * video chunks.
 */
export class OfflineMixer {
  readonly ctx: OfflineAudioContext;
  readonly buses: MixerNodes;

  constructor(opts: { sampleRate: number; channels: number; durationSec: number } & MixerOptions) {
    if (typeof OfflineAudioContext === "undefined") {
      throw new Error("OfflineAudioContext unavailable in this environment");
    }
    this.ctx = new OfflineAudioContext({
      numberOfChannels: opts.channels,
      length: Math.ceil(opts.sampleRate * opts.durationSec),
      sampleRate: opts.sampleRate,
    });
    this.buses = buildBusGraph(this.ctx, this.ctx.destination, opts);
  }

  /** Schedule a synth tone at a specific time. */
  scheduleTone(opts: {
    frequency: number;
    startSec: number;
    durationSec: number;
    waveform?: OscillatorType;
    bus?: BusName;
    gain?: number;
  }): void {
    const ctx = this.ctx;
    const osc = ctx.createOscillator();
    osc.type = opts.waveform ?? "sine";
    osc.frequency.value = opts.frequency;
    const g = ctx.createGain();
    osc.connect(g);
    g.connect(this.buses[opts.bus ?? "sfx"]);
    const peak = opts.gain ?? 0.3;
    g.gain.setValueAtTime(0, opts.startSec);
    g.gain.linearRampToValueAtTime(peak, opts.startSec + 0.005);
    g.gain.linearRampToValueAtTime(0, opts.startSec + opts.durationSec);
    osc.start(opts.startSec);
    osc.stop(opts.startSec + opts.durationSec + 0.05);
  }

  /** Schedule a pre-built AudioBuffer at a specific time. */
  scheduleBuffer(buffer: AudioBuffer, startSec: number, bus: BusName = "sfx"): void {
    const src = this.ctx.createBufferSource();
    src.buffer = buffer;
    src.connect(this.buses[bus]);
    src.start(startSec);
  }

  /** Render the entire offline graph. Returns the final AudioBuffer. */
  async render(): Promise<AudioBuffer> {
    return await this.ctx.startRendering();
  }
}
