import type { Scene } from "../../scene/scene.js";
import { Tag } from "../../scene/types.js";
import type { Entity, EntityId, Vec3 } from "../../scene/types.js";
import { GameTag, getGameComponent, type VehicleComponent } from "./components.js";
import type { GameSystem, SystemContext } from "./system.js";
import { LiveMixer } from "../../audio/mixer.js";

export class AudioSystem implements GameSystem {
  readonly id = "game.audio_system";
  private mixer: LiveMixer | null = null;
  
  // Active engine loops indexed by entity ID
  private engines = new Map<string, {
    osc: OscillatorNode;
    filter: BiquadFilterNode;
    gain: GainNode;
  }>();

  private sequencerStarted = false;
  private nextNoteTime = 0.0;
  private beatIndex = 0;

  constructor() {
    if (typeof window !== "undefined") {
      const resume = () => {
        if (this.mixer) {
          this.mixer.resume().catch(() => {});
          if (!this.sequencerStarted) {
            this.nextNoteTime = this.mixer.ctx.currentTime;
            this.sequencerStarted = true;
          }
        }
      };
      window.addEventListener("click", resume, { once: true });
      window.addEventListener("keydown", resume, { once: true });
    }
  }

  update(scene: Scene, dt: number, ctx: SystemContext): void {
    // 1. Initialize mixer lazily on first update
    if (!this.mixer) {
      try {
        this.mixer = new LiveMixer();
      } catch (err) {
        console.warn("LiveMixer initialization failed:", err);
        return;
      }
    }

    const audioCtx = this.mixer.ctx;

    // 2. Play Background Sequencer Loop (125 BPM -> 16th note is ~0.12s)
    if (this.sequencerStarted && audioCtx.state === "running") {
      const scheduleAheadTime = 0.12;
      while (this.nextNoteTime < audioCtx.currentTime + scheduleAheadTime) {
        this.playSequencerStep(this.beatIndex, this.nextNoteTime);
        this.nextNoteTime += 0.12;
        this.beatIndex = (this.beatIndex + 1) % 16;
      }
    }

    // 3. Handle Game Events (checkpoints, start, finish)
    for (const event of ctx.events) {
      if (event.type === "race-checkpoint" && event.racerId === "player") {
        this.playCheckpointChime();
      } else if (event.type === "race-started") {
        if (this.mixer) {
          this.mixer.resume().catch(() => {});
          this.nextNoteTime = audioCtx.currentTime;
          this.sequencerStarted = true;
        }
      } else if (event.type === "race-finished" && event.racerId === "player") {
        this.playRaceFinishedChime(event.rank <= 1);
      }
    }

    // 4. Update Engine Sounds for vehicles
    const playerEnt = scene.entities.get("player_vehicle" as EntityId);
    const playerPos = playerEnt?.components[Tag.Transform]?.position ?? null;

    for (const e of scene.entities.values()) {
      const vehicle = getGameComponent<VehicleComponent>(e, GameTag.Vehicle);
      const tx = e.components[Tag.Transform];
      if (!vehicle || !tx) {
        // Stop engine sound if entity no longer has vehicle physics
        this.stopEngine(e.id);
        continue;
      }

      // Calculate distance to player for volume attenuation
      let distance = 0;
      if (e.id !== "player_vehicle" && playerPos) {
        const dx = tx.position[0] - playerPos[0];
        const dy = tx.position[1] - playerPos[1];
        const dz = tx.position[2] - playerPos[2];
        distance = Math.hypot(dx, dy, dz);
      }

      const speed = Math.hypot(vehicle.velocity[0], vehicle.velocity[2]);

      // Mute vehicles too far away or stopped
      if (distance > 35.0 || (speed < 0.1 && vehicle.throttle === 0)) {
        this.stopEngine(e.id);
      } else {
        this.playOrUpdateEngine(e.id, speed, vehicle.throttle, distance);
      }
    }
  }

  private playOrUpdateEngine(id: string, speed: number, throttle: number, distance: number): void {
    if (!this.mixer) return;
    const ctx = this.mixer.ctx;
    let eng = this.engines.get(id);

    if (!eng) {
      const osc = ctx.createOscillator();
      osc.type = "sawtooth";

      const filter = ctx.createBiquadFilter();
      filter.type = "lowpass";
      filter.Q.value = 5.0; // resonant growl

      const gain = ctx.createGain();
      gain.gain.value = 0.0;

      osc.connect(filter);
      filter.connect(gain);
      gain.connect(this.mixer.buses.sfx);

      osc.start();
      eng = { osc, filter, gain };
      this.engines.set(id, eng);
    }

    // pitch: idle around 50Hz, top speed up to 230Hz
    const pitch = 50.0 + (speed / 60.0) * 180.0;
    eng.osc.frequency.setTargetAtTime(pitch, ctx.currentTime, 0.05);

    // filter cutoff: higher speeds open the filter for a whining jet sound
    const cutoff = 250.0 + (speed / 60.0) * 900.0;
    eng.filter.frequency.setTargetAtTime(cutoff, ctx.currentTime, 0.05);

    // gain attenuation by distance
    let targetVolume = id === "player_vehicle" ? 0.24 : 0.08 * (1.0 - distance / 35.0);
    if (throttle > 0) {
      targetVolume *= 1.3;
    }
    eng.gain.gain.setTargetAtTime(targetVolume, ctx.currentTime, 0.06);
  }

  private stopEngine(id: string): void {
    const eng = this.engines.get(id);
    if (eng) {
      try {
        eng.gain.gain.value = 0;
        eng.osc.stop();
      } catch (err) {}
      this.engines.delete(id);
    }
  }

  private playCheckpointChime(): void {
    if (!this.mixer) return;
    const ctx = this.mixer.ctx;
    const now = ctx.currentTime;

    const playTone = (freq: number, delay: number) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "sine";
      osc.frequency.setValueAtTime(freq, now + delay);

      gain.gain.setValueAtTime(0, now + delay);
      gain.gain.linearRampToValueAtTime(0.18, now + delay + 0.01);
      gain.gain.exponentialRampToValueAtTime(0.001, now + delay + 0.6);

      osc.connect(gain);
      gain.connect(this.mixer!.buses.ui);
      osc.start(now + delay);
      osc.stop(now + delay + 0.65);
    };

    // Upward chime chord (A4 -> C#5 -> E5)
    playTone(440.0, 0.0);
    playTone(554.37, 0.06);
    playTone(659.25, 0.12);
  }

  private playRaceFinishedChime(won: boolean): void {
    if (!this.mixer) return;
    const ctx = this.mixer.ctx;
    const now = ctx.currentTime;

    const playTone = (freq: number, duration: number, delay: number) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "triangle";
      osc.frequency.value = freq;

      gain.gain.setValueAtTime(0, now + delay);
      gain.gain.linearRampToValueAtTime(0.2, now + delay + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.001, now + delay + duration);

      osc.connect(gain);
      gain.connect(this.mixer!.buses.ui);
      osc.start(now + delay);
      osc.stop(now + delay + duration + 0.05);
    };

    if (won) {
      // Ascending triumphant chord
      playTone(523.25, 0.8, 0.0); // C5
      playTone(659.25, 0.8, 0.1); // E5
      playTone(783.99, 1.2, 0.2); // G5
      playTone(1046.50, 1.5, 0.3); // C6
    } else {
      // Descending minor chord
      playTone(392.00, 0.6, 0.0); // G4
      playTone(349.23, 0.6, 0.15); // F4
      playTone(311.13, 1.2, 0.3); // Eb4
    }
  }

  private playSequencerStep(step: number, time: number): void {
    if (!this.mixer) return;
    const ctx = this.mixer.ctx;

    // A. Kick drum (four-on-the-floor)
    if (step === 0 || step === 4 || step === 8 || step === 12) {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "sine";
      osc.connect(gain);
      gain.connect(this.mixer.buses.music);

      osc.frequency.setValueAtTime(130, time);
      osc.frequency.exponentialRampToValueAtTime(0.01, time + 0.12);

      gain.gain.setValueAtTime(0.38, time);
      gain.gain.linearRampToValueAtTime(0, time + 0.12);

      osc.start(time);
      osc.stop(time + 0.15);
    }

    // B. Hi-hat (offbeat)
    if (step === 2 || step === 6 || step === 10 || step === 14 || (step % 2 === 1 && Math.random() < 0.2)) {
      const bufferSize = ctx.sampleRate * 0.035;
      const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
      const data = buffer.getChannelData(0);
      for (let i = 0; i < bufferSize; i++) {
        data[i] = Math.random() * 2 - 1;
      }
      const noise = ctx.createBufferSource();
      noise.buffer = buffer;

      const filter = ctx.createBiquadFilter();
      filter.type = "highpass";
      filter.frequency.value = 6500;

      const gain = ctx.createGain();
      gain.gain.setValueAtTime(0.04, time);
      gain.gain.linearRampToValueAtTime(0, time + 0.035);

      noise.connect(filter);
      filter.connect(gain);
      gain.connect(this.mixer.buses.music);

      noise.start(time);
      noise.stop(time + 0.04);
    }

    // C. Talking drums (African pitch bend percussion)
    // Rhythmic syncopations
    const tomSteps = [3, 5, 7, 9, 11, 15];
    if (tomSteps.includes(step) && Math.random() < 0.85) {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "triangle";
      osc.connect(gain);
      gain.connect(this.mixer.buses.music);

      // Pitch sweep
      const startFreq = step % 4 === 3 ? 175.0 : 210.0;
      const endFreq = startFreq - (step % 2 === 0 ? 90.0 : 50.0);
      osc.frequency.setValueAtTime(startFreq, time);
      osc.frequency.exponentialRampToValueAtTime(endFreq, time + 0.14);

      gain.gain.setValueAtTime(0.15, time);
      gain.gain.linearRampToValueAtTime(0, time + 0.14);

      osc.start(time);
      osc.stop(time + 0.16);
    }
  }
}
