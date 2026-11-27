/**
 * world/tile-actors — NPCs and ambient sound that stream with their tile.
 *
 * NPCs: each tile declares its residents. They are spawned as scene entities
 * (persona, transform, capsule collider, body mesh) when the tile becomes
 * resident and removed when it is evicted. Their live state — position,
 * heading, what they were doing, their RNG, their mood — is saved on evict and
 * restored on the next load, so walking away and back finds them where they
 * were rather than reset at home.
 *
 * Ambient audio: each tile declares positional loops (wind on a ridge, the
 * beacon's hum). A voice is started through an `AmbientAudioSink` when the
 * tile loads, faded in, and faded out then released when the tile leaves.
 * Only the `maxVoices` nearest emitters hold a real voice; the rest stay
 * virtual, so the number of live voices is bounded on any walk.
 *
 * Engine-agnostic: no Web Audio here. The host supplies the sink (the smoke
 * game wires it to SpatialMixer), and the level supplies ground height and
 * walkability so NPCs stay on resident, walkable ground.
 */
import type { Scene } from "../../scene/scene.js";
import { Tag, type Entity, type EntityId, type PersonaComponent, type Vec3 } from "../../scene/types.js";

export interface NpcSpec {
  /** Stable across loads: the NPC's identity. */
  id: EntityId;
  name: string;
  home: Vec3;
  persona: Pick<PersonaComponent, "traits"> & Partial<PersonaComponent>;
  color: Vec3;
}

export interface NpcState {
  position: Vec3;
  yaw: number;
  phase: "idle" | "walk";
  target: Vec3 | null;
  /** Seconds left in the current idle. */
  timer: number;
  rng: number;
  mood: number;
}

export interface AmbientSpec {
  key: string;
  position: Vec3;
  /** Sound id the sink understands (e.g. "wind", "hum"). */
  sound: string;
  gain: number;
  /** Audible radius in metres — also the voice-priority cutoff. */
  radius: number;
}

export interface AmbientVoice {
  setGain(gain: number): void;
  stop(): void;
}

export interface AmbientAudioSink {
  start(spec: AmbientSpec): AmbientVoice;
}

export interface TileActorsOptions {
  npcsFor(tx: number, tz: number): readonly NpcSpec[];
  ambientsFor(tx: number, tz: number): readonly AmbientSpec[];
  /** Ground height at world XZ (terrain). */
  groundY(x: number, z: number): number;
  /** True where an NPC may stand (resident, walkable ground). */
  walkable(x: number, z: number): boolean;
  audio?: AmbientAudioSink;
  /** Fade time for ambient voices, seconds. Default 1.5. */
  fadeSeconds?: number;
  /** Most ambient voices alive at once. Default 8. */
  maxVoices?: number;
  /** NPC walk speed m/s. Default 1.3. */
  walkSpeed?: number;
}

export interface TileActorStats {
  npcs: number;
  peakNpcs: number;
  savedStates: number;
  spawned: number;
  despawned: number;
  restored: number;
  emitters: number;
  voices: number;
  peakVoices: number;
  voicesStarted: number;
  voicesReleased: number;
}

const NPC_HALF_HEIGHT = 0.55;
const NPC_RADIUS = 0.35;
const NPC_CENTER_Y = NPC_HALF_HEIGHT + NPC_RADIUS;

interface LiveNpc {
  spec: NpcSpec;
  tile: string;
  state: NpcState;
}

interface Emitter {
  spec: AmbientSpec;
  tile: string;
  /** Present while the emitter holds a real voice. */
  voice: AmbientVoice | null;
  /** Current faded gain 0..1 of spec.gain. */
  level: number;
  /** True once the tile left: fade to 0, then release. */
  leaving: boolean;
}

function nextRandom(state: number): [number, number] {
  let a = (state + 0x6d2b79f5) >>> 0;
  let t = a;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return [((t ^ (t >>> 14)) >>> 0) / 4294967296, a];
}

function seedOf(id: string): number {
  let h = 2166136261;
  for (let i = 0; i < id.length; i++) h = Math.imul(h ^ id.charCodeAt(i), 16777619);
  return h >>> 0;
}

export class TileActors {
  private readonly scene: Scene;
  private readonly opts: TileActorsOptions;
  private readonly fade: number;
  private readonly maxVoices: number;
  private readonly speed: number;
  private readonly live = new Map<EntityId, LiveNpc>();
  /** Saved NPC state for unloaded tiles, keyed by NPC id. */
  private readonly saved = new Map<EntityId, NpcState>();
  private readonly emitters = new Map<string, Emitter>();
  stats: TileActorStats = {
    npcs: 0, peakNpcs: 0, savedStates: 0, spawned: 0, despawned: 0, restored: 0,
    emitters: 0, voices: 0, peakVoices: 0, voicesStarted: 0, voicesReleased: 0,
  };

  constructor(scene: Scene, opts: TileActorsOptions) {
    this.scene = scene;
    this.opts = opts;
    this.fade = Math.max(0.01, opts.fadeSeconds ?? 1.5);
    this.maxVoices = Math.max(0, opts.maxVoices ?? 8);
    this.speed = opts.walkSpeed ?? 1.3;
  }

  /** Call when tile (tx, tz) becomes resident. */
  onTileResident(tx: number, tz: number): void {
    const tile = `${tx}|${tz}`;
    for (const spec of this.opts.npcsFor(tx, tz)) {
      if (this.live.has(spec.id)) continue;
      const prior = this.saved.get(spec.id);
      const state: NpcState = prior
        ? { ...prior, position: [...prior.position] as unknown as Vec3, target: prior.target ? ([...prior.target] as unknown as Vec3) : null }
        : {
            position: [spec.home[0], this.opts.groundY(spec.home[0], spec.home[2]) + NPC_CENTER_Y, spec.home[2]],
            yaw: 0, phase: "idle", target: null, timer: 1, rng: seedOf(spec.id), mood: spec.persona.mood ?? 0,
          };
      if (prior) {
        this.saved.delete(spec.id);
        this.stats.restored += 1;
      }
      this.live.set(spec.id, { spec, tile, state });
      this.scene.add(npcEntity(spec, state));
      this.stats.spawned += 1;
    }
    for (const spec of this.opts.ambientsFor(tx, tz)) {
      const existing = this.emitters.get(spec.key);
      if (existing) {
        // Re-entered before the fade-out finished: keep the voice, fade back in.
        existing.leaving = false;
        continue;
      }
      this.emitters.set(spec.key, { spec, tile, voice: null, level: 0, leaving: false });
    }
    this.refreshStats();
  }

  /** Call when tile (tx, tz) is evicted. */
  onTileEvicted(tx: number, tz: number): void {
    const tile = `${tx}|${tz}`;
    for (const [id, npc] of this.live) {
      if (npc.tile !== tile) continue;
      this.saved.set(id, npc.state);
      this.live.delete(id);
      if (this.scene.has(id)) this.scene.remove(id);
      this.stats.despawned += 1;
    }
    for (const e of this.emitters.values()) if (e.tile === tile) e.leaving = true;
    this.refreshStats();
  }

  /** Saved state for an NPC that is currently unloaded (tests, save games). */
  savedState(id: EntityId): NpcState | undefined {
    return this.saved.get(id);
  }

  /** Live state for a resident NPC. */
  liveState(id: EntityId): NpcState | undefined {
    return this.live.get(id)?.state;
  }

  /** Advance NPC behaviour and ambient fades/voice assignment. */
  update(dt: number, listener: Vec3): void {
    const step = Math.min(Math.max(dt, 0), 0.25);
    for (const npc of this.live.values()) this.stepNpc(npc, step);
    this.stepAudio(step, listener);
    this.refreshStats();
  }

  destroy(): void {
    for (const id of this.live.keys()) if (this.scene.has(id)) this.scene.remove(id);
    this.live.clear();
    for (const e of this.emitters.values()) this.release(e);
    this.emitters.clear();
    this.refreshStats();
  }

  // ── NPCs ─────────────────────────────────────────────────────────────

  private random(npc: LiveNpc): number {
    const [r, next] = nextRandom(npc.state.rng);
    npc.state.rng = next;
    return r;
  }

  private stepNpc(npc: LiveNpc, dt: number): void {
    const s = npc.state;
    const axes = npc.spec.persona.axes;
    const curiosity = axes?.curiosity ?? 0.5;
    const fear = axes?.fear ?? 0.5;
    if (s.phase === "idle") {
      s.timer -= dt;
      if (s.timer <= 0) {
        // Curious NPCs roam further; fearful ones stay near home.
        const radius = 4 + curiosity * 14 - fear * 3;
        for (let attempt = 0; attempt < 6; attempt++) {
          const a = this.random(npc) * Math.PI * 2;
          const r = radius * Math.sqrt(this.random(npc));
          const tx = npc.spec.home[0] + Math.cos(a) * r;
          const tz = npc.spec.home[2] + Math.sin(a) * r;
          if (this.opts.walkable(tx, tz)) {
            s.target = [tx, 0, tz];
            s.phase = "walk";
            break;
          }
        }
        if (s.phase === "idle") s.timer = 1 + this.random(npc) * 2;
      }
    } else if (s.target) {
      const dx = s.target[0] - s.position[0], dz = s.target[2] - s.position[2];
      const dist = Math.hypot(dx, dz);
      const speed = this.speed * (1 + fear * 0.3);
      if (dist < 0.2) {
        s.phase = "idle";
        s.target = null;
        s.timer = 2 + this.random(npc) * 4 * (1 - curiosity * 0.5);
      } else {
        const move = Math.min(dist, speed * dt);
        const nx = s.position[0] + (dx / dist) * move, nz = s.position[2] + (dz / dist) * move;
        if (this.opts.walkable(nx, nz)) {
          s.position = [nx, this.opts.groundY(nx, nz) + NPC_CENTER_Y, nz];
          s.yaw = Math.atan2(dx, dz);
        } else {
          // The way ahead is unloaded or steep: stop and choose again.
          s.phase = "idle";
          s.target = null;
          s.timer = 1;
        }
      }
    }
    const tx = this.scene.get(npc.spec.id)?.components[Tag.Transform];
    if (tx) {
      const half = s.yaw / 2;
      this.scene.setLocalTransform(npc.spec.id, s.position, [0, Math.sin(half), 0, Math.cos(half)], tx.scale);
    }
  }

  // ── ambient audio ────────────────────────────────────────────────────

  private stepAudio(dt: number, listener: Vec3): void {
    // Voice priority: nearest audible, non-leaving emitters first.
    const ranked = [...this.emitters.values()]
      .map((e) => ({ e, d: Math.hypot(e.spec.position[0] - listener[0], e.spec.position[1] - listener[1], e.spec.position[2] - listener[2]) }))
      .sort((a, b) => a.d - b.d);
    const want = new Set<Emitter>();
    for (const { e, d } of ranked) {
      if (want.size >= this.maxVoices) break;
      if (!e.leaving && d <= e.spec.radius) want.add(e);
    }
    const rate = dt / this.fade;
    for (const { e } of ranked) {
      const target = want.has(e) ? 1 : 0;
      e.level = target > e.level ? Math.min(target, e.level + rate) : Math.max(target, e.level - rate);
      if (target > 0 && !e.voice && this.opts.audio && this.liveVoices() < this.maxVoices) {
        e.voice = this.opts.audio.start(e.spec);
        this.stats.voicesStarted += 1;
      }
      if (e.voice) e.voice.setGain(e.level * e.spec.gain);
      if (e.level <= 0 && e.voice && target === 0) this.release(e);
      if (e.leaving && e.level <= 0) {
        this.release(e);
        this.emitters.delete(e.spec.key);
      }
    }
  }

  private liveVoices(): number {
    let n = 0;
    for (const e of this.emitters.values()) if (e.voice) n += 1;
    return n;
  }

  private release(e: Emitter): void {
    if (!e.voice) return;
    e.voice.stop();
    e.voice = null;
    this.stats.voicesReleased += 1;
  }

  private refreshStats(): void {
    const voices = this.liveVoices();
    this.stats.npcs = this.live.size;
    this.stats.peakNpcs = Math.max(this.stats.peakNpcs, this.live.size);
    this.stats.savedStates = this.saved.size;
    this.stats.emitters = this.emitters.size;
    this.stats.voices = voices;
    this.stats.peakVoices = Math.max(this.stats.peakVoices, voices);
  }
}

function npcEntity(spec: NpcSpec, state: NpcState): Entity {
  const persona: PersonaComponent = {
    traits: spec.persona.traits,
    voice: spec.persona.voice ?? null,
    memoryId: spec.persona.memoryId ?? null,
    goals: spec.persona.goals ?? [],
    mood: state.mood,
    ...(spec.persona.axes ? { axes: spec.persona.axes } : {}),
    ...(spec.persona.role ? { role: spec.persona.role } : {}),
    ...(spec.persona.faction ? { faction: spec.persona.faction } : {}),
  };
  const half = state.yaw / 2;
  return {
    id: spec.id,
    name: spec.name,
    components: {
      [Tag.Transform]: { position: state.position, rotation: [0, Math.sin(half), 0, Math.cos(half)], scale: [0.7, 1.8, 0.7], parent: null },
      [Tag.Mesh]: {
        geometry: { kind: "primitive", shape: "box" },
        material: { kind: "pbr", baseColor: spec.color, metallic: 0, roughness: 0.7 },
        visible: true,
      },
      [Tag.Collider]: { shape: "capsule", radius: NPC_RADIUS, halfHeight: NPC_HALF_HEIGHT },
      [Tag.Persona]: persona,
    },
  };
}
