/**
 * Maya — the canonical AI-companion primitive for AugEngine.
 *
 * AugEngine's brand promise is "the AI-native engine where NPCs actually
 * remember you." Maya is the showcase of that — a drop-in companion
 * with a real persona, real seeded memory, a dialogue script, and a
 * reaction system that mutates her mood from observation-bus events.
 *
 * Smoke-game has a project-local Maya in `examples/smoke-game/src/story/`;
 * this engine-level version exposes the same character for any game built
 * on AugEngine to spawn via `createMaya(scene)`. The local smoke-game copy
 * stays in place until it's migrated; the two are wire-compatible.
 *
 * Architectural promotion 2026-05-17: Maya is no longer "a smoke-game
 * NPC." She is a named primitive of the engine, the canonical AI-NPC
 * demo every AugEngine consumer can reuse.
 */

import { Tag } from "../scene/types.js";
import type {
  Entity,
  EntityId,
  PersonaAxes,
  PersonaComponent,
  PersonaRole,
  Vec3,
} from "../scene/types.js";
import type { Scene } from "../scene/scene.js";

/** Stable entity id for spawned Maya. Override via `createMaya(scene, { entityId })`. */
export const MAYA_DEFAULT_ENTITY_ID = "maya" as EntityId;

/** Brain memory namespace. Stable across saves + games. */
export const MAYA_MEMORY_NAMESPACE = "npc/maya";

/** Default voice ref — engine resolves to a TTS backend at runtime.
 *  `"claude-haiku-4-5"` matches the same hosted /api/chat path used by
 *  Dele's deleAi.js, so an existing AugEngine game can stream her lines
 *  without standing up new infra. */
export const MAYA_VOICE = "claude-haiku-4-5";

/** Reaction kinds Maya responds to via her observation hook. */
export type MayaReaction =
  | { kind: "player_aimed_at_her" }
  | { kind: "player_shot_at_her" }
  | { kind: "player_helped_her" }
  | { kind: "player_betrayed_her" }
  | { kind: "scene_changed"; sceneId: string }
  | { kind: "ally_killed"; allyId: EntityId }
  | { kind: "boss_seen"; bossId: EntityId };

/** Canonical Maya persona. Engine-stable across games. */
export const MAYA_PERSONA: {
  axes: PersonaAxes;
  traits: string[];
  biasTags: string[];
  mood: number;
  role: PersonaRole;
  faction: string;
} = {
  axes: { aggression: 0.18, fear: 0.40, loyalty: 0.85, curiosity: 0.72 },
  traits: ["curious", "guarded", "wry"],
  biasTags: ["betrayed-by-corp", "remembers-block-13-quarantine"],
  mood: 0.05,
  role: "ally",
  faction: "inmates",
};

/** Seeded background facts. Maya knows these on first apply. */
export const MAYA_SEED_MEMORIES: ReadonlyArray<{ fact: string; tags: ReadonlyArray<string> }> = [
  { fact: "You were a comms officer at Block 13 before the lockdown.", tags: ["backstory"] },
  { fact: "Three weeks ago, the player pulled you out of a quarantine sweep.", tags: ["backstory", "player"] },
  { fact: "You believe Warden Drev authorised the gas attack on Cell Bay 4.", tags: ["warden", "grudge"] },
  { fact: "The yard guards rotate on a 90-minute shift change. East-side crane has the longest blind spot.", tags: ["tactical", "yard"] },
];

/** Canonical dialogue script — short clauses, second-person, no exclamation points. */
export interface DialogueBeat {
  pattern: RegExp;
  line: string;
  raises?: string;
}

export interface MayaDialogueScript {
  /** First-line beats triggered on proximity-enter. Cycled round-robin. */
  intro: ReadonlyArray<string>;
  /** Keyword beats — first match wins. */
  beats: ReadonlyArray<DialogueBeat>;
  /** Fallback lines when nothing matched. Round-robin to avoid repetition. */
  seeded: ReadonlyArray<string>;
  /** Phase-change reactions. Keyed by phase id (game-specific). */
  onPhaseChange?: Partial<Record<string, string>>;
}

export const MAYA_DIALOGUE: MayaDialogueScript = {
  intro: [
    "You came back.",
    "I figured you'd find me. You always do.",
    "You're early. Or I'm slow. Either way — talk fast.",
  ],
  beats: [
    {
      pattern: /\b(warden|drev)\b/i,
      line: "Drev signed off on the Block 13 gas. I want him to look me in the face before he runs.",
      raises: "maya/named-warden",
    },
    {
      pattern: /\b(block ?13|cell ?bay ?4|quarantine)\b/i,
      line: "Cell Bay 4 was where they put the ones they wanted to forget. I was three doors down. I remember the singing.",
      raises: "maya/remembers-block-13",
    },
    {
      pattern: /\b(yard|crates?|patrol|guards?)\b/i,
      line: "Three guards on rotation. East-side crane has the longest blind spot. They go off shift at sundown.",
      raises: "maya/tactical-yard",
    },
    {
      pattern: /\b(tower|helipad|roof|escape)\b/i,
      line: "The helipad's his exit. Get to him before he gets to the chopper, this ends differently.",
      raises: "maya/pointed-at-helipad",
    },
    {
      pattern: /\b(plan|orders|what (now|next)|whats? the plan)\b/i,
      line: "Get out alive. Don't engage patrols if you don't have to. Find me at Tower 9 if I'm still walking.",
    },
    {
      pattern: /\b(trust|believe|why (should|would))/i,
      line: "You pulled me out of a sweep three weeks ago. I didn't ask. That's the only reason I'm talking.",
    },
    {
      pattern: /\b(scared|afraid|fear|worried)\b/i,
      line: "I stopped being scared somewhere around the second cell bay. Now I'm just tired. Tired's worse.",
    },
    {
      pattern: /\b(sorry|apologize|forgive)\b/i,
      line: "Apologise after. Not before. You haven't earned the discount yet.",
    },
    {
      pattern: /\b(remember|memory|forget)\b/i,
      line: "I keep what you tell me. That's the only useful thing about me right now.",
      raises: "maya/affirmed-memory",
    },
  ],
  seeded: [
    "Talk to me. Anything. The silence in here gets loud fast.",
    "If you've got a question, ask. If you don't, sit. Either way.",
    "I keep what you tell me. The brain doesn't lie, even when I want it to.",
    "I'm here. That's the part that matters.",
  ],
  onPhaseChange: {
    ally: "We're moving. Stay on my left — you're easier to cover that way.",
    tribunal: "He's up there. He won't come down. You have to go up.",
    retreat: "He's running. The helipad. He's running for the helipad.",
    finale: "It's over. Whatever it was. It's over.",
  },
};

/** Minimal brain shape Maya needs to remember + recall.
 *  Compatible with `@mnemopay/augengine`'s in-process brain AND with a remote
 *  MnemoPay client passed in by the consumer. */
export interface MayaBrain {
  remember(input: {
    namespace?: string;
    memoryId?: string;
    content: string;
    tags?: string[];
    metadata?: Record<string, unknown>;
  }): Promise<unknown>;
  recall(input: {
    namespace?: string;
    query?: string;
    limit?: number;
  }): Promise<ReadonlyArray<{ content: string; score?: number; tags?: string[] }>>;
  snapshot?(): ReadonlyArray<{ namespace: string; content: string }>;
}

export interface CreateMayaOptions {
  /** Where Maya spawns. Default [0, 0, 0]. */
  position?: Vec3;
  /** Override the entity id. Default MAYA_DEFAULT_ENTITY_ID. */
  entityId?: EntityId;
  /** Brain client to wire memory + recall through. Optional — without it
   *  she remembers in-memory only. */
  brain?: MayaBrain;
  /** Override the voice ref. Default MAYA_VOICE. */
  voice?: string | null;
  /** Skip seeding the background memories. Default false (do seed). */
  skipSeed?: boolean;
}

export interface MayaCompanion {
  /** Maya's spawned entity id. */
  readonly entityId: EntityId;
  /** Current mood, mutated by react(). Range -1..1. */
  mood(): number;
  /** Return a line in response to a player message. Uses keyword beats
   *  first, then a brain-recall-aware fallback, then a seeded line. */
  respond(playerMessage: string): Promise<{ line: string; raises?: string; source: "beat" | "recall" | "seeded" | "intro" }>;
  /** Return an intro line — round-robin. */
  introLine(): string;
  /** Return the phase-change reaction line for a given phase id. */
  phaseLine(phaseId: string): string | null;
  /** Retain a memory keyed to Maya's namespace. */
  remember(content: string, tags?: string[]): Promise<void>;
  /** Recall up to N memories matching a query. */
  recall(query: string, limit?: number): Promise<ReadonlyArray<{ content: string; tags?: string[] }>>;
  /** Push a reaction. Mutates mood, may schedule a memory write. */
  react(reaction: MayaReaction): Promise<void>;
  /** Raw seeded brain snapshot helper (useful in tests). */
  seedCount(): number;
}

/**
 * Spawn Maya into a Scene. Idempotent — calling twice on the same scene
 * with the same entityId returns the same MayaCompanion handle (the
 * second call's options are merged in).
 */
export async function createMaya(scene: Scene, opts: CreateMayaOptions = {}): Promise<MayaCompanion> {
  const entityId = opts.entityId ?? MAYA_DEFAULT_ENTITY_ID;
  const voice = opts.voice ?? MAYA_VOICE;

  if (!scene.has(entityId)) {
    const entity: Entity = {
      id: entityId,
      name: "Maya",
      components: {
        [Tag.Transform]: {
          position: opts.position ?? [0, 0, 0],
          rotation: [0, 0, 0, 1],
          scale: [1, 1, 1],
          parent: null,
        },
      },
    };
    scene.add(entity);
  }

  // Apply / merge persona.
  applyPersona(scene, entityId, voice);

  // Seed background memories if a brain is wired.
  let seeded = 0;
  if (opts.brain && !opts.skipSeed) {
    const existing = new Set(
      opts.brain.snapshot?.()
        ?.filter((r) => r.namespace === MAYA_MEMORY_NAMESPACE)
        .map((r) => r.content) ?? [],
    );
    for (const seed of MAYA_SEED_MEMORIES) {
      if (existing.has(seed.fact)) continue;
      await opts.brain.remember({
        namespace: MAYA_MEMORY_NAMESPACE,
        content: seed.fact,
        tags: [...seed.tags],
      });
      seeded++;
    }
  }

  let currentMood = MAYA_PERSONA.mood;
  let introIdx = 0;
  let seededIdx = 0;

  // Local memory writer — extracted so react() doesn't depend on `this`
  // (TS-inferred `this` on a Promise<MayaCompanion>-returning factory
  // collapses to the union, blocking method calls).
  async function rememberFact(content: string, tags?: string[]): Promise<void> {
    if (!opts.brain) return;
    const payload: Parameters<MayaBrain["remember"]>[0] = {
      namespace: MAYA_MEMORY_NAMESPACE,
      content,
    };
    if (tags && tags.length) payload.tags = tags;
    await opts.brain.remember(payload);
  }

  return {
    entityId,

    mood(): number { return currentMood; },

    introLine(): string {
      const line = MAYA_DIALOGUE.intro[introIdx % MAYA_DIALOGUE.intro.length]!;
      introIdx++;
      return line;
    },

    phaseLine(phaseId: string): string | null {
      return MAYA_DIALOGUE.onPhaseChange?.[phaseId] ?? null;
    },

    async respond(playerMessage) {
      // 1. Keyword beat match.
      const trimmed = playerMessage.trim();
      for (const beat of MAYA_DIALOGUE.beats) {
        if (beat.pattern.test(trimmed)) {
          const r: { line: string; raises?: string; source: "beat" | "recall" | "seeded" | "intro" } = {
            line: beat.line,
            source: "beat",
          };
          if (beat.raises) r.raises = beat.raises;
          return r;
        }
      }

      // 2. Brain recall — if present, use the top memory to scaffold a line.
      if (opts.brain) {
        const memories = await opts.brain.recall({
          namespace: MAYA_MEMORY_NAMESPACE,
          query: trimmed,
          limit: 2,
        });
        const top = memories[0];
        if (top) {
          return {
            line: `${top.content} You already told me that.`,
            source: "recall",
          };
        }
      }

      // 3. Seeded fallback.
      const line = MAYA_DIALOGUE.seeded[seededIdx % MAYA_DIALOGUE.seeded.length]!;
      seededIdx++;
      return { line, source: "seeded" };
    },

    async remember(content: string, tags?: string[]): Promise<void> {
      await rememberFact(content, tags);
    },

    async recall(query: string, limit = 5) {
      if (!opts.brain) return [];
      const rows = await opts.brain.recall({
        namespace: MAYA_MEMORY_NAMESPACE,
        query,
        limit,
      });
      return rows.map((r) => {
        const out: { content: string; tags?: string[] } = { content: r.content };
        if (r.tags && r.tags.length) out.tags = r.tags;
        return out;
      });
    },

    async react(reaction) {
      switch (reaction.kind) {
        case "player_aimed_at_her":
          currentMood = clamp(currentMood - 0.15, -1, 1);
          return;
        case "player_shot_at_her":
          currentMood = clamp(currentMood - 0.6, -1, 1);
          await rememberFact("The player aimed a weapon at you. You haven't forgotten.", ["grudge", "player"]);
          return;
        case "player_helped_her":
          currentMood = clamp(currentMood + 0.2, -1, 1);
          await rememberFact("The player helped you when they didn't have to.", ["debt", "player"]);
          return;
        case "player_betrayed_her":
          currentMood = clamp(currentMood - 0.8, -1, 1);
          await rememberFact("The player betrayed you. Trust resets to zero.", ["grudge", "player"]);
          return;
        case "scene_changed":
          await rememberFact(`The party moved to ${reaction.sceneId}.`, ["movement"]);
          return;
        case "ally_killed":
          currentMood = clamp(currentMood - 0.25, -1, 1);
          await rememberFact(`${reaction.allyId} did not survive.`, ["loss"]);
          return;
        case "boss_seen":
          currentMood = clamp(currentMood - 0.1, -1, 1);
          await rememberFact(`You saw ${reaction.bossId}. The hair on your neck went up.`, ["tactical", "warden"]);
          return;
        default: {
          const _: never = reaction;
          void _;
        }
      }
    },

    seedCount(): number { return seeded; },
  };
}

function applyPersona(scene: Scene, entityId: EntityId, voice: string | null): PersonaComponent | null {
  const entity = scene.get(entityId);
  if (!entity) return null;
  const existing = entity.components[Tag.Persona] as PersonaComponent | undefined;
  const persona: PersonaComponent = {
    traits: unique([...(existing?.traits ?? []), ...MAYA_PERSONA.traits]),
    voice: existing?.voice ?? voice,
    memoryId: existing?.memoryId ?? MAYA_MEMORY_NAMESPACE,
    goals: existing?.goals ?? [],
    mood: existing?.mood ?? MAYA_PERSONA.mood,
    axes: MAYA_PERSONA.axes,
    biasTags: unique([...(existing?.biasTags ?? []), ...MAYA_PERSONA.biasTags]),
    role: MAYA_PERSONA.role,
    faction: MAYA_PERSONA.faction,
  };
  entity.components[Tag.Persona] = persona;
  return persona;
}

function unique<T>(arr: ReadonlyArray<T>): T[] {
  const out: T[] = [];
  const seen = new Set<T>();
  for (const v of arr) {
    if (!seen.has(v)) {
      seen.add(v);
      out.push(v);
    }
  }
  return out;
}

function clamp(v: number, min: number, max: number): number {
  return v < min ? min : v > max ? max : v;
}
