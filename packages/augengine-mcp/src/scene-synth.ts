/**
 * Deterministic prompt -> SceneSpec synthesis.
 *
 * The engine's `AITask` union declares a `scene_gen` task, but nothing
 * implements a provider for it. That turns out not to matter here: over
 * MCP, the *caller is the language model*. Claude Desktop / Cursor / Claude
 * Code have already read the user's intent by the time they call us. This
 * synthesiser's job is to turn that intent into a valid, renderable
 * SceneSpec the agent can then refine with `add_entity` / `attach_persona`
 * — not to re-run an LLM behind the agent's back.
 *
 * Consequences worth keeping: no API key, no network, no credit burn, and
 * the same (prompt, seed) always yields a byte-identical spec. That last
 * property is what makes `create_scene`'s `cacheHit` flag mean something.
 *
 * Layout conventions (sun, sky dome, floor, placeholder cylinders for
 * characters) follow `@mnemopay/augengine-arena-alpha`'s scene synthesiser
 * so generated scenes load through the same `WebGpuLiveRenderer` pipeline
 * without surprises.
 */

import { createHash } from "node:crypto";
import {
  Tag,
  type Entity,
  type EntityId,
  type SceneSpec,
  type Vec3,
} from "@mnemopay/augengine";

export interface SynthOptions {
  prompt: string;
  /** Omit for a seed derived from the prompt — keeps identical prompts identical. */
  seed?: number;
}

export interface SynthResult {
  spec: SceneSpec;
  promptHash: string;
}

/** Deterministic 32-bit PRNG (mulberry32). */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function hashPrompt(prompt: string): string {
  return createHash("sha256").update(prompt).digest("hex");
}

function seedFromHash(hash: string): number {
  return parseInt(hash.slice(0, 8), 16) >>> 0;
}

/** Nouns that become persona-bearing characters. */
const CHARACTER_WORDS = [
  "npc", "npcs", "person", "people", "character", "characters", "figure",
  "guard", "guards", "merchant", "merchants", "trader", "villager", "villagers",
  "knight", "knights", "soldier", "soldiers", "robot", "robots", "droid",
  "child", "children", "kid", "woman", "women", "man", "men", "stranger",
  "strangers", "monk", "monks", "elder", "vendor", "dancer", "musician",
];

interface PropDef {
  shape: "box" | "sphere" | "cylinder";
  scale: Vec3;
  color: Vec3;
}

/** Nouns that become static props. */
const PROP_WORDS: Record<string, PropDef> = {
  fountain: { shape: "cylinder", scale: [1.6, 0.6, 1.6], color: [0.62, 0.64, 0.66] },
  tree: { shape: "cylinder", scale: [0.3, 2.6, 0.3], color: [0.28, 0.2, 0.12] },
  crate: { shape: "box", scale: [0.8, 0.8, 0.8], color: [0.42, 0.32, 0.2] },
  barrel: { shape: "cylinder", scale: [0.45, 0.6, 0.45], color: [0.38, 0.26, 0.16] },
  rock: { shape: "sphere", scale: [0.7, 0.55, 0.7], color: [0.4, 0.4, 0.42] },
  table: { shape: "box", scale: [1.2, 0.4, 0.8], color: [0.45, 0.33, 0.21] },
  chair: { shape: "box", scale: [0.4, 0.5, 0.4], color: [0.4, 0.3, 0.2] },
  lamp: { shape: "cylinder", scale: [0.12, 1.8, 0.12], color: [0.24, 0.24, 0.26] },
  pillar: { shape: "cylinder", scale: [0.45, 3.2, 0.45], color: [0.7, 0.68, 0.62] },
  statue: { shape: "box", scale: [0.6, 2.0, 0.6], color: [0.58, 0.56, 0.5] },
  bench: { shape: "box", scale: [1.4, 0.3, 0.5], color: [0.44, 0.32, 0.2] },
  car: { shape: "box", scale: [1.8, 0.8, 4.0], color: [0.72, 0.18, 0.16] },
  stall: { shape: "box", scale: [1.6, 1.2, 1.2], color: [0.55, 0.4, 0.25] },
};

const NUMBER_WORDS: Record<string, number> = {
  a: 1, an: 1, one: 1, two: 2, three: 3, four: 4, five: 5,
  six: 6, seven: 7, eight: 8, nine: 9, ten: 10,
  several: 3, few: 3, some: 3, many: 6, crowd: 8,
};

/** Adjectives lifted into persona traits when present. */
const TRAIT_WORDS = [
  "angry", "arguing", "calm", "cheerful", "nervous", "hostile", "friendly",
  "tired", "drunk", "suspicious", "curious", "bored", "afraid", "proud",
  "quiet", "loud", "wounded", "young", "old", "grieving", "excited",
];

interface EnvironmentTuning {
  lightColor: Vec3;
  lightIntensity: number;
  skyColor: Vec3;
  floorColor: Vec3;
}

function tuneEnvironment(text: string): EnvironmentTuning {
  if (/\b(night|midnight|dark|moonlit)\b/.test(text)) {
    return {
      lightColor: [0.55, 0.6, 0.85],
      lightIntensity: 0.35,
      skyColor: [0.06, 0.07, 0.12],
      floorColor: [0.13, 0.13, 0.16],
    };
  }
  if (/\b(dusk|sunset|evening)\b/.test(text)) {
    return {
      lightColor: [1.0, 0.72, 0.45],
      lightIntensity: 0.95,
      skyColor: [0.48, 0.32, 0.3],
      floorColor: [0.26, 0.22, 0.19],
    };
  }
  if (/\b(cave|dungeon|basement|tunnel|indoor|interior|warehouse)\b/.test(text)) {
    return {
      lightColor: [0.78, 0.76, 0.7],
      lightIntensity: 0.55,
      skyColor: [0.08, 0.08, 0.09],
      floorColor: [0.2, 0.19, 0.18],
    };
  }
  if (/\b(desert|dune|sand)\b/.test(text)) {
    return {
      lightColor: [1.0, 0.96, 0.82],
      lightIntensity: 1.7,
      skyColor: [0.66, 0.76, 0.88],
      floorColor: [0.72, 0.62, 0.42],
    };
  }
  if (/\b(forest|wood|jungle|grove)\b/.test(text)) {
    return {
      lightColor: [0.92, 1.0, 0.88],
      lightIntensity: 1.0,
      skyColor: [0.42, 0.56, 0.44],
      floorColor: [0.18, 0.26, 0.15],
    };
  }
  return {
    lightColor: [1.0, 0.97, 0.9],
    lightIntensity: 1.3,
    skyColor: [0.52, 0.62, 0.76],
    floorColor: [0.28, 0.28, 0.27],
  };
}

function tokenize(prompt: string): string[] {
  return prompt
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, " ")
    .split(/\s+/)
    .filter(Boolean);
}

/** Look backwards from a noun for a count ("two npcs", "3 guards"). */
function countBefore(tokens: string[], idx: number): number {
  for (let back = 1; back <= 3 && idx - back >= 0; back++) {
    const t = tokens[idx - back];
    if (t === undefined) continue;
    const digits = Number(t);
    if (Number.isInteger(digits) && digits > 0 && digits <= 64) return digits;
    const word = NUMBER_WORDS[t];
    if (word !== undefined) return word;
  }
  return 1;
}

function singular(word: string): string {
  if (word.endsWith("ies")) return `${word.slice(0, -3)}y`;
  if (word.endsWith("es") && word.length > 4) return word.slice(0, -2);
  if (word.endsWith("s") && !word.endsWith("ss")) return word.slice(0, -1);
  return word;
}

const eid = (s: string): EntityId => s as EntityId;

function titleCase(word: string): string {
  return `${word.charAt(0).toUpperCase()}${word.slice(1)}`;
}

export function synthesiseFromPrompt(opts: SynthOptions): SynthResult {
  const prompt = opts.prompt;
  const promptHash = hashPrompt(prompt);
  const seed = opts.seed ?? seedFromHash(promptHash);
  const rand = rng(seed);
  const lower = prompt.toLowerCase();
  const tokens = tokenize(prompt);
  const tune = tuneEnvironment(lower);

  const traits = TRAIT_WORDS.filter((t) => lower.includes(t));
  const entities: Entity[] = [];

  // Sun.
  entities.push({
    id: eid("light_main"),
    name: "Sun",
    components: {
      [Tag.Transform]: {
        position: [0, 50, 0],
        rotation: [0, 0, 0, 1],
        scale: [1, 1, 1],
        parent: null,
      },
      [Tag.Light]: {
        kind: "directional",
        color: tune.lightColor,
        intensity: tune.lightIntensity,
      },
    },
  });

  // Sky dome — inverted sphere, the same trick arena-alpha uses.
  entities.push({
    id: eid("sky_dome"),
    name: "Sky Dome",
    components: {
      [Tag.Transform]: {
        position: [0, 0, 0],
        rotation: [0, 0, 0, 1],
        scale: [-200, -200, -200],
        parent: null,
      },
      [Tag.Mesh]: {
        geometry: { kind: "primitive", shape: "sphere" },
        material: { kind: "unlit", color: tune.skyColor },
        visible: true,
      },
    },
  });

  // Ground plane.
  entities.push({
    id: eid("floor"),
    name: "Ground",
    components: {
      [Tag.Transform]: {
        position: [0, 0, 0],
        rotation: [0, 0, 0, 1],
        scale: [40, 1, 40],
        parent: null,
      },
      [Tag.Mesh]: {
        geometry: { kind: "primitive", shape: "plane" },
        material: {
          kind: "pbr",
          baseColor: tune.floorColor,
          metallic: 0.0,
          roughness: 0.92,
        },
        visible: true,
      },
      [Tag.Collider]: { shape: "box", extents: [20, 0.1, 20] },
    },
  });

  // Camera — far enough back to frame the whole ring.
  entities.push({
    id: eid("camera_main"),
    name: "Main Camera",
    components: {
      [Tag.Transform]: {
        position: [0, 1.7, 9],
        rotation: [0, 0, 0, 1],
        scale: [1, 1, 1],
        parent: null,
      },
      [Tag.Camera]: { fovY: (65 * Math.PI) / 180, near: 0.1, far: 1000, active: true },
    },
  });

  // Props first — they anchor the composition; characters arrange around them.
  const propPositions: Vec3[] = [];
  const seenProps = new Set<string>();
  tokens.forEach((tok, idx) => {
    const key = tok in PROP_WORDS ? tok : singular(tok);
    const def = PROP_WORDS[key];
    if (!def || seenProps.has(key)) return;
    seenProps.add(key);
    const count = Math.min(countBefore(tokens, idx), 12);
    for (let i = 0; i < count; i++) {
      const isLoneAnchor = count === 1 && seenProps.size === 1;
      const angle = rand() * Math.PI * 2;
      const radius = isLoneAnchor ? 0 : 2.5 + rand() * 5;
      const pos: Vec3 = [
        Number((Math.cos(angle) * radius).toFixed(3)),
        def.scale[1] / 2,
        Number((Math.sin(angle) * radius).toFixed(3)),
      ];
      propPositions.push(pos);
      entities.push({
        id: eid(`prop_${key}_${i}`),
        name: `${titleCase(key)} ${i + 1}`,
        components: {
          [Tag.Transform]: {
            position: pos,
            rotation: [0, 0, 0, 1],
            scale: def.scale,
            parent: null,
          },
          [Tag.Mesh]: {
            geometry: { kind: "primitive", shape: def.shape },
            material: {
              kind: "pbr",
              baseColor: def.color,
              metallic: 0.0,
              roughness: 0.8,
            },
            visible: true,
          },
          [Tag.Collider]: {
            shape: "box",
            extents: [def.scale[0] / 2, def.scale[1] / 2, def.scale[2] / 2],
          },
          [Tag.AffordanceSet]: { affordances: ["look", "use"] },
        },
      });
    }
  });

  // Characters — persona-bearing, ringed around the first prop.
  const anchor: Vec3 = propPositions[0] ?? [0, 0, 0];
  let charIndex = 0;
  const seenChars = new Set<string>();
  tokens.forEach((tok, idx) => {
    if (!CHARACTER_WORDS.includes(tok)) return;
    const key = singular(tok);
    if (seenChars.has(key)) return;
    seenChars.add(key);
    const count = Math.min(countBefore(tokens, idx), 24);
    for (let i = 0; i < count; i++) {
      const angle = (charIndex / Math.max(count, 2)) * Math.PI * 2 + rand() * 0.4;
      const radius = 2.0 + rand() * 1.5;
      const pos: Vec3 = [
        Number((anchor[0] + Math.cos(angle) * radius).toFixed(3)),
        1.05,
        Number((anchor[2] + Math.sin(angle) * radius).toFixed(3)),
      ];
      entities.push({
        id: eid(`npc_${key}_${i}`),
        name: `${titleCase(key)} ${i + 1}`,
        components: {
          [Tag.Transform]: {
            position: pos,
            rotation: [0, 0, 0, 1],
            scale: [0.34, 0.55, 0.22],
            parent: null,
          },
          [Tag.Mesh]: {
            geometry: { kind: "primitive", shape: "cylinder" },
            material: {
              kind: "pbr",
              baseColor: [
                Number((0.35 + rand() * 0.4).toFixed(3)),
                Number((0.35 + rand() * 0.3).toFixed(3)),
                Number((0.35 + rand() * 0.3).toFixed(3)),
              ],
              metallic: 0.05,
              roughness: 0.45,
            },
            visible: true,
          },
          [Tag.Collider]: { shape: "capsule", radius: 0.3, halfHeight: 0.9 },
          [Tag.Persona]: {
            traits: traits.length > 0 ? [...traits] : ["neutral"],
            voice: null,
            memoryId: null,
            goals: [],
            mood: 0,
          },
          [Tag.Agent]: {
            channels: ["Visual", "Dialogue", "Social"],
            viewFilter: null,
            permittedVerbs: ["say", "move", "look", "emote"],
          },
          [Tag.AffordanceSet]: { affordances: ["talk", "look"] },
        },
      });
      charIndex++;
    }
  });

  const spec: SceneSpec = {
    schemaVersion: 1,
    source: { kind: "generated", prompt, promptHash, cached: false },
    entities,
    defaultCamera: eid("camera_main"),
    environment: { kind: "color", color: tune.skyColor },
  };

  return { spec, promptHash };
}
