/**
 * AugEngine scene-graph types — the single source of truth shared by both
 * renderers (live + render) and every subsystem (characters, audio, AI).
 *
 * Per AUGENGINE-ARCHITECTURE.md decision #6: one scene loader, three sources
 * (Authored | Procgen(seed) | Generated(prompt)). The Scene structure is
 * identical regardless of which source produced it.
 */

export type EntityId = string & { readonly __brand: "EntityId" };

export const Tag = {
  Transform: "transform",
  Mesh: "mesh",
  Light: "light",
  Camera: "camera",
  Skin: "skin",
  AnimationMixer: "animation_mixer",
  RigidBody: "rigid_body",
  Reactive: "reactive",
  Collider: "collider",
  Audio: "audio",
  Persona: "persona", // Decision #2 — every entity *can* have one
  Agent: "agent",     // Decision #7 — bots use the observation bus
  Script: "script",
  AffordanceSet: "affordance_set",
  Wallet: "wallet",
  TradeIntent: "trade_intent",
} as const;

export type ComponentTag = typeof Tag[keyof typeof Tag];

export type Vec3 = readonly [number, number, number];
export type Quat = readonly [number, number, number, number]; // x, y, z, w
export type Mat4 = Float32Array; // 16 floats, column-major

export interface TransformComponent {
  position: Vec3;
  rotation: Quat;
  scale: Vec3;
  parent: EntityId | null;
}

export interface MeshComponent {
  geometry: GeometryRef;
  material: MaterialRef;
  visible: boolean;
}

export type GeometryRef =
  | { kind: "primitive"; shape: "box" | "sphere" | "plane" | "cylinder"; params?: Record<string, number> }
  | { kind: "gltf"; url: string; primitiveIndex?: number }
  | { kind: "procedural"; seed: number; generator: string };

export type MaterialRef =
  | { kind: "unlit"; color: Vec3; opacity?: number }
  | { kind: "pbr"; baseColor: Vec3; metallic: number; roughness: number; emissive?: Vec3; baseColorTexture?: string; metallicRoughnessTexture?: string; normalTexture?: string }
  | { kind: "vrm-mtoon"; configRef: string }
  | { kind: "splat"; spzUrl: string };

export interface CameraComponent {
  fovY: number;
  near: number;
  far: number;
  active: boolean;
}

export interface LightComponent {
  kind: "directional" | "point" | "spot" | "ambient";
  color: Vec3;
  intensity: number;
  // For spot/point:
  range?: number;
  // For spot:
  innerConeRad?: number;
  outerConeRad?: number;
}

export interface SkinComponent {
  jointEntities: EntityId[];
  inverseBindMatrices: Float32Array; // 16 floats per joint, packed
  rootEntity: EntityId;
}

export interface AnimationMixerComponent {
  clips: AnimationClipRef[];
  active: { clipId: string; time: number; weight: number; loop: boolean }[];
  retargetSourceRig?: string; // optional: source rig identifier for runtime retargeting
}

export interface RigidBodyComponent {
  velocity: Vec3;
  angularVelocity: Vec3;
  mass: number;
  isKinematic: boolean;
  restitution?: number;
  friction?: number;
  /** Continuous collision detection defaults to true for dynamic bodies. */
  ccd?: boolean;
  linearDamping?: number;
  angularDamping?: number;
}

/**
 * Collision shape. `radius`, `halfHeight`, `extents` and `centerOffset` are
 * WORLD-SPACE sizes: the entity's transform scale is visual only and is not
 * applied (a 2.5 x 3 x 6 container is `scale: [2.5, 3, 6]`,
 * `extents: [1.25, 1.5, 3]`). PhysicsSystem, the character controller,
 * raycast and nav all read them this way. Only `vertices` are local and
 * multiplied by scale.
 */
export interface ColliderComponent {
  shape: "sphere" | "capsule" | "box" | "mesh" | "heightfield";
  radius?: number;
  halfHeight?: number;
  extents?: Vec3;
  centerOffset?: Vec3;
  /** Explicit local triangle geometry for static mesh collision. */
  vertices?: number[];
  indices?: number[];
  /** Static terrain grid (shape "heightfield"). */
  heightfield?: HeightfieldShape;
  /**
   * Sensor: detected by `PhysicsSystem.overlapShape` but produces no contact
   * forces, and the character controller and `raycast` ignore it. Used for
   * combat hurtboxes and trigger volumes.
   */
  sensor?: boolean;
}

/**
 * A regular height grid centred on the entity's transform position. Heights
 * are offsets from the transform's y (scale and rotation are not applied); the
 * grid spans `sizeX` x `sizeZ` metres in world space. Each cell is split along the
 * (x1,z0)–(x0,z1) diagonal — Rapier's split, and the triangles the terrain
 * mesh draws.
 */
export interface HeightfieldShape {
  /** Sample count along x (cells + 1). */
  samplesX: number;
  /** Sample count along z (cells + 1). */
  samplesZ: number;
  /** Row-major heights: index = j * samplesX + i, i along x, j along z. */
  heights: ArrayLike<number>;
  sizeX: number;
  sizeZ: number;
}

export interface AnimationClipRef {
  id: string;
  url: string; // glTF / VRMA / .bvh / .vmd
  fpsHint?: number;
}

export interface AudioComponent {
  source: AudioSourceRef;
  volume: number;
  spatial: boolean;
  loop: boolean;
}

export type AudioSourceRef =
  | { kind: "buffer"; url: string }
  | { kind: "stream"; url: string }
  | { kind: "tts"; text: string; voiceRef: string }
  | { kind: "synth"; oscillator: "sine" | "saw" | "square"; freq: number };

/** Decision #2 — Persona is a component, optional on any entity. */
export interface PersonaComponent {
  /** Free-form trait list, serialized into LLM prompts. */
  traits: string[];
  /** Voice ref for TTS (engine resolves to a backend at runtime). */
  voice: string | null;
  /** MnemoPay memory key — portable across blackpig titles. */
  memoryId: string | null;
  /** Current goals, evaluated by the agent system. */
  goals: PersonaGoal[];
  /** -1..1, drives animation/voice modulation. */
  mood: number;
  /**
   * Optional psychometric axes — when present, the scheduler uses these
   * to gate state transitions instead of inferring from traits + mood.
   * All ∈ [0, 1]. Backward-compatible: absent ⇒ the existing `traits`
   * + `mood` heuristic runs unchanged.
   *
   * Lifted from the 2026-05-13 starter-pack `AICharacter.persona` shape
   * after a third-party design pass. See
   * `project_augengine_aaa_starter_review_2026_05_13.md`.
   */
  axes?: PersonaAxes;
  /**
   * Optional bias tags — narrative-side labels the LLM mixes into prompts
   * (e.g. `"hates-mercenaries"`, `"sister-of-yelena"`). Distinct from
   * `traits` because tags are world-state biases the brain memory can
   * mutate over time; traits are baseline personality.
   */
  biasTags?: string[];
  /**
   * Optional NPC role label — convention from the starter pack. Drives
   * scheduler defaults (story-critical roles get bumped to higher
   * think-tiers without explicit `storyCriticalIds`).
   */
  role?: PersonaRole;
  /**
   * Optional faction id — used by the scheduler for friend/foe scoring
   * and by the action grammar to gate verbs across factional lines.
   */
  faction?: string;
}

/** Psychometric axes — all ∈ [0, 1]. */
export interface PersonaAxes {
  /** Likelihood of escalating to direct confrontation. */
  aggression: number;
  /** Likelihood of retreating under threat. Inverse of bravery. */
  fear: number;
  /** Resistance to switching allegiances. */
  loyalty: number;
  /** Likelihood of investigating unfamiliar stimuli. */
  curiosity: number;
}

/**
 * Standard NPC role taxonomy. Free-form string allowed for game-specific
 * roles, but the canonical set drives default scheduler priorities.
 */
export type PersonaRole =
  | "infected"
  | "flanker"
  | "brute"
  | "civilian"
  | "ally"
  | "boss"
  | string;

export interface PersonaGoal {
  id: string;
  description: string;
  priority: number;
  /** Predicate evaluated against world state — done when true. */
  done: string;
}

/** Decision #7 — Agent component subscribes to observation bus channels. */
export interface AgentComponent {
  channels: ObservationChannel[];
  /** Optional view-filter predicate — limits what the agent perceives. */
  viewFilter: string | null;
  /** Action grammar verbs this agent is allowed to emit. */
  permittedVerbs: string[];
}

export type ObservationChannel =
  | "Visual"
  | "Audible"
  | "Social"
  | "Inventory"
  | "Combat"
  | "Dialogue";

export interface ScriptComponent {
  /** Prompt template path (`.prompt` file) per decision #3, or static script ID. */
  path: string;
  /** Slot fillers resolved at runtime against world state + persona. */
  slots: Record<string, string>;
}

export interface AffordanceSetComponent {
  /** Verbs an actor can apply to this entity, e.g. ["talk", "trade", "attack"]. */
  affordances: string[];
}

/** A scene is a flat list of entities with components keyed by tag. */
export interface Entity {
  id: EntityId;
  name: string;
  components: Partial<{
    [Tag.Transform]: TransformComponent;
    [Tag.Mesh]: MeshComponent;
    [Tag.Camera]: CameraComponent;
    [Tag.Light]: LightComponent;
    [Tag.Skin]: SkinComponent;
    [Tag.AnimationMixer]: AnimationMixerComponent;
    [Tag.RigidBody]: RigidBodyComponent;
    [Tag.Reactive]: import("../chemistry/types.js").ReactiveComponent;
    [Tag.Collider]: ColliderComponent;
    [Tag.Audio]: AudioComponent;
    [Tag.Persona]: PersonaComponent;
    [Tag.Agent]: AgentComponent;
    [Tag.Script]: ScriptComponent;
    [Tag.AffordanceSet]: AffordanceSetComponent;
    [Tag.Wallet]: any; // imported from augengine-economy
    [Tag.TradeIntent]: any; // imported from augengine-economy
  }>;
}

/** Decision #6 — one loader, three sources, identical downstream. */
export type SceneSource =
  | { kind: "authored"; path: string }
  | { kind: "procgen"; seed: number; generator: string }
  | { kind: "generated"; prompt: string; promptHash: string; cached: boolean };

export interface SceneSpec {
  schemaVersion: 1;
  source: SceneSource;
  entities: Entity[];
  /** Default active camera entity. */
  defaultCamera?: EntityId;
  /** Background fill / skybox / environment. */
  environment?: EnvironmentRef;
}

export type EnvironmentRef =
  | { kind: "color"; color: Vec3 }
  | { kind: "skybox"; cubemapUrl: string }
  | { kind: "ibl"; hdrUrl: string }
  | { kind: "splat-world"; spzUrl: string }; // World Labs Marble / HY-World output
