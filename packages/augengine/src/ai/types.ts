/**
 * AugEngine AI subsystem types — implements the seven AI-native design decisions
 * from AUGENGINE-ARCHITECTURE.md.
 *
 * Decisions encoded here:
 *   #1 — AIRequest: async, frame-budgeted, fallback-always
 *   #3 — PromptTemplate: every authored line is a template
 *   #4 — AIArtifactRef: content-addressable + manifest separated from save state
 *   #5 — Action: unified grammar; player + BT + agent all emit these
 *   #7 — ObservationBus: typed pub/sub; bots subscribe to the same channels as cameras
 */

import type { EntityId, ObservationChannel, Vec3 } from "../scene/types.js";

/** Decision #1 — every AI call goes through this abstraction. */
export interface AIRequest<T> {
  task: AITask;
  payload: unknown;
  /** Hard deadline. If exceeded, fallback fires and the AI result is dropped. */
  timeoutMs: number;
  /** 1 = critical (frame-blocking allowed off-thread), 3 = background. */
  priority: 1 | 2 | 3;
  /** Deterministic, instant, always defined. */
  fallback: () => T;
  /** Optional content-addressable cache key to short-circuit repeats. */
  cacheKey?: string;
}

export type AITask =
  | "dialogue"          // single-turn NPC reply
  | "ambient"           // background generation (barks, flavor text)
  | "action_plan"       // agent → list of Actions
  | "scene_gen"         // prompt → SceneSpec JSON
  | "asset_gen"         // prompt → 3D model / world / image / audio
  | "lip_sync"          // audio → blendshapes (premium path: Audio2Face-3D)
  | "stt"               // audio → text
  | "tts";              // text → audio

/** Decision #3 — every authored line is a template. Static lines have no slots. */
export interface PromptTemplate {
  /** Path of the `.prompt` file relative to the scene's authoring root. */
  path: string;
  /** The template body with `{{slot}}` placeholders. */
  body: string;
  /** Few-shot examples (input/output pairs) the runtime injects. */
  examples?: PromptExample[];
  /** Slot names the renderer expects to be filled. */
  slots: string[];
  /** Test block: list of (input, expected) pairs CI runs against a frozen LLM. */
  expectedOutputs?: PromptExample[];
}

export interface PromptExample {
  input: Record<string, string>;
  output: string;
}

/** Decision #4 — AI artifacts content-addressed; save state references hashes. */
export interface AIArtifactRef {
  /** Content-addressable hash: sha256 of (personaId, promptHash, worldStateHash). */
  hash: string;
  kind: "text" | "image" | "audio" | "scene" | "model";
  /** Storage URL or in-memory blob ref. */
  url: string;
  /** When the artifact was first generated. */
  generatedAt: string;
}

export interface AIArtifactManifest {
  artifacts: AIArtifactRef[];
  /** Deterministic state hash this manifest was captured against. */
  worldStateHash: string;
}

/** Decision #5 — unified action grammar; one IR for player + BT + agent. */
export interface Action {
  verb: string;
  actor: EntityId;
  target?: EntityId | Vec3;
  params: Record<string, unknown>;
  /** Predicate IDs that must be true before dispatch. */
  preconditions: string[];
  /** Effect IDs applied after successful dispatch. */
  effects: string[];
  /** Where this action originated — informational, not load-bearing. */
  source: ActionSource;
}

export type ActionSource =
  | { kind: "player"; deviceId: string }
  | { kind: "behavior_tree"; nodeId: string }
  | { kind: "ai_agent"; entity: EntityId; promptHash: string }
  | { kind: "script"; scriptId: string };

export interface ActionResult {
  ok: boolean;
  /** Effect IDs actually applied. */
  appliedEffects: string[];
  /** Reason for rejection if ok=false. */
  reason?: string;
}

/** Decision #7 — typed pub/sub bus shared by cameras + speakers + agents. */
export interface ObservationEvent {
  channel: ObservationChannel;
  ts: string;
  /** Payload shape varies by channel — agents and renderers narrow as needed. */
  payload: unknown;
  /** Source entity that produced the event. */
  source: EntityId | null;
  /** Spatial origin if applicable, for view-filter math. */
  origin?: Vec3;
}

export interface Unsubscribe {
  (): void;
}

export interface ObservationBus {
  publish(event: ObservationEvent): void;
  subscribe(
    channel: ObservationChannel,
    filter: ViewFilter | null,
    cb: (event: ObservationEvent) => void,
  ): Unsubscribe;
}

/** A view-filter is a serializable predicate evaluated by the bus. */
export interface ViewFilter {
  /** Maximum range from observer entity (0 = unlimited). */
  rangeMax?: number;
  /** Required line-of-sight from observer position. */
  requiresLineOfSight?: boolean;
  /** Allowed source-entity tags. */
  allowedSourceTags?: string[];
}
