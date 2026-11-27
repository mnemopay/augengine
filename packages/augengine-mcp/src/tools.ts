/**
 * AugEngine MCP tool surface — typed contracts for every tool an MCP-aware agent
 * can invoke against a running AugEngine instance.
 *
 * Implementation is deliberately deferred to the engine implementation phase
 * (AUGENGINE-ARCHITECTURE.md phase 4). This file pins the surface so the MCP
 * server's behavior is documented before any code runs.
 */

import type {
  Action,
  ActionResult,
  ObservationChannel,
  ObservationEvent,
} from "@mnemopay/augengine";
import type { Entity, EntityId, SceneSpec } from "@mnemopay/augengine";

export interface AugEngineMcpTools {
  /**
   * Scene authoring — convert a natural-language prompt to a SceneSpec via
   * the AI scheduler's scene_gen task. Returns the generated spec plus the
   * cache key so subsequent identical prompts hit the artifact cache.
   */
  "augengine.create_scene": {
    input: { prompt: string; seed?: number };
    output: { sceneId: string; spec: SceneSpec; cacheHit: boolean };
  };

  /** Add an entity to an existing scene. */
  "augengine.add_entity": {
    input: { sceneId: string; entity: Entity };
    output: { ok: true; entityId: EntityId } | { ok: false; reason: string };
  };

  /** Attach a Persona component to an entity (decision #2). */
  "augengine.attach_persona": {
    input: {
      entityId: EntityId;
      traits: string[];
      voice?: string;
      memoryId?: string;
    };
    output: { ok: true } | { ok: false; reason: string };
  };

  /** Read entities matching an optional filter. */
  "augengine.list_entities": {
    input: {
      sceneId: string;
      filter?: { tags?: string[]; namePattern?: string };
    };
    output: { entities: Entity[] };
  };

  /** Dispatch a unified-grammar Action through the engine's action pipeline (decision #5). */
  "augengine.dispatch_action": {
    input: { sceneId: string; action: Action };
    output: ActionResult;
  };

  /**
   * Subscribe to an observation channel (decision #7). Returns an SSE
   * endpoint URL the caller can stream from. The MCP transport itself
   * handles streaming via its own mechanism; the SSE URL is for raw clients.
   */
  "augengine.subscribe_observation": {
    input: {
      sceneId: string;
      channel: ObservationChannel;
      filter?: { rangeMax?: number; allowedSourceTags?: string[] };
    };
    output: { sseUrl: string; subscriptionId: string };
  };

  /** Drain queued observation events as a one-shot poll (for non-streaming clients). */
  "augengine.poll_observation": {
    input: { subscriptionId: string; maxEvents?: number };
    output: { events: ObservationEvent[]; remaining: number };
  };

  /** Render a clip from the scene through the render runtime. */
  "augengine.render_clip": {
    input: {
      sceneId: string;
      fps: number;
      frames: number;
      seed: number;
      codec?: "h264" | "hevc" | "av1";
      width?: number;
      height?: number;
    };
    output: { mp4Url: string; bytes: number; durationMs: number };
  };

  /**
   * Export the current scene to a Godot 4.4 project tree via @blackpig/game-assets.
   * Path forward for native-console targets that the browser-native runtime
   * doesn't reach.
   */
  "augengine.export_godot": {
    input: { sceneId: string };
    output: { projectUrl: string };
  };

  // ─── Phase 14 — Agent eval harness (@mnemopay/augengine-eval) ───────────────

  /**
   * List every benchmark task the running AugEngine server knows about. The
   * `tasks` array is the canonical set of reference benchmarks an MCP-aware
   * agent can run against this server.
   */
  "eval.list_tasks": {
    input: Record<string, never>;
    output: {
      tasks: {
        id: string;
        name: string;
        description: string;
        maxSteps: number;
        permittedVerbs: string[];
      }[];
    };
  };

  /**
   * Run an `AgentImpl` against an `EvalTask`. The agent is identified by
   * `agentRef` — a URI the server resolves to a concrete implementation
   * (e.g. "builtin:random", "builtin:goal-seeker", "http://my-agent/step",
   * "mcp://other-server/agent"). Returns the EvalRun verbatim, including
   * audit chain.
   */
  "eval.run_task": {
    input: {
      taskId: string;
      agentRef: string;
      seed: number;
    };
    output: {
      runId: string;
      taskId: string;
      agentName: string;
      seed: number;
      steps: number;
      completed: boolean;
      tipHash: string;
      genesisHash: string;
      score: { total: number; items: { id: string; passed: boolean; weight: number }[] };
    };
  };

  /**
   * Fetch a previously-recorded run's full audit chain. The caller verifies
   * the chain locally via `@mnemopay/augengine-eval`'s `verifyAudit()`.
   */
  "eval.get_run": {
    input: { runId: string };
    output: {
      run: unknown; // serialized EvalRun
    };
  };

  /**
   * Verify an audit chain. The server walks the chain itself and returns
   * whether the hashes match; useful when the caller doesn't want to vendor
   * augengine-eval.
   */
  "eval.verify_run": {
    input: { runId: string };
    output: { ok: boolean; failedAt: number; reason?: string };
  };

  /**
   * Top-N leaderboard for a given task. Sorted by score desc, then steps asc.
   */
  "eval.leaderboard": {
    input: { taskId: string; limit?: number };
    output: {
      taskId: string;
      entries: {
        agentName: string;
        seed: number;
        score: number;
        steps: number;
        completed: boolean;
        tipHash: string;
        finishedAt: string;
      }[];
    };
  };
}

/** Tool name discriminator type — useful for pattern matching at the dispatcher. */
export type AugEngineMcpToolName = keyof AugEngineMcpTools;
