/**
 * AugEngine MCP server — newline-delimited JSON-RPC 2.0 over stdio.
 *
 * This implements the actual MCP handshake, which the eval-server's
 * `mcp-server.ts` does not: it answers `initialize`, absorbs
 * `notifications/initialized`, and routes tools through `tools/call`. A client
 * that only ever sees bare `augengine.create_scene` JSON-RPC methods (the
 * eval-server's shape) cannot drive this server, and no real MCP client speaks
 * that way — Claude Desktop, Claude Code and Cursor all handshake first, list
 * second, and call tools through `tools/call`.
 *
 * Tool naming: the manifest advertises underscore names (`augengine_add_entity`)
 * because several clients validate tool names against `[A-Za-z0-9_-]+` and
 * reject dots. The dotted names from `tools.ts` stay valid as aliases, so the
 * typed contract in that file is still the source of truth.
 *
 * Transport deliberately avoids `@modelcontextprotocol/sdk` — same call the
 * eval-server made, and it keeps this package dependency-free apart from the
 * engine itself.
 */

import {
  Scene,
  validateSceneSpec,
  type Entity,
  type EntityId,
  type Action,
  type ObservationChannel,
  type SceneSpec,
  type ViewFilter,
  Tag,
} from "@mnemopay/augengine";
import { SceneRegistry } from "./registry.js";
import { synthesiseFromPrompt, hashPrompt } from "./scene-synth.js";
import { exportGodotProject } from "./godot-export.js";
import { BUILTIN_EFFECT_IDS, BUILTIN_PREDICATE_IDS, BASE_VERBS } from "./grammar.js";
import { DETECTIVE_TOOLS, DetectiveSessions } from "./detective-tools.js";

export const SERVER_NAME = "augengine";
export const SERVER_VERSION = "0.1.0";
const SUPPORTED_PROTOCOLS = ["2025-06-18", "2025-03-26", "2024-11-05"];
const DEFAULT_PROTOCOL = "2024-11-05";

const METHOD_NOT_FOUND = -32601;
const INVALID_PARAMS = -32602;
const INTERNAL_ERROR = -32603;

interface JsonRpcRequest {
  jsonrpc: "2.0";
  id?: string | number | null;
  method: string;
  params?: Record<string, unknown>;
}

interface JsonRpcResponse {
  jsonrpc: "2.0";
  id: string | number | null;
  result?: unknown;
  error?: { code: number; message: string; data?: unknown };
}

export interface McpServerOptions {
  registry?: SceneRegistry;
  /** Root for `export_godot` output. Defaults to the process cwd. */
  exportRoot?: string;
  stdin?: NodeJS.ReadableStream;
  stdout?: NodeJS.WritableStream;
}

const obj = { type: "object" } as const;

export const TOOL_MANIFEST = [
  {
    name: "augengine_create_scene",
    description:
      "Author a new AugEngine scene from a natural-language prompt. Returns a sceneId plus the generated SceneSpec. Layout is deterministic for a given (prompt, seed) — refine the result with augengine_add_entity and augengine_attach_persona.",
    inputSchema: {
      ...obj,
      properties: {
        prompt: { type: "string", description: "What the scene contains, in plain language." },
        seed: { type: "number", description: "Optional layout seed. Omit to derive one from the prompt." },
      },
      required: ["prompt"],
    },
  },
  {
    name: "augengine_add_entity",
    description:
      "Add one entity to an existing scene. The entity must have an id, a name, and a components object keyed by AugEngine component tags (transform, mesh, light, camera, persona, agent, collider, affordance_set).",
    inputSchema: {
      ...obj,
      properties: {
        sceneId: { type: "string" },
        entity: { type: "object", description: "An AugEngine Entity." },
      },
      required: ["sceneId", "entity"],
    },
  },
  {
    name: "augengine_attach_persona",
    description:
      "Attach or replace a Persona component on an entity — traits drive how the entity is prompted when it thinks.",
    inputSchema: {
      ...obj,
      properties: {
        sceneId: { type: "string" },
        entityId: { type: "string" },
        traits: { type: "array", items: { type: "string" } },
        voice: { type: "string" },
        memoryId: { type: "string" },
        mood: { type: "number", description: "-1..1" },
      },
      required: ["sceneId", "entityId", "traits"],
    },
  },
  {
    name: "augengine_list_entities",
    description:
      "List entities in a scene, optionally filtered by component tags or a name regex. Returns id, name, component tags and world position.",
    inputSchema: {
      ...obj,
      properties: {
        sceneId: { type: "string" },
        tags: { type: "array", items: { type: "string" } },
        namePattern: { type: "string" },
      },
      required: ["sceneId"],
    },
  },
  {
    name: "augengine_dispatch_action",
    description:
      "Dispatch a unified-grammar Action through the scene's ActionDispatcher. Verbs come from the scene's affordance sets and agent permitted-verbs. Built-in preconditions: actor_exists, target_exists, target_in_range. Built-in effects: move_to_target, set_mood, emit_dialogue, set_visible.",
    inputSchema: {
      ...obj,
      properties: {
        sceneId: { type: "string" },
        action: { type: "object", description: "An AugEngine Action: verb, actor, target?, params, preconditions, effects, source." },
      },
      required: ["sceneId", "action"],
    },
  },
  {
    name: "augengine_subscribe_observation",
    description:
      "Open a buffered subscription to one observation channel (Visual, Audible, Social, Inventory, Combat, Dialogue). Returns a subscriptionId to drain with augengine_poll_observation.",
    inputSchema: {
      ...obj,
      properties: {
        sceneId: { type: "string" },
        channel: {
          type: "string",
          enum: ["Visual", "Audible", "Social", "Inventory", "Combat", "Dialogue"],
        },
        rangeMax: { type: "number" },
        allowedSourceTags: { type: "array", items: { type: "string" } },
      },
      required: ["sceneId", "channel"],
    },
  },
  {
    name: "augengine_poll_observation",
    description:
      "Drain queued observation events for a subscription. Returns the events, how many remain queued, and how many were dropped since the last poll.",
    inputSchema: {
      ...obj,
      properties: {
        subscriptionId: { type: "string" },
        maxEvents: { type: "number" },
      },
      required: ["subscriptionId"],
    },
  },
  {
    name: "augengine_export_godot",
    description:
      "Write the scene out as a Godot 4.4 project (project.godot + Main.tscn) for native targets the browser runtime does not reach.",
    inputSchema: {
      ...obj,
      properties: {
        sceneId: { type: "string" },
        outputDir: { type: "string", description: "Directory to write into. Relative paths resolve under the server's export root." },
        projectName: { type: "string" },
      },
      required: ["sceneId"],
    },
  },
  {
    name: "augengine_describe_scene",
    description:
      "Summarise a scene: entity counts by component, the dispatchable verb set, the registered predicates and effects, and the scene source.",
    inputSchema: {
      ...obj,
      properties: { sceneId: { type: "string" } },
      required: ["sceneId"],
    },
  },
  {
    name: "augengine_render_clip",
    description:
      "Render the scene to MP4. Requires a browser host with WebGPU + WebCodecs; unavailable over stdio, where it returns an explicit error rather than a fake URL.",
    inputSchema: {
      ...obj,
      properties: {
        sceneId: { type: "string" },
        fps: { type: "number" },
        frames: { type: "number" },
      },
      required: ["sceneId"],
    },
  },
  ...DETECTIVE_TOOLS,
];

/** Dotted contract name -> advertised underscore name. */
function canonicalToolName(name: string): string {
  return name.replace(/^augengine\./, "augengine_").replace(/^eval\./, "eval_").replace(/^detective\./, "detective_");
}

export class AugEngineMcpServer {
  readonly registry: SceneRegistry;
  readonly detective: DetectiveSessions;
  private exportRoot: string;

  constructor(opts: { registry?: SceneRegistry; exportRoot?: string } = {}) {
    this.registry = opts.registry ?? new SceneRegistry();
    this.exportRoot = opts.exportRoot ?? process.cwd();
    this.detective = new DetectiveSessions();
  }

  async handle(req: JsonRpcRequest): Promise<JsonRpcResponse | null> {
    const id = req.id ?? null;
    if (req.id === undefined) return null; // notification

    try {
      const result = await this.invoke(req.method, req.params ?? {});
      return { jsonrpc: "2.0", id, result };
    } catch (err) {
      const message = (err as Error).message ?? String(err);
      return { jsonrpc: "2.0", id, error: { code: errorCode(message), message } };
    }
  }

  private async invoke(method: string, params: Record<string, unknown>): Promise<unknown> {
    switch (method) {
      case "initialize": {
        const requested = typeof params.protocolVersion === "string" ? params.protocolVersion : "";
        const protocolVersion = SUPPORTED_PROTOCOLS.includes(requested)
          ? requested
          : DEFAULT_PROTOCOL;
        return {
          protocolVersion,
          capabilities: { tools: { listChanged: false } },
          serverInfo: { name: SERVER_NAME, version: SERVER_VERSION },
        };
      }
      case "ping":
        return {};
      case "tools/list":
        return { tools: TOOL_MANIFEST };
      case "tools/call":
        return this.callTool(params);
      default:
        throw new Error(`unknown method: ${method}`);
    }
  }

  private async callTool(params: Record<string, unknown>): Promise<unknown> {
    const rawName = params.name;
    if (typeof rawName !== "string") {
      throw new Error("invalid params: name must be a string");
    }
    const name = canonicalToolName(rawName);
    const args = (params.arguments ?? {}) as Record<string, unknown>;

    try {
      const payload = await this.runTool(name, args);
      return {
        content: [{ type: "text", text: JSON.stringify(payload, null, 2) }],
        isError: false,
      };
    } catch (err) {
      // Tool-level failures come back as isError content so the calling model
      // can read the reason and retry, rather than as a transport error.
      return {
        content: [{ type: "text", text: (err as Error).message ?? String(err) }],
        isError: true,
      };
    }
  }

  async runTool(name: string, args: Record<string, unknown>): Promise<unknown> {
    switch (name) {
      case "augengine_create_scene":
        return this.createScene(args);
      case "augengine_add_entity":
        return this.addEntity(args);
      case "augengine_attach_persona":
        return this.attachPersona(args);
      case "augengine_list_entities":
        return this.listEntities(args);
      case "augengine_dispatch_action":
        return this.dispatchAction(args);
      case "augengine_subscribe_observation":
        return this.subscribeObservation(args);
      case "augengine_poll_observation":
        return this.pollObservation(args);
      case "augengine_export_godot":
        return this.exportGodot(args);
      case "augengine_describe_scene":
        return this.describeScene(args);
      case "detective_new_session":
      case "detective_status":
      case "detective_act":
        return this.detective.run(name, args);
      case "augengine_render_clip":
        throw new Error(
          "augengine_render_clip needs a browser host: the render runtime depends on WebGPU and WebCodecs, neither of which exists in this Node stdio process. Run the scene through examples/smoke-game, or export it with augengine_export_godot.",
        );
      default:
        throw new Error(`unknown tool: ${name}`);
    }
  }

  private createScene(args: Record<string, unknown>): unknown {
    const prompt = stringArg(args, "prompt");
    if (prompt.trim().length === 0) {
      throw new Error("invalid params: prompt must not be empty");
    }
    const seed = optionalNumberArg(args, "seed");
    const promptKey = `${hashPrompt(prompt)}:${seed ?? "auto"}`;

    const existing = this.registry.findByPromptKey(promptKey);
    if (existing) {
      return {
        sceneId: existing.id,
        spec: { ...existing.scene.spec, source: { ...existing.scene.spec.source, cached: true } },
        cacheHit: true,
        entityCount: existing.scene.size(),
      };
    }

    const { spec } = synthesiseFromPrompt(seed === undefined ? { prompt } : { prompt, seed });
    const handle = this.registry.create(spec, promptKey);

    // A prompt that matched no subject nouns still yields a valid scene — sun,
    // sky, ground, camera — which reads as success but is almost never what the
    // caller wanted. Say so, so the agent can notice it truncated or mangled
    // its own argument and retry, rather than proceeding against an empty set.
    const subjects = spec.entities.filter(
      (e) => e.id.startsWith("npc_") || e.id.startsWith("prop_"),
    ).length;
    const warnings: string[] = [];
    if (subjects === 0) {
      warnings.push(
        `No characters or props matched the prompt ${JSON.stringify(prompt)} — this scene is an empty scaffold (ground, sky, sun, camera). If that prompt looks shorter than what you meant to send, resend the full sentence. Otherwise add entities with augengine_add_entity.`,
      );
    }

    return {
      sceneId: handle.id,
      spec,
      cacheHit: false,
      entityCount: handle.scene.size(),
      subjectCount: subjects,
      ...(warnings.length > 0 ? { warnings } : {}),
    };
  }

  private addEntity(args: Record<string, unknown>): unknown {
    const handle = this.requireScene(args);
    const raw = args.entity;
    if (typeof raw !== "object" || raw === null) {
      throw new Error("invalid params: entity must be an object");
    }
    const entity = raw as Entity;
    if (typeof entity.id !== "string" || entity.id.length === 0) {
      throw new Error("invalid params: entity.id must be a non-empty string");
    }
    if (typeof entity.name !== "string") {
      throw new Error("invalid params: entity.name must be a string");
    }
    if (typeof entity.components !== "object" || entity.components === null) {
      throw new Error("invalid params: entity.components must be an object");
    }
    if (handle.scene.has(entity.id)) {
      throw new Error(`entity ${entity.id} already exists in ${handle.id}`);
    }
    handle.scene.add(entity);
    // New affordances/permitted verbs may have arrived with this entity.
    this.registry.rebuildDispatcher(handle.id);
    return { ok: true, entityId: entity.id, entityCount: handle.scene.size() };
  }

  private attachPersona(args: Record<string, unknown>): unknown {
    const handle = this.requireScene(args);
    const entityId = stringArg(args, "entityId") as EntityId;
    const entity = handle.scene.get(entityId);
    if (!entity) throw new Error(`unknown entity ${entityId} in ${handle.id}`);

    const traitsRaw = args.traits;
    if (!Array.isArray(traitsRaw) || traitsRaw.some((t) => typeof t !== "string")) {
      throw new Error("invalid params: traits must be an array of strings");
    }
    const mood = optionalNumberArg(args, "mood") ?? 0;
    entity.components[Tag.Persona] = {
      traits: traitsRaw as string[],
      voice: typeof args.voice === "string" ? args.voice : null,
      memoryId: typeof args.memoryId === "string" ? args.memoryId : null,
      goals: entity.components[Tag.Persona]?.goals ?? [],
      mood: Math.max(-1, Math.min(1, mood)),
    };
    return { ok: true, entityId };
  }

  private listEntities(args: Record<string, unknown>): unknown {
    const handle = this.requireScene(args);
    const tags = Array.isArray(args.tags) ? (args.tags as string[]) : null;
    const patternRaw = typeof args.namePattern === "string" ? args.namePattern : null;
    let pattern: RegExp | null = null;
    if (patternRaw !== null) {
      try {
        pattern = new RegExp(patternRaw, "i");
      } catch {
        throw new Error(`invalid params: namePattern is not a valid regex: ${patternRaw}`);
      }
    }

    const cache = handle.scene.worldTransforms();
    const entities: unknown[] = [];
    for (const entity of handle.scene) {
      const componentTags = Object.keys(entity.components);
      if (tags && !tags.every((t) => componentTags.includes(t))) continue;
      if (pattern && !pattern.test(entity.name)) continue;
      const idx = cache.indexById.get(entity.id);
      const position =
        idx === undefined
          ? null
          : [
              cache.worldMatrices[idx * 16 + 12] ?? 0,
              cache.worldMatrices[idx * 16 + 13] ?? 0,
              cache.worldMatrices[idx * 16 + 14] ?? 0,
            ];
      entities.push({ id: entity.id, name: entity.name, components: componentTags, position });
    }
    return { sceneId: handle.id, count: entities.length, entities };
  }

  private dispatchAction(args: Record<string, unknown>): unknown {
    const handle = this.requireScene(args);
    const raw = args.action;
    if (typeof raw !== "object" || raw === null) {
      throw new Error("invalid params: action must be an object");
    }
    const partial = raw as Partial<Action>;
    if (typeof partial.verb !== "string") {
      throw new Error("invalid params: action.verb must be a string");
    }
    if (typeof partial.actor !== "string") {
      throw new Error("invalid params: action.actor must be an entity id string");
    }
    const action: Action = {
      verb: partial.verb,
      actor: partial.actor as EntityId,
      ...(partial.target !== undefined ? { target: partial.target } : {}),
      params: partial.params ?? {},
      preconditions: partial.preconditions ?? [],
      effects: partial.effects ?? [],
      source: partial.source ?? { kind: "script", scriptId: "mcp" },
    };
    const result = handle.dispatcher.dispatch(action);
    return { ...result, sceneId: handle.id };
  }

  private subscribeObservation(args: Record<string, unknown>): unknown {
    const handle = this.requireScene(args);
    const channel = stringArg(args, "channel") as ObservationChannel;
    const valid: ObservationChannel[] = [
      "Visual", "Audible", "Social", "Inventory", "Combat", "Dialogue",
    ];
    if (!valid.includes(channel)) {
      throw new Error(`invalid params: channel must be one of ${valid.join(", ")}`);
    }
    const rangeMax = optionalNumberArg(args, "rangeMax");
    const allowed = Array.isArray(args.allowedSourceTags)
      ? (args.allowedSourceTags as string[])
      : undefined;
    const filter: ViewFilter | null =
      rangeMax === undefined && allowed === undefined
        ? null
        : {
            ...(rangeMax !== undefined ? { rangeMax } : {}),
            ...(allowed !== undefined ? { allowedSourceTags: allowed } : {}),
          };
    const sub = this.registry.subscribe(handle.id, channel, filter);
    return { subscriptionId: sub.id, sceneId: handle.id, channel };
  }

  private pollObservation(args: Record<string, unknown>): unknown {
    const subscriptionId = stringArg(args, "subscriptionId");
    const maxEvents = optionalNumberArg(args, "maxEvents") ?? 100;
    return this.registry.poll(subscriptionId, maxEvents);
  }

  private async exportGodot(args: Record<string, unknown>): Promise<unknown> {
    const handle = this.requireScene(args);
    const { isAbsolute, join } = await import("node:path");
    const requested = typeof args.outputDir === "string" ? args.outputDir : `${handle.id}-godot`;
    const outputDir = isAbsolute(requested) ? requested : join(this.exportRoot, requested);
    const projectName =
      typeof args.projectName === "string" ? args.projectName : `AugEngine ${handle.id}`;
    // Project-path model urls resolve against the export root.
    const result = await exportGodotProject(handle.scene.spec, outputDir, projectName, { assetRoot: this.exportRoot });
    return result;
  }

  private describeScene(args: Record<string, unknown>): unknown {
    const handle = this.requireScene(args);
    const byComponent: Record<string, number> = {};
    const verbs = new Set<string>(BASE_VERBS);
    for (const entity of handle.scene) {
      for (const tag of Object.keys(entity.components)) {
        byComponent[tag] = (byComponent[tag] ?? 0) + 1;
      }
      const affordances = entity.components[Tag.AffordanceSet];
      if (affordances) for (const v of affordances.affordances) verbs.add(v);
      const agent = entity.components[Tag.Agent];
      if (agent) for (const v of agent.permittedVerbs) verbs.add(v);
    }
    return {
      sceneId: handle.id,
      createdAt: handle.createdAt,
      source: handle.scene.spec.source,
      entityCount: handle.scene.size(),
      byComponent,
      verbs: Array.from(verbs).sort(),
      predicates: [...BUILTIN_PREDICATE_IDS],
      effects: [...BUILTIN_EFFECT_IDS],
    };
  }

  private requireScene(args: Record<string, unknown>): ReturnType<SceneRegistry["create"]> {
    const sceneId = stringArg(args, "sceneId");
    const handle = this.registry.get(sceneId);
    if (!handle) throw new Error(`unknown scene ${sceneId}`);
    return handle;
  }

  /** Load a SceneSpec authored elsewhere (used by the CLI --scene flag). */
  loadSpec(json: unknown): string {
    const spec: SceneSpec = validateSceneSpec(json);
    // Constructing a Scene here surfaces malformed specs before registration.
    void new Scene(spec);
    return this.registry.create(spec).id;
  }
}

function errorCode(message: string): number {
  if (message.startsWith("invalid params:")) return INVALID_PARAMS;
  if (message.startsWith("unknown method:")) return METHOD_NOT_FOUND;
  return INTERNAL_ERROR;
}

function stringArg(args: Record<string, unknown>, key: string): string {
  const v = args[key];
  if (typeof v !== "string") throw new Error(`invalid params: ${key} must be a string`);
  return v;
}

function optionalNumberArg(args: Record<string, unknown>, key: string): number | undefined {
  const v = args[key];
  if (v === undefined || v === null) return undefined;
  if (typeof v !== "number" || !Number.isFinite(v)) {
    throw new Error(`invalid params: ${key} must be a finite number`);
  }
  return v;
}

/** Wire an AugEngineMcpServer to a line-delimited JSON-RPC stream. */
export function startMcpServer(opts: McpServerOptions = {}): {
  server: AugEngineMcpServer;
  stop: () => void;
} {
  const server = new AugEngineMcpServer({
    ...(opts.registry !== undefined ? { registry: opts.registry } : {}),
    ...(opts.exportRoot !== undefined ? { exportRoot: opts.exportRoot } : {}),
  });
  const stdin = opts.stdin ?? process.stdin;
  const stdout = opts.stdout ?? process.stdout;

  let buffer = "";
  const onData = async (chunk: Buffer | string): Promise<void> => {
    buffer += typeof chunk === "string" ? chunk : chunk.toString("utf-8");
    for (;;) {
      const idx = buffer.indexOf("\n");
      if (idx < 0) break;
      const line = buffer.slice(0, idx).trim();
      buffer = buffer.slice(idx + 1);
      if (!line) continue;
      let req: JsonRpcRequest;
      try {
        req = JSON.parse(line) as JsonRpcRequest;
      } catch {
        continue; // malformed line — no id to answer against
      }
      const response = await server.handle(req);
      if (response) stdout.write(`${JSON.stringify(response)}\n`);
    }
  };

  stdin.on("data", (chunk: Buffer | string) => {
    void onData(chunk);
  });

  return {
    server,
    stop: () => {
      stdin.removeAllListeners("data");
    },
  };
}
