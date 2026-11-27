import { describe, expect, it } from "vitest";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { validateSceneSpec, Tag, type Entity, type EntityId } from "@mnemopay/augengine";
import { AugEngineMcpServer, TOOL_MANIFEST } from "./server.js";
import { synthesiseFromPrompt } from "./scene-synth.js";
import { renderTscn } from "./godot-export.js";

const rpc = (method: string, params?: Record<string, unknown>, id: number | string = 1) => ({
  jsonrpc: "2.0" as const,
  id,
  method,
  ...(params ? { params } : {}),
});

/** Unwrap a tools/call envelope back into the tool payload. */
function unwrap(result: unknown): { payload: any; isError: boolean } {
  const r = result as { content: { type: string; text: string }[]; isError: boolean };
  const text = r.content[0]!.text;
  let payload: unknown = text;
  try {
    payload = JSON.parse(text);
  } catch {
    /* error strings are not JSON */
  }
  return { payload, isError: r.isError };
}

async function call(
  server: AugEngineMcpServer,
  name: string,
  args: Record<string, unknown>,
): Promise<{ payload: any; isError: boolean }> {
  const res = await server.handle(rpc("tools/call", { name, arguments: args }));
  return unwrap((res as { result: unknown }).result);
}

async function newScene(server: AugEngineMcpServer, prompt: string): Promise<string> {
  const { payload, isError } = await call(server, "augengine_create_scene", { prompt });
  expect(isError).toBe(false);
  return payload.sceneId as string;
}

describe("MCP protocol", () => {
  it("answers initialize with serverInfo and a supported protocol version", async () => {
    const server = new AugEngineMcpServer();
    const res = await server.handle(
      rpc("initialize", { protocolVersion: "2024-11-05", capabilities: {}, clientInfo: {} }),
    );
    const result = (res as { result: any }).result;
    expect(result.protocolVersion).toBe("2024-11-05");
    expect(result.serverInfo.name).toBe("augengine");
    expect(result.capabilities.tools).toBeDefined();
  });

  it("falls back to a known protocol version when the client asks for an unknown one", async () => {
    const server = new AugEngineMcpServer();
    const res = await server.handle(rpc("initialize", { protocolVersion: "1999-01-01" }));
    expect((res as { result: any }).result.protocolVersion).toBe("2024-11-05");
  });

  it("returns no response for a notification", async () => {
    const server = new AugEngineMcpServer();
    const res = await server.handle({
      jsonrpc: "2.0",
      method: "notifications/initialized",
    } as never);
    expect(res).toBeNull();
  });

  it("lists tools with valid MCP names and schemas", async () => {
    const server = new AugEngineMcpServer();
    const res = await server.handle(rpc("tools/list"));
    const tools = (res as { result: any }).result.tools;
    expect(tools).toHaveLength(TOOL_MANIFEST.length);
    for (const tool of tools) {
      expect(tool.name).toMatch(/^[A-Za-z0-9_-]{1,64}$/);
      expect(tool.inputSchema.type).toBe("object");
      expect(typeof tool.description).toBe("string");
    }
  });

  it("rejects an unknown method with -32601", async () => {
    const server = new AugEngineMcpServer();
    const res = await server.handle(rpc("nope/nope"));
    expect((res as { error: any }).error.code).toBe(-32601);
  });

  it("accepts the dotted contract names from tools.ts as aliases", async () => {
    const server = new AugEngineMcpServer();
    const { payload, isError } = await call(server, "augengine.create_scene", {
      prompt: "a quiet room",
    });
    expect(isError).toBe(false);
    expect(payload.sceneId).toMatch(/^scene_/);
  });

  it("reports tool failures as isError content, not transport errors", async () => {
    const server = new AugEngineMcpServer();
    const res = await server.handle(
      rpc("tools/call", { name: "augengine_describe_scene", arguments: { sceneId: "nope" } }),
    );
    expect((res as { error?: unknown }).error).toBeUndefined();
    const { isError, payload } = unwrap((res as { result: unknown }).result);
    expect(isError).toBe(true);
    expect(String(payload)).toContain("unknown scene");
  });
});

describe("create_scene", () => {
  it("produces a spec that passes the engine validator", () => {
    const { spec } = synthesiseFromPrompt({ prompt: "two guards near a fountain" });
    expect(() => validateSceneSpec(spec)).not.toThrow();
    expect(spec.source.kind).toBe("generated");
  });

  it("is deterministic for the same prompt and seed", () => {
    const a = synthesiseFromPrompt({ prompt: "three monks in a forest", seed: 7 });
    const b = synthesiseFromPrompt({ prompt: "three monks in a forest", seed: 7 });
    expect(JSON.stringify(a.spec)).toBe(JSON.stringify(b.spec));
  });

  it("varies with the seed", () => {
    const a = synthesiseFromPrompt({ prompt: "three monks in a forest", seed: 1 });
    const b = synthesiseFromPrompt({ prompt: "three monks in a forest", seed: 2 });
    expect(JSON.stringify(a.spec)).not.toBe(JSON.stringify(b.spec));
  });

  it("reads counts out of the prompt", () => {
    const { spec } = synthesiseFromPrompt({ prompt: "two guards arguing near a fountain" });
    const npcs = spec.entities.filter((e) => e.id.startsWith("npc_"));
    expect(npcs).toHaveLength(2);
    const fountains = spec.entities.filter((e) => e.id.startsWith("prop_fountain"));
    expect(fountains).toHaveLength(1);
  });

  it("reads digit counts too", () => {
    const { spec } = synthesiseFromPrompt({ prompt: "5 robots in a warehouse" });
    expect(spec.entities.filter((e) => e.id.startsWith("npc_"))).toHaveLength(5);
  });

  it("lifts adjectives into persona traits", () => {
    const { spec } = synthesiseFromPrompt({ prompt: "two angry guards" });
    const npc = spec.entities.find((e) => e.id.startsWith("npc_"));
    expect(npc?.components[Tag.Persona]?.traits).toContain("angry");
  });

  it("dims the light for a night scene", () => {
    const day = synthesiseFromPrompt({ prompt: "a guard in a field" });
    const night = synthesiseFromPrompt({ prompt: "a guard in a field at night" });
    const intensity = (s: typeof day) =>
      s.spec.entities.find((e) => e.id === "light_main")!.components[Tag.Light]!.intensity;
    expect(intensity(night)).toBeLessThan(intensity(day));
  });

  it("always includes a camera, a light and ground", () => {
    const { spec } = synthesiseFromPrompt({ prompt: "an empty place" });
    const ids = spec.entities.map((e) => e.id);
    expect(ids).toContain("camera_main");
    expect(ids).toContain("light_main");
    expect(ids).toContain("floor");
    expect(spec.defaultCamera).toBe("camera_main");
  });

  it("warns when a prompt matched no characters or props", async () => {
    const server = new AugEngineMcpServer();
    // Exactly the degenerate argument a client sent when it truncated the
    // prompt to its first word — valid scene, but no subjects in it.
    const { payload, isError } = await call(server, "augengine_create_scene", { prompt: "three" });
    expect(isError).toBe(false);
    expect(payload.subjectCount).toBe(0);
    expect(payload.warnings).toHaveLength(1);
    expect(payload.warnings[0]).toContain("empty scaffold");
    expect(payload.warnings[0]).toContain('"three"');
  });

  it("does not warn when the prompt produced subjects", async () => {
    const server = new AugEngineMcpServer();
    const { payload } = await call(server, "augengine_create_scene", {
      prompt: "three nervous monks around a statue in a dungeon",
    });
    expect(payload.subjectCount).toBe(4);
    expect(payload.warnings).toBeUndefined();
  });

  it("rejects an empty prompt outright", async () => {
    const server = new AugEngineMcpServer();
    const { isError, payload } = await call(server, "augengine_create_scene", { prompt: "   " });
    expect(isError).toBe(true);
    expect(String(payload)).toContain("must not be empty");
  });

  it("reports a cache hit when the same prompt is authored twice", async () => {
    const server = new AugEngineMcpServer();
    const first = await call(server, "augengine_create_scene", { prompt: "a lamp at dusk" });
    const second = await call(server, "augengine_create_scene", { prompt: "a lamp at dusk" });
    expect(first.payload.cacheHit).toBe(false);
    expect(second.payload.cacheHit).toBe(true);
    expect(second.payload.sceneId).toBe(first.payload.sceneId);
  });
});

describe("scene authoring", () => {
  const boxEntity = (id: string): Entity => ({
    id: id as EntityId,
    name: `Box ${id}`,
    components: {
      [Tag.Transform]: {
        position: [1, 0.5, 1],
        rotation: [0, 0, 0, 1],
        scale: [1, 1, 1],
        parent: null,
      },
      [Tag.AffordanceSet]: { affordances: ["push"] },
    },
  });

  it("adds an entity and reflects it in list_entities", async () => {
    const server = new AugEngineMcpServer();
    const sceneId = await newScene(server, "an empty room");
    const added = await call(server, "augengine_add_entity", {
      sceneId,
      entity: boxEntity("crate_a"),
    });
    expect(added.payload.ok).toBe(true);

    const listed = await call(server, "augengine_list_entities", {
      sceneId,
      namePattern: "^Box",
    });
    expect(listed.payload.count).toBe(1);
    expect(listed.payload.entities[0].id).toBe("crate_a");
    expect(listed.payload.entities[0].position).toEqual([1, 0.5, 1]);
  });

  it("refuses a duplicate entity id", async () => {
    const server = new AugEngineMcpServer();
    const sceneId = await newScene(server, "an empty room");
    await call(server, "augengine_add_entity", { sceneId, entity: boxEntity("dup") });
    const again = await call(server, "augengine_add_entity", { sceneId, entity: boxEntity("dup") });
    expect(again.isError).toBe(true);
    expect(String(again.payload)).toContain("already exists");
  });

  it("validates entity shape", async () => {
    const server = new AugEngineMcpServer();
    const sceneId = await newScene(server, "an empty room");
    const bad = await call(server, "augengine_add_entity", {
      sceneId,
      entity: { name: "no id", components: {} },
    });
    expect(bad.isError).toBe(true);
    expect(String(bad.payload)).toContain("entity.id");
  });

  it("filters list_entities by component tag", async () => {
    const server = new AugEngineMcpServer();
    const sceneId = await newScene(server, "two guards near a fountain");
    const personas = await call(server, "augengine_list_entities", {
      sceneId,
      tags: ["persona"],
    });
    expect(personas.payload.count).toBe(2);
  });

  it("attaches a persona to an existing entity", async () => {
    const server = new AugEngineMcpServer();
    const sceneId = await newScene(server, "an empty room");
    await call(server, "augengine_add_entity", { sceneId, entity: boxEntity("golem") });
    const attached = await call(server, "augengine_attach_persona", {
      sceneId,
      entityId: "golem",
      traits: ["stoic", "ancient"],
      mood: 0.5,
    });
    expect(attached.payload.ok).toBe(true);

    const listed = await call(server, "augengine_list_entities", { sceneId, tags: ["persona"] });
    expect(listed.payload.entities.map((e: any) => e.id)).toContain("golem");
  });

  it("clamps persona mood into -1..1", async () => {
    const server = new AugEngineMcpServer();
    const sceneId = await newScene(server, "an empty room");
    await call(server, "augengine_add_entity", { sceneId, entity: boxEntity("golem") });
    await call(server, "augengine_attach_persona", {
      sceneId,
      entityId: "golem",
      traits: ["furious"],
      mood: 99,
    });
    const handle = server.registry.get(sceneId)!;
    expect(handle.scene.get("golem" as EntityId)!.components[Tag.Persona]!.mood).toBe(1);
  });
});

describe("actions and the observation bus", () => {
  it("dispatches dialogue and the subscriber sees it", async () => {
    const server = new AugEngineMcpServer();
    const sceneId = await newScene(server, "two guards near a fountain");
    const actor = server.registry
      .get(sceneId)!
      .scene.spec.entities.find((e) => e.id.startsWith("npc_"))!.id;

    const sub = await call(server, "augengine_subscribe_observation", {
      sceneId,
      channel: "Dialogue",
    });
    const subscriptionId = sub.payload.subscriptionId;

    const dispatched = await call(server, "augengine_dispatch_action", {
      sceneId,
      action: {
        verb: "say",
        actor,
        params: { text: "You took the last of the water." },
        preconditions: ["actor_exists"],
        effects: ["emit_dialogue"],
      },
    });
    expect(dispatched.payload.ok).toBe(true);
    expect(dispatched.payload.appliedEffects).toEqual(["emit_dialogue"]);

    const polled = await call(server, "augengine_poll_observation", { subscriptionId });
    expect(polled.payload.events).toHaveLength(1);
    expect(polled.payload.events[0].channel).toBe("Dialogue");
    expect(polled.payload.events[0].payload.text).toBe("You took the last of the water.");
    expect(polled.payload.remaining).toBe(0);
  });

  it("drains the queue so a second poll is empty", async () => {
    const server = new AugEngineMcpServer();
    const sceneId = await newScene(server, "a guard near a fountain");
    const actor = server.registry
      .get(sceneId)!
      .scene.spec.entities.find((e) => e.id.startsWith("npc_"))!.id;
    const sub = await call(server, "augengine_subscribe_observation", {
      sceneId,
      channel: "Dialogue",
    });
    await call(server, "augengine_dispatch_action", {
      sceneId,
      action: { verb: "say", actor, params: { text: "hm" }, preconditions: [], effects: ["emit_dialogue"] },
    });
    await call(server, "augengine_poll_observation", {
      subscriptionId: sub.payload.subscriptionId,
    });
    const second = await call(server, "augengine_poll_observation", {
      subscriptionId: sub.payload.subscriptionId,
    });
    expect(second.payload.events).toHaveLength(0);
  });

  it("rejects a verb the scene never declared", async () => {
    const server = new AugEngineMcpServer();
    const sceneId = await newScene(server, "a guard near a fountain");
    const actor = server.registry
      .get(sceneId)!
      .scene.spec.entities.find((e) => e.id.startsWith("npc_"))!.id;
    const res = await call(server, "augengine_dispatch_action", {
      sceneId,
      action: { verb: "teleport", actor, params: {}, preconditions: [], effects: [] },
    });
    expect(res.payload.ok).toBe(false);
    expect(res.payload.reason).toContain("verb not registered");
  });

  it("registers verbs that arrive with a new entity", async () => {
    const server = new AugEngineMcpServer();
    const sceneId = await newScene(server, "an empty room");
    await call(server, "augengine_add_entity", {
      sceneId,
      entity: {
        id: "lever",
        name: "Lever",
        components: {
          [Tag.Transform]: { position: [0, 1, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1], parent: null },
          [Tag.AffordanceSet]: { affordances: ["pull"] },
        },
      },
    });
    const res = await call(server, "augengine_dispatch_action", {
      sceneId,
      action: { verb: "pull", actor: "lever", params: {}, preconditions: ["actor_exists"], effects: [] },
    });
    expect(res.payload.ok).toBe(true);
  });

  it("fails a precondition when the target is out of range", async () => {
    const server = new AugEngineMcpServer();
    const sceneId = await newScene(server, "a guard near a fountain");
    const handle = server.registry.get(sceneId)!;
    const actor = handle.scene.spec.entities.find((e) => e.id.startsWith("npc_"))!.id;
    const res = await call(server, "augengine_dispatch_action", {
      sceneId,
      action: {
        verb: "look",
        actor,
        target: [500, 0, 500],
        params: { range: 2 },
        preconditions: ["target_in_range"],
        effects: [],
      },
    });
    expect(res.payload.ok).toBe(false);
    expect(res.payload.reason).toContain("target_in_range");
  });

  it("move_to_target actually moves the actor", async () => {
    const server = new AugEngineMcpServer();
    const sceneId = await newScene(server, "a guard near a fountain");
    const handle = server.registry.get(sceneId)!;
    const actorId = handle.scene.spec.entities.find((e) => e.id.startsWith("npc_"))!.id;
    const before = handle.scene.get(actorId)!.components[Tag.Transform]!.position;

    const res = await call(server, "augengine_dispatch_action", {
      sceneId,
      action: {
        verb: "move",
        actor: actorId,
        target: [10, 0, 10],
        params: { standoff: 1 },
        preconditions: ["actor_exists"],
        effects: ["move_to_target"],
      },
    });
    expect(res.payload.ok).toBe(true);
    const after = handle.scene.get(actorId)!.components[Tag.Transform]!.position;
    expect(after).not.toEqual(before);
  });

  it("rejects an unknown observation channel", async () => {
    const server = new AugEngineMcpServer();
    const sceneId = await newScene(server, "an empty room");
    const res = await call(server, "augengine_subscribe_observation", {
      sceneId,
      channel: "Telepathy",
    });
    expect(res.isError).toBe(true);
  });
});

describe("describe_scene", () => {
  it("summarises components, verbs, predicates and effects", async () => {
    const server = new AugEngineMcpServer();
    const sceneId = await newScene(server, "two guards near a fountain");
    const { payload } = await call(server, "augengine_describe_scene", { sceneId });
    expect(payload.byComponent.persona).toBe(2);
    expect(payload.verbs).toContain("talk");
    expect(payload.predicates).toContain("target_in_range");
    expect(payload.effects).toContain("emit_dialogue");
    expect(payload.source.kind).toBe("generated");
  });
});

describe("godot export", () => {
  it("renders nodes and sub-resources for a synthesised scene", () => {
    const { spec } = synthesiseFromPrompt({ prompt: "two guards near a fountain" });
    const { text, nodeCount } = renderTscn(spec);
    expect(nodeCount).toBe(spec.entities.length);
    expect(text).toContain("[gd_scene");
    expect(text).toContain('[node name="Root" type="Node3D"]');
    expect(text).toContain("MeshInstance3D");
    expect(text).toContain("DirectionalLight3D");
    expect(text).toContain("Camera3D");
    expect(text).toContain("Transform3D(");
    expect(text).toContain("metadata/persona_traits");
  });

  it("declares load_steps consistent with its sub-resources", () => {
    const { spec } = synthesiseFromPrompt({ prompt: "a crate" });
    const { text } = renderTscn(spec);
    const declared = Number(/load_steps=(\d+)/.exec(text)![1]);
    const actual = (text.match(/\[(sub|ext)_resource /g) ?? []).length;
    expect(declared).toBe(actual + 1);
  });

  it("sizes spheres and cylinders like the engine primitives (radius 1, height 2)", () => {
    const spec = {
      schemaVersion: 1 as const,
      source: { kind: "authored" as const, path: "t" },
      entities: (["sphere", "cylinder"] as const).map((shape) => ({
        id: shape as never,
        name: shape,
        components: {
          transform: { position: [0, 0, 0] as const, rotation: [0, 0, 0, 1] as const, scale: [1, 1, 1] as const, parent: null },
          mesh: { geometry: { kind: "primitive" as const, shape }, material: { kind: "pbr" as const, baseColor: [1, 1, 1] as const, metallic: 0, roughness: 1 }, visible: true },
        },
      })),
    };
    const { text } = renderTscn(spec as never);
    expect(text).toMatch(/SphereMesh[^[]*radius = 1\n[^[]*height = 2/);
    expect(text).toMatch(/CylinderMesh[^[]*top_radius = 1\n[^[]*bottom_radius = 1\n[^[]*height = 2/);
  });

  it("instances imported glTF models once and writes the .glb next to Main.tscn", async () => {
    const glbBase64 = "Z2xURgIAAAAUAAAA"; // any bytes: the exporter copies, it does not parse
    const url = `data:model/gltf-binary;base64,${glbBase64}`;
    const t = (parent: string | null) => ({ position: [1, 0, 0] as const, rotation: [0, 0, 0, 1] as const, scale: [1, 1, 1] as const, parent });
    const part = (id: string, i: number) => ({
      id, name: id,
      components: { transform: t("robot"), mesh: { geometry: { kind: "gltf", url, primitiveIndex: i }, material: { kind: "pbr", baseColor: [1, 1, 1], metallic: 0, roughness: 1 }, visible: true } },
    });
    const spec = {
      schemaVersion: 1, source: { kind: "authored", path: "t" },
      entities: [{ id: "robot", name: "Robot", components: { transform: t(null) } }, part("robot.part1", 0), part("robot.part2", 1)],
    };
    const { text, models, nodeCount } = renderTscn(spec as never);
    expect(models).toEqual([{ url, file: "models/Robot.glb" }]);
    expect(text).toContain('[ext_resource type="PackedScene" path="res://models/Robot.glb" id="1_model"]');
    expect(text).toContain('[node name="Robot" parent="." instance=ExtResource("1_model")]');
    expect(text).not.toContain("part1");
    expect(nodeCount).toBe(1);
    expect(Number(/load_steps=(\d+)/.exec(text)![1])).toBe(2);
    const dir = await mkdtemp(join(tmpdir(), "augengine-godot-glb-"));
    try {
      const { exportGodotProject } = await import("./godot-export.js");
      const out = await exportGodotProject(spec as never, dir, "Glb");
      expect(out.missingModels).toEqual([]);
      const written = await readFile(join(dir, "models", "Robot.glb"));
      expect(written.toString("base64")).toBe(glbBase64);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  it("writes a project Godot can open", async () => {
    const dir = await mkdtemp(join(tmpdir(), "augengine-godot-"));
    try {
      const server = new AugEngineMcpServer({ exportRoot: dir });
      const sceneId = await newScene(server, "a merchant beside a stall");
      const { payload, isError } = await call(server, "augengine_export_godot", {
        sceneId,
        outputDir: "out",
        projectName: "Market",
      });
      expect(isError).toBe(false);
      expect(payload.files).toHaveLength(2);
      const project = await readFile(join(dir, "out", "project.godot"), "utf-8");
      expect(project).toContain('config/name="Market"');
      expect(project).toContain('run/main_scene="res://Main.tscn"');
      const scene = await readFile(join(dir, "out", "Main.tscn"), "utf-8");
      expect(scene).toContain("[gd_scene");
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});

describe("render_clip", () => {
  it("refuses honestly instead of returning a fake URL", async () => {
    const server = new AugEngineMcpServer();
    const sceneId = await newScene(server, "a guard");
    const { payload, isError } = await call(server, "augengine_render_clip", {
      sceneId,
      fps: 30,
      frames: 60,
    });
    expect(isError).toBe(true);
    expect(String(payload)).toContain("WebGPU");
    expect(String(payload)).not.toContain("http");
  });
});
