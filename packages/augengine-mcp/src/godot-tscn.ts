/**
 * SceneSpec -> Godot 4.4 `.tscn` text. Pure (no Node APIs) so the browser
 * editor can export too; `godot-export.ts` writes these strings to disk.
 * See godot-export.ts for the mapping notes.
 */

import {
  Tag,
  type Entity,
  type Quat,
  type SceneSpec,
  type Vec3,
} from "@mnemopay/augengine";

/** Quaternion (x, y, z, w) -> basis axis columns, each scaled. */
function basisAxes(rotation: Quat, scale: Vec3): [Vec3, Vec3, Vec3] {
  const [x, y, z, w] = rotation;
  const xAxis: Vec3 = [
    1 - 2 * (y * y + z * z),
    2 * (x * y + w * z),
    2 * (x * z - w * y),
  ];
  const yAxis: Vec3 = [
    2 * (x * y - w * z),
    1 - 2 * (x * x + z * z),
    2 * (y * z + w * x),
  ];
  const zAxis: Vec3 = [
    2 * (x * z + w * y),
    2 * (y * z - w * x),
    1 - 2 * (x * x + y * y),
  ];
  const mul = (v: Vec3, s: number): Vec3 => [v[0] * s, v[1] * s, v[2] * s];
  return [mul(xAxis, scale[0]), mul(yAxis, scale[1]), mul(zAxis, scale[2])];
}

const fmt = (n: number): string => {
  const rounded = Number(n.toFixed(6));
  return Object.is(rounded, -0) ? "0" : String(rounded);
};

function transform3d(position: Vec3, rotation: Quat, scale: Vec3): string {
  const [ax, ay, az] = basisAxes(rotation, scale);
  const parts = [...ax, ...ay, ...az, ...position].map(fmt);
  return `Transform3D(${parts.join(", ")})`;
}

/** Godot node names cannot contain . : @ / or leading/trailing spaces. */
function safeName(name: string, fallback: string): string {
  const cleaned = name.replace(/[.:@/"%]/g, "_").trim();
  return cleaned.length > 0 ? cleaned : fallback;
}

interface SubResource {
  id: string;
  body: string;
}

/** Godot mesh sizes that reproduce the engine's built-in primitives. */
export const GODOT_PRIMITIVE_SIZES = {
  box: { size: 1 },
  plane: { size: 1 },
  sphere: { radius: 1, height: 2 },
  cylinder: { radius: 1, height: 2 },
} as const;

/** A glTF model the exported scene instances (written next to Main.tscn). */
export interface GodotModelFile {
  /** Geometry url in the SceneSpec (data:, blob:, or a project path). */
  url: string;
  /** Path inside the Godot project, e.g. "models/robot.glb". */
  file: string;
}

export interface RenderTscnOptions {
  /** Pick the in-project file name for a model url (default: derived from the url / entity name). */
  modelFileName?: (url: string, entityName: string) => string;
}

function gltfUrlOf(entity: Entity): string | null {
  const g = entity.components[Tag.Mesh]?.geometry;
  return g?.kind === "gltf" ? g.url : null;
}

function defaultModelFile(url: string, entityName: string): string {
  const fromPath = /^(data:|blob:)/.test(url) ? "" : (url.split(/[\\/]/).pop() ?? "");
  const base = (fromPath || entityName).replace(/\.(glb|gltf)$/i, "").replace(/[^a-z0-9_-]+/gi, "_").replace(/^_+|_+$/g, "") || "model";
  return `${base}.glb`;
}

function meshSubResource(entity: Entity, index: number): SubResource | null {
  const mesh = entity.components[Tag.Mesh];
  if (!mesh) return null;
  const geometry = mesh.geometry;
  if (geometry.kind !== "primitive") return null;
  const id = `Mesh_${index}`;
  switch (geometry.shape) {
    case "box":
      return { id, body: `[sub_resource type="BoxMesh" id="${id}"]\nsize = Vector3(1, 1, 1)\n` };
    // Engine primitives (runtime/live/webgpu/primitives.ts): sphereMesh(radius 1),
    // cylinderMesh(radius 1, height 2), box 1×1×1, plane 1×1. Godot's sizes must
    // match or every exported sphere/cylinder comes out half size.
    case "sphere":
      return { id, body: `[sub_resource type="SphereMesh" id="${id}"]\nradius = ${fmt(GODOT_PRIMITIVE_SIZES.sphere.radius)}\nheight = ${fmt(GODOT_PRIMITIVE_SIZES.sphere.height)}\n` };
    case "cylinder":
      return {
        id,
        body: `[sub_resource type="CylinderMesh" id="${id}"]\ntop_radius = ${fmt(GODOT_PRIMITIVE_SIZES.cylinder.radius)}\nbottom_radius = ${fmt(GODOT_PRIMITIVE_SIZES.cylinder.radius)}\nheight = ${fmt(GODOT_PRIMITIVE_SIZES.cylinder.height)}\n`,
      };
    case "plane":
      return { id, body: `[sub_resource type="PlaneMesh" id="${id}"]\nsize = Vector2(1, 1)\n` };
    default:
      return null;
  }
}

function materialColor(entity: Entity): Vec3 | null {
  const mesh = entity.components[Tag.Mesh];
  if (!mesh) return null;
  const material = mesh.material;
  if (material.kind === "unlit") return material.color;
  if (material.kind === "pbr") return material.baseColor;
  return null;
}

function materialSubResource(entity: Entity, index: number): SubResource | null {
  const color = materialColor(entity);
  if (!color) return null;
  const mesh = entity.components[Tag.Mesh];
  const unlit = mesh?.material.kind === "unlit";
  const id = `Mat_${index}`;
  const shading = unlit ? "\nshading_mode = 0" : "";
  return {
    id,
    body:
      `[sub_resource type="StandardMaterial3D" id="${id}"]\n` +
      `albedo_color = Color(${fmt(color[0])}, ${fmt(color[1])}, ${fmt(color[2])}, 1)${shading}\n`,
  };
}

export function renderTscn(spec: SceneSpec, opts: RenderTscnOptions = {}): { text: string; nodeCount: number; models: GodotModelFile[] } {
  const subResources: SubResource[] = [];
  const nodes: string[] = [];

  // Imported glTF models: Godot imports a .glb as a scene, so each model is
  // instanced ONCE at the entity that owns it — the entity with the gltf mesh,
  // or (Studio's multi-part import) the parent whose children are its parts.
  // The per-part child entities are then left out (the .glb already holds them).
  const byId = new Map(spec.entities.map((e) => [e.id as string, e]));
  const parentOf = (e: Entity): string | null => e.components[Tag.Transform]?.parent ?? null;
  const instanceUrl = new Map<string, string>(); // entity id -> model url it instances
  const skipped = new Set<string>();
  for (const e of spec.entities) {
    const url = gltfUrlOf(e);
    if (!url) continue;
    const parentId = parentOf(e);
    const parent = parentId ? byId.get(parentId) : undefined;
    const siblingsSameModel = parent && !parent.components[Tag.Mesh] &&
      spec.entities.filter((c) => parentOf(c) === parentId).every((c) => gltfUrlOf(c) === url);
    if (parent && siblingsSameModel) {
      instanceUrl.set(parent.id, url);
      skipped.add(e.id);
    } else {
      instanceUrl.set(e.id, url);
    }
  }
  const models: GodotModelFile[] = [];
  const extIds = new Map<string, string>();
  const usedFiles = new Set<string>();
  for (const [entityId, url] of instanceUrl) {
    if (extIds.has(url)) continue;
    let file = `models/${(opts.modelFileName ?? defaultModelFile)(url, byId.get(entityId)?.name ?? "model")}`;
    for (let n = 2; usedFiles.has(file); n++) file = file.replace(/(-\d+)?\.glb$/, `-${n}.glb`);
    usedFiles.add(file);
    models.push({ url, file });
    extIds.set(url, `${models.length}_model`);
  }

  spec.entities.forEach((entity, i) => {
    if (skipped.has(entity.id)) return;
    const modelUrl = instanceUrl.get(entity.id);
    const transform = entity.components[Tag.Transform];
    const mesh = entity.components[Tag.Mesh];
    const light = entity.components[Tag.Light];
    const camera = entity.components[Tag.Camera];
    const persona = entity.components[Tag.Persona];

    let type = "Node3D";
    if (mesh) type = "MeshInstance3D";
    else if (camera) type = "Camera3D";
    else if (light) {
      type =
        light.kind === "directional"
          ? "DirectionalLight3D"
          : light.kind === "spot"
            ? "SpotLight3D"
            : "OmniLight3D";
    }

    const lines: string[] = [];
    if (modelUrl) {
      lines.push(`[node name="${safeName(entity.name, entity.id)}" parent="." instance=ExtResource("${extIds.get(modelUrl)}")]`);
    } else {
      lines.push(`[node name="${safeName(entity.name, entity.id)}" type="${type}" parent="."]`);
    }

    if (transform) {
      lines.push(
        `transform = ${transform3d(transform.position, transform.rotation, transform.scale)}`,
      );
    }

    if (mesh && !modelUrl) {
      const meshRes = meshSubResource(entity, i);
      if (meshRes) {
        subResources.push(meshRes);
        lines.push(`mesh = SubResource("${meshRes.id}")`);
      }
      const matRes = materialSubResource(entity, i);
      if (matRes) {
        subResources.push(matRes);
        lines.push(`surface_material_override/0 = SubResource("${matRes.id}")`);
      }
      if (!mesh.visible) lines.push("visible = false");
    }

    if (light) {
      lines.push(
        `light_color = Color(${fmt(light.color[0])}, ${fmt(light.color[1])}, ${fmt(light.color[2])}, 1)`,
      );
      lines.push(`light_energy = ${fmt(light.intensity)}`);
      if (light.range !== undefined && light.kind !== "directional") {
        lines.push(`omni_range = ${fmt(light.range)}`);
      }
    }

    if (camera) {
      lines.push(`fov = ${fmt((camera.fovY * 180) / Math.PI)}`);
      lines.push(`near = ${fmt(camera.near)}`);
      lines.push(`far = ${fmt(camera.far)}`);
      if (camera.active) lines.push("current = true");
    }

    if (persona) {
      // Narrative data survives the export as node metadata.
      const traits = persona.traits.map((t) => `"${t.replace(/"/g, "")}"`).join(", ");
      lines.push(`metadata/persona_traits = [${traits}]`);
      lines.push(`metadata/persona_mood = ${fmt(persona.mood)}`);
      if (persona.role !== undefined) {
        lines.push(`metadata/persona_role = "${safeName(persona.role, "role")}"`);
      }
    }

    nodes.push(lines.join("\n"));
  });

  // load_steps counts external + sub-resources plus the scene itself.
  const header = `[gd_scene load_steps=${models.length + subResources.length + 1} format=3]\n`;
  const extResources = models.map((m) => `[ext_resource type="PackedScene" path="res://${m.file}" id="${extIds.get(m.url)}"]\n`);
  const body = [
    header,
    ...extResources,
    ...subResources.map((r) => r.body),
    `[node name="Root" type="Node3D"]\n`,
    ...nodes.map((n) => `${n}\n`),
  ].join("\n");

  return { text: body, nodeCount: nodes.length, models };
}

export const renderProjectGodot = (name: string): string =>
  `; Generated by @mnemopay/augengine-mcp — augengine.export_godot
config_version=5

[application]

config/name="${name}"
run/main_scene="res://Main.tscn"
config/features=PackedStringArray("4.4", "Forward Plus")

[rendering]

renderer/rendering_method="forward_plus"
`;
