/**
 * SceneSpec -> Godot 4.4 project tree.
 *
 * `@blackpig/game-assets` also emits Godot projects, but it generates a whole
 * 2D game from a one-line goal — it never reads a SceneSpec. This exporter is
 * the thing `augengine.export_godot` actually promises: the *current* scene
 * graph, written out as a `.tscn` Godot can open, for native targets the
 * browser runtime does not reach.
 *
 * Mapping:
 *   Transform  -> Node3D transform
 *   Mesh       -> MeshInstance3D + Box/Sphere/Cylinder/PlaneMesh sub-resource
 *                 (sizes match the engine primitives, see GODOT_PRIMITIVE_SIZES)
 *   glTF mesh  -> the model's .glb written to models/ and instanced
 *                 (ext_resource PackedScene), one instance per model
 *   Light      -> DirectionalLight3D / OmniLight3D / SpotLight3D
 *   Camera     -> Camera3D (fov converted to degrees)
 *   Persona    -> metadata on the node, so narrative data survives the trip
 *
 * Transform3D is emitted in Godot's four-vector form: x-axis, y-axis, z-axis,
 * origin. Basis axes are the rotation columns scaled componentwise.
 */

import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, isAbsolute, join } from "node:path";
import type { SceneSpec } from "@mnemopay/augengine";
import { renderProjectGodot, renderTscn } from "./godot-tscn.js";

export { renderTscn, renderProjectGodot };

export interface GodotExportResult {
  projectDir: string;
  sceneFile: string;
  nodeCount: number;
  files: string[];
  /** Model urls that could not be written (e.g. in-memory blob: urls), with why. */
  missingModels: { url: string; reason: string }[];
}

const DATA_GLB = /^data:model\/gltf-binary;base64,/;

/** Bytes of a model url: embedded data URI, or a path relative to `assetRoot`. */
async function modelBytes(url: string, assetRoot: string | undefined): Promise<Uint8Array> {
  if (DATA_GLB.test(url)) return new Uint8Array(Buffer.from(url.replace(DATA_GLB, ""), "base64"));
  if (/^blob:/i.test(url)) throw new Error("in-memory model (blob: url) — save the scene with the model embedded or in a project first");
  if (/^https?:/i.test(url)) throw new Error("remote model url — download it into the project first");
  const path = isAbsolute(url) || !assetRoot ? url : join(assetRoot, url);
  return new Uint8Array(await readFile(path));
}

export async function exportGodotProject(
  spec: SceneSpec,
  outputDir: string,
  projectName: string,
  opts: { assetRoot?: string } = {},
): Promise<GodotExportResult> {
  const { text, nodeCount, models } = renderTscn(spec);
  await mkdir(outputDir, { recursive: true });
  const sceneFile = join(outputDir, "Main.tscn");
  const projectFile = join(outputDir, "project.godot");
  await writeFile(sceneFile, text, "utf-8");
  await writeFile(projectFile, renderProjectGodot(projectName), "utf-8");
  const files = [sceneFile, projectFile];
  const missingModels: { url: string; reason: string }[] = [];
  for (const m of models) {
    try {
      const bytes = await modelBytes(m.url, opts.assetRoot);
      const target = join(outputDir, m.file);
      await mkdir(dirname(target), { recursive: true });
      await writeFile(target, bytes);
      files.push(target);
    } catch (error) {
      missingModels.push({ url: m.url.slice(0, 80), reason: error instanceof Error ? error.message : String(error) });
    }
  }
  return { projectDir: outputDir, sceneFile, nodeCount, files, missingModels };
}
