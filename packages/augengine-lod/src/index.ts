/**
 * @mnemopay/augengine-lod — bake LOD chains and the procedural city kit.
 *
 *   bakeGlbFile(in, outDir)  GLB in → `<name>.lods.glb` + `<name>.lod.json` (+ preview PNG)
 *   bakeKit(outDir)          build the kit, bake every asset, write GLBs/manifests/previews
 *   kitMeshLibrary()         in-memory meshes keyed for `renderer.uploadMesh(key, data, 8)`
 *
 * Browsers: import from "@mnemopay/augengine-lod/runtime" (this entry uses node:fs/zlib).
 */
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { basename, extname, join } from "node:path";
import { bakeLods, type BakeOptions, type BakeResult } from "./bake.js";
import { readGlb, withImpostorMaterial, writeLodGlb } from "./gltf.js";
import { buildManifest, lodKey, type LodManifest } from "./manifest.js";
import { materialSetFromKit } from "./material.js";
import type { KitMaterial, KitMesh } from "./mesh.js";
import { createImageDecoder } from "./decode.js";
import { buildKit, impostorFor } from "./kit/index.js";
import { contactSheet, encodePng, renderPreview } from "./preview.js";
import { kitMeshKey } from "./runtime.js";

// Node entry: everything in ./runtime (browser-safe) plus file output and PNG previews.
export * from "./runtime.js";
export * from "./preview.js";
export { createImageDecoder, decodeImageNode, decodeJpeg, decodeWebp, type ImageDecoder } from "./decode.js";

export interface WrittenAsset {
  manifest: LodManifest;
  glbPath: string;
  manifestPath: string;
  previewPath: string | null;
  /** UV-aware bakes: seam counts, the mode used, and island-spanning triangles at LOD1 / LOD2 (0 unless "permissive"). */
  seams?: NonNullable<BakeResult["seams"]> & { uvIslandBreaks: [number, number] };
  warnings: string[];
}

/**
 * Bake one mesh and write GLB + manifest (+ a contact-sheet PNG of all LODs).
 * Pass the source's `materials` (from `readGlb`) to carry textures through;
 * without them the asset gets one material from `material` (the kit).
 * The impostor defaults to a baked atlas here (`impostorTexture: "atlas"`).
 */
export async function writeBaked(
  name: string,
  mesh: KitMesh,
  material: KitMaterial,
  outDir: string,
  opts: BakeOptions & { meshKey?: string; preview?: boolean } = {},
): Promise<{ result: BakeResult; written: WrittenAsset }> {
  await mkdir(outDir, { recursive: true });
  let set = opts.materials ?? materialSetFromKit(material);
  // PNG, JPEG and WebP base colours are decoded for the impostor atlas and the preview.
  const decodeImage = opts.decodeImage ?? await createImageDecoder(set);
  const result = await bakeLods(mesh, { impostorTexture: "atlas", ...opts, decodeImage });
  const meshKey = opts.meshKey ?? name;
  const glbName = `${name}.lods.glb`;
  let impostorMaterial: number | undefined;
  if (result.impostor) ({ set, material: impostorMaterial } = withImpostorMaterial(set, result.impostor, encodePng));
  const glb = writeLodGlb(result.levels.map((l) => ({
    name: lodKey(meshKey, l.level),
    lod: l.level,
    mesh: l.kind === "impostor" && impostorMaterial !== undefined ? { ...l.mesh, groups: [{ material: impostorMaterial, start: 0, count: l.mesh.indices.length }] } : l.mesh,
  })), set);
  const manifest = buildManifest(name, meshKey, glbName, result, material, { set, ...(impostorMaterial !== undefined ? { impostorMaterial } : {}) });
  const glbPath = join(outDir, glbName);
  const manifestPath = join(outDir, `${name}.lod.json`);
  await writeFile(glbPath, glb);
  await writeFile(manifestPath, JSON.stringify(manifest, null, 2) + "\n");
  let previewPath: string | null = null;
  if (opts.preview !== false) {
    previewPath = join(outDir, `${name}.lods.png`);
    const shots = result.levels.map((l, i) => renderPreview(
      i === result.levels.length - 1 && result.impostor && impostorMaterial !== undefined ? { ...l.mesh, groups: [{ material: impostorMaterial, start: 0, count: l.mesh.indices.length }] } : l.mesh,
      { size: 220, materials: set, decodeImage },
    ));
    await writeFile(previewPath, encodePng(contactSheet(shots)));
  }
  const seams = result.seams
    ? { ...result.seams, uvIslandBreaks: [result.levels[1]?.uvIslandBreaks ?? 0, result.levels[2]?.uvIslandBreaks ?? 0] as [number, number] }
    : undefined;
  return { result, written: { manifest, glbPath, manifestPath, previewPath, ...(seams ? { seams } : {}), warnings: [...result.warnings] } };
}

export interface BakeFileOptions extends Omit<BakeOptions, "materials"> {
  preview?: boolean;
  /** "keep" (default): carry materials + textures; "fold": the 0.1 behaviour (factors into vertex colours, no textures). */
  materialMode?: "keep" | "fold";
  /** Sources without NORMAL: "auto"/"flat" (glTF spec) or "smooth" (organic scans). */
  normals?: "auto" | "flat" | "smooth";
  /** Output name stem (default: the input file name). */
  name?: string;
}

/** CLI/API entry for an external GLB (e.g. a Blender or Maya glTF export). */
export async function bakeGlbFile(inPath: string, outDir: string, opts: BakeFileOptions = {}): Promise<WrittenAsset> {
  const src = readGlb(new Uint8Array(await readFile(inPath)), {
    ...(opts.materialMode ? { materials: opts.materialMode } : {}),
    ...(opts.normals ? { missingNormals: opts.normals } : {}),
  });
  const name = opts.name ?? basename(inPath, extname(inPath));
  const { materialMode: _m, normals: _n, name: _nm, ...bake } = opts;
  const { written } = await writeBaked(name, src.mesh, src.material, outDir, {
    ...bake,
    materials: src.materials,
    preserveUvSeams: opts.preserveUvSeams ?? src.hasTextures,
  });
  written.warnings.unshift(...src.warnings);
  return written;
}

/** Build the kit, bake every asset, write outputs + one overview sheet. */
export async function bakeKit(outDir: string): Promise<WrittenAsset[]> {
  const out: WrittenAsset[] = [];
  const lod0s = [];
  for (const asset of buildKit()) {
    const { written, result } = await writeBaked(asset.id, asset.mesh, asset.material, outDir, { meshKey: kitMeshKey(asset.id), impostor: impostorFor(asset.kind) });
    out.push(written);
    lod0s.push(renderPreview(result.levels[0]!.mesh, { size: 200 }));
  }
  await writeFile(join(outDir, "kit-overview.png"), encodePng(contactSheet(lod0s)));
  await writeFile(join(outDir, "kit.json"), JSON.stringify(out.map((w) => w.manifest), null, 2) + "\n");
  return out;
}
