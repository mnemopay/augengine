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
import { readGlb, writeGlb } from "./gltf.js";
import { buildManifest, lodKey, type LodManifest } from "./manifest.js";
import type { KitMaterial, KitMesh } from "./mesh.js";
import { buildKit, impostorFor } from "./kit/index.js";
import { contactSheet, encodePng, renderPreview } from "./preview.js";
import { kitMeshKey } from "./runtime.js";

// Node entry: everything in ./runtime (browser-safe) plus file output and PNG previews.
export * from "./runtime.js";
export * from "./preview.js";

export interface WrittenAsset {
  manifest: LodManifest;
  glbPath: string;
  manifestPath: string;
  previewPath: string | null;
  warnings: string[];
}

/** Bake one mesh and write GLB + manifest (+ a contact-sheet PNG of all LODs). */
export async function writeBaked(
  name: string,
  mesh: KitMesh,
  material: KitMaterial,
  outDir: string,
  opts: BakeOptions & { meshKey?: string; preview?: boolean } = {},
): Promise<{ result: BakeResult; written: WrittenAsset }> {
  await mkdir(outDir, { recursive: true });
  const result = await bakeLods(mesh, opts);
  const meshKey = opts.meshKey ?? name;
  const glbName = `${name}.lods.glb`;
  const glb = writeGlb(result.levels.map((l) => ({ name: lodKey(meshKey, l.level), lod: l.level, mesh: l.mesh })), material);
  const manifest = buildManifest(name, meshKey, glbName, result, material);
  const glbPath = join(outDir, glbName);
  const manifestPath = join(outDir, `${name}.lod.json`);
  await writeFile(glbPath, glb);
  await writeFile(manifestPath, JSON.stringify(manifest, null, 2) + "\n");
  let previewPath: string | null = null;
  if (opts.preview !== false) {
    previewPath = join(outDir, `${name}.lods.png`);
    await writeFile(previewPath, encodePng(contactSheet(result.levels.map((l) => renderPreview(l.mesh, { size: 220 })))));
  }
  return { result, written: { manifest, glbPath, manifestPath, previewPath, warnings: [] } };
}

/** CLI/API entry for an external GLB (e.g. a Blender or Maya glTF export). */
export async function bakeGlbFile(inPath: string, outDir: string, opts: BakeOptions & { preview?: boolean } = {}): Promise<WrittenAsset> {
  const src = readGlb(new Uint8Array(await readFile(inPath)));
  const name = basename(inPath, extname(inPath));
  const { written } = await writeBaked(name, src.mesh, src.material, outDir, {
    ...opts,
    preserveUvSeams: opts.preserveUvSeams ?? src.hasTextures,
  });
  written.warnings.push(...src.warnings);
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
