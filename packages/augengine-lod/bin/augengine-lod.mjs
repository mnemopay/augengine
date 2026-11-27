#!/usr/bin/env node
// augengine-lod — bake LOD chains for AugEngine.
//   augengine-lod bake <in.glb> [--out dir] [--ratios 0.25,0.0625] [--error 0.02,0.08]
//                      [--uv-seams] [--lock-border] [--no-impostor] [--no-preview]
//   augengine-lod kit [--out dir]
import { bakeGlbFile, bakeKit } from "../dist/index.js";

const [cmd, ...rest] = process.argv.slice(2);
const flag = (name) => rest.includes(name);
const value = (name, fallback) => {
  const i = rest.indexOf(name);
  return i >= 0 && rest[i + 1] ? rest[i + 1] : fallback;
};
const choice = (name, allowed, fallback) => {
  const selected = value(name, fallback);
  if (!allowed.includes(selected)) throw new Error(name + " expects " + allowed.join(", "));
  return selected;
};
const pair = (s) => {
  const parts = s.split(",").map(Number);
  if (parts.length !== 2 || parts.some((n) => !Number.isFinite(n) || n <= 0)) throw new Error(`expected two positive numbers, got "${s}"`);
  return parts;
};

try {
  if (cmd === "bake") {
    const input = rest[0];
    if (!input || input.startsWith("--")) throw new Error("usage: augengine-lod bake <in.glb> [--out dir] (input file first)");
    if (/\.fbx$/i.test(input)) throw new Error("FBX is not supported. Export glTF Binary (.glb) from Blender (File > Export > glTF 2.0) or Maya (glTF exporter).");
    const w = await bakeGlbFile(input, value("--out", "lod-out"), {
      ratios: pair(value("--ratios", "0.25,0.0625")),
      maxError: pair(value("--error", "0.02,0.08")),
      ...(flag("--uv-seams") ? { preserveUvSeams: true } : {}),
      materialMode: choice("--materials", ["keep", "fold"], "keep"),
      normals: choice("--normals", ["auto", "flat", "smooth"], "auto"),
      impostorTexture: choice("--impostor-texture", ["atlas", "vertex", "none"], "atlas"),
      lockBorder: flag("--lock-border"),
      impostor: !flag("--no-impostor"),
      preview: !flag("--no-preview"),
    });
    for (const l of w.manifest.lods) console.log(`LOD${l.level} ${l.kind.padEnd(10)} ${String(l.triangles).padStart(7)} tris  err ${l.error}`);
    for (const warn of w.warnings) console.warn(`warning: ${warn}`);
    console.log(`wrote ${w.glbPath}\n      ${w.manifestPath}${w.previewPath ? `\n      ${w.previewPath}` : ""}`);
  } else if (cmd === "kit") {
    const all = await bakeKit(value("--out", "kit-out"));
    for (const w of all) console.log(`${w.manifest.name.padEnd(28)} ${w.manifest.lods.map((l) => l.triangles).join(" / ")} tris`);
  } else {
    console.log("usage:\n  augengine-lod bake <in.glb> [--out dir] [--ratios a,b] [--error a,b] [--uv-seams] [--lock-border] [--no-impostor] [--no-preview] [--materials keep|fold] [--normals auto|flat|smooth] [--impostor-texture atlas|vertex|none]\n  augengine-lod kit [--out dir]");
    process.exit(!cmd || cmd === "--help" || cmd === "-h" || cmd === "help" ? 0 : 1);
  }
} catch (e) {
  console.error(`augengine-lod: ${e instanceof Error ? e.message : e}`);
  process.exit(1);
}
