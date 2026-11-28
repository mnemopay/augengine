#!/usr/bin/env node
// augengine-lod — bake LOD chains for AugEngine. `augengine-lod --help` lists every flag.
import { bakeGlbFile, bakeKit } from "../dist/index.js";

const HELP = `usage:
  augengine-lod bake <in.glb> [options]
  augengine-lod kit [--out dir]
  augengine-lod --help

bake options:
  --out <dir>                      output directory (default lod-out)
  --name <stem>                    output file stem (default: the input file name)
  --ratios <a,b>                   LOD1/LOD2 triangle ratios of LOD0 (default 0.25,0.0625)
  --error <a,b>                    max simplification error for LOD1/LOD2, fraction of the
                                   bounding radius (default 0.02,0.08)
  --materials keep|fold            keep materials + textures (default) or fold factors into
                                   vertex colours (the 0.1 behaviour)
  --normals auto|flat|smooth       sources without NORMAL: flat (glTF default) or smooth
  --shading auto|flat|smooth       shading of the baked chain (default auto: from the source)
  --uv-seams                       weld by UV even without textures (on when textured)
  --seams preserve|lock|permissive UV seams: preserve (default) keeps seams closed and every
                                   corner on its own UV island; lock also pins seam junctions;
                                   permissive collapses across seams (more reduction, texture
                                   stretching: the triangles that span two islands are reported)
  --seam-weld <texels>             snap UV wedges closer than this many texels before
                                   simplifying (default 0.5; 0 turns it off)
  --lock-border                    never move open-border vertices (tiles that must meet)
  --impostor cross|box             LOD3 shape (default cross)
  --no-impostor                    no LOD3
  --impostor-texture atlas|vertex|none
                                   LOD3 colour: baked atlas (default), per-face colour, one tint
  --atlas-tile <px>                atlas pixels per impostor view (default 64)
  --no-preview                     skip the <name>.lods.png contact sheet

kit options:
  --out <dir>                      output directory (default kit-out)`;

const [cmd, ...rest] = process.argv.slice(2);
const VALUE_FLAGS = {
  bake: ["--out", "--name", "--ratios", "--error", "--materials", "--normals", "--shading", "--seams", "--seam-weld", "--impostor", "--impostor-texture", "--atlas-tile"],
  kit: ["--out"],
};
const BOOL_FLAGS = { bake: ["--uv-seams", "--lock-border", "--no-impostor", "--no-preview"], kit: [] };

/** Reject flags the command does not know (so --help really lists every flag). */
function checkFlags(command, args) {
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (!a.startsWith("--")) continue;
    if (VALUE_FLAGS[command].includes(a)) {
      if (args[i + 1] === undefined || args[i + 1].startsWith("--")) throw new Error(`${a} expects a value`);
      i++;
    } else if (!BOOL_FLAGS[command].includes(a)) {
      throw new Error(`unknown option ${a} (see augengine-lod --help)`);
    }
  }
}
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
const number = (name, fallback, min) => {
  const n = Number(value(name, String(fallback)));
  if (!Number.isFinite(n) || n < min) throw new Error(`${name} expects a number >= ${min}`);
  return n;
};

try {
  if (!cmd || cmd === "--help" || cmd === "-h" || cmd === "help" || rest.includes("--help") || rest.includes("-h")) {
    console.log(HELP);
    process.exit(cmd && !["--help", "-h", "help", "bake", "kit"].includes(cmd) ? 1 : 0);
  }
  if (cmd === "bake") {
    const input = rest[0];
    if (!input || input.startsWith("--")) throw new Error("usage: augengine-lod bake <in.glb> [--out dir] (input file first)");
    if (/\.fbx$/i.test(input)) throw new Error("FBX is not supported. Export glTF Binary (.glb) from Blender (File > Export > glTF 2.0) or Maya (glTF exporter).");
    checkFlags("bake", rest.slice(1));
    const impostor = flag("--no-impostor") ? false : choice("--impostor", ["cross", "box"], "cross");
    const shading = choice("--shading", ["auto", "flat", "smooth"], "auto");
    const w = await bakeGlbFile(input, value("--out", "lod-out"), {
      ratios: pair(value("--ratios", "0.25,0.0625")),
      maxError: pair(value("--error", "0.02,0.08")),
      ...(flag("--uv-seams") ? { preserveUvSeams: true } : {}),
      ...(value("--name") ? { name: value("--name") } : {}),
      materialMode: choice("--materials", ["keep", "fold"], "keep"),
      normals: choice("--normals", ["auto", "flat", "smooth"], "auto"),
      shading,
      uvSeams: choice("--seams", ["preserve", "lock", "permissive"], "preserve"),
      seamWeld: number("--seam-weld", 0.5, 0),
      impostorTexture: choice("--impostor-texture", ["atlas", "vertex", "none"], "atlas"),
      atlasTile: number("--atlas-tile", 64, 8),
      lockBorder: flag("--lock-border"),
      impostor,
      preview: !flag("--no-preview"),
    });
    for (const l of w.manifest.lods) console.log(`LOD${l.level} ${l.kind.padEnd(10)} ${String(l.triangles).padStart(7)} tris  err ${l.error}`);
    if (w.seams) {
      const s = w.seams;
      console.log(`seams ${s.mode}: ${s.seamPositions} of ${s.positions} positions on a UV seam, ${s.junctions} junctions, ${s.snapped} corners welded; island-spanning triangles LOD1 ${s.uvIslandBreaks[0]}, LOD2 ${s.uvIslandBreaks[1]}`);
    }
    for (const warn of w.warnings) console.warn(`warning: ${warn}`);
    console.log(`wrote ${w.glbPath}\n      ${w.manifestPath}${w.previewPath ? `\n      ${w.previewPath}` : ""}`);
  } else if (cmd === "kit") {
    checkFlags("kit", rest);
    const all = await bakeKit(value("--out", "kit-out"));
    for (const w of all) console.log(`${w.manifest.name.padEnd(28)} ${w.manifest.lods.map((l) => l.triangles).join(" / ")} tris`);
  } else {
    console.error(`augengine-lod: unknown command "${cmd}"\n\n${HELP}`);
    process.exit(1);
  }
} catch (e) {
  console.error(`augengine-lod: ${e instanceof Error ? e.message : e}`);
  process.exit(1);
}
