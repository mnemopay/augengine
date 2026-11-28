# Changelog

## 0.2.0-alpha.0 (unreleased, not on npm)

- Preserve material groups and embedded PBR textures through the GLB bake. Generated levels reuse the same images.
- Write a v2 manifest with per-primitive materials, texture memory estimates and impostor atlas layout.
- Choose source normals, material handling and impostor colour from the CLI. The default file bake uses an atlas.
- UV seams no longer stop simplification. Seam vertices slide along their seam with both UVs
  (meshoptimizer's seam handling) instead of being pinned; `uvSeams` / `--seams` is now
  `preserve` (default) | `lock` (pins only junctions: three or more UV islands, or a material
  boundary) | `permissive` (collapses across seams; island-spanning triangles are counted in
  `LodLevelOut.uvIslandBreaks` and warned about). A seam-weld pre-pass (`seamWeld` /
  `--seam-weld`, default half a texel) snaps near-identical UV wedges. Wedges of one position
  now get bit-identical coordinates in the simplification input, which fixes seams that
  opened into cracks when the two sides differed by float noise. Measured LOD0/LOD1/LOD2:
  TRELLIS bust 29,973/7,492/6,168 -> 29,973/7,492/1,872; Studio fishmonger 6,370/2,864/2,858
  -> 6,370/1,592/720; kit buildings unchanged (byte-identical GLBs). New exports:
  `weldUvSeams`, `seamClasses`, `uvIslands`, `uvIslandBreaks`; `BakeResult.seams`.
- The impostor atlas and the contact sheet decode JPEG (jpeg-js) and WebP (`@jsquash/webp`,
  WebAssembly) as well as PNG, so textured impostors carry real colours instead of a white
  silhouette. Large images are box-filtered before sampling. New Node exports:
  `createImageDecoder`, `decodeJpeg`, `decodeWebp`. `textureImage` follows
  `EXT_texture_webp`; `imageSize` reads WebP headers. Both decoders are pure JS/WASM (no native install).
- CLI: `--help` lists every flag and unknown flags are refused. New flags: `--seams`,
  `--seam-weld`, `--impostor cross|box`, `--shading`, `--atlas-tile`, `--name`. `bake`
  prints the seam summary.
- Known limits: on meshes cut into very many UV islands, LOD2 bottoms out near the number of
  seam junctions (fishmonger: 720 triangles, 54.8% below LOD1, short of the 1/16 target;
  `--seams permissive` reaches it with visible texture smears). KTX2/Basis base colours are
  still not decoded for the atlas. In the engine, cross-card atlas impostors draw with a
  black background (the renderer ignores `alphaMode: "MASK"`) and `EXT_texture_webp`
  textures are not loaded; both are engine-side.

- `simplifySmooth(mesh, targetTriangles, maxError)` in `@mnemopay/augengine-lod/runtime`:
  simplification for organic meshes with smooth, seam-safe normals and compacted vertices. In the
  repo since PR #23; not in the published 0.1.0-alpha.0.

## 0.1.0-alpha.0 — 2026-11-23

First public release.

- `augengine-lod bake <in.glb>`: GLB in → LOD0 source, LOD1 (~1/4 triangles) and LOD2 (~1/16)
  via meshoptimizer 1.3.0 error-bounded simplification, LOD3 impostor (crossed quads or
  bounds box), plus an `augengine-lod/1` JSON manifest and a PNG contact sheet.
- `augengine-lod kit`: an original procedural city kit (six buildings, street lamp, bench,
  barrier, three trees, water tower), generated from fixed seeds, baked through the same
  pipeline. Used by AugEngine's Loading Range demo.
- Node entry (`@mnemopay/augengine-lod`) with file output and previews; browser-safe entry
  (`@mnemopay/augengine-lod/runtime`) with in-memory baking, GLB read/write and
  `kitMeshLibrary()`.

Known limits: textures are not carried through the bake; impostors are flat-tinted; FBX,
Draco/meshopt-compressed meshes, sparse accessors and external `.bin` buffers are refused.
