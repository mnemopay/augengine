# Changelog

## Unreleased

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
