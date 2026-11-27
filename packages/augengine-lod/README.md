# @mnemopay/augengine-lod

LOD baking for [AugEngine](https://augengine.com/), plus an original procedural city kit.
A glTF Binary (`.glb`) goes in; four levels of detail come out (LOD0 source, LOD1 ~1/4
triangles, LOD2 ~1/16, LOD3 an impostor) with a JSON manifest the engine can stream.

```
npm install @mnemopay/augengine-lod
```

Node 18+. **Alpha (`0.1.0-alpha.0`):** the API may change between `0.x` releases.

| Feature | Status |
| --- | --- |
| `bake`: GLB → LOD0–LOD3 + manifest + preview sheet | alpha (verified against the published package: `kit` and `bake` both run) |
| Procedural city kit (`kit`) | alpha |
| Browser-safe in-memory baking (`/runtime`) | alpha |
| Textures carried through the bake | not yet |
| `simplifySmooth` for organic meshes | in the repo, not in the published 0.1.0-alpha.0 |

Part of [AugEngine](https://www.npmjs.com/package/@mnemopay/augengine): the engine's
`selectLod` / `selectLodHysteresis` and the streamed-level demo consume these manifests.

## CLI

```
npx augengine-lod bake model.glb --out lod-out
npx augengine-lod kit --out kit-out
```

`bake` options: `--ratios 0.25,0.0625` (LOD1/LOD2 triangle ratios), `--error 0.02,0.08`
(max simplification error as a fraction of the mesh's bounding radius), `--uv-seams` (keep UV seams; on by
default when the source has textures), `--lock-border` (for tiles that must meet
neighbours), `--no-impostor`, `--no-preview`.

Output per model:

- `<name>.lods.glb`: one mesh per level, named with the engine's `lodMeshKey` convention
  (`name`, `name#lod1`, `name#lod2`, `name#lod3`), each node tagged `extras.lod`.
- `<name>.lod.json`: the manifest (below).
- `<name>.lods.png`: software-rendered contact sheet of all four levels.

Getting a `.glb`: Blender, File > Export > glTF 2.0 (Format: glTF Binary). Maya: the
glTF exporter.

## API

Node (file output, PNG previews):

```js
import { bakeGlbFile, bakeKit } from "@mnemopay/augengine-lod";

const written = await bakeGlbFile("crate.glb", "lod-out", { ratios: [0.25, 0.0625] });
console.log(written.manifest.lods.map((l) => l.triangles));
```

Browser-safe (no `node:fs`/`node:zlib`): baking in memory, GLB read/write, the kit.

```js
import { readGlb, bakeLods, writeGlb, kitMeshLibrary } from "@mnemopay/augengine-lod/runtime";

const { mesh } = readGlb(new Uint8Array(await (await fetch("/crate.glb")).arrayBuffer()));
const { levels } = await bakeLods(mesh, { impostor: "box" });

// The kit, keyed `proc:kit-<id>:<lod>` for renderer.uploadMesh(key, data, 8)
const { catalog, meshes } = await kitMeshLibrary();
```

## Manifest (`augengine-lod/1`)

```json
{
  "format": "augengine-lod/1",
  "name": "crate",
  "meshKey": "crate",
  "glb": "crate.lods.glb",
  "bounds": { "min": [-1, 0, -1], "max": [1, 2, 1] },
  "radius": 1.73,
  "material": { "baseColor": [0.6, 0.5, 0.4], "metallic": 0, "roughness": 0.8 },
  "lods": [
    { "level": 0, "mesh": "crate", "triangles": 1036, "vertices": 702, "error": 0, "kind": "source" },
    { "level": 1, "mesh": "crate#lod1", "triangles": 258, "vertices": 180, "error": 0.004, "kind": "simplified" }
  ],
  "suggestedThresholds": [11, 34, 90]
}
```

`suggestedThresholds` are distances in metres for `selectLod` / `selectLodHysteresis`,
scaled from the engine defaults (32/96/256 m) by object size.

## The kit

Six mid-rise buildings (four facade palettes, lit/dark window grids, ledges, cornice, roof
plant, water tanks), a street lamp, a bench, a jersey barrier, three low-poly trees and a
landmark water tower. All generated in code from fixed seeds: no downloaded or third-party
assets. Per-part colour lives in vertex colours, so each asset is one mesh and one draw.

## Limits

- Textures are not carried through the bake; base-colour factors and vertex colours are.
  The LOD3 impostor is a flat-tinted box or crossed quads, not a baked texture.
- FBX is refused (export glTF Binary instead). Draco- or meshopt-compressed meshes and
  sparse accessors are rejected. Only GLB-embedded buffers are read (no external `.bin`).

## Links

- Engine: [`@mnemopay/augengine`](https://www.npmjs.com/package/@mnemopay/augengine)
- [CHANGELOG](CHANGELOG.md)
- Source: [`packages/augengine-lod`](https://github.com/mnemopay/augengine/tree/main/packages/augengine-lod)

## License

Apache-2.0. See [LICENSE](LICENSE).
