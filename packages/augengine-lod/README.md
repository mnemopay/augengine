# @mnemopay/augengine-lod

LOD baking for [AugEngine](https://augengine.com/), plus an original procedural city kit.
A glTF Binary (`.glb`) goes in; four levels of detail come out (LOD0 source, LOD1 ~1/4
triangles, LOD2 ~1/16, LOD3 an impostor) with a JSON manifest the engine can stream.

```
npm install @mnemopay/augengine-lod
```

Node 18+. **Alpha.** npm has `0.1.0-alpha.0`; this repo is at `0.2.0-alpha.0` (material-aware baking, v2 manifest), **not yet published**. The API may change between `0.x` releases.

| Feature | Status |
| --- | --- |
| `bake`: GLB → LOD0–LOD3 + manifest + preview sheet | alpha (verified against the published package: `kit` and `bake` both run) |
| Procedural city kit (`kit`) | alpha |
| Browser-safe in-memory baking (`/runtime`) | alpha |
| Textures and material groups carried through the bake, `augengine-lod/2` manifest | alpha, repo only (0.2.0-alpha.0, unpublished) |
| UV-seam-aware simplification of textured meshes (seams stay closed, no island-spanning triangles) | alpha, repo only |
| Textured impostor atlas (LOD3) from PNG, JPEG and WebP base colours | alpha, repo only |
| `simplifySmooth` for organic meshes | in the repo, not in the published 0.1.0-alpha.0 |

Part of [AugEngine](https://www.npmjs.com/package/@mnemopay/augengine): the engine's
`selectLod` / `selectLodHysteresis` and the streamed-level demo consume these manifests.

## CLI

```
npx augengine-lod bake model.glb --out lod-out
npx augengine-lod kit --out kit-out
```

`augengine-lod --help` lists every flag; unknown flags are refused. `bake` options:

| Flag | Default | What it does |
| --- | --- | --- |
| `--out <dir>` | `lod-out` | output directory |
| `--name <stem>` | input file name | output file stem |
| `--ratios a,b` | `0.25,0.0625` | LOD1 / LOD2 triangle ratios of LOD0 |
| `--error a,b` | `0.02,0.08` | max simplification error, fraction of the bounding radius |
| `--materials keep\|fold` | `keep` | keep materials + textures, or fold factors into vertex colours (0.1) |
| `--normals auto\|flat\|smooth` | `auto` | sources without `NORMAL`: flat (glTF default) or smooth |
| `--shading auto\|flat\|smooth` | `auto` | shading of the baked chain (auto: from the source) |
| `--uv-seams` | on when textured | weld by UV even without textures |
| `--seams preserve\|lock\|permissive` | `preserve` | UV seam handling (below) |
| `--seam-weld <texels>` | `0.5` | snap UV wedges closer than this before simplifying; `0` = off |
| `--lock-border` | off | never move open-border vertices (tiles that must meet) |
| `--impostor cross\|box` | `cross` | LOD3 shape |
| `--no-impostor` | | no LOD3 |
| `--impostor-texture atlas\|vertex\|none` | `atlas` | LOD3 colour: baked atlas, per-face colour, one tint |
| `--atlas-tile <px>` | `64` | atlas pixels per impostor view |
| `--no-preview` | | skip the contact sheet |

`kit` takes `--out <dir>` (default `kit-out`).

### UV seams

A textured mesh is split wherever its UV map is cut; image-to-3D output (TRELLIS,
generated characters) is cut into hundreds of islands, so 20-45% of its positions sit on a
seam. The bake handles that in three steps:

1. **Seam weld.** UV wedges at one position closer than half a texel of the material's
   base-colour image are snapped together (exporter rounding, not a cut).
2. **`preserve` (default).** Real seams stay closed: a seam vertex only slides along its
   seam with both of its UVs, and junctions (three or more islands, or a material
   boundary) stay. Every output triangle samples one UV island.
3. **`lock`** additionally pins the junctions explicitly (0.2.0-alpha.0 pinned *every*
   vertex with a UV split, which stopped these meshes near LOD1). **`permissive`** lets
   collapses cross seams: far more reduction, but triangles spanning two islands stretch
   the texture across the atlas. The CLI prints how many there are per level.

Measured (default flags, `--normals auto`; triangles LOD0 / LOD1 / LOD2, reduction per level):

| Input | 0.2.0-alpha.0 (seams locked) | Now (`preserve`) | Island-spanning triangles |
| --- | --- | --- | --- |
| Kit building (`building-1-5-23`, untextured) | 1,036 / 258 / 64 | 1,036 / 258 / 64 (75.1% / 75.2%, byte-identical GLB) | n/a |
| TRELLIS bust (`desmond_bust.glb`, 289 seam junctions) | 29,973 / 7,492 / 6,168 | 29,973 / 7,492 / 1,872 (75.0% / 75.0%) | 0 / 0 |
| Normal-mapped test sphere (two materials, u = 0/1 seam) | 1,472 / 368 / 226 | 1,472 / 368 / 92 (75.0% / 75.0%) | 0 / 0 |
| Generated character (Studio fishmonger, 199 UV islands, 343 junctions) | 6,370 / 2,864 / 2,858 | 6,370 / 1,592 / 720 (75.0% / **54.8%**) | 0 / 0 |
| same character, `--seams permissive` | | 6,370 / 1,592 / 398 (75.0% / 75.0%) | 55 / 122 (about 18% of LOD2's area) |

The character's LOD2 stops near 720 triangles because seam junctions cannot be collapsed
without breaking texture continuity, and that character has 343 of them; see Limits.

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

## Manifest (`augengine-lod/1`; 0.2 writes `augengine-lod/2`, a superset)

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

- Published 0.1.0-alpha.0: textures are not carried through the bake; base-colour factors
  and vertex colours are, and the LOD3 impostor is flat-tinted. The repo's 0.2.0-alpha.0
  keeps textures (below).
- 0.2 with textures kept: seam junctions (three or more UV islands meeting) never move,
  so a mesh cut into very many islands bottoms out near its junction count. On the Studio
  fishmonger (199 islands, 343 junctions) LOD2 is 720 triangles: 54.8% fewer than LOD1,
  short of the 1/16 target (398). `--seams permissive` reaches 398 but 122 of those
  triangles span two islands and visibly smear the texture (hands, face, boots). The
  TRELLIS busts and the kit reach the targets. A few open edges can appear (2 on the
  fishmonger's LOD1/LOD2, 2-5 on the busts, which already have 3 in the source).
- The impostor atlas decodes PNG, JPEG (jpeg-js) and WebP (libwebp as WebAssembly, via
  `@jsquash/webp`) in Node. KTX2/Basis base colours still fall back to the material factor
  (a warning is printed). The browser entry (`/runtime`) has no decoders: pass `decodeImage`.
- Engine side (not this package): AugEngine's WebGPU renderer does not honour
  `alphaMode: "MASK"`, so the default `cross` impostor's atlas draws as an opaque card with
  a black background. Use `--impostor box` or `--impostor-texture vertex` for engine
  assets until it does. The engine's glTF loader also does not decode `EXT_texture_webp`
  textures, so a WebP-only source renders untextured in the engine (the baked atlas, a PNG,
  is fine).
- FBX is refused (export glTF Binary instead). Draco- or meshopt-compressed meshes and
  sparse accessors are rejected. Only GLB-embedded buffers are read (no external `.bin`).

## Links

- Engine: [`@mnemopay/augengine`](https://www.npmjs.com/package/@mnemopay/augengine)
- [CHANGELOG](CHANGELOG.md)
- Source: [`packages/augengine-lod`](https://github.com/mnemopay/augengine/tree/main/packages/augengine-lod)

## License

Apache-2.0. See [LICENSE](LICENSE).


### Material-aware baking (0.2 alpha)

Use `augengine-lod bake model.glb --materials keep --normals auto --impostor-texture atlas`. Source images are shared between detail levels. The v2 manifest adds material assignments and image sizes to the existing level keys and distance thresholds. Use `--materials fold` for the earlier factor-to-vertex-colour behavior. The Node entry decodes PNG, JPEG and WebP base colours for the impostor atlas and the `.lods.png` contact sheet (`createImageDecoder`); other embedded formats are carried through in the GLB untouched. The engine reads both manifest versions with `loadLodIntoRenderer` (`@mnemopay/augengine`, live runtime).

Why these decoders: jpeg-js is pure JavaScript with no dependencies, and `@jsquash/webp` is libwebp compiled to WebAssembly (about 140 KB for the decoder). Neither needs a compiler or a platform binary, so `npx @mnemopay/augengine-lod` installs anywhere Node 18+ runs. sharp would have decoded all three, but it ships a native libvips build per platform (tens of MB) that can fail to install.
