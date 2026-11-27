/**
 * @mnemopay/augengine-lod/runtime — the browser-safe half of the package
 * (no node:fs / node:zlib): baking, the kit, GLB read/write, and the
 * in-memory mesh library a game uploads at load time.
 */
import { bakeLods } from "./bake.js";
import { bounds, toEngineMesh, type EngineMeshData, type KitMesh } from "./mesh.js";
import { buildKit, impostorFor, type KitAsset } from "./kit/index.js";

export * from "./mesh.js";
export * from "./bake.js";
export * from "./gltf.js";
export * from "./manifest.js";
export * from "./kit/index.js";
export { MeshBuilder, rng } from "./builder.js";

/** Engine mesh key stem for a kit asset; the renderer resolves `proc:<generator>:<seed>`. */
export function kitMeshKey(assetId: string): string {
  return `proc:kit-${assetId}`;
}

/** One entry per kit asset, shaped like the engine's `KitCatalogEntry` (runtime/world). */
export interface KitCatalogEntry {
  id: string;
  kind: KitAsset["kind"];
  /** Full extents [x, height, z] in metres (from the mesh bounds). */
  size: [number, number, number];
  metallic: number;
  roughness: number;
}

/**
 * The kit as a placement catalog for `attachStreamedLevel(scene, { kit })`.
 * Synchronous and cheap (no baking): sizes come from each asset's bounds.
 */
export function kitCatalog(assets: KitAsset[] = buildKit()): KitCatalogEntry[] {
  return assets.map((a) => {
    const b = bounds(a.mesh);
    return {
      id: a.id,
      kind: a.kind,
      size: [b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]],
      metallic: a.material.metallic,
      roughness: a.material.roughness,
    };
  });
}

/**
 * Bake the kit in memory and return every LOD as an engine mesh, keyed
 * `proc:kit-<id>:<lod>` so a `{ kind: "procedural", generator: "kit-<id>", seed: lod }`
 * geometry draws it. Upload with `renderer.uploadMesh(key, data, 8)`.
 *
 * Meshes are shifted down by half their height, so an entity placed at the
 * prop's centre (how the engine positions props and colliders) sits on the
 * ground. Horizontally they keep the asset's own origin (footprint centre;
 * the pole for lamps).
 */
export async function kitMeshLibrary(): Promise<{ assets: KitAsset[]; catalog: KitCatalogEntry[]; meshes: { key: string; data: EngineMeshData; triangles: number }[] }> {
  const assets = buildKit();
  const meshes: { key: string; data: EngineMeshData; triangles: number }[] = [];
  for (const a of assets) {
    const b = bounds(a.mesh);
    const r = await bakeLods(shiftY(a.mesh, -(b.min[1] + b.max[1]) / 2), { impostor: impostorFor(a.kind) });
    for (const l of r.levels) meshes.push({ key: `proc:kit-${a.id}:${l.level}`, data: toEngineMesh(l.mesh), triangles: l.triangles });
  }
  return { assets, catalog: kitCatalog(assets), meshes };
}

function shiftY(m: KitMesh, dy: number): KitMesh {
  const positions = m.positions.slice();
  for (let i = 1; i < positions.length; i += 3) positions[i]! += dy;
  return { ...m, positions };
}
