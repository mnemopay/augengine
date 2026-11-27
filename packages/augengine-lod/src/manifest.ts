/**
 * manifest — the JSON sidecar written next to each baked GLB. It tells the
 * engine which mesh key to upload for each LOD (`lodMeshKey` convention) and
 * carries the numbers the tile manager budgets with.
 */
import type { BakeResult } from "./bake.js";
import type { KitMaterial, Vec3 } from "./mesh.js";

export interface LodManifest {
  format: "augengine-lod/1";
  name: string;
  /** Mesh key stem; LOD n uploads as `${meshKey}#lod${n}` (LOD0 as `meshKey`). */
  meshKey: string;
  glb: string;
  bounds: { min: Vec3; max: Vec3 };
  radius: number;
  material: KitMaterial;
  lods: { level: number; mesh: string; triangles: number; vertices: number; error: number; kind: string }[];
  /** Suggested distance thresholds for `selectLod` / `selectLodHysteresis`. */
  suggestedThresholds: [number, number, number];
}

export function lodKey(stem: string, level: number): string {
  return level === 0 ? stem : `${stem}#lod${level}`;
}

export function buildManifest(name: string, meshKey: string, glb: string, r: BakeResult, material: KitMaterial): LodManifest {
  // Scale the engine defaults (32/96/256 m, tuned for building-sized props) by object size,
  // clamped so tiny props don't flip LOD at arm's length.
  const k = Math.min(2, Math.max(0.35, r.radius / 8));
  return {
    format: "augengine-lod/1",
    name,
    meshKey,
    glb,
    bounds: { min: r.bounds.min, max: r.bounds.max },
    radius: r.radius,
    material,
    lods: r.levels.map((l) => ({
      level: l.level,
      mesh: lodKey(meshKey, l.level),
      triangles: l.triangles,
      vertices: l.mesh.positions.length / 3,
      error: Number(l.error.toFixed(5)),
      kind: l.kind,
    })),
    suggestedThresholds: [Math.round(32 * k), Math.round(96 * k), Math.round(256 * k)],
  };
}
