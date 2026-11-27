/**
 * manifest — the JSON sidecar written next to each baked GLB. It tells the
 * engine which mesh key to upload for each LOD (`lodMeshKey` convention) and
 * carries the numbers the tile manager budgets with.
 *
 * `augengine-lod/2` (0.2) is a superset of `augengine-lod/1`: every v1 field
 * is still there with the same meaning (`meshKey`, `lods[].mesh`, `material`,
 * `suggestedThresholds`, ...), plus per-LOD primitives with their material,
 * the materials and the textures they use, texture memory, the shading the
 * chain was baked with and how the impostor carries colour. The engine reads
 * both (`loadLodIntoRenderer` in @mnemopay/augengine/runtime/live).
 */
import type { BakeResult } from "./bake.js";
import type { KitMaterial, Vec3 } from "./mesh.js";
import { groupsOf } from "./mesh.js";
import { imageMemory, materialSetFromKit, textureImage, textureInfo, TEXTURE_SLOTS, type ImageMemory, type LodMaterialSet, type TextureSlot } from "./material.js";

export interface LodManifestLevelV1 {
  level: number;
  /** glTF mesh name in the GLB, and the engine mesh key stem for this level. */
  mesh: string;
  triangles: number;
  vertices: number;
  error: number;
  kind: string;
}

export interface LodManifestV1 {
  format: "augengine-lod/1";
  name: string;
  /** Mesh key stem; LOD n uploads as `${meshKey}#lod${n}` (LOD0 as `meshKey`). */
  meshKey: string;
  glb: string;
  bounds: { min: Vec3; max: Vec3 };
  radius: number;
  material: KitMaterial;
  lods: LodManifestLevelV1[];
  /** Suggested distance thresholds for `selectLod` / `selectLodHysteresis`. */
  suggestedThresholds: [number, number, number];
}

export interface LodManifestMaterial {
  index: number;
  name?: string;
  baseColorFactor: [number, number, number, number];
  metallicFactor: number;
  roughnessFactor: number;
  emissiveFactor: Vec3;
  alphaMode: "OPAQUE" | "MASK" | "BLEND";
  doubleSided: boolean;
  /** Which image each texture slot samples (indices into `textures.images`). */
  textures: Partial<Record<TextureSlot, { image: number; mimeType: string }>>;
}

export interface LodManifestV2 extends Omit<LodManifestV1, "format" | "lods"> {
  format: "augengine-lod/2";
  /** How the chain is shaded: flat (faceted source) or smooth (authored normals kept). */
  shading: "flat" | "smooth";
  lods: (LodManifestLevelV1 & {
    /** One per material group, in the glTF mesh's primitive order. */
    primitives: { primitive: number; material: number; triangles: number }[];
  })[];
  materials: LodManifestMaterial[];
  /** Images in the GLB, each stored once and shared by every level. */
  textures: { images: ImageMemory[]; fileBytes: number; gpuBytes: number };
  impostor: {
    kind: "box" | "cross";
    /** "atlas": baked texture; "vertex": per-face colours; "none": one tint. */
    texture: "atlas" | "vertex" | "none";
    material: number;
    atlas: { image: number; tile: number; cols: number; rows: number; faces: string[] } | null;
  } | null;
}

/** What the baker writes today. */
export type LodManifest = LodManifestV2;

export function lodKey(stem: string, level: number): string {
  return level === 0 ? stem : `${stem}#lod${level}`;
}

export interface ManifestExtras {
  /** Materials as written into the GLB (source materials + the impostor's). */
  set?: LodMaterialSet;
  /** Material index the impostor level uses. */
  impostorMaterial?: number;
}

export function buildManifest(name: string, meshKey: string, glb: string, r: BakeResult, material: KitMaterial, extras: ManifestExtras = {}): LodManifestV2 {
  // Scale the engine defaults (32/96/256 m, tuned for building-sized props) by object size,
  // clamped so tiny props don't flip LOD at arm's length.
  const k = Math.min(2, Math.max(0.35, r.radius / 8));
  const set = extras.set ?? materialSetFromKit(material);
  const images = imageMemory(set);
  const atlasImage = r.impostor?.atlas ? set.images.length - 1 : -1;
  return {
    format: "augengine-lod/2",
    name,
    meshKey,
    glb,
    bounds: { min: r.bounds.min, max: r.bounds.max },
    radius: r.radius,
    material,
    shading: r.shading,
    lods: r.levels.map((l) => {
      const groups = l.kind === "impostor" && extras.impostorMaterial !== undefined
        ? [{ material: extras.impostorMaterial, start: 0, count: l.mesh.indices.length }]
        : groupsOf(l.mesh);
      return {
        level: l.level,
        mesh: lodKey(meshKey, l.level),
        triangles: l.triangles,
        vertices: l.mesh.positions.length / 3,
        error: Number(l.error.toFixed(5)),
        kind: l.kind,
        primitives: groups.filter((g) => g.count > 0).map((g, i) => ({ primitive: i, material: g.material, triangles: g.count / 3 })),
      };
    }),
    materials: set.materials.map((m, index): LodManifestMaterial => {
      const pbr = m?.pbrMetallicRoughness ?? {};
      const textures: LodManifestMaterial["textures"] = {};
      for (const slot of TEXTURE_SLOTS) {
        const ti = textureInfo(m, slot);
        const img = ti ? textureImage(set, ti.index) : undefined;
        if (img !== undefined) textures[slot] = { image: img, mimeType: set.images[img]?.mimeType ?? "image/png" };
      }
      return {
        index,
        ...(m?.name ? { name: m.name } : {}),
        baseColorFactor: pbr.baseColorFactor ?? [1, 1, 1, 1],
        metallicFactor: pbr.metallicFactor ?? 1,
        roughnessFactor: pbr.roughnessFactor ?? 1,
        emissiveFactor: m?.emissiveFactor ?? [0, 0, 0],
        alphaMode: m?.alphaMode ?? "OPAQUE",
        doubleSided: m?.doubleSided ?? false,
        textures,
      };
    }),
    textures: {
      images,
      fileBytes: images.reduce((s, i) => s + i.fileBytes, 0),
      gpuBytes: images.reduce((s, i) => s + (i.gpuBytes ?? 0), 0),
    },
    impostor: r.impostor
      ? {
        kind: r.impostor.kind,
        texture: r.impostor.texture,
        material: extras.impostorMaterial ?? 0,
        atlas: r.impostor.layout && atlasImage >= 0 ? { image: atlasImage, ...r.impostor.layout } : null,
      }
      : null,
    suggestedThresholds: [Math.round(32 * k), Math.round(96 * k), Math.round(256 * k)],
  };
}
