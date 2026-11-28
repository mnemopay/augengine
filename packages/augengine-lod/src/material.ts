/**
 * material — the source's glTF materials, carried through the bake.
 *
 * Materials, textures, samplers and images are kept as the source authored
 * them (JSON objects copied verbatim, image bytes copied byte-for-byte), so
 * every PBR channel survives: baseColor, metallicRoughness, normal,
 * occlusion and emissive textures and factors, alpha mode, double-sided, and
 * material/texture extensions (KHR_texture_transform, KHR_materials_*,
 * KHR_texture_basisu). KTX2/Basis images are passed through as-is, never
 * re-encoded. Every LOD's primitives point at the same material indices, so
 * an image is stored once per GLB however many levels use it.
 */
import type { KitMaterial, Vec3 } from "./mesh.js";
import { imageSize } from "./png.js";

export interface LodImage {
  bytes: Uint8Array;
  mimeType: string;
  name?: string;
  /** External URI the source referenced instead of embedding (kept as-is). */
  uri?: string;
}

export interface LodMaterialSet {
  /** glTF material objects, verbatim. Texture `index` fields point into `textures`. */
  materials: any[];
  /** glTF texture objects, verbatim (`source`, `sampler`, `extensions.KHR_texture_basisu`). */
  textures: any[];
  /** glTF sampler objects, verbatim. */
  samplers: any[];
  images: LodImage[];
  /** Material/texture extensions the set relies on (copied into the output's `extensionsUsed`). */
  extensionsUsed: string[];
  /** Subset of `extensionsUsed` the source marked required. */
  extensionsRequired: string[];
}

/** Texture slots a glTF material can reference. */
export const TEXTURE_SLOTS = ["baseColor", "metallicRoughness", "normal", "occlusion", "emissive"] as const;
export type TextureSlot = (typeof TEXTURE_SLOTS)[number];

/** The textureInfo object for a slot, or undefined. */
export function textureInfo(mat: any, slot: TextureSlot): { index: number; texCoord?: number; extensions?: Record<string, unknown> } | undefined {
  switch (slot) {
    case "baseColor": return mat?.pbrMetallicRoughness?.baseColorTexture;
    case "metallicRoughness": return mat?.pbrMetallicRoughness?.metallicRoughnessTexture;
    case "normal": return mat?.normalTexture;
    case "occlusion": return mat?.occlusionTexture;
    case "emissive": return mat?.emissiveTexture;
  }
}

/**
 * Image index a texture draws from: the plain `source` (the fallback image
 * when an extension is also present), else the EXT_texture_webp or
 * KHR_texture_basisu one.
 */
export function textureImage(set: LodMaterialSet, textureIndex: number): number | undefined {
  const t = set.textures[textureIndex];
  if (!t) return undefined;
  return t.source ?? t.extensions?.EXT_texture_webp?.source ?? t.extensions?.KHR_texture_basisu?.source;
}

export function hasAnyTexture(set: LodMaterialSet): boolean {
  return set.materials.some((m) => TEXTURE_SLOTS.some((s) => textureInfo(m, s) !== undefined));
}

/** A one-material set from the kit's simple PBR parameters (what v1 bakes wrote). */
export function materialSetFromKit(material: KitMaterial, name = "kit"): LodMaterialSet {
  return {
    materials: [{
      name,
      pbrMetallicRoughness: { baseColorFactor: [...material.baseColor, 1], metallicFactor: material.metallic, roughnessFactor: material.roughness },
      ...(material.emissive ? { emissiveFactor: material.emissive } : {}),
      doubleSided: true,
    }],
    textures: [],
    samplers: [],
    images: [],
    extensionsUsed: [],
    extensionsRequired: [],
  };
}

/** Average of the set's factors, for the v1 `material` field and quick previews. */
export function summarizeMaterials(set: LodMaterialSet, materialsUsed: number[] = set.materials.map((_, i) => i)): KitMaterial {
  const used = materialsUsed.map((i) => set.materials[i]).filter(Boolean);
  if (used.length === 0) return { baseColor: [1, 1, 1], metallic: 1, roughness: 1 };
  const avg = (f: (m: any) => number) => used.reduce((s, m) => s + f(m), 0) / used.length;
  const bc = (k: number) => avg((m) => m.pbrMetallicRoughness?.baseColorFactor?.[k] ?? 1);
  const em: Vec3 = [0, 1, 2].map((k) => avg((m) => m.emissiveFactor?.[k] ?? 0)) as Vec3;
  return {
    baseColor: [bc(0), bc(1), bc(2)],
    metallic: avg((m) => m.pbrMetallicRoughness?.metallicFactor ?? 1),
    roughness: avg((m) => m.pbrMetallicRoughness?.roughnessFactor ?? 1),
    ...(em.some((x) => x > 0) ? { emissive: em } : {}),
  };
}

export interface ImageMemory {
  image: number;
  mimeType: string;
  width: number | null;
  height: number | null;
  fileBytes: number;
  /**
   * Estimated GPU bytes once uploaded: RGBA8 with a full mip chain (x4/3).
   * KTX2 is counted the same way (its transcoded size depends on the GPU's
   * compressed formats, which are 4-8x smaller); null when the size is unknown.
   */
  gpuBytes: number | null;
}

export function imageMemory(set: LodMaterialSet): ImageMemory[] {
  return set.images.map((img, i) => {
    const size = img.bytes.length > 0 ? imageSize(img.bytes, img.mimeType) : null;
    return {
      image: i,
      mimeType: img.mimeType,
      width: size?.width ?? null,
      height: size?.height ?? null,
      fileBytes: img.bytes.length,
      gpuBytes: size ? Math.round(size.width * size.height * 4 * (4 / 3)) : null,
    };
  });
}
