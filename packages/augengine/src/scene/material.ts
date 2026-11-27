/**
 * Material system v2 — data layer.
 *
 * Phase 9 partial. The MaterialKind discriminated union expands the
 * Phase 1 trio (`unlit | pbr | vrm-mtoon | splat`) with concrete
 * parameter sets per kind. The pipeline factory (next session) takes
 * a MaterialKind + scene context and produces a GPURenderPipeline.
 *
 * Why split: the renderer currently uses a single hard-coded pipeline.
 * Switching to per-material pipelines requires a refactor of the live
 * runtime's draw loop. Landing the data types now unblocks that refactor
 * without breaking the existing smoke-cube.
 */

import type { Vec3 } from "./types.js";

export type MaterialKind =
  | UnlitMaterial
  | PBRMaterial
  | CelMaterial
  | OutlineMaterial
  | MToonMaterial
  | SplatMaterial;

export interface UnlitMaterial {
  kind: "unlit";
  color: Vec3;
  opacity?: number;
  /** Optional texture URL — sampled, no lighting. */
  texture?: string;
  /** Self-illumination added on top of the flat color (HDR — values >1 bloom). */
  emissive?: Vec3;
}

export interface PBRMaterial {
  kind: "pbr";
  baseColor: Vec3;
  metallic: number;
  roughness: number;
  emissive?: Vec3;
  /** Texture URLs. PBR runtime samples + multiplies with the scalar params. */
  baseColorTexture?: string;
  metallicRoughnessTexture?: string;
  normalTexture?: string;
  emissiveTexture?: string;
}

export interface CelMaterial {
  kind: "cel";
  baseColor: Vec3;
  /** Discrete light steps; 2-4 reads as anime, 5-8 as soft-cel. */
  steps: number;
  /** Self-illumination added on top of the stepped lighting (HDR — values >1 bloom). */
  emissive?: Vec3;
  /** Where the cel breakpoints live on the lambert scale (0-1). Length = steps. */
  thresholds?: number[];
  /** Outline color (drawn as a separate pass; see OutlineMaterial). */
  outlineColor?: Vec3;
  outlineWidth?: number;
}

export interface OutlineMaterial {
  kind: "outline";
  /** Inverted-hull / screen-space depth-edge / Sobel filter. */
  technique: "inverted-hull" | "depth-edge" | "sobel";
  color: Vec3;
  width: number;
}

export interface MToonMaterial {
  kind: "vrm-mtoon";
  baseColor: Vec3;
  shadeColor: Vec3;
  shadingShift?: number;
  shadingToony?: number;
  rimColor?: Vec3;
  rimLightingMix?: number;
  /** Texture URLs sourced from VRM file's MToon extension. */
  litTexture?: string;
  shadeTexture?: string;
  rimTexture?: string;
}

export interface SplatMaterial {
  kind: "splat";
  /** SPZ-4 URL (Niantic compressed splat container). */
  spzUrl: string;
  /** SH degree to render — 0 (just color) through 4 (full). */
  shDegree?: number;
}

// ──────────────────────────────────────────────────────────────────────
// Material registry — string id → MaterialKind. Lets scenes reference
// materials by id instead of inlining the full definition. Phase 9 work
// on the renderer will read from this registry.
// ──────────────────────────────────────────────────────────────────────

export class MaterialRegistry {
  private byId = new Map<string, MaterialKind>();

  register(id: string, material: MaterialKind): void {
    this.byId.set(id, material);
  }

  get(id: string): MaterialKind | undefined {
    return this.byId.get(id);
  }

  has(id: string): boolean {
    return this.byId.has(id);
  }

  all(): IterableIterator<[string, MaterialKind]> {
    return this.byId.entries();
  }
}

/** Convenience constructors. */
export const Materials = {
  unlit(color: Vec3, opacity = 1): UnlitMaterial {
    return { kind: "unlit", color, opacity };
  },
  pbr(baseColor: Vec3, metallic = 0, roughness = 0.7, emissive?: Vec3): PBRMaterial {
    return { kind: "pbr", baseColor, metallic, roughness, ...(emissive ? { emissive } : {}) };
  },
  cel(baseColor: Vec3, steps = 3): CelMaterial {
    return { kind: "cel", baseColor, steps };
  },
  outline(color: Vec3, width = 0.02, technique: OutlineMaterial["technique"] = "inverted-hull"): OutlineMaterial {
    return { kind: "outline", technique, color, width };
  },
  mtoon(baseColor: Vec3, shadeColor: Vec3): MToonMaterial {
    return { kind: "vrm-mtoon", baseColor, shadeColor };
  },
  splat(spzUrl: string, shDegree = 2): SplatMaterial {
    return { kind: "splat", spzUrl, shDegree };
  },
};
