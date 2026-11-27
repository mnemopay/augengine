/**
 * Transform hierarchy — computes world matrices for every entity from its
 * (position, rotation, scale) components and parent chain. Walks the tree
 * parent-first so each child sees a finalized parent world transform.
 *
 * The output `worldMatrices` is a flat Float32Array(16 * entityCount) keyed
 * by entity index — friendly for direct upload to a GPU storage buffer in
 * phase 9 (skinning compute path).
 */

import type { Entity, EntityId, Mat4, TransformComponent } from "./types.js";
import { Tag } from "./types.js";
import { m4FromTRS, m4Identity, m4Mul } from "./math.js";

export interface TransformWorldCache {
  /** Flat 16-floats-per-entity world matrix array, keyed by `entityIndex`. */
  worldMatrices: Float32Array;
  /** entityId → contiguous index into worldMatrices. */
  indexById: Map<EntityId, number>;
  /** Entity order, parent-first. */
  order: EntityId[];
}

// Reused across calls. The walk is synchronous and not reentrant.
const topoKnown = new Set<EntityId>();
const topoById = new Map<EntityId, Entity>();
const topoVisited = new Set<EntityId>();
const topoOnStack = new Set<EntityId>();
const topoEntityById = new Map<EntityId, Entity>();
const localScratch: Mat4 = new Float32Array(16);

/**
 * Build a parent-first traversal order from a flat entity list. Detects
 * cycles and entities referencing missing parents (treated as roots).
 * `out` is cleared and filled when provided, so a scene can reuse one array.
 */
export function topoSortParentFirst(entities: Entity[], out: EntityId[] = []): EntityId[] {
  topoKnown.clear();
  topoById.clear();
  topoVisited.clear();
  topoOnStack.clear();
  out.length = 0;
  for (const e of entities) {
    topoKnown.add(e.id);
    topoById.set(e.id, e);
  }

  const visit = (id: EntityId): void => {
    if (topoVisited.has(id)) return;
    if (topoOnStack.has(id)) {
      // Cycle — break it; subsequent visits finalize the chain.
      return;
    }
    topoOnStack.add(id);
    const e = topoById.get(id);
    const parent = (e?.components[Tag.Transform] as TransformComponent | undefined)?.parent ?? null;
    if (parent && topoKnown.has(parent) && !topoVisited.has(parent)) {
      visit(parent);
    }
    topoOnStack.delete(id);
    topoVisited.add(id);
    out.push(id);
  };

  for (const e of entities) visit(e.id);
  return out;
}

/**
 * Allocate or refresh world-transform cache for a scene. Call once per
 * scene mutation; reuse across many frames where transforms are unchanged.
 */
export function computeWorldTransforms(
  entities: Entity[],
  scratch?: TransformWorldCache,
): TransformWorldCache {
  const order = topoSortParentFirst(entities, scratch?.order ?? []);
  const indexById = scratch?.indexById ?? new Map<EntityId, number>();
  indexById.clear();
  const worldMatrices =
    scratch?.worldMatrices && scratch.worldMatrices.length === order.length * 16
      ? scratch.worldMatrices
      : new Float32Array(order.length * 16);

  topoEntityById.clear();
  for (const e of entities) topoEntityById.set(e.id, e);

  for (let i = 0; i < order.length; i++) {
    const id = order[i]!;
    indexById.set(id, i);
    const entity = topoEntityById.get(id);
    const tx = entity?.components[Tag.Transform] as TransformComponent | undefined;
    if (!tx) {
      // No transform component — identity world.
      m4Identity(worldMatrices.subarray(i * 16, i * 16 + 16) as Mat4);
      continue;
    }
    m4FromTRS(localScratch, tx.position, tx.rotation, tx.scale);
    if (tx.parent) {
      const parentIdx = indexById.get(tx.parent);
      if (parentIdx !== undefined) {
        const parentWorld = worldMatrices.subarray(
          parentIdx * 16,
          parentIdx * 16 + 16,
        ) as Mat4;
        m4Mul(
          worldMatrices.subarray(i * 16, i * 16 + 16) as Mat4,
          parentWorld,
          localScratch,
        );
        continue;
      }
    }
    // Root or unresolved parent — local matrix becomes world.
    worldMatrices.set(localScratch, i * 16);
  }

  if (scratch) {
    scratch.worldMatrices = worldMatrices;
    scratch.indexById = indexById;
    scratch.order = order;
    return scratch;
  }
  return { worldMatrices, indexById, order };
}

/** Get a borrowed view of an entity's world matrix. Do not mutate. */
export function getWorldMatrix(
  cache: TransformWorldCache,
  id: EntityId,
): Float32Array | null {
  const idx = cache.indexById.get(id);
  if (idx === undefined) return null;
  return cache.worldMatrices.subarray(idx * 16, idx * 16 + 16);
}
