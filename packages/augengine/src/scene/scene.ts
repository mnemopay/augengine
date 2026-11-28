/**
 * Scene — the in-memory representation of a SceneSpec. Owns the entity
 * registry and the transform cache. Both renderers (live + render) and
 * every subsystem (characters, audio, AI) read from this single source
 * of truth.
 *
 * Decision #6 — one loader, one Scene shape, regardless of whether the
 * spec came from `.scene` JSON, a procgen seed, or an LLM prompt.
 */

import type {
  Entity,
  EntityId,
  SceneSpec,
  ComponentTag,
  Vec3,
  Quat,
} from "./types.js";
import { Tag } from "./types.js";
import {
  computeWorldTransforms,
  type TransformWorldCache,
} from "./transform.js";
import { m4FromTRS, m4Mul } from "./math.js";
import type { Mat4 } from "./types.js";

export class Scene {
  readonly spec: SceneSpec;
  readonly entities: Map<EntityId, Entity>;
  private worldCache: TransformWorldCache | null = null;
  /** Filled and reused by worldTransforms() so a dirty frame does not allocate the entity list. */
  private entityList: Entity[] = [];
  private dirty = true;
  /**
   * Set when entities are added/removed or callers mark transforms dirty by
   * hand (they may have re-parented). While false, a dirty cache is refreshed
   * along the cached parent-first order instead of re-sorting every entity.
   */
  private structureDirty = true;
  /** Ids that some entity names as its parent, as of the last full world-transform pass. */
  private readonly parentIds = new Set<EntityId>();
  /** Parent of each cache slot as of the last full pass (re-parenting forces a full pass). */
  private cachedParents: (EntityId | null)[] = [];
  private readonly localScratch: Mat4 = new Float32Array(16);
  private activeCameraOverride: EntityId | null = null;

  constructor(spec: SceneSpec) {
    this.spec = spec;
    this.entities = new Map();
    for (const e of spec.entities) this.entities.set(e.id, e);
  }

  /** Entity id of the currently active camera. Falls back to `spec.defaultCamera`. */
  get activeCamera(): EntityId | null {
    return this.activeCameraOverride ?? this.spec.defaultCamera ?? null;
  }

  /** Runtime override of the active camera (null to revert to `spec.defaultCamera`). */
  set activeCamera(id: EntityId | null) {
    this.activeCameraOverride = id;
  }

  /** Load a SceneSpec from a JSON value, validating required structure. */
  static fromJson(json: unknown): Scene {
    const spec = validateSceneSpec(json);
    return new Scene(spec);
  }

  /** Iterate all entities. */
  *[Symbol.iterator](): IterableIterator<Entity> {
    yield* this.entities.values();
  }

  has(id: EntityId): boolean {
    return this.entities.has(id);
  }

  get(id: EntityId): Entity | undefined {
    return this.entities.get(id);
  }

  add(entity: Entity): void {
    this.entities.set(entity.id, entity);
    this.dirty = true;
    this.structureDirty = true;
  }

  remove(id: EntityId): boolean {
    const removed = this.entities.delete(id);
    if (removed) {
      this.dirty = true;
      this.structureDirty = true;
    }
    return removed;
  }

  /** Find entities that have a given component tag. */
  *withComponent(tag: ComponentTag): IterableIterator<Entity> {
    for (const e of this.entities.values()) {
      if (tag in e.components) yield e;
    }
  }

  /** Mark the world transform cache dirty (after mutating any transform). */
  markTransformsDirty(): void {
    this.dirty = true;
    this.structureDirty = true;
  }

  /** Update entity transform. Cheaper than full add() — only marks dirty. */
  setLocalTransform(
    id: EntityId,
    position: Vec3,
    rotation: Quat,
    scale: Vec3,
  ): boolean {
    const e = this.entities.get(id);
    if (!e) return false;
    const tx = e.components[Tag.Transform];
    if (!tx) return false;
    e.components[Tag.Transform] = {
      ...tx,
      position,
      rotation,
      scale,
    };
    // Fast path: a root with no children only changes its own world matrix,
    // so patch the clean cache in place instead of forcing a full pass over
    // every entity (hurtbox sensors, particles, cameras, props move every
    // frame next to hundreds of static entities).
    if (!this.dirty && this.worldCache && tx.parent === null && !this.parentIds.has(id)) {
      const i = this.worldCache.indexById.get(id);
      if (i !== undefined) {
        m4FromTRS(this.worldCache.worldMatrices.subarray(i * 16, i * 16 + 16) as Mat4, position, rotation, scale);
        return true;
      }
    }
    this.dirty = true;
    return true;
  }

  /** Borrow the world transform cache, recomputing if needed. */
  worldTransforms(): TransformWorldCache {
    if (this.dirty && this.worldCache && !this.structureDirty && this.refreshAlongOrder()) {
      this.dirty = false;
      return this.worldCache;
    }
    if (this.dirty || !this.worldCache) {
      const list = this.entityList;
      list.length = 0;
      for (const e of this.entities.values()) list.push(e);
      this.worldCache = computeWorldTransforms(list, this.worldCache ?? undefined);
      this.parentIds.clear();
      for (const e of list) {
        const p = e.components[Tag.Transform]?.parent;
        if (p) this.parentIds.add(p);
      }
      const order = this.worldCache.order;
      this.cachedParents.length = order.length;
      for (let i = 0; i < order.length; i++) this.cachedParents[i] = this.entities.get(order[i]!)?.components[Tag.Transform]?.parent ?? null;
      this.dirty = false;
      this.structureDirty = false;
    }
    return this.worldCache;
  }

  /**
   * Recompute world matrices along the cached parent-first order (no sort).
   * Returns false, leaving the cache for a full pass, if any entity's parent
   * changed since that order was built.
   */
  private refreshAlongOrder(): boolean {
    const cache = this.worldCache!;
    const order = cache.order;
    const wm = cache.worldMatrices;
    for (let i = 0; i < order.length; i++) {
      const tx = this.entities.get(order[i]!)?.components[Tag.Transform];
      if ((tx?.parent ?? null) !== this.cachedParents[i]) return false;
    }
    const local = this.localScratch;
    for (let i = 0; i < order.length; i++) {
      const tx = this.entities.get(order[i]!)?.components[Tag.Transform];
      const out = wm.subarray(i * 16, i * 16 + 16) as Mat4;
      if (!tx) {
        out.fill(0);
        out[0] = out[5] = out[10] = out[15] = 1;
        continue;
      }
      const pi = tx.parent ? cache.indexById.get(tx.parent) : undefined;
      if (pi === undefined) {
        m4FromTRS(out, tx.position, tx.rotation, tx.scale);
      } else {
        m4FromTRS(local, tx.position, tx.rotation, tx.scale);
        m4Mul(out, wm.subarray(pi * 16, pi * 16 + 16) as Mat4, local);
      }
    }
    return true;
  }

  /** Stable count for buffer sizing. */
  size(): number {
    return this.entities.size;
  }
}

/**
 * Defensive structural validator. Throws on malformed input. Does NOT
 * sanitize component contents beyond their presence — those are the
 * subsystems' jobs (renderer rejects invalid mesh refs, AI rejects
 * malformed personas, etc).
 */
export function validateSceneSpec(value: unknown): SceneSpec {
  if (typeof value !== "object" || value === null) {
    throw new Error("SceneSpec must be an object");
  }
  const v = value as Record<string, unknown>;
  if (v.schemaVersion !== 1) {
    throw new Error(
      `SceneSpec.schemaVersion must be 1, got ${String(v.schemaVersion)}`,
    );
  }
  if (!v.source || typeof v.source !== "object") {
    throw new Error("SceneSpec.source is required");
  }
  if (!Array.isArray(v.entities)) {
    throw new Error("SceneSpec.entities must be an array");
  }
  for (const [i, e] of v.entities.entries()) {
    if (typeof e !== "object" || e === null) {
      throw new Error(`SceneSpec.entities[${i}] must be an object`);
    }
    const ent = e as Record<string, unknown>;
    if (typeof ent.id !== "string") {
      throw new Error(`SceneSpec.entities[${i}].id must be a string`);
    }
    if (typeof ent.name !== "string") {
      throw new Error(`SceneSpec.entities[${i}].name must be a string`);
    }
    if (typeof ent.components !== "object" || ent.components === null) {
      throw new Error(`SceneSpec.entities[${i}].components must be an object`);
    }
  }
  return v as unknown as SceneSpec;
}
