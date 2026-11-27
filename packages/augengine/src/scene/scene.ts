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

export class Scene {
  readonly spec: SceneSpec;
  readonly entities: Map<EntityId, Entity>;
  private worldCache: TransformWorldCache | null = null;
  /** Filled and reused by worldTransforms() so a dirty frame does not allocate the entity list. */
  private entityList: Entity[] = [];
  private dirty = true;
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
  }

  remove(id: EntityId): boolean {
    const removed = this.entities.delete(id);
    if (removed) this.dirty = true;
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
    this.dirty = true;
    return true;
  }

  /** Borrow the world transform cache, recomputing if needed. */
  worldTransforms(): TransformWorldCache {
    if (this.dirty || !this.worldCache) {
      const list = this.entityList;
      list.length = 0;
      for (const e of this.entities.values()) list.push(e);
      this.worldCache = computeWorldTransforms(list, this.worldCache ?? undefined);
      this.dirty = false;
    }
    return this.worldCache;
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
