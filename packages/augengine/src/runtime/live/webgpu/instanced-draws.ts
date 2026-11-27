/**
 * Instanced draws — pure planning for the live renderer.
 *
 * The renderer used to issue one drawIndexed per visible entity, each with
 * its own 112-byte draw uniform. Entities that share a mesh and a pipeline
 * variant (and need nothing per-entity beyond the draw uniform: no skin, no
 * morph, no textures, no vertex colours) can instead share one storage
 * buffer of draw records and one drawIndexed(indexCount, n).
 *
 * This module holds the GPU-free parts so they are testable in Node:
 * grouping, the threshold, buffer capacity, and packing one record.
 * Culling stays per entity: only entities whose mesh is visible this frame
 * are offered for grouping, so a hidden member is simply absent from the
 * instance buffer that frame.
 */
import type { Mat4 } from "../../../scene/types.js";

/** Floats per draw record: mat4 model + color + params + emissive (112 bytes). */
export const DRAW_RECORD_FLOATS = 28;

/** Below this many members a group draws per entity (break-even on Intel UHD). */
export const INSTANCE_MIN_GROUP = 4;

export interface GroupPlan<T> {
  /** Groups large enough to draw instanced, in first-seen order. */
  groups: { key: string; items: T[] }[];
  /** Members of small groups: draw these individually. */
  singles: T[];
}

/**
 * Group candidates by key (pipeline variant + mesh). Stable: groups keep
 * first-seen order and members keep input order, so instance indices are
 * deterministic frame to frame for an unchanged scene.
 */
export function planInstanceGroups<T>(
  candidates: readonly { key: string; item: T }[],
  minGroup = INSTANCE_MIN_GROUP,
): GroupPlan<T> {
  const byKey = new Map<string, T[]>();
  for (const c of candidates) {
    let arr = byKey.get(c.key);
    if (!arr) { arr = []; byKey.set(c.key, arr); }
    arr.push(c.item);
  }
  const groups: { key: string; items: T[] }[] = [];
  const singles: T[] = [];
  for (const [key, items] of byKey) {
    if (items.length >= minGroup) groups.push({ key, items });
    else singles.push(...items);
  }
  return { groups, singles };
}

/** Storage capacity (records) for a group of `n`: next power of two, min 16. */
export function instanceCapacity(n: number): number {
  let c = 16;
  while (c < n) c *= 2;
  return c;
}

/** Write one draw record at slot `i` of `out` (the same layout as the per-entity uniform). */
export function packDrawRecord(
  out: Float32Array,
  i: number,
  model: Mat4 | Float32Array,
  color: readonly number[],
  params: readonly number[],
  emissive: readonly number[],
): void {
  const o = i * DRAW_RECORD_FLOATS;
  out.set(model as ArrayLike<number>, o);
  out[o + 16] = color[0]!; out[o + 17] = color[1]!; out[o + 18] = color[2]!; out[o + 19] = 1;
  out[o + 20] = params[0]!; out[o + 21] = params[1]!; out[o + 22] = params[2]!; out[o + 23] = params[3]!;
  out[o + 24] = emissive[0]!; out[o + 25] = emissive[1]!; out[o + 26] = emissive[2]!; out[o + 27] = 0;
}
