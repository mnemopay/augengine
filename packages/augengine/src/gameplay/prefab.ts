/**
 * Prefabs: compose entities from JSON.
 *
 * A prefab is a small entity tree with local ids. Instantiating it adds real
 * entities to a scene under a unique id prefix, at a placement. Prefabs can
 * nest other prefabs (cycles are rejected), and per-instance overrides patch
 * components by local id. Gameplay behaviours that are not scene components
 * (trigger volumes, spawners, checkpoints) ride along under `gameplay` and are
 * returned resolved to world space for the game to wire up.
 *
 * Physics requires collider/rigid-body entities to be scene roots, so by
 * default their world transform is baked at instantiation and they are
 * attached to no parent (`flattenPhysics: false` keeps the hierarchy).
 *
 * `parsePrefab` validates untrusted JSON and reports every problem with its
 * path ("entities[2].transform.scale: expected 3 numbers").
 */

import type { Scene } from "../scene/scene.js";
import { Tag, type Entity, type EntityId, type Quat, type Vec3 } from "../scene/types.js";

export interface PrefabTransform {
  position?: Vec3;
  /** Quaternion [x, y, z, w]; or use `yaw` (degrees about +Y). */
  rotation?: Quat;
  yaw?: number;
  scale?: Vec3;
}

export interface TriggerSpec {
  shape: "box" | "sphere";
  /** Box half extents (world metres). */
  halfExtents?: Vec3;
  radius?: number;
  /** Fire once, then disable. */
  once?: boolean;
  /** Free-form tag the game routes on ("start-wave", "kill-zone"). */
  event: string;
}

export interface SpawnerSpec {
  /** What to spawn (the game maps it to a prefab or a character). */
  kind: string;
  /** Spawn points relative to the entity (default: the entity origin). */
  points?: Vec3[];
  /** Total to spawn (default Infinity). */
  count?: number;
  /** Seconds between spawns (default 1). */
  interval?: number;
  /** Never exceed this many alive (default 4). */
  maxAlive?: number;
  /** Start only when this trigger event fires (default: active at once). */
  activateOn?: string;
  seed?: number;
}

export interface CheckpointSpec {
  /** Ordered: reaching a higher order makes it the respawn point. */
  order: number;
  radius?: number;
}

export interface PrefabEntitySpec {
  /** Local id, unique within the prefab. */
  id: string;
  name?: string;
  /** Local id of the parent within this prefab. */
  parent?: string;
  transform?: PrefabTransform;
  /** Scene components by tag (mesh, collider, light, rigid_body, …), copied as-is. */
  components?: Partial<Entity["components"]>;
  /** Instantiate another prefab here (its root becomes this entity's child). */
  prefab?: string;
  gameplay?: { trigger?: TriggerSpec; spawner?: SpawnerSpec; checkpoint?: CheckpointSpec };
}

export interface PrefabSpec {
  name: string;
  entities: PrefabEntitySpec[];
}

export interface PrefabPlacement {
  position?: Vec3;
  rotation?: Quat;
  yaw?: number;
  scale?: Vec3;
  /** Patch components of local ids: { "door": { mesh: { visible: false } } } (shallow per component). */
  overrides?: Record<string, Partial<Record<string, unknown>>>;
  flattenPhysics?: boolean;
}

export interface PlacedTrigger extends TriggerSpec { id: EntityId; center: Vec3 }
export interface PlacedSpawner extends SpawnerSpec { id: EntityId; worldPoints: Vec3[] }
export interface PlacedCheckpoint extends CheckpointSpec { id: EntityId; position: Vec3 }

export interface PrefabInstance {
  rootId: EntityId;
  /** Every entity added, in creation order. */
  entityIds: EntityId[];
  /** Local id path ("gate", "tower/door") → scene id. */
  ids: Map<string, EntityId>;
  triggers: PlacedTrigger[];
  spawners: PlacedSpawner[];
  checkpoints: PlacedCheckpoint[];
}

// ──────────────────────────────────────────────────────────────────────
// Validation
// ──────────────────────────────────────────────────────────────────────

export class PrefabError extends Error {
  constructor(readonly problems: string[]) {
    super(`Invalid prefab:\n  ${problems.join("\n  ")}`);
  }
}

const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const isVec = (v: unknown, n: number) => Array.isArray(v) && v.length === n && v.every(isNum);
const KNOWN_COMPONENTS = new Set<string>(Object.values(Tag));

/** Validate untrusted JSON as a PrefabSpec. Throws PrefabError listing every problem. */
export function parsePrefab(json: unknown): PrefabSpec {
  const p: string[] = [];
  if (typeof json !== "object" || json === null || Array.isArray(json)) throw new PrefabError(["prefab: expected an object"]);
  const o = json as Record<string, unknown>;
  if (typeof o.name !== "string" || !o.name) p.push("name: expected a non-empty string");
  if (!Array.isArray(o.entities) || o.entities.length === 0) {
    p.push("entities: expected a non-empty array");
    throw new PrefabError(p);
  }
  const ids = new Set<string>();
  o.entities.forEach((e, i) => {
    const at = `entities[${i}]`;
    if (typeof e !== "object" || e === null) { p.push(`${at}: expected an object`); return; }
    const ent = e as Record<string, unknown>;
    if (typeof ent.id !== "string" || !ent.id) p.push(`${at}.id: expected a non-empty string`);
    else if (ent.id.includes("/")) p.push(`${at}.id: "/" is reserved for nested paths`);
    else if (ids.has(ent.id)) p.push(`${at}.id: duplicate "${ent.id}"`);
    else ids.add(ent.id);
    if (ent.name !== undefined && typeof ent.name !== "string") p.push(`${at}.name: expected a string`);
    if (ent.parent !== undefined && typeof ent.parent !== "string") p.push(`${at}.parent: expected a local id`);
    if (ent.prefab !== undefined && typeof ent.prefab !== "string") p.push(`${at}.prefab: expected a prefab name`);
    const t = ent.transform as Record<string, unknown> | undefined;
    if (t !== undefined) {
      if (typeof t !== "object" || t === null) p.push(`${at}.transform: expected an object`);
      else {
        if (t.position !== undefined && !isVec(t.position, 3)) p.push(`${at}.transform.position: expected 3 numbers`);
        if (t.scale !== undefined && (!isVec(t.scale, 3) || (t.scale as number[]).some((x) => x <= 0))) p.push(`${at}.transform.scale: expected 3 positive numbers`);
        if (t.rotation !== undefined && (!isVec(t.rotation, 4) || Math.abs(Math.hypot(...(t.rotation as number[])) - 1) > 1e-3)) p.push(`${at}.transform.rotation: expected a unit quaternion [x, y, z, w]`);
        if (t.yaw !== undefined && !isNum(t.yaw)) p.push(`${at}.transform.yaw: expected degrees`);
        if (t.yaw !== undefined && t.rotation !== undefined) p.push(`${at}.transform: give rotation or yaw, not both`);
      }
    }
    const c = ent.components as Record<string, unknown> | undefined;
    if (c !== undefined) {
      if (typeof c !== "object" || c === null) p.push(`${at}.components: expected an object`);
      else for (const k of Object.keys(c)) {
        if (!KNOWN_COMPONENTS.has(k)) p.push(`${at}.components.${k}: unknown component (known: ${[...KNOWN_COMPONENTS].join(", ")})`);
        if (k === Tag.Transform) p.push(`${at}.components.transform: use "transform" on the entity, not a component`);
      }
    }
    const g = ent.gameplay as Record<string, unknown> | undefined;
    if (g !== undefined) {
      const tr = g.trigger as Record<string, unknown> | undefined;
      if (tr) {
        if (tr.shape !== "box" && tr.shape !== "sphere") p.push(`${at}.gameplay.trigger.shape: expected "box" or "sphere"`);
        if (tr.shape === "box" && !isVec(tr.halfExtents, 3)) p.push(`${at}.gameplay.trigger.halfExtents: expected 3 numbers`);
        if (tr.shape === "sphere" && !(isNum(tr.radius) && tr.radius > 0)) p.push(`${at}.gameplay.trigger.radius: expected a positive number`);
        if (typeof tr.event !== "string" || !tr.event) p.push(`${at}.gameplay.trigger.event: expected a string`);
      }
      const sp = g.spawner as Record<string, unknown> | undefined;
      if (sp) {
        if (typeof sp.kind !== "string" || !sp.kind) p.push(`${at}.gameplay.spawner.kind: expected a string`);
        if (sp.points !== undefined && (!Array.isArray(sp.points) || !sp.points.every((q) => isVec(q, 3)))) p.push(`${at}.gameplay.spawner.points: expected an array of 3-number points`);
        for (const k of ["count", "interval", "maxAlive", "seed"] as const) if (sp[k] !== undefined && !(isNum(sp[k]) && (sp[k] as number) >= 0)) p.push(`${at}.gameplay.spawner.${k}: expected a non-negative number`);
      }
      const cp = g.checkpoint as Record<string, unknown> | undefined;
      if (cp && !isNum(cp.order)) p.push(`${at}.gameplay.checkpoint.order: expected a number`);
    }
  });
  o.entities.forEach((e, i) => {
    const ent = e as Record<string, unknown>;
    if (typeof ent?.parent === "string" && !ids.has(ent.parent)) p.push(`entities[${i}].parent: unknown local id "${ent.parent}"`);
  });
  // Parent cycles inside the prefab.
  const parentOf = new Map<string, string>();
  for (const e of o.entities as Array<Record<string, unknown>>) if (typeof e?.id === "string" && typeof e.parent === "string") parentOf.set(e.id, e.parent);
  for (const id of parentOf.keys()) {
    const seen = new Set<string>();
    for (let c: string | undefined = id; c; c = parentOf.get(c)) {
      if (seen.has(c)) { p.push(`entities: parent cycle through "${id}"`); break; }
      seen.add(c);
    }
  }
  if (p.length) throw new PrefabError(p);
  return json as PrefabSpec;
}

// ──────────────────────────────────────────────────────────────────────
// Instantiation
// ──────────────────────────────────────────────────────────────────────

type M = Float64Array;

function trs(t: Vec3, r: Quat, s: Vec3): M {
  const [x, y, z, w] = r;
  const m = new Float64Array(16);
  m[0] = (1 - 2 * (y * y + z * z)) * s[0]; m[1] = 2 * (x * y + w * z) * s[0]; m[2] = 2 * (x * z - w * y) * s[0];
  m[4] = 2 * (x * y - w * z) * s[1]; m[5] = (1 - 2 * (x * x + z * z)) * s[1]; m[6] = 2 * (y * z + w * x) * s[1];
  m[8] = 2 * (x * z + w * y) * s[2]; m[9] = 2 * (y * z - w * x) * s[2]; m[10] = (1 - 2 * (x * x + y * y)) * s[2];
  m[12] = t[0]; m[13] = t[1]; m[14] = t[2]; m[15] = 1;
  return m;
}

function mul(a: M, b: M): M {
  const o = new Float64Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) o[c * 4 + r] = a[r]! * b[c * 4]! + a[4 + r]! * b[c * 4 + 1]! + a[8 + r]! * b[c * 4 + 2]! + a[12 + r]! * b[c * 4 + 3]!;
  return o;
}

function decompose(m: M): { t: Vec3; r: Quat; s: Vec3 } {
  const sx = Math.hypot(m[0]!, m[1]!, m[2]!), sy = Math.hypot(m[4]!, m[5]!, m[6]!), sz = Math.hypot(m[8]!, m[9]!, m[10]!);
  const r00 = m[0]! / sx, r10 = m[1]! / sx, r20 = m[2]! / sx, r01 = m[4]! / sy, r11 = m[5]! / sy, r21 = m[6]! / sy, r02 = m[8]! / sz, r12 = m[9]! / sz, r22 = m[10]! / sz;
  const tr = r00 + r11 + r22;
  let q: [number, number, number, number];
  if (tr > 0) { const s = Math.sqrt(tr + 1) * 2; q = [(r21 - r12) / s, (r02 - r20) / s, (r10 - r01) / s, 0.25 * s]; }
  else if (r00 > r11 && r00 > r22) { const s = Math.sqrt(1 + r00 - r11 - r22) * 2; q = [0.25 * s, (r01 + r10) / s, (r02 + r20) / s, (r21 - r12) / s]; }
  else if (r11 > r22) { const s = Math.sqrt(1 + r11 - r00 - r22) * 2; q = [(r01 + r10) / s, 0.25 * s, (r12 + r21) / s, (r02 - r20) / s]; }
  else { const s = Math.sqrt(1 + r22 - r00 - r11) * 2; q = [(r02 + r20) / s, (r12 + r21) / s, 0.25 * s, (r10 - r01) / s]; }
  const l = Math.hypot(...q);
  return { t: [m[12]!, m[13]!, m[14]!], r: [q[0] / l, q[1] / l, q[2] / l, q[3] / l], s: [sx, sy, sz] };
}

const yawQ = (deg: number): Quat => { const h = (deg * Math.PI) / 360; return [0, Math.sin(h), 0, Math.cos(h)]; };
const point = (m: M, p: Vec3): Vec3 => [m[0]! * p[0] + m[4]! * p[1] + m[8]! * p[2] + m[12]!, m[1]! * p[0] + m[5]! * p[1] + m[9]! * p[2] + m[13]!, m[2]! * p[0] + m[6]! * p[1] + m[10]! * p[2] + m[14]!];

export class PrefabLibrary {
  private readonly prefabs = new Map<string, PrefabSpec>();
  private counter = 0;

  /** Register a prefab (validated). Re-registering a name replaces it. */
  register(spec: unknown): PrefabSpec {
    const p = parsePrefab(spec);
    this.prefabs.set(p.name, p);
    return p;
  }

  has(name: string): boolean {
    return this.prefabs.has(name);
  }

  /**
   * Add a prefab's entities to the scene. Ids are `<idPrefix>/<local path>`;
   * `idPrefix` defaults to `<name>#<n>`. Returns the created ids and the
   * placed gameplay behaviours.
   */
  instantiate(scene: Pick<Scene, "add">, name: string, placement: PrefabPlacement = {}, idPrefix?: string): PrefabInstance {
    const prefix = idPrefix ?? `${name}#${this.counter++}`;
    const out: PrefabInstance = { rootId: `${prefix}` as EntityId, entityIds: [], ids: new Map(), triggers: [], spawners: [], checkpoints: [] };
    const rot = placement.rotation ?? (placement.yaw !== undefined ? yawQ(placement.yaw) : [0, 0, 0, 1]);
    const rootM = trs(placement.position ?? [0, 0, 0], rot, placement.scale ?? [1, 1, 1]);
    const d = decompose(rootM);
    scene.add({ id: out.rootId, name: `${name} instance`, components: { [Tag.Transform]: { position: d.t, rotation: d.r, scale: d.s, parent: null } } });
    out.entityIds.push(out.rootId);
    this.expand(scene, name, out.rootId, rootM, prefix, "", placement, out, [name]);
    return out;
  }

  private expand(
    scene: Pick<Scene, "add">,
    name: string,
    parentId: EntityId,
    parentM: M,
    prefix: string,
    pathPrefix: string,
    placement: PrefabPlacement,
    out: PrefabInstance,
    stack: string[],
  ): void {
    const spec = this.prefabs.get(name);
    if (!spec) throw new Error(`prefab "${name}" is not registered`);
    const flatten = placement.flattenPhysics ?? true;
    const worldM = new Map<string, M>();
    const sceneId = new Map<string, EntityId>();
    // Parents before children.
    const order: PrefabEntitySpec[] = [];
    const placed = new Set<string>();
    while (order.length < spec.entities.length) {
      const before = order.length;
      for (const e of spec.entities) if (!placed.has(e.id) && (!e.parent || placed.has(e.parent))) { order.push(e); placed.add(e.id); }
      if (order.length === before) throw new Error(`prefab "${name}": parent cycle`);
    }
    for (const e of order) {
      const path = pathPrefix ? `${pathPrefix}/${e.id}` : e.id;
      const id = `${prefix}/${path}` as EntityId;
      const t = e.transform ?? {};
      const local = trs(t.position ?? [0, 0, 0], t.rotation ?? (t.yaw !== undefined ? yawQ(t.yaw) : [0, 0, 0, 1]), t.scale ?? [1, 1, 1]);
      const pM = e.parent ? worldM.get(e.parent)! : parentM;
      const wM = mul(pM, local);
      worldM.set(e.id, wM);
      const components: Record<string, unknown> = structuredClone(e.components ?? {});
      for (const [k, patch] of Object.entries(placement.overrides?.[path] ?? {})) {
        const cur = components[k];
        components[k] = cur && typeof cur === "object" && patch && typeof patch === "object" ? { ...(cur as object), ...(patch as object) } : patch;
      }
      const physical = !!(components[Tag.Collider] || components[Tag.RigidBody]);
      const parentScene = e.parent ? sceneId.get(e.parent)! : parentId;
      let tx: { position: Vec3; rotation: Quat; scale: Vec3; parent: EntityId | null };
      if (physical && flatten) {
        const d = decompose(wM);
        tx = { position: d.t, rotation: d.r, scale: d.s, parent: null };
      } else {
        const d = decompose(local);
        tx = { position: d.t, rotation: d.r, scale: d.s, parent: parentScene };
      }
      scene.add({ id, name: e.name ?? path, components: { ...components, [Tag.Transform]: tx } as Entity["components"] });
      out.entityIds.push(id);
      out.ids.set(path, id);
      sceneId.set(e.id, id);
      const g = e.gameplay;
      if (g?.trigger) out.triggers.push({ ...g.trigger, id, center: point(wM, [0, 0, 0]) });
      if (g?.spawner) out.spawners.push({ ...g.spawner, id, worldPoints: (g.spawner.points ?? [[0, 0, 0]]).map((q) => point(wM, q)) });
      if (g?.checkpoint) out.checkpoints.push({ ...g.checkpoint, id, position: point(wM, [0, 0, 0]) });
      if (e.prefab) {
        if (stack.includes(e.prefab)) throw new Error(`prefab cycle: ${[...stack, e.prefab].join(" → ")}`);
        this.expand(scene, e.prefab, id, wM, prefix, path, { ...placement, overrides: placement.overrides ?? {} }, out, [...stack, e.prefab]);
      }
    }
  }
}
