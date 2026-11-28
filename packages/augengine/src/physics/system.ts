import type * as Rapier from "@dimforge/rapier3d-deterministic-compat";
import type { Scene } from "../scene/scene.js";
import {
  Tag,
  type Entity,
  type EntityId,
  type Vec3,
  type Quat,
} from "../scene/types.js";
import { FixedStepClock } from "../runtime/fixed-step.js";
let loading: Promise<typeof Rapier> | undefined;
function loadRapier(): Promise<typeof Rapier> {
  return (loading ??= import("@dimforge/rapier3d-deterministic-compat")
    .then(async (R) => {
      await R.init();
      return R;
    })
    .catch((e) => {
      loading = undefined;
      throw e;
    }));
}
/**
 * Heightfield signatures: a terrain tile carries ~1k heights, and `sync` runs
 * every fixed step, so the heights are digested once per array, not
 * re-stringified per step.
 */
const heightDigests = new WeakMap<object, string>();
/** Heights arrays already checked finite (validation runs every fixed step; a terrain has thousands of samples). Like the digest above, keyed by array identity: replace the array to change a terrain. */
const finiteHeights = new WeakSet<object>();
function heightDigest(heights: ArrayLike<number>): string {
  const key = heights as unknown as object;
  let d = heightDigests.get(key);
  if (d === undefined) {
    let h = 2166136261;
    for (let i = 0; i < heights.length; i++) {
      h = Math.imul(h ^ (Math.fround(heights[i]!) * 1000 | 0), 16777619);
    }
    d = `${heights.length}:${h >>> 0}`;
    heightDigests.set(key, d);
  }
  return d;
}
const v = (a: Vec3) => ({ x: a[0], y: a[1], z: a[2] });
const q = (a: Quat) => ({ x: a[0], y: a[1], z: a[2], w: a[3] });
const tuple = (a: Rapier.Vector): Vec3 => [a.x, a.y, a.z];
function finite(a: readonly number[], label: string): void {
  if (
    !Array.isArray(a) ||
    !a.every((n) => Number.isFinite(n) && Number.isFinite(Math.fround(n)))
  )
    throw new Error(`Non-finite or unrepresentable ${label}`);
}
interface Binding {
  body: Rapier.RigidBody;
  collider?: Rapier.Collider;
  signature: string;
  pose: string;
  velocity: string;
}
export interface PhysicsContact {
  a: EntityId;
  b: EntityId;
  normal: Vec3;
  point: Vec3;
  separation: number;
  /** Accumulated contact exposure when chemistry runs at a lower rate. */
  durationSeconds?: number;
}
/** Query shape for `PhysicsSystem.overlapShape` (capsule axis is local Y). */
export type OverlapShape =
  | { kind: "sphere"; radius: number }
  | { kind: "capsule"; radius: number; halfHeight: number }
  | { kind: "box"; halfExtents: Vec3 };
export interface PhysicsSnapshot {
  version: 1;
  rapierVersion: "0.21.0";
  bytes: number[];
  gravity: number;
  clock: ReturnType<FixedStepClock["snapshot"]>;
  bindings: {
    id: EntityId;
    body: number;
    collider?: number;
    signature: string;
    pose: string;
    velocity: string;
  }[];
}
/** Rapier/WASM rigid bodies. Await init() before stepping. SI units. */
export class PhysicsSystem {
  gravity = -9.81;
  readonly clock = new FixedStepClock();
  contacts: PhysicsContact[] = [];
  private api: typeof Rapier | undefined;
  private world: Rapier.World | undefined;
  private scene: Scene | undefined;
  private bindings = new Map<EntityId, Binding>();
  private disposed = false;
  async init(): Promise<void> {
    if (this.disposed) throw new Error("Cannot initialize disposed physics");
    if (!Number.isFinite(Math.fround(this.gravity))) throw new Error("Invalid gravity");
    if (this.world) return;
    const R = await loadRapier();
    if (this.disposed) return;
    this.api = R;
    this.world ??= new R.World({ x: 0, y: this.gravity, z: 0 });
    this.world.numSolverIterations = 8;
  }
  get ready(): boolean {
    return Boolean(this.world) && !this.disposed;
  }
  get bodyCount(): number {
    return this.bindings.size;
  }
  step(scene: Scene, dt: number): void {
    this.clock.advance(dt, (h) => this.stepFixed(scene, h));
  }
  stepFixed(scene: Scene, dt: number): void {
    if (!this.world || !this.api || this.disposed)
      throw new Error("Await PhysicsSystem.init() before stepping");
    if (
      !Number.isFinite(dt) ||
      dt <= 0 ||
      dt > 0.1 ||
      !Number.isFinite(this.gravity)
    )
      throw new Error("Invalid physics timestep/gravity");
    const entities = [...scene.entities.values()]
      .filter((e) => e.components[Tag.Collider] || e.components[Tag.RigidBody])
      .sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
    for (const e of entities) this.validate(e);
    if (this.scene && this.scene !== scene) {
      for (const b of this.bindings.values())
        this.world.removeRigidBody(b.body);
      this.bindings.clear();
    }
    this.scene = scene;
    const ids = new Set(entities.map((e) => e.id));
    for (const [id, b] of this.bindings)
      if (!ids.has(id)) {
        this.world.removeRigidBody(b.body);
        this.bindings.delete(id);
      }
    for (const e of entities) this.sync(e);
    this.world.gravity.y = this.gravity;
    this.world.timestep = dt;
    this.world.step();
    this.contacts = [];
    const handles = new Map<number, EntityId>();
    for (const [id, b] of this.bindings)
      if (b.collider) handles.set(b.collider.handle, id);
    for (const [id, b] of this.bindings) {
      const e = scene.get(id)!;
      const rb = e.components[Tag.RigidBody];
      const tx = e.components[Tag.Transform]!;
      if (rb && !rb.isKinematic) {
        const r = b.body.rotation();
        scene.setLocalTransform(
          id,
          tuple(b.body.translation()),
          [r.x, r.y, r.z, r.w],
          tx.scale,
        );
        rb.velocity = tuple(b.body.linvel());
        rb.angularVelocity = tuple(b.body.angvel());
      }
      b.pose = JSON.stringify([
        e.components[Tag.Transform]!.position,
        e.components[Tag.Transform]!.rotation,
      ]);
      b.velocity = JSON.stringify(rb ? [rb.velocity, rb.angularVelocity] : []);
      if (!b.collider) continue;
      this.world.contactPairsWith(b.collider, (other) => {
        const otherId = handles.get(other.handle);
        if (!otherId || id >= otherId) return;
        this.world!.contactPair(b.collider!, other, (m, flipped) => {
          for (let i = 0; i < m.numSolverContacts(); i++) {
            const p = m.solverContactPoint(i);
            if (!p) continue;
            const n = m.normal();
            const s = flipped ? -1 : 1;
            this.contacts.push({
              a: id,
              b: otherId,
              normal: [n.x * s, n.y * s, n.z * s],
              point: tuple(p),
              separation: m.solverContactDist(i),
            });
          }
        });
      });
    }
    this.contacts.sort((a, b) =>
      a.a < b.a ? -1 : a.a > b.a ? 1 : a.b < b.b ? -1 : a.b > b.b ? 1 : 0,
    );
  }
  private validate(e: Entity): void {
    const tx = e.components[Tag.Transform],
      rb = e.components[Tag.RigidBody],
      c = e.components[Tag.Collider];
    if (!tx) throw new Error(`Physics entity ${e.id} needs a transform`);
    if (
      tx.position.length !== 3 ||
      tx.rotation.length !== 4 ||
      tx.scale.length !== 3
    )
      throw new Error("Physics transform has invalid vector dimensions");
    if (tx.parent !== null)
      throw new Error(
        `Physics entity ${e.id} must be root; bake its world transform first`,
      );
    finite([...tx.position, ...tx.rotation, ...tx.scale], "transform");
    if (
      tx.scale.some((s) => s <= 0) ||
      Math.abs(Math.hypot(...tx.rotation) - 1) > 1e-4
    )
      throw new Error("Physics requires positive scale and unit quaternion");
    if (rb) {
      if (rb.velocity.length !== 3 || rb.angularVelocity.length !== 3)
        throw new Error("Physics velocity must have three components");
      finite(
        [
          ...rb.velocity,
          ...rb.angularVelocity,
          rb.mass,
          rb.friction ?? 0.5,
          rb.restitution ?? 0,
          rb.linearDamping ?? 0,
          rb.angularDamping ?? 0,
        ],
        "rigid body",
      );
      if (
        Math.fround(rb.mass) <= 0 ||
        (rb.friction ?? 0) < 0 ||
        (rb.restitution ?? 0) < 0 ||
        (rb.restitution ?? 0) > 1 ||
        (rb.linearDamping ?? 0) < 0 ||
        (rb.angularDamping ?? 0) < 0
      )
        throw new Error("Invalid mass/friction/restitution/damping");
    }
    if (!c) return;
    if (c.centerOffset) {
      if (c.centerOffset.length !== 3)
        throw new Error("Collider offset must have three components");
      finite(c.centerOffset, "collider offset");
    }
    if (c.shape === "sphere" || c.shape === "capsule") {
      if (!Number.isFinite(c.radius) || c.radius! <= 0)
        throw new Error("Collider radius must be positive");
      if (!Number.isFinite(Math.fround(c.radius!)) || Math.fround(c.radius!) <= 0)
        throw new Error("Radius is outside the physics numeric range");
      if (
        c.shape === "capsule" &&
        (!Number.isFinite(c.halfHeight) || c.halfHeight! < 0 || !Number.isFinite(Math.fround(c.halfHeight!)))
      )
        throw new Error("Capsule halfHeight must be nonnegative");
    } else if (c.shape === "box") {
      if (
        !c.extents ||
        c.extents.length !== 3 ||
        c.extents.some((s) => !Number.isFinite(Math.fround(s)) || Math.fround(s) <= 0)
      )
        throw new Error("Box needs positive half extents");
    } else if (c.shape === "mesh") {
      if (rb)
        throw new Error(
          "Triangle meshes must be static; use primitives for dynamic bodies",
        );
      if (
        !c.vertices?.length ||
        c.vertices.length % 3 ||
        !c.indices?.length ||
        c.indices.length % 3
      )
        throw new Error("Mesh needs triangle vertices/indices");
      finite(
        c.vertices.map((n, i) => n * tx.scale[i % 3]!),
        "scaled mesh",
      );
      if (
        c.indices.some(
          (i) =>
            !Number.isSafeInteger(i) || i < 0 || i >= c.vertices!.length / 3,
        )
      )
        throw new Error("Invalid mesh index");
    } else if (c.shape === "heightfield") {
      if (rb) throw new Error("Heightfields must be static");
      const hf = c.heightfield;
      if (
        !hf ||
        !Number.isInteger(hf.samplesX) || hf.samplesX < 2 ||
        !Number.isInteger(hf.samplesZ) || hf.samplesZ < 2 ||
        hf.heights.length !== hf.samplesX * hf.samplesZ ||
        !(hf.sizeX > 0) || !(hf.sizeZ > 0)
      )
        throw new Error("Heightfield needs samplesX*samplesZ heights and a positive size");
      finite([hf.sizeX, hf.sizeZ], "heightfield size");
      const key = hf.heights as unknown as object;
      if (!finiteHeights.has(key)) {
        finite(Array.from(hf.heights), "heightfield");
        finiteHeights.add(key);
      }
    } else throw new Error("Unsupported collider shape");
  }
  private sync(e: Entity): void {
    const R = this.api!,
      world = this.world!,
      tx = e.components[Tag.Transform]!,
      rb = e.components[Tag.RigidBody],
      c = e.components[Tag.Collider];
    const signature = JSON.stringify([
      c?.shape === "heightfield" && c.heightfield
        ? { ...c, heightfield: { ...c.heightfield, heights: heightDigest(c.heightfield.heights) } }
        : c,
      tx.scale,
      rb
        ? [
            rb.mass,
            rb.isKinematic,
            rb.friction,
            rb.restitution,
            rb.ccd,
            rb.linearDamping,
            rb.angularDamping,
          ]
        : null,
    ]);
    let b = this.bindings.get(e.id);
    if (b && b.signature !== signature) {
      world.removeRigidBody(b.body);
      this.bindings.delete(e.id);
      b = undefined;
    }
    const pose = JSON.stringify([tx.position, tx.rotation]),
      velocity = JSON.stringify(rb ? [rb.velocity, rb.angularVelocity] : []);
    if (!b) {
      const desc = rb
        ? rb.isKinematic
          ? R.RigidBodyDesc.kinematicPositionBased()
          : R.RigidBodyDesc.dynamic()
        : R.RigidBodyDesc.fixed();
      desc.setTranslation(...tx.position).setRotation(q(tx.rotation));
      if (rb) {
        desc
          .setLinvel(...rb.velocity)
          .setAngvel(v(rb.angularVelocity))
          .setCcdEnabled(rb.ccd ?? true)
          .setLinearDamping(rb.linearDamping ?? 0)
          .setAngularDamping(rb.angularDamping ?? 0);
        if (!c) desc.setAdditionalMass(rb.mass);
      }
      const body = world.createRigidBody(desc);
      b = { body, signature, pose, velocity };
      if (c) {
        let cd: Rapier.ColliderDesc;
        const s = tx.scale;
        // Primitive sizes are world-space: transform scale is visual only
        // (see ColliderComponent). Only mesh vertices are local and scaled.
        if (c.shape === "sphere") cd = R.ColliderDesc.ball(c.radius!);
        else if (c.shape === "capsule")
          cd = R.ColliderDesc.capsule(c.halfHeight!, c.radius!);
        else if (c.shape === "box")
          cd = R.ColliderDesc.cuboid(c.extents![0], c.extents![1], c.extents![2]);
        else if (c.shape === "heightfield") {
          // Rapier takes cell counts (rows along z, columns along x) and a
          // column-major (samplesZ x samplesX) matrix; ours is row-major by z.
          const hf = c.heightfield!;
          const cm = new Float32Array(hf.samplesX * hf.samplesZ);
          for (let j = 0; j < hf.samplesZ; j++)
            for (let i = 0; i < hf.samplesX; i++)
              cm[i * hf.samplesZ + j] = hf.heights[j * hf.samplesX + i]!;
          cd = R.ColliderDesc.heightfield(hf.samplesZ - 1, hf.samplesX - 1, cm, { x: hf.sizeX, y: 1, z: hf.sizeZ });
        } else
          cd = R.ColliderDesc.trimesh(
            Float32Array.from(c.vertices!, (x, i) => x * s[i % 3]!),
            Uint32Array.from(c.indices!),
          );
        if (c.centerOffset)
          cd.setTranslation(c.centerOffset[0], c.centerOffset[1], c.centerOffset[2]);
        if (c.sensor) cd.setSensor(true);
        cd.setFriction(rb?.friction ?? 0.5).setRestitution(
          rb?.restitution ?? 0,
        );
        if (rb) cd.setMass(rb.mass);
        b.collider = world.createCollider(cd, body);
      }
      this.bindings.set(e.id, b);
    } else if (pose !== b.pose) {
      if (rb?.isKinematic) {
        b.body.setNextKinematicTranslation(v(tx.position));
        b.body.setNextKinematicRotation(q(tx.rotation));
      } else {
        b.body.setTranslation(v(tx.position), true);
        b.body.setRotation(q(tx.rotation), true);
      }
    }
    if (rb && !rb.isKinematic && velocity !== b.velocity) {
      b.body.setLinvel(v(rb.velocity), true);
      b.body.setAngvel(v(rb.angularVelocity), true);
    }
  }
  /**
   * Entities whose colliders (sensors included) overlap a query shape at the
   * given pose, sorted by id. Reflects the poses of the last `stepFixed`
   * (kinematic bodies included). `exclude` skips entities; `filter` decides
   * per candidate. Used by combat hitboxes and trigger volumes.
   */
  overlapShape(
    shape: OverlapShape,
    position: Vec3,
    rotation: Quat = [0, 0, 0, 1],
    opts: { exclude?: ReadonlySet<EntityId>; filter?: (id: EntityId) => boolean } = {},
  ): EntityId[] {
    if (!this.world || !this.api) throw new Error("Await PhysicsSystem.init() before querying");
    finite([...position, ...rotation], "query pose");
    const R = this.api;
    const s =
      shape.kind === "sphere"
        ? new R.Ball(shape.radius)
        : shape.kind === "capsule"
          ? new R.Capsule(shape.halfHeight, shape.radius)
          : new R.Cuboid(shape.halfExtents[0], shape.halfExtents[1], shape.halfExtents[2]);
    const byHandle = new Map<number, EntityId>();
    for (const [id, b] of this.bindings) if (b.collider) byHandle.set(b.collider.handle, id);
    const out: EntityId[] = [];
    this.world.intersectionsWithShape(v(position), q(rotation), s, (collider) => {
      const id = byHandle.get(collider.handle);
      if (id && !opts.exclude?.has(id) && (!opts.filter || opts.filter(id))) out.push(id);
      return true;
    });
    return out.sort();
  }
  applyImpulse(id: EntityId, impulse: Vec3): void {
    finite(impulse, "impulse");
    const b = this.bindings.get(id);
    if (!b) throw new Error(`Unknown body ${id}`);
    b.body.applyImpulse(v(impulse), true);
  }
  isSleeping(id: EntityId): boolean {
    return this.bindings.get(id)?.body.isSleeping() ?? false;
  }
  snapshot(): PhysicsSnapshot {
    if (!this.world)
      throw new Error("Physics must be initialized before snapshot");
    return {
      version: 1,
      rapierVersion: "0.21.0",
      bytes: Array.from(this.world.takeSnapshot()),
      gravity: this.gravity,
      clock: this.clock.snapshot(),
      bindings: [...this.bindings].map(([id, b]) => ({
        id,
        body: b.body.handle,
        ...(b.collider ? { collider: b.collider.handle } : {}),
        signature: b.signature,
        pose: b.pose,
        velocity: b.velocity,
      })),
    };
  }
  restore(scene: Scene, state: PhysicsSnapshot): void {
    if (!this.api || this.disposed)
      throw new Error("Initialize physics before restoring");
    if (
      state.version !== 1 ||
      state.rapierVersion !== "0.21.0" ||
      !Number.isFinite(state.gravity) ||
      !state.bytes.length ||
      state.bytes.some((n) => !Number.isInteger(n) || n < 0 || n > 255)
    )
      throw new Error("Invalid physics snapshot");
    const clock = new FixedStepClock();
    clock.restore(state.clock);
    for (const e of scene.entities.values())
      if (e.components[Tag.Collider] || e.components[Tag.RigidBody])
        this.validate(e);
    const restored = this.api.World.restoreSnapshot(
      Uint8Array.from(state.bytes),
    );
    const bindings = new Map<EntityId, Binding>();
    try {
      for (const entry of state.bindings) {
        if (!scene.has(entry.id) || bindings.has(entry.id))
          throw new Error("Invalid snapshot entity binding");
        const body = restored.getRigidBody(entry.body);
        if (!body?.isValid()) throw new Error("Invalid snapshot body handle");
        const binding: Binding = {
          body,
          signature: entry.signature,
          pose: entry.pose,
          velocity: entry.velocity,
        };
        if (entry.collider !== undefined) {
          const collider = restored.getCollider(entry.collider);
          if (!collider?.isValid())
            throw new Error("Invalid snapshot collider handle");
          binding.collider = collider;
        }
        bindings.set(entry.id, binding);
      }
    } catch (error) {
      restored.free();
      throw error;
    }
    this.world?.free();
    this.world = restored;
    this.scene = scene;
    this.bindings = bindings;
    this.gravity = state.gravity;
    this.clock.restore(state.clock);
    this.contacts = [];
  }
  dispose(): void {
    this.disposed = true;
    this.world?.free();
    this.world = undefined;
    this.bindings.clear();
    this.contacts = [];
  }
}
