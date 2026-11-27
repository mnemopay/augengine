/** Physics tier 2 — colliders, ragdoll, particles. Phase 11. */

export {
  aabbForCollider,
  aabbOverlap,
  sphereSphere,
  sphereCapsule,
  capsuleCapsule,
  sphereMesh,
  sphereAabb,
  capsuleAabb,
  buildBvh,
  type AABB,
  type Sphere,
  type Capsule,
  type MeshCollider,
  type Collider,
  type ContactManifold,
  type BvhNode,
} from "./colliders.js";

export {
  KinematicCharacterController,
  type CharacterControllerOptions,
} from "./character-controller.js";

export {
  raycast,
  rayAabb,
  raySphere,
  rayCapsule,
  type RayHit,
  type RaycastOptions,
} from "./raycast.js";

export {
  Ragdoll,
  type RagdollBone,
  type RagdollPinJoint,
  type RagdollOptions,
} from "./ragdoll.js";

export {
  ParticleBuffer,
  PARTICLE_STEP_WGSL,
  type Particle,
  type EmitterDef,
  type EmitterShape,
} from "./particles.js";

export { PhysicsSystem } from "./system.js";
export type { PhysicsContact, PhysicsSnapshot } from "./system.js";
