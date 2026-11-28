/**
 * Gameplay kit (alpha): animation graph, and — in follow-up modules — combat
 * primitives, behaviour trees, prefabs/triggers/spawners and procedural
 * terrain. Import from `@mnemopay/augengine/gameplay`. See docs/GAMEPLAY.md.
 */

export {
  AnimGraph,
  createAnimRig,
  createPose,
  copyPose,
  bindClip,
  findClip,
  sampleClip,
  rootDelta,
  blendPoses,
  applyAdditive,
  blend1DWeights,
  crossFadeWeight,
  boneMask,
  rotateByYaw,
  type AnimRig,
  type AnimRigNode,
  type CreateAnimRigOptions,
  type RigPose,
  type BoundClip,
  type AnimEvent,
  type AnimEventSpec,
  type AnimGraphSpec,
  type StateSpec,
  type MotionSpec,
  type TransitionSpec,
  type ConditionSpec,
  type LayerSpec,
  type PlayLayerOptions,
} from "./anim-graph.js";

export {
  CombatWorld,
  Hitbox,
  HitStop,
  CameraShake,
  createHealth,
  applyDamage,
  tickHealth,
  knockbackVelocity,
  horizontalDir,
  resolveBones,
  pickLockTarget,
  cycleLockTarget,
  shouldBreakLock,
  signedAngle,
  type BoneShapeSpec,
  type HurtboxSpec,
  type HitEvent,
  type CombatantRig,
  type Health,
  type DamageOptions,
  type DamageResult,
  type LockCandidate,
  type LockOptions,
} from "./combat.js";

export {
  BehaviorTree,
  MeleeBrain,
  action,
  condition,
  sequence,
  selector,
  inverter,
  type BTNode,
  type BTStatus,
  type MeleeState,
  type MeleeInput,
  type MeleeCommand,
  type MeleeBrainOptions,
} from "./behavior-tree.js";

export {
  PrefabLibrary,
  PrefabError,
  parsePrefab,
  type PrefabSpec,
  type PrefabEntitySpec,
  type PrefabTransform,
  type PrefabPlacement,
  type PrefabInstance,
  type TriggerSpec,
  type SpawnerSpec,
  type CheckpointSpec,
  type PlacedTrigger,
  type PlacedSpawner,
  type PlacedCheckpoint,
} from "./prefab.js";

export {
  TriggerSystem,
  Spawner,
  CheckpointTracker,
  insideTrigger,
  mulberry32,
  type TriggerEvent,
  type Subject,
  type SpawnRequest,
} from "./triggers.js";

export {
  generateTerrain,
  scatter,
  grassTuftMesh,
  rockMesh,
  type Terrain,
  type TerrainOptions,
  type ScatterOptions,
  type Placement,
} from "./terrain-gen.js";
