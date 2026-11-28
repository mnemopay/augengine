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
