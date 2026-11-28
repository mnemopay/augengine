/** Character system barrel. */

export {
  buildBoneMatrices,
  skinVertexLBS,
} from "./skinning.js";

export {
  buildRetargetConfig,
  retargetFrame,
  prepareSkeleton,
  forceTpose,
  detectMirrorX,
  legLength,
  makeIdentityFrame,
  MIXAMO_TO_VRM_DEFAULT,
  type Skeleton,
  type SkeletonJoint,
  type JointMap,
  type MotionFrame,
  type RetargetConfig,
} from "./retarget.js";

export {
  solveTwoBoneIK,
  footLock,
  type TwoBoneIKInput,
  type TwoBoneIKResult,
} from "./ik.js";

export {
  sampleTrack,
  clipTime,
  AnimStateMachine,
  type JointTrack,
  type AnimationClip,
  type AnimState,
  type AnimTransition,
  type BlendTree,
  type BlendChild,
} from "./animation.js";

export {
  MEDIAPIPE_POSE_LANDMARKS,
  MEDIAPIPE_SKELETON_JOINTS,
  MEDIAPIPE_TO_VRM,
  MEDIAPIPE_TO_MIXAMO,
  makeMediaPipeSkeleton,
  mediaPipeFrameToMotionFrame,
  type MediaPipeFrame,
} from './mediapipe.js';

export { fitMediaPipeFrame } from './mediapipe-fit.js';

export {
  createMaya,
  MAYA_DEFAULT_ENTITY_ID,
  MAYA_MEMORY_NAMESPACE,
  MAYA_VOICE,
  MAYA_PERSONA,
  MAYA_SEED_MEMORIES,
  MAYA_DIALOGUE,
  type MayaCompanion,
  type MayaBrain,
  type MayaReaction,
  type CreateMayaOptions,
  type DialogueBeat,
  type MayaDialogueScript,
} from "./maya.js";

export {
  MotionMatchingDatabase,
  MotionMatchingController,
  type DatabaseFrame,
  type MotionMatchingQuery,
} from "./motion-matching.js";


export {
  HUMANOID_BONES,
  HUMANOID_PARENT,
  REQUIRED_BONES,
  ARKIT_52,
  humanoidBoneOf,
  mapHumanoidBones,
  resolveHumanoidRig,
  missingHumanoidBones,
  arkitCoverage,
  type HumanoidBone,
} from "./humanoid.js";

export {
  retargetClip,
  humanoidNodes,
  rigFromGltfRoot,
  type RigNode,
  type RigSource,
  type RetargetClipOptions,
  type RetargetClipResult,
} from "./retarget-clip.js";

export {
  IK_TWO_BONE,
  LOOK_AT,
  solveTwoBoneWorld,
  lookAtDelta,
  applyTwoBone,
  applyLookAt,
  applyIkComponents,
  isDescendant,
  twoBoneChainProblem,
  type IkTwoBoneComponent,
  type LookAtComponent,
  type TwoBoneSolution,
  type ApplyIkOptions,
} from "./ik-runtime.js";

export {
  FACE_RIG_SHAPES,
  FACE_RIG_INDEX,
  FACE_RIG_APPROXIMATE,
  VISEME_SHAPES,
  faceRigWeights,
  remapFaceRigWeights,
  visemeFrameToFaceRig,
  findHeadBox,
  faceMaps,
  proportionalLandmarks,
  detectFaceLandmarks,
  faceLandmarksFromMeshPoints,
  FACE_MESH_POINTS,
  FACE_MESH_OVAL,
  OUTLINE_GROW,
  buildFaceRig,
  shapeDelta,
  OrthoCaster,
  type FaceRigShape,
  type FaceSourceMesh,
  type FaceTextureSampler,
  type HeadBox,
  type FaceLandmarks,
  type FaceMaps,
  type FaceRigOptions,
  type FaceRigResult,
} from "./face-rig.js";
