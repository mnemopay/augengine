/** Character Proof Room runtime helpers. */

export {
  createLocomotion,
  tickLocomotion,
  yawToQuat,
  GAIT_DURATIONS,
  type Gait,
  type LocomotionState,
  type LocomotionPose,
} from "./locomotion.js";

export {
  PROOF_CLOSEUP_CAMERA_ID,
  PROOF_WIDE_CAMERA_ID,
  setProofShot,
  frameCloseup,
  type ProofShot,
} from "./dialogue-camera.js";
