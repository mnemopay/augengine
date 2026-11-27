/**
 * Dialogue close-up — swaps the active camera onto a head-height plate
 * looking at the NPC during viseme playback.
 */

import type { Scene } from "../../scene/scene.js";
import type { EntityId, Vec3 } from "../../scene/types.js";
import { Tag } from "../../scene/types.js";
import { lookAtTransform } from "../live/webgpu/renderer.js";

export const PROOF_CLOSEUP_CAMERA_ID = "camera.closeup" as EntityId;
export const PROOF_WIDE_CAMERA_ID = "camera.main" as EntityId;

export type ProofShot = "wide" | "closeup";

export function setProofShot(scene: Scene, shot: ProofShot): EntityId {
  const id = shot === "closeup" ? PROOF_CLOSEUP_CAMERA_ID : PROOF_WIDE_CAMERA_ID;
  if (scene.has(id)) scene.activeCamera = id;
  return id;
}

/** Place the close-up camera 1.4 m in front of `lookAt`, eye height 1.55 m. */
export function frameCloseup(scene: Scene, lookAt: Vec3, along: Vec3 = [0, 0, 1.4]): void {
  const cam = scene.get(PROOF_CLOSEUP_CAMERA_ID);
  if (!cam) return;
  const transform = cam.components[Tag.Transform];
  if (!transform) return;
  const position: Vec3 = [lookAt[0] + along[0], lookAt[1] + 0.15, lookAt[2] + along[2]];
  const look = lookAtTransform(position, lookAt, [0, 1, 0]);
  scene.setLocalTransform(PROOF_CLOSEUP_CAMERA_ID, look.position, look.rotation, transform.scale);
  scene.activeCamera = PROOF_CLOSEUP_CAMERA_ID;
}
