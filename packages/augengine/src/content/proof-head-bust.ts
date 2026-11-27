/**
 * Proof head as a bust: head + neck + shoulders on a plinth.
 *
 * `proofHeadEntity()` places a free-floating 21 cm head in the camera's
 * foreground. Kael's locomotion carries his stand-in through that space, so in
 * captures the head read as stuck on the front of his body. The bust gives the
 * head something to sit on and moves it to the room's left side, clear of the
 * walking area, turned to face the proof-room camera at [0, 1.7, 5.6].
 */
import { Tag, type Entity, type EntityId, type Quat, type Vec3 } from "../scene/types.js";
import { proofHeadEntity } from "./proof-head.js";

/** Where the head sits on the bust (head mesh origin). */
export const PROOF_BUST_HEAD_POSITION: Vec3 = [-1.35, 1.36, 2.7];
/** Proof-room wide camera the face turns toward. */
const CAMERA: Vec3 = [0, 1.7, 5.6];

/** Yaw that turns the head's +Z face toward the camera. */
export function proofBustYaw(head: Vec3 = PROOF_BUST_HEAD_POSITION, camera: Vec3 = CAMERA): number {
  return Math.atan2(camera[0] - head[0], camera[2] - head[2]);
}

function yawQuat(yaw: number): Quat {
  return [0, Math.sin(yaw / 2), 0, Math.cos(yaw / 2)];
}

function part(
  id: string,
  name: string,
  shape: "box" | "cylinder",
  position: Vec3,
  rotation: Quat,
  scale: Vec3,
  baseColor: Vec3,
  roughness: number,
): Entity {
  return {
    id: id as EntityId,
    name,
    components: {
      [Tag.Transform]: { position, rotation, scale, parent: null },
      [Tag.Mesh]: {
        geometry: { kind: "primitive", shape },
        material: { kind: "pbr", baseColor, metallic: 0.02, roughness },
        visible: true,
      },
    },
  };
}

/**
 * Head (same id, mesh and blendshape slots as `proofHeadEntity()`) plus the
 * bust underneath it. Drop-in replacement for `[proofHeadEntity()]`.
 */
export function proofHeadBustEntities(head: Vec3 = PROOF_BUST_HEAD_POSITION): Entity[] {
  const yaw = proofBustYaw(head);
  const rot = yawQuat(yaw);
  const headEntity = proofHeadEntity();
  const tx = headEntity.components[Tag.Transform]!;
  headEntity.components[Tag.Transform] = { ...tx, position: head, rotation: rot };
  const [x, y, z] = head;
  const skin: Vec3 = [0.73, 0.51, 0.4];
  const cloth: Vec3 = [0.16, 0.2, 0.26];
  const stone: Vec3 = [0.2, 0.21, 0.23];
  const plinthTop = y - 0.29;
  return [
    headEntity,
    // Primitive cylinder is radius 1, height 2; box is a unit cube.
    part("proof.head.neck", "Proof head neck", "cylinder", [x, y - 0.15, z], rot, [0.05, 0.065, 0.05], skin, 0.55),
    part("proof.head.shoulders", "Proof head shoulders", "box", [x, y - 0.24, z], rot, [0.38, 0.1, 0.19], cloth, 0.7),
    part("proof.head.plinth", "Proof head plinth", "box", [x, plinthTop / 2, z], rot, [0.32, plinthTop, 0.32], stone, 0.62),
  ];
}
