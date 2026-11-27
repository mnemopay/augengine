/**
 * Walk / turn / stop gait for the Character Proof Room.
 *
 * Pure transform math. The host writes the resulting pose onto a lead
 * entity each frame. Cycle: walk 3.0s → turn 1.2s → stop 1.5s.
 */

import type { EntityId, Quat, Vec3 } from "../../scene/types.js";

export type Gait = "walk" | "turn" | "stop";

export interface LocomotionState {
  entityId: EntityId;
  origin: Vec3;
  position: Vec3;
  yaw: number;
  gait: Gait;
  gaitAge: number;
  distance: number;
}

export const GAIT_DURATIONS: Record<Gait, number> = {
  walk: 3.0,
  turn: 1.2,
  stop: 1.5,
};

const WALK_SPEED = 1.15; // m/s
const TURN_RATE = Math.PI / 1.2; // 180° during the turn beat

export function createLocomotion(entityId: EntityId, origin: Vec3, yaw = 0): LocomotionState {
  return { entityId, origin: [...origin] as Vec3, position: [...origin] as Vec3, yaw, gait: "walk", gaitAge: 0, distance: 0 };
}

export function yawToQuat(yaw: number): Quat {
  const h = yaw * 0.5;
  return [0, Math.sin(h), 0, Math.cos(h)];
}

function nextGait(gait: Gait): Gait {
  if (gait === "walk") return "turn";
  if (gait === "turn") return "stop";
  return "walk";
}

export interface LocomotionPose {
  position: Vec3;
  rotation: Quat;
  gait: Gait;
}

export function tickLocomotion(state: LocomotionState, dt: number): LocomotionPose {
  if (!Number.isFinite(dt)) throw new Error("Locomotion dt must be finite");
  let remaining = Math.max(0, dt);
  while (remaining > 0) {
    const duration = GAIT_DURATIONS[state.gait];
    const step = Math.min(remaining, duration - state.gaitAge);
    if (state.gait === "walk") {
      const distance = WALK_SPEED * step;
      state.distance += distance;
      state.position = [
        state.position[0] + Math.sin(state.yaw) * distance,
        state.position[1],
        state.position[2] + Math.cos(state.yaw) * distance,
      ];
    } else if (state.gait === "turn") {
      state.yaw += TURN_RATE * step;
    }
    state.gaitAge += step;
    remaining -= step;
    if (state.gaitAge >= duration - 1e-10) {
      state.gaitAge = 0;
      state.gait = nextGait(state.gait);
    }
  }
  return {
    position: [...state.position] as Vec3,
    rotation: yawToQuat(state.yaw),
    gait: state.gait,
  };
}
