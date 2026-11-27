import type { Scene } from "../../scene/scene.js";
import { Tag, type EntityId } from "../../scene/types.js";
import { GameTag, getGameComponent, type VehicleComponent } from "./components.js";
import type { GameSystem, SystemContext } from "./system.js";
import {
  v3, v3Add, v3Lerp, qMul, qSlerp, qFromAxisAngle, qRotate,
  type MutVec3, type MutQuat
} from "../../scene/math.js";

/**
 * ChaseCameraSystem — Track 5 game-native ECS system.
 * 
 * Automatically tracks the active player vehicle and smoothly updates the active
 * camera's position and rotation to create a dampened chase-cam follow effect.
 */
export class ChaseCameraSystem implements GameSystem {
  readonly id = "game.chase_camera_system";

  /** Configurable chase camera distance behind the vehicle. */
  distance = 6.0;
  /** Configurable height offset above the vehicle. */
  height = 2.5;
  /** Damping speed for position interpolation. */
  lerpSpeed = 5.0;
  /** Damping speed for rotation interpolation. */
  slerpSpeed = 5.0;
  /** Look-down tilt angle in radians. */
  lookDownAngle = 0.22;

  update(scene: Scene, dt: number, _ctx: SystemContext): void {
    const camId = scene.activeCamera;
    if (!camId) return;

    const cameraEntity = scene.get(camId);
    if (!cameraEntity) return;

    const camTx = cameraEntity.components[Tag.Transform];
    if (!camTx) return;

    // Find the first vehicle entity in the scene to chase
    let targetVehicleEntity = null;
    let targetVehicleComponent = null;
    for (const entity of scene.withComponent(Tag.Transform)) {
      const vehicle = getGameComponent<VehicleComponent>(entity, GameTag.Vehicle);
      if (vehicle) {
        targetVehicleEntity = entity;
        targetVehicleComponent = vehicle;
        break;
      }
    }

    if (!targetVehicleEntity || !targetVehicleComponent) return;

    const targetTx = targetVehicleEntity.components[Tag.Transform]!;

    // 1. Calculate Target Position (behind vehicle based on its yaw)
    const qYaw: MutQuat = [0, 0, 0, 1];
    qFromAxisAngle(qYaw, [0, 1, 0], targetVehicleComponent.yaw);

    // Behind is +Z local in glTF forward (-Z) convention
    const localOffset: [number, number, number] = [0, this.height, this.distance];
    const worldOffset: MutVec3 = [0, 0, 0];
    qRotate(worldOffset, qYaw, localOffset);

    const targetPos: MutVec3 = [0, 0, 0];
    v3Add(targetPos, targetTx.position, worldOffset);

    // 2. Interpolate Position (Lerp)
    const nextPos: MutVec3 = [0, 0, 0];
    v3Lerp(nextPos, camTx.position, targetPos, Math.min(1.0, this.lerpSpeed * dt));

    // 3. Calculate Target Rotation (Face forward + pitch down)
    const qPitch: MutQuat = [0, 0, 0, 1];
    qFromAxisAngle(qPitch, [1, 0, 0], -this.lookDownAngle);

    const targetRot: MutQuat = [0, 0, 0, 1];
    qMul(targetRot, qYaw, qPitch);

    // 4. Interpolate Rotation (Slerp)
    const nextRot: MutQuat = [0, 0, 0, 1];
    qSlerp(nextRot, camTx.rotation, targetRot, Math.min(1.0, this.slerpSpeed * dt));

    // 5. Update Camera Transform
    scene.setLocalTransform(camId, nextPos, nextRot, camTx.scale);
  }
}
