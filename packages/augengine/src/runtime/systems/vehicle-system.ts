import type { Scene } from "../../scene/scene.js";
import { Tag } from "../../scene/types.js";
import { GameTag, getGameComponent, type VehicleComponent } from "./components.js";
import type { GameSystem, SystemContext } from "./system.js";
import {
  v3, v3Add, v3Sub, v3Scale, v3Normalize, v3Length, v3Dot, v3Cross,
  qMul, qFromAxisAngle, qRotate, type MutVec3, type MutQuat
} from "../../scene/math.js";
import { raycast } from "../../physics/raycast.js";

/**
 * VehicleSystem — Track 5 game-native ECS system.
 * 
 * Simulates vehicle dynamics: engine thrust, brakes, drag, lateral tire friction
 * (drift sliding), suspension spring lift, and ground slope alignment.
 */
export class VehicleSystem implements GameSystem {
  readonly id = "game.vehicle_system";

  update(scene: Scene, dt: number, _ctx: SystemContext): void {
    for (const entity of scene.withComponent(Tag.Transform)) {
      const vehicle = getGameComponent<VehicleComponent>(entity, GameTag.Vehicle);
      if (!vehicle) continue;

      const tx = entity.components[Tag.Transform]!;

      // 1. Resolve Local Directions
      const forward: MutVec3 = [0, 0, 0];
      const right: MutVec3 = [0, 0, 0];
      const up: MutVec3 = [0, 0, 0];
      
      qRotate(forward, tx.rotation, [0, 0, -1]); // Facing -Z in glTF
      qRotate(right, tx.rotation, [1, 0, 0]);    // Right (+X)
      qRotate(up, tx.rotation, [0, 1, 0]);       // Up (+Y)

      // 2. Suspension Downward Raycast
      // Offset the ray origin upwards by 1.0m to prevent falling inside colliders.
      const raycastStartOffset = 1.0;
      const rayOrigin = v3(
        tx.position[0],
        tx.position[1] + raycastStartOffset,
        tx.position[2]
      );
      const rayDir: [number, number, number] = [0, -1, 0];
      const hit = raycast(scene, rayOrigin, rayDir, {
        maxDistance: raycastStartOffset - vehicle.centerOfMassOffset + vehicle.suspensionRestLength + 1.0,
        ignore: [entity.id]
      });

      // Calculate distance relative to vehicle center of mass
      const adjustedDistance = hit ? hit.distance - (raycastStartOffset - vehicle.centerOfMassOffset) : Infinity;

      // Detect Surface Type
      let surfaceType: "road" | "water" | "timber" = "road";
      if (hit && adjustedDistance <= vehicle.suspensionRestLength + 1.0) {
        const hitEntity = scene.get(hit.entityId);
        const raceSurface = hitEntity ? (hitEntity.components as any)["game.race_surface"] : undefined;
        if (raceSurface) {
          if (raceSurface.surface === "water_glide") {
            surfaceType = "water";
          } else if (raceSurface.surface === "timber_deck") {
            surfaceType = "timber";
          }
        }
      }
      vehicle.surfaceType = surfaceType;

      // Adjust handling and parameters based on surface
      let grip = vehicle.tireGrip;
      let brakeForce = vehicle.brakeForce;
      let stiffness = vehicle.suspensionStiffness;
      let damping = vehicle.suspensionDamping;
      let dragMultiplier = 1.0;
      
      if (surfaceType === "water") {
        grip = vehicle.tireGrip * 0.4;
        brakeForce = vehicle.brakeForce * 0.3;
        stiffness = vehicle.suspensionStiffness * 0.5;
        damping = vehicle.suspensionDamping * 0.7;
        dragMultiplier = 3.0;
        
        // waterSpray scales with speed, up to 1.0
        const currentSpeed = v3Length(vehicle.velocity);
        vehicle.waterSpray = Math.min(1.0, currentSpeed / (vehicle.maxSpeed || 30.0));
      } else {
        vehicle.waterSpray = 0.0;
        if (surfaceType === "timber") {
          grip = vehicle.tireGrip * 1.1; // slightly higher grip on timber
        }
      }

      let suspensionForceY = 0;
      if (hit && adjustedDistance <= vehicle.suspensionRestLength) {
        vehicle.isGrounded = true;
        vehicle.groundNormal = [hit.normal[0], hit.normal[1], hit.normal[2]];

        const compression = vehicle.suspensionRestLength - adjustedDistance;
        const relativeVelocityY = v3Dot(vehicle.velocity, up);
        suspensionForceY = (compression * stiffness) - (relativeVelocityY * damping);
        suspensionForceY = Math.max(0, suspensionForceY);
      } else {
        vehicle.isGrounded = false;
        vehicle.groundNormal = [0, 1, 0];
      }

      // 3. Steering/Heading Yaw Update
      const forwardVelocity = v3Dot(vehicle.velocity, forward);
      const speedFactor = Math.min(1.0, Math.abs(forwardVelocity) / 5.0);
      const speedCapFactor = Math.max(0.2, 1.0 - (Math.abs(forwardVelocity) / (vehicle.maxSpeed * 1.2)));
      const yawDelta = -vehicle.steering * vehicle.turnSpeed * speedFactor * speedCapFactor * dt;
      vehicle.yaw += yawDelta;

      // 4. Linear Propulsion (Engine / Brake)
      const propulsion: MutVec3 = [0, 0, 0];
      if (vehicle.isGrounded) {
        const isBraking = (forwardVelocity > 0.5 && vehicle.throttle < 0) || (forwardVelocity < -0.5 && vehicle.throttle > 0);
        if (isBraking) {
          const brakeDir = forwardVelocity > 0 ? -1 : 1;
          v3Scale(propulsion, forward, brakeDir * brakeForce);
        } else {
          v3Scale(propulsion, forward, vehicle.throttle * vehicle.enginePower);
        }
      }

      // 5. Lateral Friction & Drifts
      const lateralVel = v3Dot(vehicle.velocity, right);
      const lateralFriction: MutVec3 = [0, 0, 0];
      if (vehicle.isGrounded) {
        const dynamicGrip = vehicle.handbrake ? (grip * 0.2) : grip;
        const lateralDecel = -lateralVel * (dynamicGrip / dt);
        v3Scale(lateralFriction, right, lateralDecel * vehicle.mass);
      }

      // 6. Environmental Forces (Drag, Resistance, Gravity, Lift)
      const currentSpeed = v3Length(vehicle.velocity);
      const dragMag = currentSpeed * currentSpeed * vehicle.aerodynamicDrag * dragMultiplier;
      const dragForce: MutVec3 = [0, 0, 0];
      if (currentSpeed > 0.01) {
        const velDir: MutVec3 = [0, 0, 0];
        v3Normalize(velDir, vehicle.velocity);
        v3Scale(dragForce, velDir, -dragMag);
      }

      const rollingForce: MutVec3 = [0, 0, 0];
      v3Scale(rollingForce, forward, -forwardVelocity * vehicle.rollingResistance);

      const gravityForce: MutVec3 = [0, 0, 0];
      v3Scale(gravityForce, [0, -9.81, 0], vehicle.mass);

      const liftForce: MutVec3 = [0, 0, 0];
      v3Scale(liftForce, up, suspensionForceY);

      // 7. Sum Forces and Accelerate
      const totalForce: MutVec3 = [0, 0, 0];
      v3Add(totalForce, totalForce, propulsion);
      v3Add(totalForce, totalForce, lateralFriction);
      v3Add(totalForce, totalForce, dragForce);
      v3Add(totalForce, totalForce, rollingForce);
      v3Add(totalForce, totalForce, gravityForce);
      v3Add(totalForce, totalForce, liftForce);

      const acc: MutVec3 = [0, 0, 0];
      v3Scale(acc, totalForce, 1 / vehicle.mass);

      const dv: MutVec3 = [0, 0, 0];
      v3Scale(dv, acc, dt);
      
      const newVelocity: MutVec3 = [
        vehicle.velocity[0] + dv[0],
        vehicle.velocity[1] + dv[1],
        vehicle.velocity[2] + dv[2]
      ];
      vehicle.velocity = newVelocity;

      // 8. Move Entity Position
      const dp: MutVec3 = [0, 0, 0];
      v3Scale(dp, vehicle.velocity, dt);

      const nextPos: MutVec3 = [
        tx.position[0] + dp[0],
        tx.position[1] + dp[1],
        tx.position[2] + dp[2]
      ];

      // 9. Align Rotation to Yaw and Slope
      const qYaw: MutQuat = [0, 0, 0, 1];
      qFromAxisAngle(qYaw, [0, 1, 0], vehicle.yaw);

      let nextRot: MutQuat = qYaw;
      if (vehicle.isGrounded) {
        const rotAxis: MutVec3 = [0, 0, 0];
        v3Cross(rotAxis, up, vehicle.groundNormal);
        const rotAxisLen = v3Length(rotAxis);
        if (rotAxisLen > 0.001) {
          v3Normalize(rotAxis, rotAxis);
          const angle = Math.asin(rotAxisLen);
          const qSlope: MutQuat = [0, 0, 0, 1];
          qFromAxisAngle(qSlope, rotAxis, angle);
          
          const qCombined: MutQuat = [0, 0, 0, 1];
          qMul(qCombined, qSlope, qYaw);
          nextRot = qCombined;
        }
      }

      // 10. Write Back to Scene Graph
      scene.setLocalTransform(entity.id, nextPos, nextRot, tx.scale);
    }
  }
}
