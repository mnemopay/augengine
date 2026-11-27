import type { Scene } from "../../scene/scene.js";
import { Tag, type EntityId, type Vec3 } from "../../scene/types.js";
import { v3Distance, v3Sub, v3Normalize, v3Dot } from "../../scene/math.js";
import {
  GameTag,
  getGameComponent,
  setGameComponent,
  type RacerComponent,
  type RaceStateComponent,
  type VehicleComponent,
} from "./components.js";
import type { GameSystem, SystemContext } from "./system.js";

export class RaceSystem implements GameSystem {
  readonly id = "race";

  private trackCache: Map<string, Array<{ pos: Vec3; routeIndex: number }>> = new Map();

  update(scene: Scene, dt: number, ctx: SystemContext): void {
    // 1. Get or Initialize RaceState
    let stateEntity = Array.from(scene.entities.values()).find((e) =>
      getGameComponent(e, GameTag.RaceState)
    );

    if (!stateEntity) {
      // Find stage ID from checkpoint if present
      let stageId = "lagoon-crown";
      for (const e of scene.entities.values()) {
        const cp = (e.components as any)["game.race_checkpoint"];
        if (cp?.stageId) {
          stageId = cp.stageId;
          break;
        }
      }

      stateEntity = {
        id: "game.race_state_singleton" as EntityId,
        name: "Race State",
        components: {
          [Tag.Transform]: {
            position: [0, 0, 0],
            rotation: [0, 0, 0, 1],
            scale: [1, 1, 1],
            parent: null,
          },
        },
      };
      setGameComponent<RaceStateComponent>(stateEntity, GameTag.RaceState, {
        countdown: 3.0,
        active: false,
        finishedRacerIds: [],
        stageId,
        rankings: [],
      });
      scene.add(stateEntity);
    }

    const state = getGameComponent<RaceStateComponent>(stateEntity, GameTag.RaceState)!;

    // 2. Countdown Logic
    const countdownBefore = state.countdown;
    if (state.countdown > 0) {
      state.countdown = Math.max(0, state.countdown - dt);
      
      // Lock all throttles to 0 during countdown
      for (const entity of scene.entities.values()) {
        const vehicle = getGameComponent<VehicleComponent>(entity, GameTag.Vehicle);
        if (vehicle) {
          vehicle.throttle = 0;
        }
      }

      if (state.countdown <= 0 && countdownBefore > 0) {
        state.active = true;
        ctx.events.push({ type: "race-started", stageId: state.stageId });
      }
      return;
    }

    // 3. Rebuild track node cache for route tracking
    let trackNodes = this.trackCache.get(state.stageId) ?? [];
    if (trackNodes.length === 0) {
      trackNodes = Array.from(scene.entities.values())
        .filter((e) => e.id.startsWith("track_"))
        .map((e) => ({
          pos: e.components[Tag.Transform]?.position ?? [0, 0, 0],
          routeIndex: (e.components as any)["game.race_surface"]?.routeIndex ?? 0,
        }))
        .sort((a, b) => a.routeIndex - b.routeIndex);
      if (trackNodes.length > 0) {
        this.trackCache.set(state.stageId, trackNodes);
      }
    }

    // 4. Update Checkpoint Collisions
    const checkpointEntities = Array.from(scene.entities.values()).filter(
      (e) => (e.components as any)["game.race_checkpoint"]
    );

    const racerEntities = Array.from(scene.entities.values()).filter(
      (e) => getGameComponent(e, GameTag.Racer) && getGameComponent(e, GameTag.Vehicle)
    );

    // Number of checkpoints on this track
    const totalCheckpoints = checkpointEntities.length;

    for (const racerEnt of racerEntities) {
      const racer = getGameComponent<RacerComponent>(racerEnt, GameTag.Racer)!;
      const vehicle = getGameComponent<VehicleComponent>(racerEnt, GameTag.Vehicle)!;
      const racerPos = racerEnt.components[Tag.Transform]?.position ?? [0, 0, 0];

      // Dialogue Cooldown Tick
      if (racer.dialogueCooldown && racer.dialogueCooldown > 0) {
        racer.dialogueCooldown = Math.max(0, racer.dialogueCooldown - dt * 1000);
      }

      // Check collision with checkpoints
      for (const cpEnt of checkpointEntities) {
        const cp = (cpEnt.components as any)["game.race_checkpoint"];
        const cpPos = cpEnt.components[Tag.Transform]?.position ?? [0, 0, 0];
        const cpScale = cpEnt.components[Tag.Transform]?.scale ?? [1, 1, 1];

        // Simple AABB overlap check
        const halfX = cpScale[0] / 2;
        const halfY = cpScale[1] / 2;
        const halfZ = cpScale[2] / 2;

        if (
          Math.abs(racerPos[0] - cpPos[0]) <= halfX + 0.95 &&
          Math.abs(racerPos[1] - cpPos[1]) <= halfY + 0.35 &&
          Math.abs(racerPos[2] - cpPos[2]) <= halfZ + 2.15
        ) {
          // Collision detected! Check if this is the next checkpoint
          const expectedIndex = (racer.currentCheckpointIndex + 1) % totalCheckpoints;
          if (cp.split === expectedIndex) {
            const wasStart = racer.currentCheckpointIndex === -1;
            racer.currentCheckpointIndex = expectedIndex;
            ctx.events.push({
              type: "race-checkpoint",
              racerId: racer.racerId,
              checkpointId: cp.id,
              elapsedSeconds: (ctx.nowMs / 1000),
            });

            // If crossed start/finish and completed a lap
            if (expectedIndex === 0 && racer.currentCheckpointIndex === 0) {
              if (wasStart) {
                racer.currentLap = 1;
              } else {
                racer.currentLap += 1;
                ctx.events.push({
                  type: "race-lap-completed",
                  racerId: racer.racerId,
                  lap: racer.currentLap - 1,
                });
              }

              // Check win condition
              const targetLaps = (cp as any).stageLaps || 3;
              if (racer.currentLap > targetLaps && !state.finishedRacerIds.includes(racer.racerId)) {
                state.finishedRacerIds.push(racer.racerId);
                const rank = state.finishedRacerIds.length;
                ctx.events.push({
                  type: "race-finished",
                  racerId: racer.racerId,
                  rank,
                  elapsedSeconds: (ctx.nowMs / 1000),
                });

                // Trigger winner/loser dialogue remarks
                const dialogue = (racerEnt.components as any)["game.dialogue"];
                if (dialogue) {
                  racer.lastActiveDialogue = rank === 1 ? dialogue.victory[0] : dialogue.defeat[0];
                  racer.dialogueCooldown = 8000;
                }
              }
            }
          }
        }
      }

      // 5. AI Autopilot Steering
      if (racer.racerId !== "player" && trackNodes.length > 0 && !state.finishedRacerIds.includes(racer.racerId)) {
        // Find closest track node to current racer pos
        let closestIdx = 0;
        let minDist = Infinity;
        for (let t = 0; t < trackNodes.length; t++) {
          const node = trackNodes[t];
          if (!node) continue;
          const d = v3Distance(racerPos, node.pos);
          if (d < minDist) {
            minDist = d;
            closestIdx = t;
          }
        }

        // Target path coordinates a few nodes ahead (lookahead)
        const targetNodeIdx = (closestIdx + 2) % trackNodes.length;
        const targetNode = trackNodes[targetNodeIdx];
        if (targetNode) {
          const targetPos = targetNode.pos;

          // Steering logic: Vector from vehicle position to target
          const toTarget = v3Sub([0, 0, 0], targetPos, racerPos);
          v3Normalize(toTarget, toTarget);

          // Compute local target vector (steering direction relative to vehicle heading)
          const yawAngle = vehicle.yaw;
          const localX = toTarget[0] * Math.cos(-yawAngle) - toTarget[2] * Math.sin(-yawAngle);
          const localZ = toTarget[0] * Math.sin(-yawAngle) + toTarget[2] * Math.cos(-yawAngle);

          // Simple steering controller: steer towards local X axis
          const steerTarget = Math.max(-1.0, Math.min(1.0, localX * 4.0));
          vehicle.steering = steerTarget;

          // Autopilot throttle: slow down if steering angle is sharp, otherwise accelerate
          const steeringAngle = Math.abs(Math.atan2(localX, localZ));
          if (steeringAngle > 0.45) {
            vehicle.throttle = 0.25; // slow turn speed
            vehicle.handbrake = steeringAngle > 0.95; // pull handbrake for drifts
          } else {
            vehicle.throttle = 1.0;
            vehicle.handbrake = false;
          }
        }
      }
    }

    // 6. Dynamic Rank Calculation
    if (racerEntities.length > 0 && trackNodes.length > 0) {
      const sorted = racerEntities
        .map((ent) => {
          const racer = getGameComponent<RacerComponent>(ent, GameTag.Racer)!;
          const pos = ent.components[Tag.Transform]?.position ?? [0, 0, 0];

          // Compute distance along route: closest track node index + fraction
          let closestIdx = 0;
          let minDist = Infinity;
          for (let t = 0; t < trackNodes.length; t++) {
            const node = trackNodes[t];
            if (!node) continue;
            const d = v3Distance(pos, node.pos);
            if (d < minDist) {
              minDist = d;
              closestIdx = t;
            }
          }

          // Calculate score based on Lap + Checkpoint + distance fraction
          const lapScore = racer.currentLap * 10000;
          const cpScore = racer.currentCheckpointIndex * 1000;
          const trackScore = closestIdx * 10 - minDist;
          const totalScore = lapScore + cpScore + trackScore;

          return {
            racerId: racer.racerId,
            totalScore,
            racer,
          };
        })
        .sort((a, b) => b.totalScore - a.totalScore);

      state.rankings = sorted.map((s) => s.racerId);
      sorted.forEach((s, idx) => {
        s.racer.currentRank = idx + 1;
      });
    }
  }
}
