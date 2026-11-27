import type { Scene } from "../../scene/scene.js";
import { Tag } from "../../scene/types.js";
import type { Entity, EntityId, Vec3 } from "../../scene/types.js";
import { GameTag, getGameComponent, type VehicleComponent } from "./components.js";
import type { GameSystem, SystemContext } from "./system.js";

export interface VfxParticleComponent {
  velocity: Vec3;
  color: [number, number, number];
  startOpacity: number;
  startScale: Vec3;
  age: number;
  lifetime: number;
}

export class VfxSystem implements GameSystem {
  readonly id = "game.vfx_system";
  private emitTimers = new Map<string, number>();

  update(scene: Scene, dt: number, _ctx: SystemContext): void {
    const activeParticles: Entity[] = [];
    const vehicles: Entity[] = [];

    // 1. Gather all vehicles and existing particles
    for (const e of scene.entities.values()) {
      if (getGameComponent(e, "game.vfx_particle" as any)) {
        activeParticles.push(e);
      } else if (getGameComponent(e, GameTag.Vehicle)) {
        vehicles.push(e);
      }
    }

    // 2. Step existing particles
    for (const p of activeParticles) {
      const part = getGameComponent<VfxParticleComponent>(p, "game.vfx_particle" as any);
      if (!part) continue;

      part.age += dt;
      if (part.age >= part.lifetime) {
        scene.entities.delete(p.id);
        continue;
      }

      const ratio = 1.0 - part.age / part.lifetime;
      const trans = p.components[Tag.Transform] as any;
      if (trans) {
        // Move by velocity
        trans.position[0] += part.velocity[0] * dt;
        trans.position[1] += part.velocity[1] * dt;
        trans.position[2] += part.velocity[2] * dt;

        // Shrink scale
        trans.scale[0] = part.startScale[0] * ratio;
        trans.scale[1] = part.startScale[1] * ratio;
        trans.scale[2] = part.startScale[2] * ratio;
      }

      const mesh = p.components[Tag.Mesh];
      if (mesh && mesh.material) {
        // Fade opacity
        (mesh.material as any).opacity = part.startOpacity * ratio;
      }
    }

    // 3. Emit new particles from active vehicles
    for (const vehicleEnt of vehicles) {
      const vehicle = getGameComponent<VehicleComponent>(vehicleEnt, GameTag.Vehicle);
      const tx = vehicleEnt.components[Tag.Transform];
      if (!vehicle || !tx) continue;

      const pos = tx.position;
      const speed = Math.hypot(vehicle.velocity[0], vehicle.velocity[2]);
      const forward: Vec3 = [Math.sin(vehicle.yaw), 0, Math.cos(vehicle.yaw)];
      const rightVec: Vec3 = [Math.cos(vehicle.yaw), 0, -Math.sin(vehicle.yaw)];

      let timer = this.emitTimers.get(vehicleEnt.id) ?? 0;
      timer += dt;

      // Emit intervals
      const emitInterval = 0.05; // 20 particles per second max
      if (timer >= emitInterval) {
        timer = 0;

        // A. Engine Jet Thruster Flame Sparks
        if (vehicle.throttle > 0 || speed > 1.0) {
          const isYemi = vehicleEnt.id.includes("yemi_soyinka");
          const flameColor = isYemi ? [0.0, 0.8, 1.0] : [1.0, 0.45, 0.15];
          const rearPos: Vec3 = [
            pos[0] - forward[0] * 2.15,
            pos[1],
            pos[2] - forward[2] * 2.15
          ];

          const pVel: Vec3 = [
            -forward[0] * (speed * 0.4 + 4.0) + (Math.random() - 0.5) * 1.5,
            (Math.random() - 0.5) * 0.8,
            -forward[2] * (speed * 0.4 + 4.0) + (Math.random() - 0.5) * 1.5
          ];

          const pId = `particle_flame_${vehicleEnt.id}_${Math.random().toString(36).substr(2, 6)}`;
          scene.entities.set(pId as EntityId, {
            id: pId as EntityId,
            name: "Flame Particle",
            components: {
              transform: {
                position: rearPos,
                rotation: [0, 0, 0, 1],
                scale: [0.15, 0.15, 0.25],
                parent: null
              },
              mesh: {
                geometry: { kind: "primitive", shape: "box" },
                material: { kind: "unlit", color: flameColor, opacity: 0.8 },
                visible: true
              },
              "game.vfx_particle": {
                velocity: pVel,
                color: flameColor,
                startOpacity: 0.8,
                startScale: [0.15, 0.15, 0.25],
                age: 0,
                lifetime: 0.25
              }
            }
          } as any);
        }

        // B. Water Spray Splashes
        if (vehicle.surfaceType === "water" && speed > 3.0) {
          // Spawn at left and right pontoons
          const leftPontoon: Vec3 = [
            pos[0] - rightVec[0] * 1.15,
            pos[1] - 0.35,
            pos[2] - rightVec[2] * 1.15
          ];
          const rightPontoon: Vec3 = [
            pos[0] + rightVec[0] * 1.15,
            pos[1] - 0.35,
            pos[2] + rightVec[2] * 1.15
          ];

          const spawnSpray = (pPos: Vec3) => {
            const sVel: Vec3 = [
              -forward[0] * speed * 0.25 + (Math.random() - 0.5) * 2.5,
              2.0 + Math.random() * 2.5,
              -forward[2] * speed * 0.25 + (Math.random() - 0.5) * 2.5
            ];
            const pId = `particle_spray_${vehicleEnt.id}_${Math.random().toString(36).substr(2, 6)}`;
            scene.entities.set(pId as EntityId, {
              id: pId as EntityId,
              name: "Water Spray Particle",
              components: {
                transform: {
                  position: pPos,
                  rotation: [0, 0, 0, 1],
                  scale: [0.22, 0.22, 0.22],
                  parent: null
                },
                mesh: {
                  geometry: { kind: "primitive", shape: "box" },
                  material: {
                    kind: "pbr",
                    baseColor: [0.88, 0.94, 0.96],
                    baseColorTexture: "/textures/water.png",
                    metallic: 0.1,
                    roughness: 0.1,
                    opacity: 0.65
                  },
                  visible: true
                },
                "game.vfx_particle": {
                  velocity: sVel,
                  color: [0.88, 0.94, 0.96],
                  startOpacity: 0.65,
                  startScale: [0.22, 0.22, 0.22],
                  age: 0,
                  lifetime: 0.45
                }
              }
            } as any);
          };

          spawnSpray(leftPontoon);
          spawnSpray(rightPontoon);
        }
      }
      this.emitTimers.set(vehicleEnt.id, timer);
    }
  }
}
