/**
 * Character Proof Room — the graphics bar and the official Studio template.
 *
 * Textured stand-in lead (Kael) + NPC (Maya), walk/turn/stop origin,
 * dialogue close-up camera, IBL environment, contact shadows. GLB skins
 * replace the stand-in meshes when the host uploads them.
 */

import { Tag, type Entity, type EntityId, type SceneSpec, type Vec3 } from "../scene/types.js";
import { yawToQuat } from "../runtime/proof/locomotion.js";
import { PROOF_CLOSEUP_CAMERA_ID, PROOF_WIDE_CAMERA_ID } from "../runtime/proof/dialogue-camera.js";

const id = (s: string): EntityId => s as EntityId;

export const PROOF_LEAD_ID = id("lead.kael");
export const PROOF_NPC_ID = id("maya");
export const PROOF_PLAYER_ID = id("player");

export const PROOF_ROOM_SKY = {
  zenith: [0.07, 0.09, 0.14] as Vec3,
  horizon: [0.18, 0.16, 0.14] as Vec3,
  horizonGlow: [0.42, 0.28, 0.16] as Vec3,
  exponent: 1.45,
  glowFalloff: 6.2,
};

export const PROOF_ROOM_IBL = {
  irradiance: [0.34, 0.32, 0.30] as Vec3,
  intensity: 1.2,
};

interface Palette {
  coat: Vec3;
  skin: Vec3;
  accent: Vec3;
}

const KAEL: Palette = {
  coat: [0.07, 0.28, 0.32],
  skin: [0.62, 0.42, 0.30],
  accent: [0.78, 0.18, 0.42],
};

const MAYA: Palette = {
  coat: [0.12, 0.55, 0.38],
  skin: [0.72, 0.50, 0.36],
  accent: [0.95, 0.72, 0.28],
};

function part(
  entityId: string,
  name: string,
  parent: EntityId | null,
  position: Vec3,
  scale: Vec3,
  color: Vec3,
  shape: "box" | "sphere" | "cylinder" | "plane" = "box",
  metallic = 0.08,
  roughness = 0.45,
): Entity {
  return {
    id: id(entityId),
    name,
    components: {
      // Character box dimensions are authored as half-extents; the box
      // primitive has unit edge lengths (spheres/cylinders use unit radii).
      [Tag.Transform]: { position, rotation: [0, 0, 0, 1], scale: shape === "box" ? scale.map((v) => v * 2) as unknown as Vec3 : scale, parent },
      [Tag.Mesh]: {
        geometry: { kind: "primitive", shape },
        material: { kind: "pbr", baseColor: color, metallic, roughness },
        visible: true,
      },
    },
  };
}

function standIn(rootId: EntityId, name: string, origin: Vec3, yaw: number, palette: Palette): Entity[] {
  const root: Entity = {
    id: rootId,
    name,
    components: {
      [Tag.Transform]: {
        position: origin,
        rotation: yawToQuat(yaw),
        scale: [1, 1, 1],
        parent: null,
      },
      [Tag.Collider]: { shape: "capsule", radius: 0.28, halfHeight: 0.85 },
    },
  };
  return [
    root,
    part(`${rootId}.torso`, `${name} torso`, rootId, [0, 1.05, 0], [0.38, 0.55, 0.22], palette.coat, "box", 0.12, 0.38),
    part(`${rootId}.head`, `${name} head`, rootId, [0, 1.62, 0], [0.16, 0.18, 0.16], palette.skin, "sphere", 0.04, 0.42),
    part(`${rootId}.arm.l`, `${name} left arm`, rootId, [-0.32, 1.05, 0], [0.08, 0.42, 0.08], palette.coat, "box", 0.1, 0.4),
    part(`${rootId}.arm.r`, `${name} right arm`, rootId, [0.32, 1.05, 0], [0.08, 0.42, 0.08], palette.coat, "box", 0.1, 0.4),
    part(`${rootId}.leg.l`, `${name} left leg`, rootId, [-0.12, 0.42, 0], [0.1, 0.42, 0.1], [0.08, 0.09, 0.1], "box", 0.05, 0.7),
    part(`${rootId}.leg.r`, `${name} right leg`, rootId, [0.12, 0.42, 0], [0.1, 0.42, 0.1], [0.08, 0.09, 0.1], "box", 0.05, 0.7),
    part(`${rootId}.accent`, `${name} tally`, rootId, [0.14, 1.15, 0.14], [0.05, 0.05, 0.05], palette.accent, "box", 0.6, 0.18),
    part(`${rootId}.shadow`, `${name} contact shadow`, rootId, [0, 0.02, 0], [0.55, 0.01, 0.4], [0.02, 0.02, 0.025], "cylinder", 0, 1),
  ];
}

function room(): Entity[] {
  const surfaces: Array<{ id: string; name: string; position: Vec3; scale: Vec3; color: Vec3; metallic: number; roughness: number }> = [
    { id: "proof_floor", name: "Proof Floor", position: [0, -0.05, 0], scale: [12, 0.1, 14], color: [0.16, 0.17, 0.18], metallic: 0.22, roughness: 0.28 },
    { id: "proof_back", name: "Cyclorama", position: [0, 3.2, -5], scale: [12, 6.4, 0.2], color: [0.12, 0.13, 0.16], metallic: 0.05, roughness: 0.72 },
    { id: "proof_left", name: "Left bounce", position: [-6, 2.6, 1], scale: [0.18, 5.2, 10], color: [0.22, 0.18, 0.14], metallic: 0.04, roughness: 0.65 },
    { id: "proof_right", name: "Right bounce", position: [6, 2.6, 1], scale: [0.18, 5.2, 10], color: [0.10, 0.16, 0.22], metallic: 0.04, roughness: 0.65 },
    { id: "proof_dais", name: "Dais", position: [0, 0.12, 0], scale: [3.2, 0.24, 2.6], color: [0.18, 0.2, 0.22], metallic: 0.38, roughness: 0.22 },
  ];
  return surfaces.map((s) => ({
    id: id(s.id),
    name: s.name,
    components: {
      [Tag.Transform]: { position: s.position, rotation: [0, 0, 0, 1], scale: s.scale, parent: null },
      [Tag.Mesh]: {
        geometry: { kind: "primitive", shape: "box" },
        material: { kind: "pbr", baseColor: s.color, metallic: s.metallic, roughness: s.roughness },
        visible: true,
      },
      [Tag.Collider]: { shape: "box", extents: s.scale.map((v) => v * 0.5) as unknown as Vec3 },
    },
  }));
}

// Directional lights with an identity rotation shine from their position
// toward the world origin (the dais) — see runtime/live/webgpu/lights.ts. The
// renderer makes the brightest directional light (the warm key) the shadow
// caster. The IBL proxy is an ambient light: it replaces the frame ambient.
function lights(): Entity[] {
  const specs: Array<{ id: string; name: string; position: Vec3; color: Vec3; intensity: number; kind: "directional" | "point" | "ambient" }> = [
    { id: "proof_key", name: "Warm key", position: [-4.5, 6.5, 5], color: [1, 0.9, 0.74], intensity: 3.4, kind: "directional" },
    { id: "proof_fill", name: "Cool fill", position: [5, 3.8, 4], color: [0.45, 0.62, 0.95], intensity: 1.35, kind: "directional" },
    { id: "proof_rim", name: "Rim", position: [0, 5.5, -3.5], color: [0.55, 0.95, 0.82], intensity: 2.1, kind: "directional" },
    { id: "proof_ibl_proxy", name: "IBL ambient", position: [0, 8, 0], color: PROOF_ROOM_IBL.irradiance, intensity: PROOF_ROOM_IBL.intensity, kind: "ambient" },
  ];
  return specs.map((l) => ({
    id: id(l.id),
    name: l.name,
    components: {
      [Tag.Transform]: { position: l.position, rotation: [0, 0, 0, 1], scale: [1, 1, 1], parent: null },
      [Tag.Light]: { kind: l.kind, color: l.color, intensity: l.intensity },
    },
  }));
}

export function buildCharacterProofRoomEntities(): Entity[] {
  const leadOrigin: Vec3 = [-0.7, 0.24, 0.4];
  const npcOrigin: Vec3 = [0.85, 0.24, -0.35];
  return [
    {
      id: PROOF_WIDE_CAMERA_ID,
      name: "Proof wide",
      components: {
        [Tag.Transform]: { position: [0, 1.7, 5.6], rotation: yawToQuat(0), scale: [1, 1, 1], parent: null },
        [Tag.Camera]: { fovY: (48 * Math.PI) / 180, near: 0.08, far: 80, active: true },
      },
    },
    {
      id: PROOF_CLOSEUP_CAMERA_ID,
      name: "Dialogue close-up",
      components: {
        [Tag.Transform]: { position: [0.2, 1.58, 1.15], rotation: yawToQuat(0), scale: [1, 1, 1], parent: null },
        [Tag.Camera]: { fovY: (32 * Math.PI) / 180, near: 0.05, far: 40, active: false },
      },
    },
    {
      id: PROOF_PLAYER_ID,
      name: "Player",
      components: {
        // Same yaw as the wide camera: yaw 0 looks down -Z toward the dais.
        [Tag.Transform]: { position: [0, 1.7, 5.6], rotation: yawToQuat(0), scale: [1, 1, 1], parent: null },
        [Tag.Camera]: { fovY: (48 * Math.PI) / 180, near: 0.08, far: 80, active: false },
        [Tag.RigidBody]: { velocity: [0, 0, 0], angularVelocity: [0, 0, 0], mass: 80, isKinematic: true },
        [Tag.Collider]: { shape: "capsule", radius: 0.4, halfHeight: 0.9 },
      },
    },
    ...room(),
    ...standIn(PROOF_LEAD_ID, "Kael Varr", leadOrigin, Math.PI * 0.15, KAEL),
    ...standIn(PROOF_NPC_ID, "Maya", npcOrigin, Math.PI + 0.2, MAYA),
    ...lights(),
  ];
}

export const CHARACTER_PROOF_ROOM: SceneSpec = {
  schemaVersion: 1,
  source: { kind: "authored", path: "studio:proof-room" },
  defaultCamera: PROOF_WIDE_CAMERA_ID,
  environment: { kind: "ibl", hdrUrl: "studio:proof-room-ibl" },
  entities: buildCharacterProofRoomEntities(),
};
