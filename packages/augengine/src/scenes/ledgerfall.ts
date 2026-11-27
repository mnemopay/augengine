/**
 * LEDGERFALL vertical slice — scaffold on WebGpuLiveRenderer.
 *
 * Three authored beats (mezzanine arrival, informant booth, substation
 * extract) sharing one material set. Art is primitive stand-in; the
 * scene graph, cameras, and beat list are the production contract.
 */

import { Tag, type Entity, type EntityId, type SceneSpec, type Vec3 } from "../scene/types.js";

const id = (s: string): EntityId => s as EntityId;

export type LedgerfallBeatId = "mezzanine" | "booth" | "substation";

export interface LedgerfallBeat {
  id: LedgerfallBeatId;
  name: string;
  durationSeconds: number;
  kind: "cinematic" | "traversal" | "dialogue" | "combat";
  scenePath: string;
}

export const LEDGERFALL_BEATS: readonly LedgerfallBeat[] = [
  { id: "mezzanine", name: "Mezzanine Arrival", durationSeconds: 300, kind: "cinematic", scenePath: "ledgerfall/mezzanine" },
  { id: "booth", name: "Informant Booth", durationSeconds: 180, kind: "dialogue", scenePath: "ledgerfall/booth" },
  { id: "substation", name: "Substation Extract", durationSeconds: 360, kind: "combat", scenePath: "ledgerfall/substation" },
];

const WET_CONCRETE: Vec3 = [0.18, 0.2, 0.22];
const SCUFFED_STEEL: Vec3 = [0.32, 0.34, 0.36];
const PAINTED_LINE: Vec3 = [0.85, 0.72, 0.18];
const GLASS: Vec3 = [0.35, 0.55, 0.62];
const EMISSIVE: Vec3 = [0.15, 0.85, 0.9];

function box(entityId: string, name: string, position: Vec3, scale: Vec3, color: Vec3, metallic = 0.2, roughness = 0.45, emissive?: Vec3): Entity {
  return {
    id: id(entityId),
    name,
    components: {
      [Tag.Transform]: { position, rotation: [0, 0, 0, 1], scale, parent: null },
      [Tag.Mesh]: {
        geometry: { kind: "primitive", shape: "box" },
        material: { kind: "pbr", baseColor: color, metallic, roughness, ...(emissive ? { emissive } : {}) },
        visible: true,
      },
    },
  };
}

function camera(entityId: string, position: Vec3): Entity {
  return {
    id: id(entityId),
    name: "Camera",
    components: {
      [Tag.Transform]: { position, rotation: [0, 0, 0, 1], scale: [1, 1, 1], parent: null },
      [Tag.Camera]: { fovY: 1.0472, near: 0.1, far: 220, active: true },
    },
  };
}

function player(position: Vec3): Entity {
  return {
    id: id("player"),
    name: "Kael Varr",
    components: {
      [Tag.Transform]: { position, rotation: [0, 0, 0, 1], scale: [0.7, 1.7, 0.45], parent: null },
      [Tag.Mesh]: {
        geometry: { kind: "primitive", shape: "box" },
        material: { kind: "pbr", baseColor: [0.07, 0.28, 0.32], metallic: 0.12, roughness: 0.4, emissive: [0.78, 0.18, 0.42] },
        visible: true,
      },
    },
  };
}

function keyLight(position: Vec3): Entity {
  return {
    id: id("light.key"),
    name: "Key",
    components: {
      [Tag.Transform]: { position, rotation: [0, 0, 0, 1], scale: [1, 1, 1], parent: null },
      [Tag.Light]: { kind: "directional", color: [0.95, 0.92, 0.88], intensity: 2.8 },
    },
  };
}

export function buildMezzanineArrival(): SceneSpec {
  return {
    schemaVersion: 1,
    source: { kind: "authored", path: "ledgerfall/mezzanine" },
    defaultCamera: id("camera.main"),
    environment: { kind: "ibl", hdrUrl: "ledgerfall:rain-terminal" },
    entities: [
      camera("camera.main", [0, 3.2, 14]),
      player([0, 1, 8]),
      keyLight([6, 12, 8]),
      box("floor", "Terminal floor", [0, -0.1, 0], [40, 0.2, 28], WET_CONCRETE, 0.28, 0.22),
      box("mezz", "Drop-ship mezzanine", [0, 4.2, -6], [18, 0.35, 8], SCUFFED_STEEL, 0.55, 0.3),
      box("line", "Safety line", [0, 0.02, 4], [18, 0.02, 0.18], PAINTED_LINE, 0.05, 0.4),
      box("glass.l", "Curtain glass L", [-10, 3, 0], [0.12, 6, 12], GLASS, 0.7, 0.08),
      box("glass.r", "Curtain glass R", [10, 3, 0], [0.12, 6, 12], GLASS, 0.7, 0.08),
      box("sign.a", "LED board A", [-4, 5.4, -5.8], [3.2, 0.9, 0.08], [0.04, 0.06, 0.07], 0.1, 0.25, EMISSIVE),
      box("sign.b", "LED board B", [4, 5.4, -5.8], [3.2, 0.9, 0.08], [0.04, 0.06, 0.07], 0.1, 0.25, EMISSIVE),
      box("rain.proxy", "Rain sheet", [0, 6, 2], [36, 0.02, 20], [0.25, 0.3, 0.35], 0.0, 0.15),
    ],
  };
}

export function buildInformantBooth(): SceneSpec {
  return {
    schemaVersion: 1,
    source: { kind: "authored", path: "ledgerfall/booth" },
    defaultCamera: id("camera.main"),
    environment: { kind: "color", color: [0.04, 0.045, 0.05] },
    entities: [
      camera("camera.main", [0, 1.55, 2.8]),
      camera("camera.closeup", [0.15, 1.52, 1.2]),
      player([-0.6, 1, 1.6]),
      {
        id: id("mira"),
        name: "Mira",
        components: {
          [Tag.Transform]: { position: [0.55, 1, -0.2], rotation: [0, 0, 0, 1], scale: [0.6, 1.65, 0.4], parent: null },
          [Tag.Mesh]: {
            geometry: { kind: "primitive", shape: "box" },
            material: { kind: "pbr", baseColor: [0.55, 0.22, 0.2], metallic: 0.08, roughness: 0.48 },
            visible: true,
          },
          [Tag.Persona]: {
            traits: ["informant", "guarded"],
            voice: "mira",
            memoryId: "npc/mira",
            goals: [],
            mood: 0.1,
            role: "ally",
          },
        },
      },
      keyLight([-2, 4, 3]),
      box("floor", "Corridor floor", [0, -0.1, 0], [8, 0.2, 10], WET_CONCRETE, 0.2, 0.3),
      box("wall.back", "Booth wall", [0, 2, -2.2], [6, 4, 0.2], SCUFFED_STEEL, 0.4, 0.45),
      box("door", "Service door", [2.4, 1.4, -2.05], [0.9, 2.6, 0.12], [0.12, 0.12, 0.13], 0.6, 0.35),
      box("crt", "CRT bank", [-1.4, 1.3, -1.7], [1.6, 1.1, 0.35], [0.08, 0.09, 0.1], 0.3, 0.4, [0.2, 0.7, 0.35]),
      box("table", "Booth table", [0.2, 0.7, 0.2], [1.4, 0.08, 0.7], SCUFFED_STEEL, 0.5, 0.3),
    ],
  };
}

export function buildSubstationExtract(): SceneSpec {
  return {
    schemaVersion: 1,
    source: { kind: "authored", path: "ledgerfall/substation" },
    defaultCamera: id("camera.main"),
    environment: { kind: "ibl", hdrUrl: "ledgerfall:substation" },
    entities: [
      camera("camera.main", [0, 4.5, 16]),
      player([0, 1, 10]),
      keyLight([8, 14, 4]),
      box("floor.lower", "Lower deck", [0, -0.1, 4], [28, 0.2, 22], WET_CONCRETE, 0.25, 0.25),
      box("floor.upper", "Catwalk", [0, 4.2, -6], [22, 0.16, 6], SCUFFED_STEEL, 0.6, 0.28),
      box("pipe.a", "Pipe run A", [-6, 2.4, 0], [0.35, 0.35, 18], SCUFFED_STEEL, 0.7, 0.32),
      box("pipe.b", "Pipe run B", [6, 2.4, 0], [0.35, 0.35, 18], SCUFFED_STEEL, 0.7, 0.32),
      box("lightbank", "Destructible light bank", [0, 6.4, -4], [8, 0.25, 0.4], [0.9, 0.85, 0.6], 0.2, 0.2, [1, 0.9, 0.55]),
      box("extract", "Extraction pad", [0, 0.05, -8], [4, 0.08, 4], PAINTED_LINE, 0.05, 0.35, EMISSIVE),
      box("sealbreaker", "Sealbreaker", [3, 1.1, -2], [0.9, 2.2, 0.7], [0.12, 0.08, 0.08], 0.45, 0.4),
    ],
  };
}

export const LEDGERFALL_SCENES: Record<LedgerfallBeatId, () => SceneSpec> = {
  mezzanine: buildMezzanineArrival,
  booth: buildInformantBooth,
  substation: buildSubstationExtract,
};

export function ledgerfallGoldenPathSeconds(): number {
  return LEDGERFALL_BEATS.reduce((sum, b) => sum + b.durationSeconds, 0);
}
