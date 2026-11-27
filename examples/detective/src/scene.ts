/**
 * The house, built from engine primitives: five rooms side by side along +X,
 * one camera per room, warm point lights, the clue objects as props, and the
 * four suspects as textured 3D busts on plinths in the parlour.
 */
import { Tag, type Entity, type EntityId, type SceneSpec, type Vec3, type Quat } from "@mnemopay/augengine";
import { assetUrl } from "./asset-url.js";
import { lookAtTransform } from "@mnemopay/augengine/runtime/live";
import { OBJECTS, ROOMS, SUSPECTS, type RoomId, type SuspectId } from "@mnemopay/augengine-detective";

export const ROOM_SPACING = 14;
export const SUSPECT_ORDER: SuspectId[] = ["mara", "desmond", "ines", "teodor"];

const id = (s: string) => s as EntityId;
const Q0: Quat = [0, 0, 0, 1];

export function roomX(room: RoomId): number {
  return ROOMS.findIndex((r) => r.id === room) * ROOM_SPACING;
}

export const CLOSEUP_CAMERA = id("cam.closeup");

export function cameraId(room: RoomId): EntityId {
  return id(`cam.${room}`);
}

export function headId(s: SuspectId): EntityId {
  return id(`head.${s}`);
}

export function objectEntityId(objectId: string): EntityId {
  return id(`obj.${objectId}`);
}

function yawQuat(yaw: number): Quat {
  return [0, Math.sin(yaw / 2), 0, Math.cos(yaw / 2)];
}

function box(name: string, label: string, pos: Vec3, scale: Vec3, color: Vec3, roughness = 0.7, metallic = 0.02, rot: Quat = Q0, shape: "box" | "cylinder" | "sphere" = "box"): Entity {
  return {
    id: id(name),
    name: label,
    components: {
      [Tag.Transform]: { position: pos, rotation: rot, scale, parent: null },
      [Tag.Mesh]: {
        geometry: { kind: "primitive", shape },
        material: { kind: "pbr", baseColor: color, metallic, roughness },
        visible: true,
      },
    },
  };
}

function light(name: string, pos: Vec3, color: Vec3, intensity: number, range: number): Entity {
  return {
    id: id(name),
    name,
    components: {
      [Tag.Transform]: { position: pos, rotation: Q0, scale: [1, 1, 1], parent: null },
      [Tag.Light]: { kind: "point", color, intensity, range },
    },
  };
}

const PALETTE: Record<RoomId, { floor: Vec3; wall: Vec3; lamp: Vec3 }> = {
  parlour: { floor: [0.22, 0.13, 0.09], wall: [0.28, 0.2, 0.17], lamp: [1.0, 0.72, 0.45] },
  dome: { floor: [0.16, 0.17, 0.2], wall: [0.14, 0.16, 0.22], lamp: [0.65, 0.75, 1.0] },
  library: { floor: [0.2, 0.12, 0.08], wall: [0.18, 0.2, 0.16], lamp: [1.0, 0.62, 0.32] },
  kitchen: { floor: [0.34, 0.3, 0.26], wall: [0.42, 0.38, 0.32], lamp: [1.0, 0.85, 0.62] },
  conservatory: { floor: [0.2, 0.22, 0.2], wall: [0.3, 0.38, 0.36], lamp: [0.7, 0.9, 0.8] },
};

/** Where each clue/flavour object sits inside its room (relative to room centre). */
const OBJECT_LAYOUT: Record<string, { at: Vec3; scale: Vec3; color: Vec3; shape?: "box" | "cylinder" | "sphere"; metallic?: number }> = {
  "dome.flask": { at: [0.7, 0.98, -0.9], scale: [0.09, 0.18, 0.09], color: [0.7, 0.72, 0.75], shape: "cylinder", metallic: 0.9 },
  "dome.log": { at: [-0.9, 0.83, -1.1], scale: [0.42, 0.05, 0.3], color: [0.45, 0.32, 0.2] },
  "dome.desk": { at: [-0.2, 0.82, -0.8], scale: [0.22, 0.03, 0.16], color: [0.9, 0.88, 0.8] },
  "dome.telescope": { at: [2.6, 1.6, -2.4], scale: [0.16, 1.2, 0.16], color: [0.75, 0.55, 0.25], shape: "cylinder", metallic: 0.85 },
  "library.drawer": { at: [-0.7, 0.86, -1.0], scale: [0.5, 0.1, 0.3], color: [0.3, 0.18, 0.1] },
  "library.coat": { at: [2.2, 0.75, -0.2], scale: [0.4, 0.7, 0.14], color: [0.32, 0.12, 0.14] },
  "library.fire": { at: [-2.8, 0.5, -3.2], scale: [1.0, 0.6, 0.35], color: [0.85, 0.32, 0.1], metallic: 0 },
  "kitchen.cabinet": { at: [-2.2, 1.6, -3.3], scale: [0.9, 0.9, 0.35], color: [0.38, 0.28, 0.18] },
  "kitchen.note": { at: [0.1, 0.82, -0.8], scale: [0.2, 0.02, 0.14], color: [0.95, 0.93, 0.86] },
  "kitchen.kettle": { at: [0.9, 0.93, -1.1], scale: [0.16, 0.13, 0.16], color: [0.72, 0.42, 0.26], shape: "sphere", metallic: 0.8 },
  "conservatory.planter": { at: [-1.9, 0.35, -0.4], scale: [0.45, 0.35, 0.45], color: [0.35, 0.22, 0.14], shape: "cylinder" },
  "conservatory.orchids": { at: [0.6, 0.98, -1.1], scale: [0.2, 0.2, 0.2], color: [0.85, 0.5, 0.75], shape: "sphere" },
};

function buildRoom(room: RoomId): Entity[] {
  const x = roomX(room);
  const p = PALETTE[room];
  const out: Entity[] = [
    box(`room.${room}.floor`, `${room} floor`, [x, -0.05, 0], [10, 0.1, 8], p.floor, 0.85),
    box(`room.${room}.back`, `${room} back wall`, [x, 1.8, -3.6], [10, 3.6, 0.2], p.wall, 0.9),
    box(`room.${room}.left`, `${room} left wall`, [x - 4.9, 1.8, 0], [0.2, 3.6, 7.2], p.wall, 0.9),
    box(`room.${room}.right`, `${room} right wall`, [x + 4.9, 1.8, 0], [0.2, 3.6, 7.2], p.wall, 0.9),
    box(`room.${room}.ceiling`, `${room} ceiling`, [x, 3.65, 0], [10, 0.1, 8], [p.wall[0] * 0.6, p.wall[1] * 0.6, p.wall[2] * 0.6], 0.95),
    light(`room.${room}.lamp`, [x, 2.9, 0.6], p.lamp, 7, 9),
    light(`room.${room}.fill`, [x + 2.5, 1.6, 2.6], [0.55, 0.6, 0.75], 2.2, 7),
  ];
  // Furniture: a table where objects rest (dome desk, library desk, kitchen counter...).
  if (room !== "parlour") out.push(box(`room.${room}.table`, `${room} table`, [x, 0.38, -1.0], [2.6, 0.76, 0.9], [0.26, 0.17, 0.11], 0.6));
  if (room === "dome") out.push(box("room.dome.chair", "observing chair", [x + 1.9, 0.45, -1.6], [0.5, 0.9, 0.5], [0.16, 0.07, 0.06], 0.5));
  if (room === "library") {
    for (let i = 0; i < 4; i++) out.push(box(`room.library.shelf${i}`, "bookshelf", [x - 3.5 + i * 2.3, 1.6, -3.3], [1.9, 3.0, 0.4], [0.24 + i * 0.02, 0.15, 0.09], 0.8));
  }
  if (room === "conservatory") {
    out.push(box("room.conservatory.glass", "glass wall", [x, 1.8, -3.5], [9.6, 3.4, 0.05], [0.55, 0.75, 0.8], 0.05, 0.1));
    for (let i = 0; i < 3; i++) out.push(box(`room.conservatory.fern${i}`, "fern", [x + 2.8 - i * 1.4, 0.6, 1.8], [0.4, 0.6, 0.4], [0.16, 0.42, 0.2], 0.9, 0, Q0, "sphere"));
  }
  for (const o of OBJECTS.filter((ob) => ob.room === room)) {
    const l = OBJECT_LAYOUT[o.id];
    if (!l) continue;
    out.push(box(objectEntityId(o.id), o.name, [x + l.at[0], l.at[1], l.at[2]], l.scale, l.color, 0.5, l.metallic ?? 0.05, Q0, l.shape ?? "box"));
  }
  // Camera: standing in the doorway, looking at the table.
  const camPos: Vec3 = room === "parlour" ? [x, 1.62, 2.3] : [x + 0.4, 1.65, 2.4];
  const target: Vec3 = room === "parlour" ? [x, 1.2, -0.6] : [x - 0.1, 0.85, -1.3];
  out.push({
    id: cameraId(room),
    name: `${room} camera`,
    components: {
      [Tag.Transform]: { position: camPos, rotation: lookAtTransform(camPos, target, [0, 1, 0]).rotation, scale: [1, 1, 1], parent: null },
      [Tag.Camera]: { fovY: 0.95, near: 0.05, far: 80, active: room === "parlour" },
    },
  });
  return out;
}

/** URL of a suspect's 3D bust (TRELLIS image-to-3D from their painted portrait; see public/portraits/SOURCES.md). */
export function bustUrl(sid: SuspectId): string {
  return assetUrl(`/busts/${sid}.glb`);
}

/** Bust height in metres (the source meshes are ~1 unit tall). */
const BUST_SCALE = 0.62;
/** Half-extents of the bust meshes in their own units (from the GLB bounds, all four within these). */
export const BUST_HALF: [number, number, number] = [0.42, 0.5, 0.28];
const PLINTH_TOP = 1.0;

/** Centre of a suspect's bust (the close-up camera frames this). */
export function headPosition(sid: SuspectId): Vec3 {
  const i = SUSPECT_ORDER.indexOf(sid);
  return [roomX("parlour") - 1.65 + i * 1.1, PLINTH_TOP + BUST_SCALE / 2, -0.9];
}

/** Every entity that belongs to a suspect (bust, plinth, name plate) — used for click picking. */
export function suspectEntityIds(sid: SuspectId): EntityId[] {
  return [headId(sid), id(`bust.${sid}.plinth`), id(`bust.${sid}.plate`)];
}

/** Each suspect is a textured 3D bust on a plinth in the parlour, turned toward the doorway. */
function buildSuspects(): Entity[] {
  const x = roomX("parlour");
  const out: Entity[] = [];
  SUSPECT_ORDER.forEach((sid) => {
    const s = SUSPECTS[sid];
    const c: Vec3 = headPosition(sid);
    const yaw = Math.atan2(x - c[0], 2.3 - c[2]) * 0.7;
    const rot = yawQuat(yaw);
    const [, accent] = s.palette;
    out.push({
      id: headId(sid),
      name: s.name,
      components: {
        [Tag.Transform]: { position: c, rotation: rot, scale: [BUST_SCALE, BUST_SCALE, BUST_SCALE], parent: null },
        [Tag.Mesh]: {
          geometry: { kind: "gltf", url: bustUrl(sid), primitiveIndex: 0 },
          material: { kind: "pbr", baseColor: [1, 1, 1], metallic: 0, roughness: 0.85 },
          visible: true,
        },
      },
    });
    out.push(box(`bust.${sid}.plinth`, `${s.name} plinth`, [c[0], PLINTH_TOP / 2, c[2]], [0.4, PLINTH_TOP, 0.34], [0.16, 0.13, 0.11], 0.6, 0.02, rot));
    // Brass name plate in the suspect's accent colour on the plinth's face.
    out.push(box(`bust.${sid}.plate`, `${s.name} plate`, [c[0] + Math.sin(yaw) * 0.175, PLINTH_TOP - 0.16, c[2] + Math.cos(yaw) * 0.175], [0.26, 0.06, 0.012], accent, 0.35, 0.7, rot));
  });
  // One warm key light over the row of busts, from the upper left like the paintings.
  out.push(light("parlour.bust-light", [x - 1.2, 2.6, 0.8], [1.0, 0.82, 0.6], 5, 6));
  // A rug and a sofa so the parlour reads as a room.
  out.push(box("parlour.rug", "rug", [x, 0.01, 0.6], [6.5, 0.02, 3.2], [0.38, 0.12, 0.1], 0.95));
  out.push(box("parlour.sofa", "sofa", [x - 3.4, 0.4, -2.6], [2.6, 0.8, 0.9], [0.2, 0.26, 0.3], 0.8));
  return out;
}

export function buildHouseSpec(): SceneSpec {
  const entities: Entity[] = [];
  for (const r of ROOMS) entities.push(...buildRoom(r.id));
  entities.push(...buildSuspects());
  entities.push({
    id: CLOSEUP_CAMERA,
    name: "dialogue close-up",
    components: {
      [Tag.Transform]: { position: [0, 1.6, 2], rotation: Q0, scale: [1, 1, 1], parent: null },
      [Tag.Camera]: { fovY: 0.6, near: 0.05, far: 60, active: false },
    },
  });
  entities.push({
    id: id("moon"),
    name: "moonlight",
    components: {
      [Tag.Transform]: { position: [0, 10, 5], rotation: Q0, scale: [1, 1, 1], parent: null },
      [Tag.Light]: { kind: "directional", color: [0.55, 0.62, 0.85], intensity: 0.35 },
    },
  });
  return {
    schemaVersion: 1,
    source: { kind: "authored", path: "examples/detective/wrenfield" },
    entities,
    defaultCamera: cameraId("parlour"),
    environment: { kind: "color", color: [0.03, 0.035, 0.05] },
  };
}
