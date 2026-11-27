/**
 * A case's building, made from engine primitives: its five rooms side by side
 * along +X, one camera per room, warm point lights, the clue objects as
 * props, and the four suspects as textured 3D busts on plinths in the
 * interview room. Room palettes, prop positions and furniture come from the
 * case's layout (scene-layouts.ts); everything else is shared.
 */
import { Tag, type Entity, type EntityId, type SceneSpec, type Vec3, type Quat } from "@mnemopay/augengine";
import { assetUrl } from "./asset-url.js";
import { lookAtTransform } from "@mnemopay/augengine/runtime/live";
import type { CaseDef, RoomId, SuspectId } from "@mnemopay/augengine-detective";
import { layoutFor, type Shape } from "./scene-layouts.js";

export const ROOM_SPACING = 14;

const id = (s: string) => s as EntityId;
const Q0: Quat = [0, 0, 0, 1];

export function roomX(c: CaseDef, room: RoomId): number {
  return c.rooms.findIndex((r) => r.id === room) * ROOM_SPACING;
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

export function box(
  name: string,
  label: string,
  pos: Vec3,
  scale: Vec3,
  color: Vec3,
  roughness = 0.7,
  metallic = 0.02,
  rot: Quat = Q0,
  shape: Shape = "box",
  emissive?: Vec3,
): Entity {
  return {
    id: id(name),
    name: label,
    components: {
      [Tag.Transform]: { position: pos, rotation: rot, scale, parent: null },
      [Tag.Mesh]: {
        geometry: { kind: "primitive", shape },
        material: { kind: "pbr", baseColor: color, metallic, roughness, ...(emissive ? { emissive } : {}) },
        visible: true,
      },
    },
  };
}

export function light(name: string, pos: Vec3, color: Vec3, intensity: number, range: number): Entity {
  return {
    id: id(name),
    name,
    components: {
      [Tag.Transform]: { position: pos, rotation: Q0, scale: [1, 1, 1], parent: null },
      [Tag.Light]: { kind: "point", color, intensity, range },
    },
  };
}

function buildRoom(c: CaseDef, room: RoomId): Entity[] {
  const layout = layoutFor(c.id);
  const x = roomX(c, room);
  const p = layout.palette[room] ?? { floor: [0.2, 0.2, 0.2], wall: [0.3, 0.3, 0.3], lamp: [1, 0.8, 0.6] };
  const out: Entity[] = [
    box(`room.${room}.floor`, `${room} floor`, [x, -0.05, 0], [10, 0.1, 8], p.floor, 0.85),
    box(`room.${room}.back`, `${room} back wall`, [x, 1.8, -3.6], [10, 3.6, 0.2], p.wall, 0.9),
    box(`room.${room}.left`, `${room} left wall`, [x - 4.9, 1.8, 0], [0.2, 3.6, 7.2], p.wall, 0.9),
    box(`room.${room}.right`, `${room} right wall`, [x + 4.9, 1.8, 0], [0.2, 3.6, 7.2], p.wall, 0.9),
    box(`room.${room}.ceiling`, `${room} ceiling`, [x, 3.65, 0], [10, 0.1, 8], [p.wall[0] * 0.6, p.wall[1] * 0.6, p.wall[2] * 0.6], 0.95),
    light(`room.${room}.lamp`, [x, 2.9, 0.6], p.lamp, 7, 9),
    light(`room.${room}.fill`, [x + 2.5, 1.6, 2.6], [0.55, 0.6, 0.75], 2.2, 7),
  ];
  // A table in every searchable room where the objects rest (desk, counter, bench...).
  if (room !== c.interviewRoom) out.push(box(`room.${room}.table`, `${room} table`, [x, 0.38, -1.0], [2.6, 0.76, 0.9], layout.table ?? [0.26, 0.17, 0.11], 0.6));
  out.push(...layout.furniture(room, x));
  for (const o of c.objects.filter((ob) => ob.room === room)) {
    const l = layout.objects[o.id];
    if (!l) continue;
    out.push(box(objectEntityId(o.id), o.name, [x + l.at[0], l.at[1], l.at[2]], l.scale, l.color, 0.5, l.metallic ?? 0.05, Q0, l.shape ?? "box"));
  }
  // Camera: standing in the doorway, looking at the table. The interview room's camera is refitted to the screen at runtime.
  const camPos: Vec3 = room === c.interviewRoom ? [x, 1.62, 2.3] : [x + 0.4, 1.65, 2.4];
  const target: Vec3 = room === c.interviewRoom ? [x, 1.2, -0.6] : [x - 0.1, 0.85, -1.3];
  out.push({
    id: cameraId(room),
    name: `${room} camera`,
    components: {
      [Tag.Transform]: { position: camPos, rotation: lookAtTransform(camPos, target, [0, 1, 0]).rotation, scale: [1, 1, 1], parent: null },
      [Tag.Camera]: { fovY: 0.95, near: 0.05, far: 80, active: room === c.interviewRoom },
    },
  });
  return out;
}

/** URL of a suspect's 3D bust (TRELLIS image-to-3D from their painted portrait; see public/portraits/SOURCES.md). */
export function bustUrl(c: CaseDef, sid: SuspectId): string {
  return assetUrl(`/busts/${c.assetPrefix}${sid}.glb`);
}

/** Bust height in metres (the source meshes are ~1 unit tall). */
const BUST_SCALE = 0.62;
/** Half-extents of the bust meshes in their own units (from the GLB bounds, all within these). */
export const BUST_HALF: [number, number, number] = [0.42, 0.5, 0.28];
const PLINTH_TOP = 1.0;
/** Distance between bust centres. Tight enough that the whole row fits between the side panels. */
export const BUST_SPACING = 0.85;
/** Half the width of the bust row, outer bust edges included (metres). */
export const BUST_ROW_HALF = 1.5 * BUST_SPACING + BUST_HALF[0] * BUST_SCALE;
const BUST_Z = -0.9;

/** Centre of a suspect's bust (the close-up camera frames this). */
export function headPosition(c: CaseDef, sid: SuspectId): Vec3 {
  const i = c.suspectOrder.indexOf(sid);
  return [roomX(c, c.interviewRoom) + (i - 1.5) * BUST_SPACING, PLINTH_TOP + BUST_SCALE / 2, BUST_Z];
}

/** Centre of the bust row, the point the interview camera frames. */
export function bustRowCenter(c: CaseDef): Vec3 {
  return [roomX(c, c.interviewRoom), PLINTH_TOP + BUST_SCALE / 2, BUST_Z];
}

/** Every entity that belongs to a suspect (bust, plinth, name plate) — used for click picking. */
export function suspectEntityIds(sid: SuspectId): EntityId[] {
  return [headId(sid), id(`bust.${sid}.plinth`), id(`bust.${sid}.plate`)];
}

/** Each suspect is a textured 3D bust on a plinth in the interview room, turned toward the doorway. */
function buildSuspects(c: CaseDef): Entity[] {
  const x = roomX(c, c.interviewRoom);
  const out: Entity[] = [];
  for (const sid of c.suspectOrder) {
    const s = c.suspects[sid]!;
    const pos: Vec3 = headPosition(c, sid);
    const yaw = Math.atan2(x - pos[0], 3.2 - pos[2]) * 0.7;
    const rot = yawQuat(yaw);
    const [, accent] = s.palette;
    out.push({
      id: headId(sid),
      name: s.name,
      components: {
        [Tag.Transform]: { position: pos, rotation: rot, scale: [BUST_SCALE, BUST_SCALE, BUST_SCALE], parent: null },
        [Tag.Mesh]: {
          geometry: { kind: "gltf", url: bustUrl(c, sid), primitiveIndex: 0 },
          material: { kind: "pbr", baseColor: [1, 1, 1], metallic: 0, roughness: 0.85 },
          visible: true,
        },
      },
    });
    out.push(box(`bust.${sid}.plinth`, `${s.name} plinth`, [pos[0], PLINTH_TOP / 2, pos[2]], [0.4, PLINTH_TOP, 0.34], [0.16, 0.13, 0.11], 0.6, 0.02, rot));
    // Brass name plate in the suspect's accent colour on the plinth's face.
    out.push(box(`bust.${sid}.plate`, `${s.name} plate`, [pos[0] + Math.sin(yaw) * 0.175, PLINTH_TOP - 0.16, pos[2] + Math.cos(yaw) * 0.175], [0.26, 0.06, 0.012], accent, 0.35, 0.7, rot));
  }
  // One warm key light over the row of busts, from the upper left like the paintings.
  out.push(light(`${c.interviewRoom}.bust-light`, [x - 1.2, 2.6, 0.8], [1.0, 0.82, 0.6], 5, 6));
  return out;
}

export function buildCaseSpec(c: CaseDef): SceneSpec {
  const layout = layoutFor(c.id);
  const entities: Entity[] = [];
  for (const r of c.rooms) entities.push(...buildRoom(c, r.id));
  entities.push(...buildSuspects(c));
  entities.push({
    id: CLOSEUP_CAMERA,
    name: "dialogue close-up",
    components: {
      [Tag.Transform]: { position: [0, 1.6, 2], rotation: Q0, scale: [1, 1, 1], parent: null },
      [Tag.Camera]: { fovY: 0.6, near: 0.05, far: 60, active: false },
    },
  });
  entities.push({
    id: id("sky-light"),
    name: layout.skyLight.name,
    components: {
      [Tag.Transform]: { position: [0, 10, 5], rotation: Q0, scale: [1, 1, 1], parent: null },
      [Tag.Light]: { kind: "directional", color: layout.skyLight.color, intensity: layout.skyLight.intensity },
    },
  });
  return {
    schemaVersion: 1,
    source: { kind: "authored", path: `examples/detective/${c.id}` },
    entities,
    defaultCamera: cameraId(c.interviewRoom),
    environment: { kind: "color", color: layout.clearColor },
  };
}
