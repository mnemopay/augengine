/**
 * Per-case set dressing: room palettes, where each clue object sits (relative
 * to its room's centre; the table top is at y≈0.76, z≈-1.0), and extra
 * furniture. Pure data plus small builders, so a new case only adds a layout.
 */
import type { Entity, Vec3 } from "@mnemopay/augengine";
import { box, light } from "./scene.js";

export type Shape = "box" | "cylinder" | "sphere";

export interface ObjectLayout {
  at: Vec3;
  scale: Vec3;
  color: Vec3;
  shape?: Shape;
  metallic?: number;
}

export interface CaseLayout {
  palette: Record<string, { floor: Vec3; wall: Vec3; lamp: Vec3 }>;
  objects: Record<string, ObjectLayout>;
  /** Colour of the table in searchable rooms. */
  table?: Vec3;
  /** Extra furniture for a room whose centre is at x. */
  furniture: (room: string, x: number) => Entity[];
  skyLight: { name: string; color: Vec3; intensity: number };
  clearColor: Vec3;
}

const Q0: [number, number, number, number] = [0, 0, 0, 1];

const WRENFIELD: CaseLayout = {
  palette: {
    parlour: { floor: [0.22, 0.13, 0.09], wall: [0.28, 0.2, 0.17], lamp: [1.0, 0.72, 0.45] },
    dome: { floor: [0.16, 0.17, 0.2], wall: [0.14, 0.16, 0.22], lamp: [0.65, 0.75, 1.0] },
    library: { floor: [0.2, 0.12, 0.08], wall: [0.18, 0.2, 0.16], lamp: [1.0, 0.62, 0.32] },
    kitchen: { floor: [0.34, 0.3, 0.26], wall: [0.42, 0.38, 0.32], lamp: [1.0, 0.85, 0.62] },
    conservatory: { floor: [0.2, 0.22, 0.2], wall: [0.3, 0.38, 0.36], lamp: [0.7, 0.9, 0.8] },
  },
  objects: {
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
  },
  furniture(room, x) {
    const out: Entity[] = [];
    if (room === "dome") out.push(box("room.dome.chair", "observing chair", [x + 1.9, 0.45, -1.6], [0.5, 0.9, 0.5], [0.16, 0.07, 0.06], 0.5));
    if (room === "library") {
      for (let i = 0; i < 4; i++) out.push(box(`room.library.shelf${i}`, "bookshelf", [x - 3.5 + i * 2.3, 1.6, -3.3], [1.9, 3.0, 0.4], [0.24 + i * 0.02, 0.15, 0.09], 0.8));
    }
    if (room === "conservatory") {
      out.push(box("room.conservatory.glass", "glass wall", [x, 1.8, -3.5], [9.6, 3.4, 0.05], [0.55, 0.75, 0.8], 0.05, 0.1));
      for (let i = 0; i < 3; i++) out.push(box(`room.conservatory.fern${i}`, "fern", [x + 2.8 - i * 1.4, 0.6, 1.8], [0.4, 0.6, 0.4], [0.16, 0.42, 0.2], 0.9, 0, Q0, "sphere"));
    }
    if (room === "parlour") {
      // A rug and a sofa so the parlour reads as a room.
      out.push(box("parlour.rug", "rug", [x, 0.01, 0.6], [6.5, 0.02, 3.2], [0.38, 0.12, 0.1], 0.95));
      out.push(box("parlour.sofa", "sofa", [x - 3.4, 0.4, -2.6], [2.6, 0.8, 0.9], [0.2, 0.26, 0.3], 0.8));
    }
    return out;
  },
  skyLight: { name: "moonlight", color: [0.55, 0.62, 0.85], intensity: 0.35 },
  clearColor: [0.03, 0.035, 0.05],
};

/** Lagos, 1957: terrazzo, louvred windows, painted plaster, valve glow. */
const DEAD_AIR: CaseLayout = {
  palette: {
    lobby: { floor: [0.4, 0.37, 0.32], wall: [0.3, 0.4, 0.33], lamp: [1.0, 0.78, 0.52] },
    studio: { floor: [0.17, 0.13, 0.11], wall: [0.33, 0.25, 0.2], lamp: [1.0, 0.66, 0.42] },
    bandroom: { floor: [0.3, 0.19, 0.11], wall: [0.46, 0.34, 0.18], lamp: [1.0, 0.7, 0.4] },
    engineering: { floor: [0.24, 0.24, 0.25], wall: [0.2, 0.26, 0.3], lamp: [0.72, 0.85, 1.0] },
    office: { floor: [0.27, 0.17, 0.1], wall: [0.5, 0.44, 0.33], lamp: [1.0, 0.84, 0.58] },
  },
  table: [0.3, 0.2, 0.12],
  objects: {
    "studio.mic": { at: [0.2, 1.15, -0.75], scale: [0.03, 0.42, 0.03], color: [0.82, 0.82, 0.85], shape: "cylinder", metallic: 0.95 },
    "studio.runsheet": { at: [-0.55, 0.78, -0.85], scale: [0.24, 0.01, 0.32], color: [0.93, 0.9, 0.8] },
    "studio.jacket": { at: [2.2, 0.75, -0.3], scale: [0.42, 0.66, 0.14], color: [0.8, 0.74, 0.58] },
    "studio.turntable": { at: [0.85, 0.8, -1.05], scale: [0.2, 0.02, 0.2], color: [0.08, 0.08, 0.09], shape: "cylinder", metallic: 0.3 },
    "engineering.bench": { at: [-0.4, 0.85, -1.1], scale: [0.8, 0.18, 0.4], color: [0.36, 0.3, 0.22] },
    "engineering.rack": { at: [2.6, 1.2, -3.1], scale: [1.0, 2.4, 0.6], color: [0.22, 0.24, 0.26], metallic: 0.6 },
    "engineering.ladder": { at: [-3.6, 1.6, -2.8], scale: [0.5, 3.2, 0.08], color: [0.45, 0.32, 0.2] },
    "office.ledger": { at: [-0.7, 0.8, -1.0], scale: [0.36, 0.06, 0.26], color: [0.26, 0.18, 0.32] },
    "office.memo": { at: [0.85, 0.77, -0.85], scale: [0.2, 0.01, 0.28], color: [0.95, 0.94, 0.88] },
    "office.basket": { at: [1.9, 0.18, -0.6], scale: [0.18, 0.18, 0.18], color: [0.55, 0.42, 0.24], shape: "cylinder" },
    "office.typewriter": { at: [0.1, 0.84, -1.05], scale: [0.4, 0.14, 0.3], color: [0.12, 0.12, 0.13], metallic: 0.5 },
    "bandroom.case": { at: [-1.9, 0.22, -0.4], scale: [0.6, 0.22, 0.38], color: [0.42, 0.2, 0.12] },
    "bandroom.piano": { at: [1.4, 1.42, -3.15], scale: [0.5, 0.04, 0.3], color: [0.92, 0.88, 0.78] },
    "bandroom.window": { at: [-1.6, 2.0, -3.45], scale: [1.6, 1.2, 0.04], color: [0.35, 0.5, 0.72], metallic: 0.1 },
  },
  furniture(room, x) {
    const out: Entity[] = [];
    if (room === "lobby") {
      out.push(box("lobby.bench", "waiting bench", [x - 3.2, 0.25, -2.7], [2.6, 0.5, 0.6], [0.34, 0.22, 0.13], 0.7));
      out.push(box("lobby.counter", "front desk", [x + 3.2, 0.55, -2.4], [2.4, 1.1, 0.7], [0.3, 0.2, 0.12], 0.6));
      out.push(box("lobby.wireless", "wireless set", [x + 3.2, 1.3, -2.45], [0.6, 0.4, 0.3], [0.45, 0.28, 0.14], 0.5, 0.1, Q0, "box", [0.18, 0.1, 0.03]));
      out.push(box("lobby.fan", "ceiling fan", [x, 3.45, 0.2], [0.9, 0.03, 0.9], [0.4, 0.38, 0.34], 0.5, 0.5, Q0, "cylinder"));
      out.push(box("lobby.rug", "mat", [x, 0.01, 0.6], [6.0, 0.02, 3.0], [0.5, 0.3, 0.12], 0.95));
    }
    if (room === "studio") {
      out.push(box("studio.felt", "felt panels", [x, 1.9, -3.45], [8.0, 2.4, 0.08], [0.24, 0.17, 0.13], 0.98));
      out.push(box("studio.onair", "ON AIR lamp", [x - 2.0, 2.9, -3.35], [0.5, 0.18, 0.06], [0.5, 0.06, 0.04], 0.4, 0, Q0, "box", [0.9, 0.08, 0.04]));
      out.push(box("studio.chair", "presenter's chair", [x + 2.2, 0.45, -0.4], [0.5, 0.9, 0.5], [0.18, 0.1, 0.07], 0.5));
      out.push(light("studio.onair-glow", [x - 2.0, 2.6, -2.8], [1.0, 0.25, 0.15], 1.6, 4));
    }
    if (room === "bandroom") {
      out.push(box("bandroom.upright", "upright piano", [x + 1.4, 0.7, -3.1], [1.5, 1.4, 0.55], [0.14, 0.08, 0.05], 0.5, 0.05));
      out.push(box("bandroom.drum", "conga drum", [x - 0.2, 0.42, 1.4], [0.22, 0.42, 0.22], [0.5, 0.3, 0.14], 0.6, 0, Q0, "cylinder"));
      out.push(box("bandroom.drum2", "conga drum", [x + 0.3, 0.38, 1.6], [0.2, 0.38, 0.2], [0.46, 0.26, 0.12], 0.6, 0, Q0, "cylinder"));
      out.push(light("bandroom.night", [x - 1.6, 2.0, -2.8], [0.45, 0.55, 0.85], 1.4, 4));
    }
    if (room === "engineering") {
      for (let i = 0; i < 3; i++) out.push(box(`engineering.valve${i}`, "valve", [x + 2.3 + i * 0.3, 2.0, -2.75], [0.06, 0.12, 0.06], [0.6, 0.35, 0.2], 0.3, 0.2, Q0, "cylinder", [0.9, 0.4, 0.1]));
      out.push(box("engineering.hookboard", "hook board", [x - 0.4, 1.8, -3.45], [1.4, 0.8, 0.05], [0.4, 0.33, 0.22], 0.8));
    }
    if (room === "office") {
      out.push(box("office.desk2", "Femi's desk", [x - 2.8, 0.38, 0.6], [1.6, 0.76, 0.8], [0.24, 0.15, 0.08], 0.6));
      out.push(box("office.letters", "pile of letters", [x - 2.8, 0.82, 0.6], [0.6, 0.1, 0.4], [0.9, 0.88, 0.8], 0.9));
      out.push(box("office.cabinet", "filing cabinet", [x + 3.4, 0.75, -3.0], [0.6, 1.5, 0.6], [0.3, 0.34, 0.3], 0.4, 0.6));
      out.push(box("office.painting", "painting of the lagoon", [x - 0.5, 2.2, -3.45], [1.6, 0.9, 0.04], [0.55, 0.42, 0.3], 0.8));
    }
    return out;
  },
  skyLight: { name: "night over the lagoon", color: [0.6, 0.6, 0.8], intensity: 0.3 },
  clearColor: [0.04, 0.035, 0.05],
};

const LAYOUTS: Record<string, CaseLayout> = { wrenfield: WRENFIELD, deadair: DEAD_AIR };

export function layoutFor(caseId: string): CaseLayout {
  return LAYOUTS[caseId] ?? WRENFIELD;
}
