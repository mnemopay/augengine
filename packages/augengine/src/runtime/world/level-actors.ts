/**
 * world/level-actors — who lives in each tile of The Loading Range, and what
 * it sounds like. Pure and deterministic per tile coordinate (like
 * level-content), so a tile streamed back in has the same residents and
 * emitters; their live state is restored by TileActors.
 */
import type { EntityId, PersonaAxes, Vec3 } from "../../scene/types.js";
import type { AmbientSpec, NpcSpec } from "./tile-actors.js";
import { buildTileContent, LANDMARKS, LEVEL, TILE_SIZE, tileCenter } from "./level-content.js";
import { terrainHeight } from "./terrain.js";

function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const TRAITS = ["curious", "guarded", "chatty", "tired", "watchful", "cheerful", "restless", "patient"];
const COLORS: Vec3[] = [[0.75, 0.42, 0.28], [0.3, 0.45, 0.7], [0.55, 0.6, 0.3], [0.7, 0.62, 0.5], [0.45, 0.3, 0.55]];
const NAMES = ["Ada", "Bola", "Chidi", "Dami", "Efe", "Funmi", "Gbenga", "Halima", "Ife", "Jide", "Kemi", "Lola"];

export function npcId(tx: number, tz: number, k: number): EntityId {
  return `stream.${tx}.${tz}.npc${k}` as EntityId;
}

/** True when (x, z) is clear of every prop footprint in its tile (plus margin). */
function clearOfProps(tx: number, tz: number, x: number, z: number, margin: number): boolean {
  for (const p of buildTileContent(tx, tz).props) {
    const hx = p.size[0] / 2 + margin, hz = p.size[2] / 2 + margin;
    if (Math.abs(x - p.center[0]) < hx && Math.abs(z - p.center[2]) < hz) return false;
  }
  return true;
}

export function levelNpcs(tx: number, tz: number): NpcSpec[] {
  const rand = rng((tx * 92837111) ^ (tz * 689287499) ^ 0xa11ce);
  const center = tileCenter(tx, tz);
  const landmark = LANDMARKS[`${tx}|${tz}`];
  const onRoad = Math.abs(center[2] - LEVEL.roadZ) < TILE_SIZE / 2;
  const count = landmark ? 3 : onRoad ? 2 : rand() < 0.3 ? 1 : 0;
  const out: NpcSpec[] = [];
  for (let k = 0; k < count; k++) {
    let home: Vec3 | null = null;
    for (let attempt = 0; attempt < 20 && !home; attempt++) {
      const x = center[0] + (rand() - 0.5) * (TILE_SIZE - 12);
      const z = onRoad
        ? LEVEL.roadZ + (rand() < 0.5 ? -1 : 1) * (LEVEL.roadHalfWidth + 2 + rand() * 6)
        : center[2] + (rand() - 0.5) * (TILE_SIZE - 12);
      if (clearOfProps(tx, tz, x, z, 2)) home = [x, 0, z];
    }
    if (!home) continue;
    const axes: PersonaAxes = { aggression: rand() * 0.4, fear: rand(), loyalty: rand(), curiosity: rand() };
    out.push({
      id: npcId(tx, tz, k),
      name: NAMES[Math.floor(rand() * NAMES.length)]!,
      home,
      persona: {
        traits: [TRAITS[Math.floor(rand() * TRAITS.length)]!, TRAITS[Math.floor(rand() * TRAITS.length)]!],
        mood: rand() * 0.6 - 0.2,
        axes,
        role: landmark ? "guide" : "civilian",
      },
      color: COLORS[Math.floor(rand() * COLORS.length)]!,
    });
  }
  return out;
}

const LANDMARK_SOUNDS: Readonly<Record<string, { sound: string; gain: number; radius: number; dz: number; y: number }>> = {
  "West Gate": { sound: "wind", gain: 0.5, radius: 55, dz: 0, y: 10 },
  "Water Tower": { sound: "drip", gain: 0.6, radius: 40, dz: -16, y: 18 },
  "Signal Spire": { sound: "hum-high", gain: 0.5, radius: 70, dz: -14, y: 30 },
  "Monolith Ring": { sound: "drone", gain: 0.55, radius: 60, dz: 26, y: 4 },
  "East Beacon": { sound: "hum", gain: 0.6, radius: 60, dz: 0, y: 4 },
};

export function levelAmbients(tx: number, tz: number): AmbientSpec[] {
  const out: AmbientSpec[] = [];
  const c = tileCenter(tx, tz);
  const landmark = LANDMARKS[`${tx}|${tz}`];
  if (landmark) {
    const s = LANDMARK_SOUNDS[landmark];
    if (s) out.push({ key: `amb.${tx}.${tz}.landmark`, position: [c[0], s.y, LEVEL.roadZ + s.dz], sound: s.sound, gain: s.gain, radius: s.radius });
  }
  // Wind on the tile's highest ground, when it is a real hill.
  let best: Vec3 = [c[0], 0, c[2]];
  for (let j = 0; j <= 4; j++) {
    for (let i = 0; i <= 4; i++) {
      const x = c[0] - TILE_SIZE / 2 + (i * TILE_SIZE) / 4, z = c[2] - TILE_SIZE / 2 + (j * TILE_SIZE) / 4;
      const h = terrainHeight(x, z);
      if (h > best[1]) best = [x, h, z];
    }
  }
  if (best[1] > 10) out.push({ key: `amb.${tx}.${tz}.wind`, position: [best[0], best[1] + 2, best[2]], sound: "wind", gain: 0.35, radius: 45 });
  return out;
}
