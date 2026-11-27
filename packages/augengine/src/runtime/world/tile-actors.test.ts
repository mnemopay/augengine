import { describe, expect, it } from "vitest";
import { Scene } from "../../scene/scene.js";
import { Tag, type EntityId, type Vec3 } from "../../scene/types.js";
import { LEVEL, TILE_SIZE } from "./level-content.js";
import { levelAmbients, levelNpcs, npcId } from "./level-actors.js";
import { terrainHeight } from "./terrain.js";
import { TileActors, type AmbientAudioSink, type AmbientSpec } from "./tile-actors.js";

function emptyScene(): Scene {
  return new Scene({ schemaVersion: 1, source: { kind: "authored", path: "test:tile-actors" }, entities: [] });
}

class FakeSink implements AmbientAudioSink {
  live = new Map<number, { spec: AmbientSpec; gain: number; stopped: boolean }>();
  next = 0;
  start(spec: AmbientSpec) {
    const id = this.next++;
    const v = { spec, gain: 0, stopped: false };
    this.live.set(id, v);
    return {
      setGain: (g: number) => { if (v.stopped) throw new Error("gain after stop"); v.gain = g; },
      stop: () => { v.stopped = true; this.live.delete(id); },
    };
  }
}

function makeActors(scene: Scene, resident: Set<string>, sink?: AmbientAudioSink) {
  return new TileActors(scene, {
    npcsFor: levelNpcs,
    ambientsFor: levelAmbients,
    groundY: terrainHeight,
    walkable: (x, z) => resident.has(`${Math.floor(x / TILE_SIZE)}|${Math.floor(z / TILE_SIZE)}`),
    audio: sink,
    fadeSeconds: 1,
    maxVoices: 4,
  });
}

describe("tile actors: NPCs stream with their tile", () => {
  it("spawns a landmark's residents with personas, and restores their state after an unload", () => {
    const scene = emptyScene();
    const resident = new Set(["8|8"]);
    const actors = makeActors(scene, resident);
    actors.onTileResident(8, 8);
    const ids = levelNpcs(8, 8).map((n) => n.id);
    expect(ids.length).toBe(3);
    for (const id of ids) {
      const e = scene.get(id)!;
      expect(e.components[Tag.Persona]?.traits.length).toBeGreaterThan(0);
      expect(e.components[Tag.Collider]?.shape).toBe("capsule");
    }
    for (let f = 0; f < 60 * 20; f++) actors.update(1 / 60, [544, 2, 512]);
    const before = new Map(ids.map((id) => [id, JSON.parse(JSON.stringify(actors.liveState(id)))]));
    // Somebody actually moved off home.
    expect(ids.some((id) => {
      const s = actors.liveState(id)!, home = levelNpcs(8, 8).find((n) => n.id === id)!.home;
      return Math.hypot(s.position[0] - home[0], s.position[2] - home[2]) > 0.5;
    })).toBe(true);

    resident.delete("8|8");
    actors.onTileEvicted(8, 8);
    for (const id of ids) {
      expect(scene.has(id)).toBe(false);
      expect(actors.savedState(id)).toEqual(before.get(id));
    }
    resident.add("8|8");
    actors.onTileResident(8, 8);
    for (const id of ids) {
      expect(actors.liveState(id)).toEqual(before.get(id));
      expect(scene.get(id)!.components[Tag.Transform]!.position).toEqual(before.get(id).position);
    }
    expect(actors.stats.restored).toBe(3);
  });

  it("keeps NPCs on resident ground and on the terrain surface", () => {
    const scene = emptyScene();
    const resident = new Set(["4|8"]);
    const actors = makeActors(scene, resident);
    actors.onTileResident(4, 8);
    for (let f = 0; f < 60 * 30; f++) {
      actors.update(1 / 60, [288, 2, 512]);
      for (const n of levelNpcs(4, 8)) {
        const p = actors.liveState(n.id)!.position;
        expect(Math.floor(p[0] / TILE_SIZE)).toBe(4);
        expect(p[1]).toBeCloseTo(terrainHeight(p[0], p[2]) + 0.9, 5);
      }
    }
  });
});

describe("tile actors: ambient audio fades in, fades out, and is released", () => {
  it("fades a landmark voice in on load and releases it after the tile leaves", () => {
    const scene = emptyScene();
    const sink = new FakeSink();
    const actors = makeActors(scene, new Set(["15|8"]), sink);
    actors.onTileResident(15, 8);
    const beacon: Vec3 = [15 * TILE_SIZE + 32, 4, LEVEL.roadZ];
    actors.update(0.25, beacon);
    const first = [...sink.live.values()][0]!;
    expect(first.gain).toBeGreaterThan(0);
    expect(first.gain).toBeLessThan(first.spec.gain);
    for (let k = 0; k < 8; k++) actors.update(0.25, beacon);
    expect(first.gain).toBeCloseTo(first.spec.gain, 5);
    actors.onTileEvicted(15, 8);
    actors.update(0.25, beacon);
    expect(first.stopped).toBe(false); // still fading out
    for (let k = 0; k < 8; k++) actors.update(0.25, beacon);
    expect(first.stopped).toBe(true);
    expect(sink.live.size).toBe(0);
    expect(actors.stats.emitters).toBe(0);
  });
});

describe("tile actors: bounded over a long walk", () => {
  it("never leaks NPCs, emitters or voices over a 2 km out-and-back walk", () => {
    const scene = emptyScene();
    const sink = new FakeSink();
    const resident = new Set<string>();
    const actors = makeActors(scene, resident, sink);
    const radius = 1.75;
    let peakResidentNpcSlots = 0;
    const route: number[] = [];
    for (let x = 8; x <= 1016; x += 2) route.push(x);
    for (let x = 1016; x >= 8; x -= 2) route.push(x);
    for (const x of route) {
      const pos: Vec3 = [x, 2, LEVEL.roadZ];
      const want = new Set<string>();
      const cx = x / TILE_SIZE, cz = LEVEL.roadZ / TILE_SIZE;
      for (let tz = 0; tz < LEVEL.gridSize; tz++) {
        for (let tx = 0; tx < LEVEL.gridSize; tx++) {
          if (Math.hypot(tx + 0.5 - cx, tz + 0.5 - cz) <= radius) want.add(`${tx}|${tz}`);
        }
      }
      for (const k of [...resident]) if (!want.has(k)) {
        resident.delete(k);
        const [tx, tz] = k.split("|").map(Number) as [number, number];
        actors.onTileEvicted(tx, tz);
      }
      for (const k of want) if (!resident.has(k)) {
        resident.add(k);
        const [tx, tz] = k.split("|").map(Number) as [number, number];
        actors.onTileResident(tx, tz);
      }
      let slots = 0;
      for (const k of resident) {
        const [tx, tz] = k.split("|").map(Number) as [number, number];
        slots += levelNpcs(tx, tz).length;
      }
      peakResidentNpcSlots = Math.max(peakResidentNpcSlots, slots);
      // ~1.4 m/s walking pace between route points.
      for (let f = 0; f < 3; f++) actors.update(0.5, pos);
      expect(actors.stats.voices).toBeLessThanOrEqual(4);
      expect(sink.live.size).toBe(actors.stats.voices);
    }
    expect(actors.stats.peakNpcs).toBeLessThanOrEqual(peakResidentNpcSlots);
    expect(actors.stats.peakNpcs).toBeLessThan(30);
    expect(actors.stats.voicesStarted).toBeGreaterThan(4);
    expect(actors.stats.voicesStarted - actors.stats.voicesReleased).toBe(actors.stats.voices);
    // Every NPC in the scene belongs to a resident tile.
    for (const e of scene.withComponent(Tag.Persona)) {
      const [, tx, tz] = (e.id as string).split(".");
      expect(resident.has(`${tx}|${tz}`)).toBe(true);
    }
    expect(scene.get(npcId(0, 8, 0) as EntityId) !== undefined || resident.has("0|8")).toBe(true);
  });
});
