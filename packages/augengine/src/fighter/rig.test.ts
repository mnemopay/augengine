import { describe, expect, it } from "vitest";
import { BUTTON } from "./input.js";
import { clipForFighter, rigMarkers, sampleSkinned } from "./rig.js";
import { createFight } from "./sim.js";

function vertex(mesh: Float32Array, index: number): [number, number, number] {
  return [mesh[index * 3] ?? 0, mesh[index * 3 + 1] ?? 0, mesh[index * 3 + 2] ?? 0];
}

function maxAbsZ(mesh: Float32Array): number {
  let max = 0;
  for (let i = 2; i < mesh.length; i += 3) max = Math.max(max, Math.abs(mesh[i] ?? 0));
  return max;
}

describe("skinned rig sampled at poseFrame/60", () => {
  it("keeps the bind pose at idle frame 0 and deforms the punch by frame 8", () => {
    const markers = rigMarkers("mara");
    const idle = sampleSkinned("mara", "idle", 0);
    const again = sampleSkinned("mara", "idle", 0);
    expect(again).toEqual(idle);

    const punched = sampleSkinned("mara", "punch", 8);
    const handRest = vertex(idle, markers.hand);
    const handHit = vertex(punched, markers.hand);
    expect(handHit[0]).toBeGreaterThan(handRest[0] + 6);
    expect(sampleSkinned("mara", "punch", 8)).toEqual(punched);

    const held = sampleSkinned("mara", "punch", 8);
    expect(vertex(held, markers.hand)).toEqual(handHit);
  });

  it("loops the walk on a one-second clip and drops the head for a knockdown", () => {
    const markers = rigMarkers("mara");
    const step0 = vertex(sampleSkinned("mara", "walk", 0), markers.foot);
    const step30 = vertex(sampleSkinned("mara", "walk", 30), markers.foot);
    const step60 = vertex(sampleSkinned("mara", "walk", 60), markers.foot);
    expect(step30[0]).not.toBeCloseTo(step0[0], 1);
    expect(step60[0]).toBeCloseTo(step0[0], 3);
    expect(step60[1]).toBeCloseTo(step0[1], 3);

    const standing = vertex(sampleSkinned("mara", "idle", 0), markers.head);
    const down = vertex(sampleSkinned("mara", "down", 0), markers.head);
    expect(down[1]).toBeLessThan(standing[1] - 20);
  });

  it("builds Holt wider than Mara and picks the clip from the tick", () => {
    const mara = sampleSkinned("mara", "idle", 0);
    const holt = sampleSkinned("holt", "idle", 0);
    expect(maxAbsZ(holt)).toBeGreaterThan(maxAbsZ(mara) + 1);

    const fight = createFight();
    expect(clipForFighter(fight.fighters[0])).toBe("idle");
    fight.fighters[0].held = BUTTON.RIGHT;
    expect(clipForFighter(fight.fighters[0])).toBe("walk");
    fight.fighters[0].crouching = true;
    expect(clipForFighter(fight.fighters[0])).toBe("crouch");
    fight.fighters[0].y = 12;
    expect(clipForFighter(fight.fighters[0])).toBe("air");
    fight.fighters[0].phase = "attack";
    fight.fighters[0].move = fight.fighters[0].character.moves.find((move) => move.id === "shin") ?? null;
    expect(clipForFighter(fight.fighters[0])).toBe("kick");
    fight.fighters[0].phase = "knockdown";
    expect(clipForFighter(fight.fighters[0])).toBe("down");
  });
});
