import { describe, it, expect } from "vitest";
import { Tag } from "../scene/types.js";
import { Scene } from "../scene/scene.js";
import { CHARACTER_PROOF_ROOM, PROOF_LEAD_ID, PROOF_NPC_ID, buildCharacterProofRoomEntities } from "./character-proof-room.js";
import { PROOF_CLOSEUP_CAMERA_ID, frameCloseup } from "../runtime/proof/dialogue-camera.js";

describe("Character Proof Room", () => {
  it("updates cached camera transforms when framing a dialogue close-up", () => {
    const scene = new Scene(structuredClone(CHARACTER_PROOF_ROOM));
    const first = scene.worldTransforms();
    const matrices = first.worldMatrices;
    frameCloseup(scene, [2, 1.6, -1]);
    const worlds = scene.worldTransforms();
    expect(worlds).toBe(first);
    expect(worlds.worldMatrices).toBe(matrices);
    const index = worlds.indexById.get(PROOF_CLOSEUP_CAMERA_ID)! * 16;
    expect(worlds.worldMatrices[index + 12]).toBeCloseTo(2);
    expect(worlds.worldMatrices[index + 14]).toBeCloseTo(0.4);
    expect(scene.activeCamera).toBe(PROOF_CLOSEUP_CAMERA_ID);
  });
  it("grounds character feet at their root instead of floating above the dais", () => {
    const leg = CHARACTER_PROOF_ROOM.entities.find((e) => e.id === `${PROOF_LEAD_ID}.leg.l`)!.components[Tag.Transform]!;
    expect(leg.position[1] - leg.scale[1] / 2).toBeCloseTo(0);
  });
  it("points the wide camera at the characters rather than away from the room", () => {
    const scene = new Scene(structuredClone(CHARACTER_PROOF_ROOM));
    const worlds = scene.worldTransforms();
    const camera = worlds.indexById.get(scene.activeCamera!)! * 16;
    const lead = worlds.indexById.get(PROOF_LEAD_ID)! * 16;
    const m = worlds.worldMatrices;
    const toward = [m[lead + 12]! - m[camera + 12]!, m[lead + 13]! - m[camera + 13]!, m[lead + 14]! - m[camera + 14]!];
    const forwardDot = -m[camera + 8]! * toward[0]! - m[camera + 9]! * toward[1]! - m[camera + 10]! * toward[2]!;
    expect(forwardDot).toBeGreaterThan(0);
  });
  it("spawns the player facing the characters, not out of the room", () => {
    const scene = new Scene(structuredClone(CHARACTER_PROOF_ROOM));
    const worlds = scene.worldTransforms();
    const player = CHARACTER_PROOF_ROOM.entities.find((e) => e.name === "Player")!;
    const camera = worlds.indexById.get(player.id)! * 16;
    const lead = worlds.indexById.get(PROOF_LEAD_ID)! * 16;
    const m = worlds.worldMatrices;
    const toward = [m[lead + 12]! - m[camera + 12]!, m[lead + 13]! - m[camera + 13]!, m[lead + 14]! - m[camera + 14]!];
    const forwardDot = -m[camera + 8]! * toward[0]! - m[camera + 9]! * toward[1]! - m[camera + 10]! * toward[2]!;
    expect(forwardDot).toBeGreaterThan(0);
  });
  it("ships a textured lead, NPC, close-up camera, and IBL environment", () => {
    const ids = CHARACTER_PROOF_ROOM.entities.map((e) => e.id as string);
    expect(ids).toContain(PROOF_LEAD_ID as string);
    expect(ids).toContain(PROOF_NPC_ID as string);
    expect(ids).toContain(PROOF_CLOSEUP_CAMERA_ID as string);
    expect(CHARACTER_PROOF_ROOM.environment?.kind).toBe("ibl");
    const leadParts = CHARACTER_PROOF_ROOM.entities.filter((e) => (e.id as string).startsWith("lead.kael"));
    expect(leadParts.length).toBeGreaterThanOrEqual(8);
    const mesh = CHARACTER_PROOF_ROOM.entities.find((e) => e.id === `${PROOF_LEAD_ID}.torso`)?.components[Tag.Mesh];
    expect(mesh?.material.kind).toBe("pbr");
    expect(CHARACTER_PROOF_ROOM.defaultCamera).toBeDefined();
    expect(buildCharacterProofRoomEntities().length).toBe(CHARACTER_PROOF_ROOM.entities.length);
  });
});
