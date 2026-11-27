import { describe, expect, it } from "vitest";
import { Tag } from "../scene/types.ts";
import { PROOF_HEAD_ENTITY_ID } from "./proof-head.ts";
import { PROOF_BUST_HEAD_POSITION, proofBustYaw, proofHeadBustEntities } from "./proof-head-bust.ts";

describe("proof head bust", () => {
  const ents = proofHeadBustEntities();
  const byId = new Map(ents.map((e) => [e.id as string, e]));

  it("keeps the head entity id (blendshapes keep targeting it) and moves it onto the bust", () => {
    const head = byId.get(PROOF_HEAD_ENTITY_ID)!;
    expect(head.components[Tag.Transform]!.position).toEqual(PROOF_BUST_HEAD_POSITION);
  });

  it("turns the face (+Z) toward the proof-room camera", () => {
    const yaw = proofBustYaw();
    const face = [Math.sin(yaw), 0, Math.cos(yaw)];
    const [x, , z] = PROOF_BUST_HEAD_POSITION;
    const toCam = [0 - x, 0, 5.6 - z];
    const len = Math.hypot(toCam[0]!, toCam[2]!);
    expect(face[0]! * toCam[0]! / len + face[2]! * toCam[2]! / len).toBeGreaterThan(0.999);
  });

  it("stacks neck, shoulders and plinth under the head with no gaps down to the floor", () => {
    const pos = (id: string) => byId.get(id)!.components[Tag.Transform]!;
    const neck = pos("proof.head.neck");
    const shoulders = pos("proof.head.shoulders");
    const plinth = pos("proof.head.plinth");
    const headY = PROOF_BUST_HEAD_POSITION[1];
    // Neck (cylinder height = 2·scale.y) reaches into the 21 cm head (chin ≈ headY − 0.105).
    expect(neck.position[1] + neck.scale[1]).toBeGreaterThan(headY - 0.105);
    // Shoulders top meets the neck bottom; plinth top meets the shoulders bottom.
    expect(shoulders.position[1] + shoulders.scale[1] / 2).toBeGreaterThanOrEqual(neck.position[1] - neck.scale[1] - 1e-6);
    expect(plinth.position[1] + plinth.scale[1] / 2).toBeCloseTo(shoulders.position[1] - shoulders.scale[1] / 2, 6);
    expect(plinth.position[1] - plinth.scale[1] / 2).toBeCloseTo(0, 6);
  });

  it("sits clear of the dais and the lead's start point", () => {
    const [x, , z] = PROOF_BUST_HEAD_POSITION;
    // Dais spans x ±1.6, z ±1.3; Kael starts at (−0.7, 0.4).
    expect(Math.abs(x) > 1.6 + 0.3 || Math.abs(z) > 1.3 + 0.3).toBe(true);
    expect(Math.hypot(x + 0.7, z - 0.4)).toBeGreaterThan(2.0);
  });
});
