import { describe, it, expect } from "vitest";
import { createLocomotion, tickLocomotion, yawToQuat } from "./locomotion.js";
import type { EntityId } from "../../scene/types.js";

describe("proof-room locomotion", () => {
  it("cycles walk → turn → stop", () => {
    const state = createLocomotion("lead.kael" as EntityId, [0, 0, 0], 0);
    expect(state.gait).toBe("walk");
    tickLocomotion(state, 3.05);
    expect(state.gait).toBe("turn");
    tickLocomotion(state, 1.25);
    expect(state.gait).toBe("stop");
    tickLocomotion(state, 1.55);
    expect(state.gait).toBe("walk");
  });

  it("walks forward and turns yaw", () => {
    const walking = createLocomotion("lead.kael" as EntityId, [0, 0, 0], 0);
    const pose = tickLocomotion(walking, 1);
    expect(pose.position[2]).toBeGreaterThan(0.5);
    const turning = createLocomotion("lead.kael" as EntityId, [0, 0, 0], 0);
    turning.gait = "turn";
    const afterTurn = tickLocomotion(turning, 1.2);
    expect(Math.abs(afterTurn.rotation[1])).toBeGreaterThan(0.5);
  });

  it("builds a unit yaw quaternion", () => {
    const q = yawToQuat(0);
    expect(q[3]).toBeCloseTo(1);
  });

  it("turns in place after walking instead of orbiting the starting point", () => {
    const state = createLocomotion("lead.kael" as EntityId, [0, 0, 0]);
    const walked = tickLocomotion(state, 3);
    const turning = tickLocomotion(state, 0.6);
    expect(turning.position).toEqual(walked.position);
  });

  it("produces the same pose when a frame crosses multiple gait boundaries", () => {
    const whole = createLocomotion("lead.kael" as EntityId, [1, 2, 3]);
    const split = createLocomotion("lead.kael" as EntityId, [1, 2, 3]);
    const expected = tickLocomotion(whole, 7);
    let actual = tickLocomotion(split, 0);
    for (let i = 0; i < 140; i++) actual = tickLocomotion(split, 0.05);
    expect(actual.gait).toBe(expected.gait);
    actual.position.forEach((value, i) => expect(value).toBeCloseTo(expected.position[i]!, 8));
    actual.rotation.forEach((value, i) => expect(value).toBeCloseTo(expected.rotation[i]!, 8));
  });
});
