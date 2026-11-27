import { describe, expect, it } from "vitest";
import { pick, rayBox, rayFromNdc, rotate, type PickBox, type PickCamera, type Q4 } from "./pick.js";

const I: Q4 = [0, 0, 0, 1];
const yaw = (a: number): Q4 => [0, Math.sin(a / 2), 0, Math.cos(a / 2)];
const cam: PickCamera = { position: [0, 0, 5], rotation: I, fovY: Math.PI / 3, aspect: 16 / 9 };

describe("pick", () => {
  it("the screen centre looks down −Z", () => {
    const { dir } = rayFromNdc(cam, 0, 0);
    expect(dir[0]).toBeCloseTo(0);
    expect(dir[2]).toBeCloseTo(-1);
  });

  it("rotate matches a 90° yaw (−Z forward turns to −X)", () => {
    const v = rotate(yaw(Math.PI / 2), [0, 0, -1]);
    expect(v[0]).toBeCloseTo(-1);
    expect(v[2]).toBeCloseTo(0);
  });

  it("hits a box in front and misses one to the side", () => {
    const a: PickBox = { id: "a", center: [0, 0, 0], rotation: I, half: [0.5, 0.5, 0.5] };
    const b: PickBox = { id: "b", center: [3, 0, 0], rotation: I, half: [0.5, 0.5, 0.5] };
    expect(pick(cam, 0, 0, [a, b])?.id).toBe("a");
    expect(rayBox([0, 0, 5], [0, 0, -1], b)).toBeNull();
  });

  it("returns the nearest of two overlapping boxes", () => {
    const far: PickBox = { id: "far", center: [0, 0, -2], rotation: I, half: [1, 1, 1] };
    const near: PickBox = { id: "near", center: [0, 0, 1], rotation: I, half: [0.2, 0.2, 0.2] };
    const hit = pick(cam, 0, 0, [far, near]);
    expect(hit?.id).toBe("near");
    expect(hit?.distance).toBeCloseTo(3.8);
  });

  it("respects box rotation (a thin plank turned 90° is hit edge-on)", () => {
    const plank: PickBox = { id: "p", center: [0.9, 0, 0], rotation: yaw(Math.PI / 2), half: [1, 0.2, 0.05] };
    // Unrotated it would span x ∈ [−0.1, 1.9]; turned 90° it spans only x ∈ [0.85, 0.95].
    expect(rayBox([0.3, 0, 5], [0, 0, -1], plank)).toBeNull();
    expect(rayBox([0.9, 0, 5], [0, 0, -1], plank)).not.toBeNull();
  });

  it("maps screen edges through the field of view", () => {
    // A box at the top edge of the view, 5 m away.
    const top = 5 * Math.tan(Math.PI / 6);
    const b: PickBox = { id: "t", center: [0, top, 0], rotation: I, half: [0.1, 0.1, 0.1] };
    expect(pick(cam, 0, 1, [b])?.id).toBe("t");
    expect(pick(cam, 0, 0, [b])).toBeNull();
  });

  it("ignores boxes behind the camera", () => {
    const behind: PickBox = { id: "x", center: [0, 0, 8], rotation: I, half: [0.5, 0.5, 0.5] };
    expect(pick(cam, 0, 0, [behind])).toBeNull();
  });
});
