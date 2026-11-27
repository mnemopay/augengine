import { describe, it, expect } from "vitest";
import { ARKIT_BLENDSHAPE_COUNT } from "../runtime/live/webgpu/blendshape-constants.js";
import {
  CHERRY_MORPH_COUNT,
  CHERRY_VISEMES,
  PROCEDURAL_CHANNELS,
  cherryToArkit,
  emptyCherryWeights,
  packCherryWeights,
  proceduralIdleFace,
} from "./cherry-visemes.js";

describe("Cherry 12 visemes + 8 procedural", () => {
  it("is a 20-channel v1 set", () => {
    expect(CHERRY_VISEMES).toHaveLength(12);
    expect(PROCEDURAL_CHANNELS).toHaveLength(8);
    expect(CHERRY_MORPH_COUNT).toBe(20);
  });

  it("packs 20 floats and projects onto ARKit 52", () => {
    const frame = emptyCherryWeights();
    frame.visemes.aa = 0.8;
    frame.procedural.blinkL = 1;
    const packed = packCherryWeights(frame);
    expect(packed).toHaveLength(20);
    expect(packed[8]).toBeCloseTo(0.8); // aa
    const arkit = cherryToArkit(frame);
    expect(arkit).toHaveLength(ARKIT_BLENDSHAPE_COUNT);
    expect(arkit[23]).toBeGreaterThan(0.5); // jawOpen from aa
    expect(arkit[8]).toBe(1); // blinkL
  });

  it("idle face blinks and keeps sil dominant", () => {
    const open = proceduralIdleFace(0);
    expect(open.visemes.sil).toBeGreaterThan(0.5);
    const blinking = proceduralIdleFace(Math.PI / (2 * 1.7));
    expect(blinking.procedural.blinkL).toBeGreaterThanOrEqual(0);
  });
});
