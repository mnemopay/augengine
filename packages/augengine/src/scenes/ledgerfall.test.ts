import { describe, it, expect } from "vitest";
import {
  LEDGERFALL_BEATS,
  LEDGERFALL_SCENES,
  ledgerfallGoldenPathSeconds,
  buildInformantBooth,
} from "./ledgerfall.js";

describe("LEDGERFALL slice scaffold", () => {
  it("covers the 12–20 minute golden path with three named beats", () => {
    expect(LEDGERFALL_BEATS).toHaveLength(3);
    const seconds = ledgerfallGoldenPathSeconds();
    expect(seconds).toBeGreaterThanOrEqual(12 * 60);
    expect(seconds).toBeLessThanOrEqual(20 * 60);
  });

  it("builds each beat as a WebGPU scene spec with a camera and player", () => {
    for (const beat of LEDGERFALL_BEATS) {
      const spec = LEDGERFALL_SCENES[beat.id]();
      expect(spec.schemaVersion).toBe(1);
      expect(spec.entities.some((e) => e.id === "player")).toBe(true);
      expect(spec.defaultCamera).toBeDefined();
    }
    const booth = buildInformantBooth();
    expect(booth.entities.some((e) => e.id === "mira")).toBe(true);
    expect(booth.entities.some((e) => e.id === "camera.closeup")).toBe(true);
  });
});
