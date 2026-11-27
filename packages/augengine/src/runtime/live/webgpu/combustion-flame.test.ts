/**
 * Combustion → flame look mapping.
 *
 * Pure numeric mapping, so these are real behavioural assertions: the look
 * must respond monotonically to simulated state, go out when fuel runs out,
 * and never emit a non-finite uniform value.
 */

import { describe, it, expect } from "vitest";
import { flameFromCombustion } from "./combustion-flame.js";
import { resolveEmitterSpec, FIRE_PRESET } from "./particle-pass.js";

const IGNITION_K = 900;

function look(temperatureK: number, reactionMolesPerSecond: number) {
  return flameFromCombustion({ temperatureK, reactionMolesPerSecond, ignitionK: IGNITION_K });
}

describe("flameFromCombustion", () => {
  it("does not burn below the ignition temperature", () => {
    const cold = look(IGNITION_K - 1, 0.02);
    expect(cold.burning).toBe(false);
    expect(cold.options.active).toBe(false);
    expect(cold.heat).toBe(0);
  });

  it("burns at or above the ignition temperature while fuel is being consumed", () => {
    const hot = look(IGNITION_K, 0.02);
    expect(hot.burning).toBe(true);
    expect(hot.options.active).toBe(true);
  });

  it("goes out when the reaction stalls even though the body is still hot", () => {
    // This is the oxygen-exhaustion case: temperature stays high, but with no
    // reaction there is no flame.
    const stalled = look(1600, 0);
    expect(stalled.burning).toBe(false);
    expect(stalled.options.active).toBe(false);
    // Heat is still reported — a caller may keep glowing embers on it.
    expect(stalled.heat).toBeGreaterThan(0);
  });

  it("deactivates rather than zeroing the rate, so the flame fades out", () => {
    // Live particles must be allowed to finish their lifetime; an emission
    // rate of 0 with active:true would strand the pool instead.
    const out = look(1600, 0);
    expect(out.options.active).toBe(false);
  });

  it("ramps heat linearly from ignition to saturation", () => {
    const mid = flameFromCombustion({
      temperatureK: 1350,
      reactionMolesPerSecond: 0.02,
      ignitionK: 900,
      saturationK: 1800,
    });
    expect(mid.heat).toBeCloseTo(0.5, 6);
  });

  it("clamps heat at saturation instead of growing without bound", () => {
    const a = flameFromCombustion({ temperatureK: 1800, reactionMolesPerSecond: 0.02, ignitionK: 900, saturationK: 1800 });
    const b = flameFromCombustion({ temperatureK: 50_000, reactionMolesPerSecond: 0.02, ignitionK: 900, saturationK: 1800 });
    expect(a.heat).toBe(1);
    expect(b.heat).toBe(1);
    expect(b.options.intensity).toBeCloseTo(a.options.intensity!, 6);
  });

  it("brightens monotonically with temperature", () => {
    const temps = [900, 1100, 1300, 1500, 1800];
    const intensities = temps.map((t) => look(t, 0.02).options.intensity!);
    for (let i = 1; i < intensities.length; i++) {
      expect(intensities[i]!).toBeGreaterThan(intensities[i - 1]!);
    }
  });

  it("rises faster when hotter", () => {
    expect(look(1800, 0.02).options.buoyancy!).toBeGreaterThan(look(950, 0.02).options.buoyancy!);
  });

  it("emits more particles as fuel throughput rises, but sub-linearly", () => {
    const low = look(1500, 0.002).options.emissionRate!;
    const high = look(1500, 0.02).options.emissionRate!;
    expect(high).toBeGreaterThan(low);
    // sqrt softening: a 10× fuel increase must not be a 10× particle count.
    expect(high / low).toBeLessThan(10);
  });

  it("keeps a visible pilot flame at a trace fuel rate", () => {
    const trace = look(1500, 1e-9);
    expect(trace.burning).toBe(true);
    expect(trace.options.emissionRate!).toBeGreaterThan(0);
    expect(trace.options.emissionRate!).toBeLessThan(FIRE_PRESET.emissionRate);
  });

  it("clamps fuel drive at the reference rate so a spike cannot exceed the preset", () => {
    const huge = look(1500, 1e6);
    expect(huge.options.emissionRate!).toBeLessThanOrEqual(FIRE_PRESET.emissionRate);
  });

  it("never produces a non-finite uniform value, even for absurd input", () => {
    const cases = [
      { temperatureK: Number.NaN, reactionMolesPerSecond: 0.02, ignitionK: 900 },
      { temperatureK: 1500, reactionMolesPerSecond: Number.NaN, ignitionK: 900 },
      { temperatureK: Number.POSITIVE_INFINITY, reactionMolesPerSecond: Number.POSITIVE_INFINITY, ignitionK: 900 },
      { temperatureK: -500, reactionMolesPerSecond: -5, ignitionK: 900 },
      // Degenerate authoring: saturation at ignition would divide by zero.
      { temperatureK: 1500, reactionMolesPerSecond: 0.02, ignitionK: 900, saturationK: 900 },
      { temperatureK: 1500, reactionMolesPerSecond: 0.02, ignitionK: 900, referenceMolesPerSecond: 0 },
    ];
    for (const input of cases) {
      const result = flameFromCombustion(input);
      expect(Number.isFinite(result.heat), JSON.stringify(input)).toBe(true);
      for (const [key, value] of Object.entries(result.options)) {
        if (typeof value === "number") {
          expect(Number.isFinite(value), `${key} for ${JSON.stringify(input)}`).toBe(true);
        } else if (Array.isArray(value)) {
          for (const v of value) expect(Number.isFinite(v), `${key} for ${JSON.stringify(input)}`).toBe(true);
        }
      }
    }
  });

  it("produces options the emitter resolver accepts unchanged in range", () => {
    const spec = resolveEmitterSpec(look(1500, 0.02).options, FIRE_PRESET);
    expect(spec.active).toBe(true);
    expect(spec.emissionRate).toBeGreaterThan(0);
    expect(spec.intensity).toBeGreaterThan(1); // still hot enough to bloom
    expect(spec.sizeRange[0]).toBeLessThanOrEqual(spec.sizeRange[1]);
  });

  it("passes the emitter origin through when supplied, and omits it otherwise", () => {
    expect(look(1500, 0.02).options.origin).toBeUndefined();
    const placed = flameFromCombustion({
      temperatureK: 1500,
      reactionMolesPerSecond: 0.02,
      ignitionK: 900,
      origin: [1, 2, 3],
    });
    expect(placed.options.origin).toEqual([1, 2, 3]);
  });
});
