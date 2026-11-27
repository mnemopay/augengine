/**
 * HUD formatter tests — Track 6. The DOM `mountFrameBudgetHud` is
 * exercised by the proof-room capture e2e (`hud_present` assertion);
 * this unit file covers the pure string formatter so layout changes
 * fail fast.
 */
import { describe, expect, it } from "vitest";
import { formatHud } from "./hud.ts";

describe("formatHud", () => {
  it("shows a placeholder line when the budget is null", () => {
    const out = formatHud(null);
    expect(out).toContain("augengine perf");
    expect(out).toContain("waiting");
  });

  it("renders fps + cpu p50/p95/p99 + gpu + face + source", () => {
    const out = formatHud({
      fps: 59.81,
      p50Ms: 14.22,
      p95Ms: 17.84,
      p99Ms: 22.10,
      maxMs: 31.50,
      budgetMissCount: 4,
      count: 600,
      gpu_ms: 6.41,
      gpu_ms_p95: 8.10,
      gpu_ms_source: "timestamp-query",
      face_ms: 4.20,
      face_provider: "cherry",
    });
    expect(out).toContain("fps     59.8");
    expect(out).toContain("cpu p50 14.22 ms");
    expect(out).toContain("cpu p95 17.84 ms");
    expect(out).toContain("cpu p99 22.10 ms");
    expect(out).toContain("cpu max 31.50 ms");
    expect(out).toContain("gpu     6.41 ms (ts)");
    expect(out).toContain("gpu p95 8.10 ms");
    expect(out).toContain("face    4.20 ms (cherry)");
    expect(out).toContain("miss    4/600");
  });

  it("renders em-dash for missing optional fields", () => {
    const out = formatHud({ fps: 60, p50Ms: 16.67, p95Ms: 16.67, p99Ms: 16.67, maxMs: 16.67, budgetMissCount: 0, count: 60 });
    expect(out).toContain("gpu     — ms (—)");
    expect(out).toContain("face    — ms (—)");
  });

  it("labels CPU fallback as (cpu)", () => {
    const out = formatHud({
      fps: 30,
      p50Ms: 33,
      p95Ms: 50,
      p99Ms: 60,
      maxMs: 99,
      budgetMissCount: 1,
      count: 60,
      gpu_ms: 4.2,
      gpu_ms_p95: 5.1,
      gpu_ms_source: "cpu-fallback",
    });
    expect(out).toContain("gpu     4.20 ms (cpu)");
  });
});
