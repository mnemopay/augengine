import { describe, expect, it } from "vitest";
import { KeyGuard, limitsFromEnv, openRouterStatusFetcher, type GuardLimits, type KeyStatus } from "./key-guard.js";

const LIMITS: GuardLimits = { dailyCapUsd: 3, monthlyCapUsd: 12, floorUsd: 10, instanceDailyUsd: 1, cacheMs: 30_000 };
const ok: KeyStatus = { usageDaily: 0.5, usageMonthly: 4, limitRemaining: 25 };

function clock(start = Date.UTC(2026, 10, 24, 12)) {
  let t = start;
  return { now: () => t, advance: (ms: number) => (t += ms) };
}

describe("KeyGuard", () => {
  it("is open while the key is under every line", async () => {
    const g = new KeyGuard(async () => ok, LIMITS);
    expect(await g.check()).toEqual({ open: true });
  });

  it("closes on the key-wide daily cap, monthly cap and remaining floor", async () => {
    expect(await new KeyGuard(async () => ({ ...ok, usageDaily: 3 }), LIMITS).check()).toEqual({ open: false, reason: "key daily cap reached" });
    expect(await new KeyGuard(async () => ({ ...ok, usageMonthly: 12.5 }), LIMITS).check()).toEqual({ open: false, reason: "key monthly cap reached" });
    expect(await new KeyGuard(async () => ({ ...ok, limitRemaining: 9.99 }), LIMITS).check()).toEqual({ open: false, reason: "key floor reached" });
  });

  it("treats an unlimited key (no remaining figure) as above the floor", async () => {
    expect(await new KeyGuard(async () => ({ ...ok, limitRemaining: null }), LIMITS).check()).toEqual({ open: true });
  });

  it("fails safe when the key status cannot be read", async () => {
    const d = await new KeyGuard(async () => {
      throw new Error("boom");
    }, LIMITS).check();
    expect(d.open).toBe(false);
  });

  it("closes once this instance spends its daily game budget, and reopens the next UTC day", async () => {
    const c = clock();
    const g = new KeyGuard(async () => ok, LIMITS, c.now);
    g.record(0.6);
    expect((await g.check()).open).toBe(true);
    g.record(0.4);
    expect(await g.check()).toEqual({ open: false, reason: "instance daily budget reached" });
    c.advance(24 * 3600_000);
    expect(g.instanceSpentToday).toBe(0);
    expect((await g.check()).open).toBe(true);
  });

  it("ignores non-finite or negative costs", () => {
    const g = new KeyGuard(async () => ok, LIMITS);
    g.record(Number.NaN);
    g.record(-1);
    expect(g.instanceSpentToday).toBe(0);
  });

  it("caches the key status and shares one in-flight request", async () => {
    const c = clock();
    let calls = 0;
    const g = new KeyGuard(async () => {
      calls++;
      return ok;
    }, LIMITS, c.now);
    await Promise.all([g.check(), g.check(), g.check()]);
    await g.check();
    expect(calls).toBe(1);
    c.advance(30_001);
    await g.check();
    expect(calls).toBe(2);
  });

  it("re-reads after the cache expires and closes when the key crosses a line", async () => {
    const c = clock();
    let status = ok;
    const g = new KeyGuard(async () => status, LIMITS, c.now);
    expect((await g.check()).open).toBe(true);
    status = { ...ok, usageDaily: 5 };
    expect((await g.check()).open).toBe(true); // still cached
    c.advance(30_001);
    expect(await g.check()).toEqual({ open: false, reason: "key daily cap reached" });
  });
});

describe("openRouterStatusFetcher", () => {
  const resp = (body: unknown, status = 200) =>
    (async () => new Response(JSON.stringify(body), { status })) as unknown as typeof fetch;

  it("reads usage_daily, usage_monthly and limit_remaining", async () => {
    const s = await openRouterStatusFetcher("k", resp({ data: { usage_daily: 1.2, usage_monthly: 3.5, limit_remaining: 26 } }))();
    expect(s).toEqual({ usageDaily: 1.2, usageMonthly: 3.5, limitRemaining: 26 });
  });

  it("throws on HTTP errors and malformed bodies (so the guard closes)", async () => {
    await expect(openRouterStatusFetcher("k", resp({}, 401))()).rejects.toThrow();
    await expect(openRouterStatusFetcher("k", resp({ data: { usage_daily: "x" } }))()).rejects.toThrow();
  });
});

describe("limitsFromEnv", () => {
  it("uses defaults and ignores bad values", () => {
    expect(limitsFromEnv({})).toEqual({ dailyCapUsd: 3, monthlyCapUsd: 12, floorUsd: 10, instanceDailyUsd: 1, cacheMs: 30_000 });
    expect(limitsFromEnv({ DETECTIVE_KEY_DAILY_USD: "2", DETECTIVE_KEY_FLOOR_USD: "-5" } as NodeJS.ProcessEnv).dailyCapUsd).toBe(2);
    expect(limitsFromEnv({ DETECTIVE_KEY_FLOOR_USD: "-5" } as NodeJS.ProcessEnv).floorUsd).toBe(10);
  });
});
