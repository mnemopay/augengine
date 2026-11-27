import { PassThrough } from "node:stream";
import type { IncomingMessage, ServerResponse } from "node:http";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { KeyGuard, type GuardLimits } from "./key-guard.js";
import { handleDetectiveApi, setKeyGuardForTests } from "./chat-api.js";

const LIMITS: GuardLimits = { dailyCapUsd: 3, monthlyCapUsd: 12, floorUsd: 10, instanceDailyUsd: 1, cacheMs: 30_000 };
const KEY = "test-key-not-real-0123456789";

function req(url: string, method: string, headers: Record<string, string> = {}, body?: unknown): IncomingMessage & { body?: unknown } {
  const s = new PassThrough() as unknown as IncomingMessage & { body?: unknown };
  Object.assign(s, { url, method, headers: { "x-forwarded-for": `10.0.0.${Math.floor(Math.random() * 250)}`, ...headers }, socket: { remoteAddress: "127.0.0.1" } });
  if (body !== undefined) s.body = body;
  return s;
}

function res() {
  const out = { status: 0, headers: {} as Record<string, unknown>, body: "" };
  const r = {
    headersSent: false,
    writeHead(status: number, headers: Record<string, unknown> = {}) {
      out.status = status;
      out.headers = headers;
      r.headersSent = true;
      return r;
    },
    write(chunk: string) {
      out.body += chunk;
      return true;
    },
    end(chunk?: string) {
      if (chunk) out.body += chunk;
    },
    on() {
      return r;
    },
  };
  return { r: r as unknown as ServerResponse, out };
}

describe("detective API spend guard", () => {
  beforeEach(() => {
    process.env.OPENROUTER_API_KEY = KEY;
    delete process.env.DETECTIVE_TEST_SECRET;
  });
  afterEach(() => {
    setKeyGuardForTests(null);
    delete process.env.OPENROUTER_API_KEY;
    delete process.env.DETECTIVE_TEST_SECRET;
  });

  it("health reports resting (llm=false) when the key-wide daily cap is reached", async () => {
    setKeyGuardForTests(new KeyGuard(async () => ({ usageDaily: 3.5, usageMonthly: 4, limitRemaining: 25 }), LIMITS), KEY);
    const { r, out } = res();
    await handleDetectiveApi(req("/api/detective/health", "GET"), r);
    const j = JSON.parse(out.body);
    expect(j.llm).toBe(false);
    expect(j.resting).toBe("key daily cap reached");
    expect(j.budget.instanceDailyUsd).toBe(1);
    expect(JSON.stringify(j)).not.toContain(KEY);
  });

  it("health is open when the key is under every line", async () => {
    setKeyGuardForTests(new KeyGuard(async () => ({ usageDaily: 0.2, usageMonthly: 4, limitRemaining: 25 }), LIMITS), KEY);
    const { r, out } = res();
    await handleDetectiveApi(req("/api/detective/health", "GET"), r);
    expect(JSON.parse(out.body)).toMatchObject({ llm: true, resting: null });
  });

  it("chat refuses with 429 resting before any model call when the guard is closed", async () => {
    let fetched = false;
    setKeyGuardForTests(new KeyGuard(async () => {
      fetched = true;
      return { usageDaily: 0.1, usageMonthly: 1, limitRemaining: 5 };
    }, LIMITS), KEY);
    const { r, out } = res();
    await handleDetectiveApi(req("/api/detective/chat", "POST", { "x-dtv-session": "abcd1234efgh" }, {}), r);
    expect(fetched).toBe(true);
    expect(out.status).toBe(429);
    expect(JSON.parse(out.body)).toEqual({ error: "resting", reason: "key floor reached" });
  });

  it("the force-offline switch only works with the server-side secret", async () => {
    setKeyGuardForTests(new KeyGuard(async () => ({ usageDaily: 0.1, usageMonthly: 1, limitRemaining: 25 }), LIMITS), KEY);
    const secret = "s3cret-for-tests-only-123456";
    process.env.DETECTIVE_TEST_SECRET = secret;

    const wrong = res();
    await handleDetectiveApi(req("/api/detective/health", "GET", { "x-dtv-force-offline": "s3cret-for-tests-only-12345X" }), wrong.r);
    expect(JSON.parse(wrong.out.body).llm).toBe(true);

    const right = res();
    await handleDetectiveApi(req("/api/detective/chat", "POST", { "x-dtv-session": "abcd1234efgh", "x-dtv-force-offline": secret }, {}), right.r);
    expect(right.out.status).toBe(429);
    expect(JSON.parse(right.out.body).error).toBe("resting");
  });

  it("the switch is ignored when no secret (or a short one) is configured", async () => {
    setKeyGuardForTests(new KeyGuard(async () => ({ usageDaily: 0.1, usageMonthly: 1, limitRemaining: 25 }), LIMITS), KEY);
    process.env.DETECTIVE_TEST_SECRET = "short";
    const { r, out } = res();
    await handleDetectiveApi(req("/api/detective/health", "GET", { "x-dtv-force-offline": "short" }), r);
    expect(JSON.parse(out.body).llm).toBe(true);
  });
});
