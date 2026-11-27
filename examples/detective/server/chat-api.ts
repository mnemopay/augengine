/**
 * Server side of the detective's dialogue. Runs in Node only (Vite dev
 * middleware, `vite preview`, or a Vercel function) — the OpenRouter key never
 * reaches the browser.
 *
 *   GET  /api/detective/health → { llm: boolean, model, spentUsd }
 *   POST /api/detective/chat   → text/plain stream of the suspect's line
 *
 * The request may only carry a gist the rule engine can produce
 * (`checkPayload`), so this endpoint cannot be used as a general LLM proxy.
 */
import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import type { IncomingMessage, ServerResponse } from "node:http";
import { timingSafeEqual } from "node:crypto";
import { Budget, DEFAULT_MODEL, buildMessages, checkPayload, streamChat } from "@mnemopay/augengine-detective";
import { KeyGuard, openRouterStatusFetcher } from "./key-guard.js";

export function loadApiKey(): string | null {
  const env = process.env.OPENROUTER_API_KEY?.trim();
  if (env) return env;
  if (process.env.DETECTIVE_KEY_FILE === "0") return null;
  try {
    const k = readFileSync(join(homedir(), "openrouter-key.txt"), "utf8").trim();
    return k.length > 10 ? k : null;
  } catch {
    return null;
  }
}

const MODEL = process.env.DETECTIVE_MODEL || DEFAULT_MODEL;
export const budget = new Budget({
  turnsPerSession: 80,
  tokensPerSession: 80_000,
  costPerSession: Number(process.env.DETECTIVE_SESSION_USD || 0.05),
  globalCost: Number(process.env.DETECTIVE_GLOBAL_USD || 1),
});

let guard: KeyGuard | null = null;
let guardKey: string | null = null;
/** One guard per key per instance (the key is read lazily; tests may swap it). */
export function keyGuard(key: string): KeyGuard {
  if (!guard || guardKey !== key) {
    guard = new KeyGuard(openRouterStatusFetcher(key));
    guardKey = key;
  }
  return guard;
}
/** Tests only: replace the guard (e.g. with a fake status fetcher). */
export function setKeyGuardForTests(g: KeyGuard | null, key: string | null = null): void {
  guard = g;
  guardKey = key;
}

/** A server-side switch to force the "resting" path, for live checks. Needs DETECTIVE_TEST_SECRET. */
function forcedOffline(req: IncomingMessage): boolean {
  const secret = process.env.DETECTIVE_TEST_SECRET ?? "";
  const given = String(req.headers["x-dtv-force-offline"] ?? "");
  if (secret.length < 16 || given.length !== secret.length) return false;
  return timingSafeEqual(Buffer.from(given), Buffer.from(secret));
}

const hits = new Map<string, number[]>();
function rateLimited(ip: string): boolean {
  const now = Date.now();
  const arr = (hits.get(ip) ?? []).filter((t) => now - t < 60_000);
  arr.push(now);
  hits.set(ip, arr);
  return arr.length > 30;
}

async function readBody(req: IncomingMessage & { body?: unknown }): Promise<unknown> {
  if (req.body !== undefined) return typeof req.body === "string" ? JSON.parse(req.body) : req.body;
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const c of req) {
    size += (c as Buffer).length;
    if (size > 16_000) throw new Error("body too large");
    chunks.push(c as Buffer);
  }
  return JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}");
}

function json(res: ServerResponse, status: number, body: unknown): void {
  res.writeHead(status, { "content-type": "application/json", "cache-control": "no-store" });
  res.end(JSON.stringify(body));
}

/** Returns true when the request was for this API. */
export async function handleDetectiveApi(req: IncomingMessage & { body?: unknown }, res: ServerResponse): Promise<boolean> {
  const url = (req.url ?? "").split("?")[0] ?? "";
  if (url === "/api/detective/health" && req.method === "GET") {
    const key = loadApiKey();
    let resting: string | null = key ? null : "no key";
    if (key && !budget.allow("health").ok) resting = "instance budget reached";
    if (key && !resting) {
      const d = await keyGuard(key).check();
      if (!d.open) resting = d.reason;
    }
    if (forcedOffline(req)) resting = "forced offline (test)";
    const g = key ? keyGuard(key) : null;
    json(res, 200, {
      llm: resting === null,
      resting,
      model: MODEL,
      spentUsd: Number(budget.globalCost.toFixed(5)),
      budget: g
        ? {
            instanceSpentTodayUsd: Number(g.instanceSpentToday.toFixed(5)),
            instanceDailyUsd: g.limits.instanceDailyUsd,
            keyDailyCapUsd: g.limits.dailyCapUsd,
            keyMonthlyCapUsd: g.limits.monthlyCapUsd,
            keyFloorUsd: g.limits.floorUsd,
            sessionUsd: budget.limits.costPerSession,
          }
        : null,
    });
    return true;
  }
  if (url !== "/api/detective/chat") return false;
  if (req.method !== "POST") {
    json(res, 405, { error: "POST only" });
    return true;
  }
  const ip = String(req.headers["x-forwarded-for"] ?? req.socket.remoteAddress ?? "?").split(",")[0]!.trim();
  if (rateLimited(ip)) {
    json(res, 429, { error: "slow down" });
    return true;
  }
  const session = String(req.headers["x-dtv-session"] ?? "");
  if (!/^[a-z0-9-]{8,64}$/i.test(session)) {
    json(res, 400, { error: "bad session" });
    return true;
  }
  const key = loadApiKey();
  if (!key) {
    json(res, 503, { error: "llm unavailable" });
    return true;
  }
  if (forcedOffline(req)) {
    json(res, 429, { error: "resting", reason: "forced offline (test)" });
    return true;
  }
  const gate = await keyGuard(key).check();
  if (!gate.open) {
    json(res, 429, { error: "resting", reason: gate.reason });
    return true;
  }
  const allowed = budget.allow(session);
  if (!allowed.ok) {
    json(res, 429, { error: allowed.reason });
    return true;
  }
  let body: unknown;
  try {
    body = await readBody(req);
  } catch {
    json(res, 400, { error: "bad body" });
    return true;
  }
  const checked = checkPayload(body);
  if (!checked.ok) {
    json(res, 400, { error: checked.error });
    return true;
  }
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), 15_000);
  res.on("close", () => ac.abort());
  try {
    let started = false;
    for await (const chunk of streamChat({ apiKey: key, model: MODEL, messages: buildMessages(checked.ctx), signal: ac.signal })) {
      if ("delta" in chunk) {
        if (!started) {
          res.writeHead(200, { "content-type": "text/plain; charset=utf-8", "cache-control": "no-store", "x-dtv-source": "llm" });
          started = true;
        }
        res.write(chunk.delta);
      } else {
        budget.record(session, chunk.usage);
        keyGuard(key).record(chunk.usage.cost);
      }
    }
    if (!started) res.writeHead(200, { "content-type": "text/plain; charset=utf-8", "x-dtv-source": "llm" });
    res.end();
  } catch (err) {
    if (!res.headersSent) json(res, 502, { error: (err as Error).message.slice(0, 120) });
    else res.end();
  } finally {
    clearTimeout(timer);
  }
  return true;
}
