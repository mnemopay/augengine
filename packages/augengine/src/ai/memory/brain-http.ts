/**
 * Brain HTTP client — AugEngine personas reach an external knowledge brain.
 *
 * Where `createInProcessBrain()` gives each persona its OWN local memory
 * (player names, conversations, NPC state), `createBrainHttpClient()` lets
 * a persona query a shared, read-only knowledge brain over HTTP (a
 * `brain serve --http` endpoint), so it can cite real, curated facts.
 *
 * Pattern: compose with `createInProcessBrain()` via `compositeBrain()` so
 * NPC Maya remembers the *player* (in-process) AND can cite real things
 * (operator brain).
 *
 * Read-only — `remember()` is a no-op. AugEngine NPCs should not write to the
 * operator brain; that's reserved for `brain ingest` / `brain skills run
 * meeting-ingestion`.
 */

import type { MnemoPayClientLike } from "./mnemopay.js";

export interface BrainHttpClientOptions {
  /** Brain serve --http endpoint, e.g. https://brain.mnemopay.com or http://localhost:8787 */
  endpoint: string;
  /** Per-call abort timeout in ms. Default: 4000. */
  timeoutMs?: number;
  /** Optional callback for instrumentation. */
  onError?: (err: Error) => void;
}

interface BrainQueryHit {
  name: string;
  type: string;
  snippet: string;
  score: number;
  path: string;
}

interface BrainQueryResponse {
  hits?: BrainQueryHit[];
  query?: string;
  error?: string;
}

export function createBrainHttpClient(opts: BrainHttpClientOptions): MnemoPayClientLike {
  const endpoint = opts.endpoint.replace(/\/$/, "");
  const timeoutMs = opts.timeoutMs ?? 4000;

  return {
    async recall(input) {
      const k = Math.min(Math.max(input.limit ?? 5, 1), 10);
      const q = String(input.query || "").trim();
      if (!q || q.length < 2) return { memories: [] };
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), timeoutMs);
      try {
        const res = await fetch(`${endpoint}/query`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ q, k, site: input.namespace ?? "augengine" }),
          signal: ctrl.signal,
        });
        if (!res.ok) {
          const err = new Error(`brain http ${res.status}`);
          opts.onError?.(err);
          return { memories: [] };
        }
        const data = await res.json() as BrainQueryResponse;
        const hits = data.hits ?? [];
        return {
          memories: hits.map((h) => ({
            content: `[${h.type}] ${h.name}: ${h.snippet}`,
            text: h.snippet,
            score: h.score,
          })),
        };
      } catch (e) {
        opts.onError?.(e as Error);
        return { memories: [] };
      } finally {
        clearTimeout(timer);
      }
    },

    async remember(_input) {
      // Operator brain is write-protected for personas. Silently no-op.
      // To capture knowledge into the operator brain, use `brain ingest` or
      // `brain skills run meeting-ingestion` from the brain CLI.
      return { id: "noop" };
    },

    async summarize(input) {
      // The HTTP brain doesn't expose a summarize endpoint. Return a
      // best-effort concatenation under the budget.
      const budget = input.maxTokens ?? 1500;
      const joined = input.docs.slice(0, 6).map((line, i) => `${i + 1}. ${line}`).join("\n");
      const sliced = joined.length > budget * 4 ? joined.slice(0, budget * 4) : joined;
      return { summary: sliced };
    },
  };
}

/**
 * Composite brain — writes go to `primary` (typically `createInProcessBrain`),
 * recalls merge results from BOTH backends. The result is an NPC that knows
 * the player history (local) AND can cite real-world ops knowledge (operator
 * brain over HTTP).
 *
 * Conflict resolution: results from `primary` rank first (player-specific
 * memory always beats generic ops trivia), then fallback hits.
 */
export function compositeBrain(primary: MnemoPayClientLike, fallback: MnemoPayClientLike): MnemoPayClientLike {
  return {
    async recall(input) {
      const [p, f] = await Promise.all([
        primary.recall(input).catch(() => ({ memories: [] as any[] })),
        fallback.recall(input).catch(() => ({ memories: [] as any[] })),
      ]);
      const primMems = normalizeMemories(p);
      const fallMems = normalizeMemories(f);
      const seen = new Set<string>();
      const merged: Array<{ content?: string; text?: string; summary?: string; score?: number }> = [];
      for (const m of [...primMems, ...fallMems]) {
        const key = m.content ?? m.text ?? "";
        if (!key || seen.has(key)) continue;
        seen.add(key);
        merged.push(m);
      }
      const limit = input.limit ?? 5;
      return { memories: merged.slice(0, limit) };
    },

    async remember(input) {
      // Writes always go to primary (in-process / writable backend).
      return primary.remember(input);
    },

    async summarize(input) {
      if (primary.summarize) return primary.summarize(input);
      if (fallback.summarize) return fallback.summarize(input);
      return { summary: input.docs.slice(0, 5).join("\n") };
    },
  };
}

function normalizeMemories(resp: unknown): Array<{ content?: string; text?: string; summary?: string; score?: number }> {
  if (Array.isArray(resp)) {
    return resp.map((row) => typeof row === "string" ? { content: row } : (row as any));
  }
  const r = resp as { memories?: any[]; results?: any[] };
  return r.memories ?? r.results ?? [];
}
