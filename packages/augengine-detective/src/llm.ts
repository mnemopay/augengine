/**
 * Server-side LLM plumbing (never imported by browser code that holds a key).
 *
 * - `checkPayload` makes sure a dialogue request can only ever restate a gist
 *   the rule engine could have produced, so the endpoint cannot be used as a
 *   general-purpose LLM proxy.
 * - `streamChat` calls OpenRouter's chat completions with streaming and yields
 *   text deltas plus the final usage/cost.
 * - `Budget` enforces per-session and global caps.
 */

import { EVIDENCE, SUSPECTS, type EvidenceId, type SuspectId } from "./case.js";
import type { ChatMessage, DialogueContext } from "./dialogue.js";
import { isEvidenceId, isSuspectId, type Utterance } from "./rules.js";

export const DEFAULT_MODEL = "openai/gpt-4o-mini";
export const MAX_REPLY_TOKENS = 140;

/** Every gist the rules can produce for a suspect. */
export function allowedGists(suspect: SuspectId): Set<string> {
  const s = SUSPECTS[suspect];
  const out = new Set<string>();
  for (const a of Object.values(s.answers)) if (a) out.add(a.line);
  for (const l of s.lies) {
    out.add(l.lie);
    out.add(l.truth);
    out.add(l.crackLine);
  }
  // Multiple lies can crack on one presentation; their lines are joined.
  const cracks = s.lies.map((l) => l.crackLine);
  for (let i = 0; i < cracks.length; i++) {
    for (let j = 0; j < cracks.length; j++) if (i !== j) out.add(`${cracks[i]} ${cracks[j]}`);
  }
  if (cracks.length >= 3) out.add(cracks.join(" "));
  out.add(s.shrug);
  out.add("I'm not sure what you're asking, detective. Ask me about tonight.");
  return out;
}

const MEMORY_LINE = [
  /^You accused me last time, detective\. In front of everyone\. I haven't forgotten\.$/,
  /^You named me last time\. And yet here I am, still talking to you\.$/,
  /^Back again\. Last time you showed me the [a-z' .]{3,40}\. I told you what I know\.$/,
  /^We've done this \d{1,3} times now, detective\. My story hasn't changed\.$/,
  /^You again\. I remember your questions from last time\.$/,
];

export interface DialoguePayload {
  suspect: SuspectId;
  gist: string;
  topic: string;
  playerText: string;
  notebook: EvidenceId[];
  memoryLine: string | null;
}

/** Validate an untrusted request body; returns a DialogueContext or an error. */
export function checkPayload(body: unknown): { ok: true; ctx: DialogueContext } | { ok: false; error: string } {
  if (!body || typeof body !== "object") return { ok: false, error: "bad body" };
  const b = body as Record<string, unknown>;
  if (!isSuspectId(b.suspect)) return { ok: false, error: "bad suspect" };
  if (typeof b.gist !== "string" || !allowedGists(b.suspect).has(b.gist)) return { ok: false, error: "gist not produced by the rules" };
  if (typeof b.playerText !== "string" || b.playerText.length > 400) return { ok: false, error: "bad playerText" };
  if (typeof b.topic !== "string" || b.topic.length > 60) return { ok: false, error: "bad topic" };
  const notebook = Array.isArray(b.notebook) ? b.notebook.filter(isEvidenceId) : [];
  if (notebook.length > Object.keys(EVIDENCE).length) return { ok: false, error: "bad notebook" };
  let memoryLine: string | null = null;
  if (b.memoryLine != null) {
    if (typeof b.memoryLine !== "string" || !MEMORY_LINE.some((re) => re.test(b.memoryLine as string))) {
      return { ok: false, error: "bad memoryLine" };
    }
    memoryLine = b.memoryLine;
  }
  const utterance: Utterance = {
    suspect: b.suspect,
    topic: b.topic as Utterance["topic"],
    gist: b.gist,
    revealed: [],
    cracked: [],
    lying: false,
  };
  return { ok: true, ctx: { utterance, playerText: b.playerText, notebook, memoryLine } };
}

export interface Usage {
  promptTokens: number;
  completionTokens: number;
  /** USD as reported by OpenRouter, else estimated. */
  cost: number;
}

/** Rough price table (USD per token) used only when the provider omits cost. */
const PRICE: Record<string, [number, number]> = {
  "openai/gpt-4o-mini": [0.15e-6, 0.6e-6],
};

export function estimateCost(model: string, u: { promptTokens: number; completionTokens: number }): number {
  const [pi, po] = PRICE[model] ?? [1e-6, 2e-6];
  return u.promptTokens * pi + u.completionTokens * po;
}

export class Budget {
  private sessions = new Map<string, { turns: number; tokens: number; cost: number }>();
  globalCost = 0;
  constructor(
    readonly limits = { turnsPerSession: 60, tokensPerSession: 60_000, costPerSession: 0.05, globalCost: 2 },
  ) {}
  allow(sessionId: string): { ok: true } | { ok: false; reason: string } {
    if (this.globalCost >= this.limits.globalCost) return { ok: false, reason: "global budget reached" };
    const s = this.sessions.get(sessionId);
    if (!s) return { ok: true };
    if (s.turns >= this.limits.turnsPerSession) return { ok: false, reason: "session turn cap" };
    if (s.tokens >= this.limits.tokensPerSession) return { ok: false, reason: "session token cap" };
    if (s.cost >= this.limits.costPerSession) return { ok: false, reason: "session cost cap" };
    return { ok: true };
  }
  record(sessionId: string, u: Usage): void {
    const s = this.sessions.get(sessionId) ?? { turns: 0, tokens: 0, cost: 0 };
    s.turns += 1;
    s.tokens += u.promptTokens + u.completionTokens;
    s.cost += u.cost;
    this.sessions.set(sessionId, s);
    this.globalCost += u.cost;
  }
  session(sessionId: string) {
    return this.sessions.get(sessionId) ?? { turns: 0, tokens: 0, cost: 0 };
  }
}

export interface StreamChatOptions {
  apiKey: string;
  messages: ChatMessage[];
  model?: string;
  maxTokens?: number;
  signal?: AbortSignal;
  fetchImpl?: typeof fetch;
  referer?: string;
}

export type ChatChunk = { delta: string } | { usage: Usage };

/** Stream a completion from OpenRouter. Yields text deltas, then one usage chunk. */
export async function* streamChat(opts: StreamChatOptions): AsyncGenerator<ChatChunk> {
  const model = opts.model ?? DEFAULT_MODEL;
  const f = opts.fetchImpl ?? fetch;
  const res = await f("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: {
      authorization: `Bearer ${opts.apiKey}`,
      "content-type": "application/json",
      "http-referer": opts.referer ?? "https://augengine.com/detective",
      "x-title": "AugEngine Detective",
    },
    body: JSON.stringify({
      model,
      messages: opts.messages,
      max_tokens: opts.maxTokens ?? MAX_REPLY_TOKENS,
      temperature: 0.7,
      stream: true,
      usage: { include: true },
    }),
    ...(opts.signal ? { signal: opts.signal } : {}),
  });
  if (!res.ok || !res.body) {
    const detail = await res.text().catch(() => "");
    throw new Error(`openrouter ${res.status}: ${detail.slice(0, 200)}`);
  }
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  let usage: Usage | null = null;
  let chars = 0;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let nl: number;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (!line.startsWith("data:")) continue;
      const data = line.slice(5).trim();
      if (data === "[DONE]") continue;
      try {
        const j = JSON.parse(data) as {
          choices?: { delta?: { content?: string } }[];
          usage?: { prompt_tokens?: number; completion_tokens?: number; cost?: number };
        };
        const d = j.choices?.[0]?.delta?.content;
        if (d) {
          chars += d.length;
          yield { delta: d };
        }
        if (j.usage) {
          const promptTokens = j.usage.prompt_tokens ?? 0;
          const completionTokens = j.usage.completion_tokens ?? 0;
          usage = {
            promptTokens,
            completionTokens,
            cost: typeof j.usage.cost === "number" ? j.usage.cost : estimateCost(model, { promptTokens, completionTokens }),
          };
        }
      } catch {
        /* keep-alive comments and partial frames */
      }
    }
  }
  yield {
    usage:
      usage ?? {
        promptTokens: 700,
        completionTokens: Math.ceil(chars / 4),
        cost: estimateCost(model, { promptTokens: 700, completionTokens: Math.ceil(chars / 4) }),
      },
  };
}

