/**
 * Browser side of dialogue. Sends only the rule-engine context to our own
 * server (never a key, never free-form prompts), streams the reply, then runs
 * the guardrail. Falls back to the offline gist whenever anything is off.
 */
import { finalizeLine, type DialogueContext } from "@mnemopay/augengine-detective";

export type DialogueMode = "llm" | "offline";

let resting = false;

/** True once the server said the storyteller is resting; the rest of the session stays offline. */
export function isResting(): boolean {
  return resting;
}

/** Tests only. */
export function resetResting(): void {
  resting = false;
}

const NOTE_ID = "dtv-resting-note";
function showRestingNote(): void {
  if (typeof document === "undefined" || document.getElementById(NOTE_ID)) return;
  const el = document.createElement("div");
  el.id = NOTE_ID;
  el.setAttribute("role", "status");
  el.textContent = "The storyteller is resting. The suspects will answer from their notes for now.";
  el.style.cssText =
    "position:fixed;left:50%;bottom:16px;transform:translateX(-50%);z-index:50;padding:8px 14px;border-radius:6px;" +
    "background:rgba(20,16,12,.92);color:#e8dcc4;font:13px/1.4 Georgia,serif;border:1px solid #6b5a3e;max-width:90vw;text-align:center";
  document.body.appendChild(el);
}

function goResting(): void {
  resting = true;
  showRestingNote();
}

export async function detectMode(forceOffline: boolean): Promise<DialogueMode> {
  if (forceOffline) return "offline";
  try {
    const r = await fetch("/api/detective/health", { cache: "no-store" });
    if (!r.ok) return "offline";
    const j = (await r.json()) as { llm?: boolean; resting?: string | null };
    if (!j.llm && j.resting && j.resting !== "no key") goResting();
    return j.llm ? "llm" : "offline";
  } catch {
    return "offline";
  }
}

export interface SpokenLine {
  text: string;
  source: "llm" | "offline";
  rejected: string[];
}

/** Streams partial text to `onPartial`; resolves with the guarded final line. */
export async function speak(
  ctx: DialogueContext,
  mode: DialogueMode,
  sessionId: string,
  onPartial: (text: string) => void,
): Promise<SpokenLine> {
  if (mode === "offline" || resting) {
    const f = finalizeLine(null, ctx);
    return { text: f.text, source: "offline", rejected: [] };
  }
  let acc = "";
  try {
    const res = await fetch("/api/detective/chat", {
      method: "POST",
      headers: { "content-type": "application/json", "x-dtv-session": sessionId },
      body: JSON.stringify({
        suspect: ctx.utterance.suspect,
        gist: ctx.utterance.gist,
        topic: ctx.utterance.topic,
        playerText: ctx.playerText,
        notebook: ctx.notebook,
        memoryLine: ctx.memoryLine,
      }),
    });
    if (res.status === 429) {
      const j = (await res.json().catch(() => ({}))) as { error?: string };
      if (j.error === "resting") goResting();
    }
    if (!res.ok || !res.body) throw new Error(String(res.status));
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      acc += dec.decode(value, { stream: true });
      onPartial(acc);
    }
  } catch {
    const f = finalizeLine(null, ctx);
    return { text: f.text, source: "offline", rejected: ["network"] };
  }
  const f = finalizeLine(acc, ctx);
  return { text: f.text, source: f.source, rejected: f.validation && !f.validation.ok ? f.validation.reasons : [] };
}
