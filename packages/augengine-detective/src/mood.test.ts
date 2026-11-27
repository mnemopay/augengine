import { describe, expect, it } from "vitest";
import { act, newGame, type Action, type GameState } from "./rules.js";
import { OBJECTS } from "./case.js";
import { MOODS, moodFor, portraitPath, type Mood } from "./mood.js";

function run(s: GameState, actions: Action[]): GameState {
  for (const a of actions) {
    const r = act(s, a);
    if (!r.ok) throw new Error(r.error);
    s = r.state;
  }
  return s;
}

function examineAll(s: GameState): GameState {
  for (const room of ["dome", "library", "kitchen", "conservatory"] as const) {
    s = run(s, [{ type: "move", room }]);
    for (const o of OBJECTS.filter((x) => x.room === room)) s = run(s, [{ type: "examine", objectId: o.id }]);
  }
  return run(s, [{ type: "move", room: "parlour" }]);
}

function moodOf(s: GameState, a: Extract<Action, { type: "ask" | "present" }>): { mood: Mood; state: GameState } {
  const r = act(s, a);
  if (!r.ok || !r.utterance) throw new Error(r.ok ? "no utterance" : r.error);
  const mood = moodFor({
    utterance: r.utterance,
    presented: a.type === "present" ? a.evidence : undefined,
    playerText: a.type === "ask" ? a.text : undefined,
  });
  return { mood, state: r.state };
}

describe("portrait mood (decided by the rules)", () => {
  it("is calm for an honest answer", () => {
    expect(moodOf(newGame(), { type: "ask", suspect: "mara", text: "Who are you?" }).mood).toBe("calm");
  });

  it("is evasive while a suspect tells a scripted lie", () => {
    const { mood } = moodOf(examineAll(newGame()), { type: "ask", suspect: "desmond", text: "Where were you this evening?" });
    expect(mood).toBe("evasive");
  });

  it("is angry when shown evidence that points at them", () => {
    const { mood } = moodOf(examineAll(newGame()), { type: "present", suspect: "desmond", evidence: "C_WILL" });
    expect(mood).toBe("angry");
  });

  it("is angry when openly accused in a question", () => {
    expect(moodOf(newGame(), { type: "ask", suspect: "ines", text: "You poisoned him, didn't you?" }).mood).toBe("angry");
  });

  it("is cracked on the turn a lie breaks, overriding anger", () => {
    // Desmond's IOU lie cracks when shown the IOUs, which also point at him.
    const { mood } = moodOf(examineAll(newGame()), { type: "present", suspect: "desmond", evidence: "C_IOU" });
    expect(mood).toBe("cracked");
  });

  it("is deterministic for the same input", () => {
    const s = examineAll(newGame());
    const a = { type: "ask" as const, suspect: "desmond" as const, text: "Where were you this evening?" };
    expect(moodOf(s, a).mood).toBe(moodOf(s, a).mood);
  });

  it("maps every suspect and mood to a portrait path; the victim only has calm", () => {
    for (const sid of ["mara", "desmond", "ines", "teodor"] as const)
      for (const m of MOODS) expect(portraitPath(sid, m)).toBe(`/portraits/${sid}/${m}.webp`);
    expect(portraitPath("vance", "angry")).toBe("/portraits/vance/calm.webp");
  });
});
