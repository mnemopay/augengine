import { describe, it, expect } from "vitest";
import {
  ensurePersonaAxes,
  renderPersonaSystem,
} from "./persona.js";
import type { PersonaComponent } from "../scene/types.js";

function basePersona(overrides: Partial<PersonaComponent> = {}): PersonaComponent {
  return {
    traits: [],
    voice: null,
    memoryId: null,
    goals: [],
    mood: 0,
    ...overrides,
  };
}

describe("renderPersonaSystem", () => {
  it("emits role + faction when present", () => {
    const s = renderPersonaSystem(basePersona({ role: "boss", faction: "infected" }));
    expect(s).toContain("Role: boss (infected)");
  });

  it("emits role without faction parentheses when faction is absent", () => {
    const s = renderPersonaSystem(basePersona({ role: "ally" }));
    expect(s).toContain("Role: ally.");
    expect(s).not.toContain("(undefined)");
  });

  it("emits biasTags when present", () => {
    const s = renderPersonaSystem(
      basePersona({ biasTags: ["hates-mercenaries", "sister-of-yelena"] }),
    );
    expect(s).toContain("Biases: hates-mercenaries, sister-of-yelena.");
  });

  it("emits disposition only for axes that deviate from neutral 0.5", () => {
    const s = renderPersonaSystem(
      basePersona({
        axes: { aggression: 0.9, fear: 0.1, loyalty: 0.5, curiosity: 0.5 },
      }),
    );
    expect(s).toContain("aggression=high");
    expect(s).toContain("fear=low");
    expect(s).not.toContain("loyalty=");
    expect(s).not.toContain("curiosity=");
  });

  it("skips the disposition line entirely when every axis is neutral", () => {
    const s = renderPersonaSystem(
      basePersona({
        axes: { aggression: 0.5, fear: 0.5, loyalty: 0.5, curiosity: 0.5 },
      }),
    );
    expect(s).not.toContain("Disposition");
  });

  it("traits + mood + goals still render (backward compat)", () => {
    const s = renderPersonaSystem(
      basePersona({
        traits: ["calm", "observant"],
        mood: -0.8,
        goals: [{ id: "g1", description: "guard the gate", priority: 5, done: "gate.closed" }],
      }),
    );
    expect(s).toContain("Traits: calm, observant.");
    expect(s).toContain("Current mood: angry");
    expect(s).toContain("guard the gate");
  });
});

describe("ensurePersonaAxes", () => {
  it("is a no-op when axes already present", () => {
    const p = basePersona({ axes: { aggression: 0.9, fear: 0.1, loyalty: 0.5, curiosity: 0.5 } });
    const out = ensurePersonaAxes(p);
    expect(out).toBe(p); // same reference — no clone
  });

  it("fills neutral 0.5 axes when missing", () => {
    const p = basePersona();
    const out = ensurePersonaAxes(p);
    expect(out.axes).toEqual({
      aggression: 0.5,
      fear: 0.5,
      loyalty: 0.5,
      curiosity: 0.5,
    });
    // Doesn't mutate the original.
    expect(p.axes).toBeUndefined();
  });
});
