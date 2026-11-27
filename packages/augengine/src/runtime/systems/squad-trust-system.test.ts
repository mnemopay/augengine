/**
 * Track 5 — SquadTrustSystem unit tests.
 *
 * Covers: tier thresholds, register idempotency, score clamp [0, 100],
 * audit trail cap, trust-changed events.
 */

import { describe, it, expect } from "vitest";
import { Scene } from "../../scene/scene.js";
import { Tag, type EntityId, type SceneSpec } from "../../scene/types.js";
import { GameTag, getGameComponent, trustTierForScore, type SquadTrustComponent } from "./components.js";
import { SquadTrustSystem } from "./squad-trust-system.js";
import { buildSystemContext } from "./system.js";

const NPC = "npc.maya" as EntityId;

function makeScene(): Scene {
  const spec: SceneSpec = {
    schemaVersion: 1, source: { kind: "authored", path: "t" },
    entities: [{
      id: NPC, name: "Maya",
      components: { [Tag.Transform]: { position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1], parent: null } },
    }],
  };
  return new Scene(spec);
}

describe("SquadTrustSystem", () => {
  it("tier thresholds map score → tier", () => {
    expect(trustTierForScore(0)).toBe("hostile");
    expect(trustTierForScore(19)).toBe("hostile");
    expect(trustTierForScore(20)).toBe("distrustful");
    expect(trustTierForScore(40)).toBe("neutral");
    expect(trustTierForScore(60)).toBe("cooperative");
    expect(trustTierForScore(80)).toBe("loyal");
    expect(trustTierForScore(100)).toBe("loyal");
  });

  it("register is idempotent", () => {
    const scene = makeScene();
    SquadTrustSystem.register(scene, NPC, "Maya");
    SquadTrustSystem.register(scene, NPC, "Maya 2"); // no-op
    const trust = getGameComponent<SquadTrustComponent>(scene.get(NPC)!, GameTag.SquadTrust)!;
    expect(trust.name).toBe("Maya");
  });

  it("recordCoveringFire bumps score + emits trust-changed", () => {
    const scene = makeScene();
    SquadTrustSystem.register(scene, NPC, "Maya");
    const ctx = buildSystemContext(0, null, 1);
    SquadTrustSystem.recordCoveringFire(scene, NPC, ctx);
    const trust = getGameComponent<SquadTrustComponent>(scene.get(NPC)!, GameTag.SquadTrust)!;
    expect(trust.score).toBe(55);
    expect(ctx.events.some((e) => e.type === "trust-changed")).toBe(true);
  });

  it("friendly fire clamps at 0", () => {
    const scene = makeScene();
    SquadTrustSystem.register(scene, NPC, "Maya", 10);
    const ctx = buildSystemContext(0, null, 1);
    SquadTrustSystem.recordFriendlyFire(scene, NPC, ctx); // -25, clamps to 0
    const trust = getGameComponent<SquadTrustComponent>(scene.get(NPC)!, GameTag.SquadTrust)!;
    expect(trust.score).toBe(0);
    expect(trust.tier).toBe("hostile");
  });

  it("data core collection clamps at 100", () => {
    const scene = makeScene();
    SquadTrustSystem.register(scene, NPC, "Maya", 95);
    const ctx = buildSystemContext(0, null, 1);
    SquadTrustSystem.recordDataCoreCollection(scene, NPC, ctx); // +15 → clamps to 100
    const trust = getGameComponent<SquadTrustComponent>(scene.get(NPC)!, GameTag.SquadTrust)!;
    expect(trust.score).toBe(100);
    expect(trust.tier).toBe("loyal");
  });

  it("audit trail cap drops oldest entries", () => {
    const scene = makeScene();
    SquadTrustSystem.register(scene, NPC, "Maya");
    const ctx = buildSystemContext(0, null, 1);
    for (let i = 0; i < 40; i++) {
      SquadTrustSystem.recordCoveringFire(scene, NPC, ctx);
    }
    const trust = getGameComponent<SquadTrustComponent>(scene.get(NPC)!, GameTag.SquadTrust)!;
    expect(trust.auditTrail.length).toBeLessThanOrEqual(32);
  });

  it("squadStates iterator yields registered entities only", () => {
    const scene = makeScene();
    SquadTrustSystem.register(scene, NPC, "Maya");
    const ids = [...SquadTrustSystem.squadStates(scene)].map((s) => s.id);
    expect(ids).toEqual([NPC]);
  });
});
