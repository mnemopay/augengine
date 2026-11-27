/**
 * Track 5 — Squad trust system.
 *
 * The previous `SquadTrustManager` (a shell-side class) lived outside
 * the scene graph and persisted its ledger to localStorage. The system
 * now stores per-NPC trust state as a `SquadTrustComponent` so the
 * squad HUD reads the data straight off the entity and the persistence
 * layer round-trips it for free with the rest of the scene (via
 * SaveSystem) — no shell-owned localStorage required.
 *
 * The system itself is event-driven — the smoke-game shell calls the
 * static recorders (`recordCoveringFire`, `recordFriendlyFire`, etc) in
 * response to weapon hits / objective completions. `update()` is a
 * no-op today, reserved for trust decay across long sessions.
 */

import type { Scene } from "../../scene/scene.js";
import type { EntityId } from "../../scene/types.js";
import {
  GameTag,
  getGameComponent,
  setGameComponent,
  trustTierForScore,
  type SquadTrustComponent,
} from "./components.js";
import type { GameSystem, SystemContext } from "./system.js";

const AUDIT_TRAIL_CAP = 32;

export class SquadTrustSystem implements GameSystem {
  readonly id = "squad-trust";

  /** Initialise (or replace) a trust component on an NPC entity. */
  static register(scene: Scene, npc: EntityId, name: string, initialScore = 50): void {
    const entity = scene.get(npc);
    if (!entity) return;
    const existing = getGameComponent<SquadTrustComponent>(entity, GameTag.SquadTrust);
    if (existing) return; // idempotent
    setGameComponent<SquadTrustComponent>(entity, GameTag.SquadTrust, {
      name,
      score: initialScore,
      tier: trustTierForScore(initialScore),
      auditTrail: [`[Genesis] Registered ${name} with initial trust ${initialScore}`],
    });
  }

  static recordCoveringFire(scene: Scene, npc: EntityId, ctx: SystemContext): void {
    SquadTrustSystem.adjust(scene, npc, +5, "Player provided effective covering fire", ctx);
  }

  static recordAmmoSharing(scene: Scene, npc: EntityId, ctx: SystemContext): void {
    SquadTrustSystem.adjust(scene, npc, +10, "Player shared rare thermal ammunition", ctx);
  }

  static recordDataCoreCollection(scene: Scene, npc: EntityId, ctx: SystemContext): void {
    SquadTrustSystem.adjust(scene, npc, +15, "Player secured data containment core", ctx);
  }

  static recordFriendlyFire(scene: Scene, npc: EntityId, ctx: SystemContext): void {
    SquadTrustSystem.adjust(scene, npc, -25, "WARNING: Friendly fire incident recorded", ctx);
  }

  /** Walk the scene yielding `(entityId, trustState)` pairs for HUDs. */
  static *squadStates(scene: Scene): Iterable<{ id: EntityId; trust: SquadTrustComponent }> {
    for (const entity of scene) {
      const trust = getGameComponent<SquadTrustComponent>(entity, GameTag.SquadTrust);
      if (trust) yield { id: entity.id, trust };
    }
  }

  update(_scene: Scene, _dt: number, _ctx: SystemContext): void {
    /* trust decay reserved for v2 */
  }

  private static adjust(
    scene: Scene,
    npcId: EntityId,
    delta: number,
    description: string,
    ctx: SystemContext,
  ): void {
    const entity = scene.get(npcId);
    if (!entity) return;
    const state = getGameComponent<SquadTrustComponent>(entity, GameTag.SquadTrust);
    if (!state) return;
    const oldScore = state.score;
    const newScore = Math.max(0, Math.min(100, oldScore + delta));
    const oldTier = state.tier;
    const newTier = trustTierForScore(newScore);
    state.score = newScore;
    state.tier = newTier;
    const timestamp = new Date(ctx.nowMs).toISOString().slice(11, 19);
    let logMsg = `[${timestamp}] ${description} (Score: ${oldScore} -> ${newScore})`;
    if (oldTier !== newTier) logMsg += ` [TIER → ${newTier.toUpperCase()}]`;
    state.auditTrail.push(logMsg);
    while (state.auditTrail.length > AUDIT_TRAIL_CAP) state.auditTrail.shift();
    ctx.events.push({
      type: "trust-changed",
      npc: npcId,
      delta,
      score: newScore,
      tier: newTier,
      reason: description,
    });
  }
}
