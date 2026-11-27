/**
 * Track 5 — Maya memory consequences system.
 *
 * Listens to the per-frame event stream and writes narrative
 * consequence lines onto Maya's `MayaMemoryComponent`. The smoke-game
 * shell forwards the same lines to the brain memory recorder so the
 * dialogue panel surfaces them across reloads — but the *state itself*
 * lives on the entity, not in a closure.
 *
 * Consequences fired in the smoke-game's prior code:
 *   - Player friendly-fires Maya twice → `betrayed = true`.
 *   - Player secures an objective Maya could see → witnessed bump.
 *   - Player kills a recruit Maya could see → witnessed bump.
 *
 * The system is intentionally allocation-light — it only walks the
 * event log once per frame, no scene scan.
 */

import type { Scene } from "../../scene/scene.js";
import type { EntityId } from "../../scene/types.js";
import {
  GameTag,
  getGameComponent,
  setGameComponent,
  type MayaMemoryComponent,
} from "./components.js";
import type { GameSystem, SystemContext } from "./system.js";

export interface MayaMemorySystemOptions {
  mayaEntityId: EntityId;
  /** Optional sink — receives every consequence line for brain wiring. */
  onConsequence?: (line: string) => void;
}

/** Initialise (or replace) the memory component on Maya's entity. */
export function ensureMayaMemoryComponent(scene: Scene, mayaId: EntityId): void {
  const entity = scene.get(mayaId);
  if (!entity) return;
  if (getGameComponent<MayaMemoryComponent>(entity, GameTag.MayaMemory)) return;
  setGameComponent<MayaMemoryComponent>(entity, GameTag.MayaMemory, {
    friendlyFireCount: 0,
    objectivesWitnessed: 0,
    killsWitnessed: 0,
    betrayed: false,
    lastConsequence: null,
    lastConsequenceAt: -1,
  });
}

export class MayaMemorySystem implements GameSystem {
  readonly id = "maya-memory";

  constructor(private readonly opts: MayaMemorySystemOptions) {}

  update(scene: Scene, _dt: number, ctx: SystemContext): void {
    const entity = scene.get(this.opts.mayaEntityId);
    if (!entity) return;
    const mem = getGameComponent<MayaMemoryComponent>(entity, GameTag.MayaMemory);
    if (!mem) return;
    for (const event of ctx.events) {
      if (event.type === "damage" && event.target === this.opts.mayaEntityId && event.source === "friendly-fire") {
        mem.friendlyFireCount++;
        const line = mem.friendlyFireCount >= 2
          ? "Maya: 'That's twice now. I won't forget.'"
          : "Maya: 'Watch your fire, operator.'";
        this.write(mem, line, ctx);
        if (mem.friendlyFireCount >= 2 && !mem.betrayed) {
          mem.betrayed = true;
        }
      } else if (event.type === "objective-completed") {
        mem.objectivesWitnessed++;
        if (mem.objectivesWitnessed === 1) {
          this.write(mem, "Maya: 'Good. That's one less for the Tribunal.'", ctx);
        }
      } else if (event.type === "shot-hit" && event.killed) {
        mem.killsWitnessed++;
        if (mem.killsWitnessed === 1) {
          this.write(mem, "Maya: 'First kill. Don't let it stack up.'", ctx);
        }
      }
    }
  }

  private write(mem: MayaMemoryComponent, line: string, ctx: SystemContext): void {
    mem.lastConsequence = line;
    mem.lastConsequenceAt = ctx.nowMs;
    ctx.events.push({ type: "maya-consequence", line, betrayed: mem.betrayed });
    if (this.opts.onConsequence) {
      try { this.opts.onConsequence(line); } catch { /* sink never blocks gameplay */ }
    }
  }
}
