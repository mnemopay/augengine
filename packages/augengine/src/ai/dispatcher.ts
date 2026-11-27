/**
 * ActionDispatcher — the unified-grammar entry point for ALL gameplay
 * mutation. Player input emits Actions. Behavior trees emit Actions.
 * AI agents emit Actions. The dispatcher is the only thing allowed to
 * apply effects to world state.
 *
 * Decision #5 of AUGENGINE-ARCHITECTURE.md. Pre-conditions and effects are
 * registered by gameplay code as predicates/effects keyed by ID; the
 * dispatcher resolves them at runtime.
 */

import type { Action, ActionResult } from "./types.js";
import type { Scene } from "../scene/scene.js";
import type { ObservationBus } from "./bus.js";

export type Predicate = (scene: Scene, action: Action) => boolean;
export type Effect = (scene: Scene, action: Action) => void;

export interface DispatcherOptions {
  scene: Scene;
  bus?: ObservationBus;
}

export class ActionDispatcher {
  private predicates = new Map<string, Predicate>();
  private effects = new Map<string, Effect>();
  private allowedVerbs = new Set<string>();
  private scene: Scene;
  private bus?: ObservationBus;

  constructor(opts: DispatcherOptions) {
    this.scene = opts.scene;
    if (opts.bus !== undefined) this.bus = opts.bus;
  }

  /** Register a verb as dispatchable. Required before any Action with that verb is accepted. */
  registerVerb(verb: string): void {
    this.allowedVerbs.add(verb);
  }

  registerPredicate(id: string, predicate: Predicate): void {
    this.predicates.set(id, predicate);
  }

  registerEffect(id: string, effect: Effect): void {
    this.effects.set(id, effect);
  }

  validate(action: Action): { ok: true } | { ok: false; reason: string } {
    if (!this.allowedVerbs.has(action.verb)) {
      return { ok: false, reason: `verb not registered: ${action.verb}` };
    }
    for (const pid of action.preconditions) {
      const p = this.predicates.get(pid);
      if (!p) return { ok: false, reason: `unknown predicate: ${pid}` };
      if (!p(this.scene, action)) {
        return { ok: false, reason: `predicate failed: ${pid}` };
      }
    }
    return { ok: true };
  }

  dispatch(action: Action): ActionResult {
    const v = this.validate(action);
    if (!v.ok) {
      return { ok: false, appliedEffects: [], reason: v.reason };
    }
    const applied: string[] = [];
    for (const eid of action.effects) {
      const e = this.effects.get(eid);
      if (!e) {
        return {
          ok: false,
          appliedEffects: applied,
          reason: `unknown effect: ${eid}`,
        };
      }
      e(this.scene, action);
      applied.push(eid);
    }
    if (this.bus) {
      this.bus.publish({
        channel: "Social",
        ts: new Date().toISOString(),
        payload: { kind: "action.dispatched", action, appliedEffects: applied },
        source: action.actor,
      });
    }
    return { ok: true, appliedEffects: applied };
  }
}
