/**
 * Track 5 — Health + Armor system.
 *
 * Walks entities with `HealthComponent` once per frame. The system does
 * not auto-damage anything; other systems (RecruitSystem, WeaponSystem)
 * call `applyDamage` to deduct HP and the system emits the death event
 * once HP crosses zero.
 *
 * Armor handling: if an entity also has `ArmorComponent`, incoming
 * damage is first absorbed at `armor.absorptionFraction` until armor is
 * exhausted, then bleeds through to HP. Matches the smoke-game's prior
 * 0.5 fraction.
 */

import type { Scene } from "../../scene/scene.js";
import type { EntityId } from "../../scene/types.js";
import {
  GameTag,
  getGameComponent,
  type ArmorComponent,
  type HealthComponent,
} from "./components.js";
import type { DamageSource, GameSystem, SystemContext } from "./system.js";

export interface ApplyDamageOptions {
  source: DamageSource;
  /** When set, suppress the death event (caller is staging it themselves). */
  skipDeathEvent?: boolean;
}

export class HealthSystem implements GameSystem {
  readonly id = "health";

  /**
   * Deduct `amount` damage from `target`. Pushes a "damage" and (on
   * death) "death" event into `ctx.events`. Returns the post-hit HP.
   * Defensive against missing components — returns null instead of
   * throwing so callers can decide whether to ignore or surface.
   */
  static applyDamage(
    scene: Scene,
    target: EntityId,
    amount: number,
    ctx: SystemContext,
    opts: ApplyDamageOptions,
  ): number | null {
    const entity = scene.get(target);
    if (!entity) return null;
    const health = getGameComponent<HealthComponent>(entity, GameTag.Health);
    if (!health || health.dead) return health?.hp ?? null;
    const armor = getGameComponent<ArmorComponent>(entity, GameTag.Armor);
    let remaining = amount;
    if (armor && armor.value > 0 && armor.absorptionFraction > 0) {
      const desiredAbsorb = remaining * armor.absorptionFraction;
      const absorbed = Math.min(armor.value, desiredAbsorb);
      armor.value = Math.max(0, armor.value - absorbed);
      remaining = Math.max(0, remaining - absorbed);
    }
    health.hp = Math.max(0, health.hp - remaining);
    health.lastDamageAt = ctx.nowMs;
    ctx.events.push({
      type: "damage",
      target,
      amount,
      remainingHp: health.hp,
      source: opts.source,
    });
    if (health.hp <= 0) {
      health.dead = true;
      if (!opts.skipDeathEvent) {
        ctx.events.push({ type: "death", target, source: opts.source });
      }
    }
    return health.hp;
  }

  /** Replenish HP (and optionally armor). Caps at max. */
  static heal(
    scene: Scene,
    target: EntityId,
    hpDelta: number,
    armorDelta = 0,
  ): void {
    const entity = scene.get(target);
    if (!entity) return;
    const health = getGameComponent<HealthComponent>(entity, GameTag.Health);
    if (health && !health.dead) {
      health.hp = Math.min(health.maxHp, health.hp + hpDelta);
    }
    if (armorDelta > 0) {
      const armor = getGameComponent<ArmorComponent>(entity, GameTag.Armor);
      if (armor) armor.value = Math.min(armor.max, armor.value + armorDelta);
    }
  }

  /**
   * Per-frame sweep. Currently a no-op — the system reacts to explicit
   * `applyDamage` / `heal` calls and only walks entities to keep the
   * surface symmetric with other systems for future regen / DOT effects.
   */
  update(_scene: Scene, _dt: number, _ctx: SystemContext): void {
    /* reserved for regen / DOT — see TODO in components.ts */
  }
}
