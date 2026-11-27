/**
 * Track 5 — Weapon system.
 *
 * Owns magazine / reserve / cooldown / reload state for every entity
 * with a `WeaponComponent`. The smoke-game shell calls:
 *
 *   - `WeaponSystem.fire(scene, weaponId, ctx)`   — left click
 *   - `WeaponSystem.reload(scene, weaponId, ctx)` — R key
 *
 * and the `update(scene, dt, ctx)` per frame decrements the timers and
 * completes the reload draw when the lockout expires. The system emits
 * `shot-fired` / `shot-hit` / `reload-start` / `reload-complete` events
 * that the shell forwards to the mixer + HUD.
 *
 * The system is render-agnostic — it does NOT raycast for hits. Callers
 * pass an optional `hit` payload to `fire()` (the smoke-game's
 * crosshair raycast supplies it). If a hit is provided AND the target
 * has a HealthComponent, the system also deducts damage via
 * HealthSystem.applyDamage.
 */

import type { Scene } from "../../scene/scene.js";
import { Tag, type EntityId, type Vec3 } from "../../scene/types.js";
import {
  GameTag,
  getGameComponent,
  type WeaponComponent,
} from "./components.js";
import { HealthSystem } from "./health-system.js";
import type { GameSystem, SystemContext } from "./system.js";

export interface FireOptions {
  /** Optional hit target (resolved by caller from raycast). */
  hit?: { target: EntityId; isFriendly?: boolean } | null;
  /** Override damage (e.g. headshot multiplier). Defaults to weapon's. */
  damageOverride?: number;
}

export type FireResult =
  | { fired: false; reason: "cooldown" | "reloading" | "empty" | "no-weapon" }
  | {
      fired: true;
      hit: boolean;
      killed: boolean;
      remainingAmmo: number;
    };

export class WeaponSystem implements GameSystem {
  readonly id = "weapon";

  /**
   * Try to fire `weaponId`. Returns a typed result. Pushes the events
   * (and, when applicable, hands off to HealthSystem.applyDamage which
   * pushes its own damage/death events).
   */
  static fire(
    scene: Scene,
    weaponId: EntityId,
    ctx: SystemContext,
    opts: FireOptions = {},
  ): FireResult {
    const entity = scene.get(weaponId);
    if (!entity) return { fired: false, reason: "no-weapon" };
    const weapon = getGameComponent<WeaponComponent>(entity, GameTag.Weapon);
    if (!weapon) return { fired: false, reason: "no-weapon" };
    if (weapon.reloadTimer > 0) return { fired: false, reason: "reloading" };
    if (weapon.cooldownTimer > 0) return { fired: false, reason: "cooldown" };
    if (weapon.ammo <= 0) return { fired: false, reason: "empty" };

    weapon.ammo--;
    weapon.cooldownTimer = weapon.fireCooldown;
    weapon.recoilKick = 0.022;
    const tx = entity.components[Tag.Transform];
    const pos: Vec3 = tx ? (tx.position as Vec3) : [0, 0, 0];
    ctx.events.push({ type: "shot-fired", weapon: weaponId, position: pos });

    if (!opts.hit) {
      return { fired: true, hit: false, killed: false, remainingAmmo: weapon.ammo };
    }
    const damage = opts.damageOverride ?? weapon.shotDamage;
    const remaining = HealthSystem.applyDamage(scene, opts.hit.target, damage, ctx, {
      source: opts.hit.isFriendly ? "friendly-fire" : "player",
    });
    const killed = remaining !== null && remaining <= 0;
    ctx.events.push({
      type: "shot-hit",
      weapon: weaponId,
      target: opts.hit.target,
      damage,
      killed,
    });
    return { fired: true, hit: true, killed, remainingAmmo: weapon.ammo };
  }

  /** Begin a reload. Idempotent if already reloading or mag full. */
  static reload(scene: Scene, weaponId: EntityId, ctx: SystemContext): boolean {
    const entity = scene.get(weaponId);
    if (!entity) return false;
    const weapon = getGameComponent<WeaponComponent>(entity, GameTag.Weapon);
    if (!weapon) return false;
    if (weapon.reloadTimer > 0) return false;
    if (weapon.reserve <= 0 || weapon.ammo >= weapon.magSize) return false;
    weapon.reloadTimer = weapon.reloadLockout;
    ctx.events.push({ type: "reload-start", weapon: weaponId, lockoutSec: weapon.reloadLockout });
    return true;
  }

  /**
   * Per-frame timer integration. Completes the reload draw when the
   * lockout expires. Decays recoil kick so the shell can read
   * `weapon.recoilKick` once per frame and apply it to camera pitch.
   */
  update(scene: Scene, dt: number, ctx: SystemContext): void {
    for (const entity of scene) {
      const weapon = getGameComponent<WeaponComponent>(entity, GameTag.Weapon);
      if (!weapon) continue;
      if (weapon.cooldownTimer > 0) {
        weapon.cooldownTimer = Math.max(0, weapon.cooldownTimer - dt);
      }
      if (weapon.reloadTimer > 0) {
        weapon.reloadTimer -= dt;
        if (weapon.reloadTimer <= 0) {
          const needed = weapon.magSize - weapon.ammo;
          const draw = Math.min(needed, weapon.reserve);
          weapon.ammo += draw;
          weapon.reserve -= draw;
          weapon.reloadTimer = 0;
          ctx.events.push({
            type: "reload-complete",
            weapon: entity.id,
            loaded: draw,
          });
        }
      }
      if (weapon.recoilKick > 0) {
        weapon.recoilKick = Math.max(0, weapon.recoilKick - dt * 0.08);
      }
    }
  }
}
