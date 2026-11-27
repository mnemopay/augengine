import { BUTTON } from "./input.js";
import type { Fight } from "./sim.js";

/**
 * Buttons for the side with nobody on the keys. Pure function of the fight,
 * so the same state always produces the same press.
 */
export function cpuBits(fight: Fight, index: 0 | 1): number {
  if (fight.roundOver) return 0;
  const self = fight.fighters[index];
  const other = fight.fighters[index === 0 ? 1 : 0];
  const forward = self.facing === 1 ? BUTTON.RIGHT : BUTTON.LEFT;
  const back = self.facing === 1 ? BUTTON.LEFT : BUTTON.RIGHT;
  if (self.phase === "thrown" && self.tech > 0) return BUTTON.LP | BUTTON.LK;
  if (self.phase === "blockstun") return back;
  if (other.phase === "attack" && other.move && self.y === 0) {
    const gap = Math.abs(other.x - self.x);
    if (gap < 120 && other.moveFrame >= other.move.startup - 1) {
      if (other.move.guard === "throw") return 0;
      if (other.move.guard === "low") return back | BUTTON.DOWN;
      return back;
    }
  }
  if (self.phase !== "idle" || self.y > 0) return 0;
  const gap = Math.abs(other.x - self.x);
  if (gap > 90) return forward;
  if (other.phase === "hitstun") return BUTTON.LP;
  if (fight.frame % 48 === 10) return BUTTON.MP;
  return 0;
}
