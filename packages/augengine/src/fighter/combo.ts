import { BUTTON } from "./input.js";
import { replay, type Fight } from "./sim.js";

/**
 * Mara's cross (+6) links her 6-frame jab, buffered 4 frames early,
 * then a quarter-circle forward + LP cancels into arc and knocks Holt down.
 * The input list is found by replaying the same deterministic tick.
 */
export function teachCombo(): Array<[number, number]> {
  const scout: Array<[number, number]> = [[BUTTON.MP, 0]];
  let link = -1;
  for (let i = 0; i < 120 && link < 0; i++) {
    const fight = replay(scout);
    const mara = fight.fighters[0];
    const holt = fight.fighters[1];
    if (mara.phase === "idle" && holt.phase === "hitstun" && fight.advantage[0] === 6) {
      link = scout.length - 1;
    } else {
      scout.push([0, 0]);
    }
  }
  if (link < 4) {
    throw new Error(`no +6 link window (frame ${link})`);
  }

  const linked = scout.map((pair): [number, number] => [pair[0], pair[1]]);
  const early = linked[link - 4];
  if (!early) throw new Error("early buffer slot missing");
  early[0] |= BUTTON.LP;

  let connect = -1;
  const withJab = linked.map((pair): [number, number] => [pair[0], pair[1]]);
  for (let i = 0; i < 50 && connect < 0; i++) {
    const fight = replay(withJab);
    if (fight.hitsLanded >= 2) connect = withJab.length - 1;
    else withJab.push([0, 0]);
  }
  if (connect < 0) throw new Error("jab did not link");

  const motion = [
    BUTTON.DOWN,
    BUTTON.DOWN,
    BUTTON.DOWN | BUTTON.RIGHT,
    BUTTON.DOWN | BUTTON.RIGHT,
    BUTTON.RIGHT,
    BUTTON.RIGHT | BUTTON.LP,
  ];
  const script = withJab.map((pair): [number, number] => [pair[0], pair[1]]);
  for (let i = 0; i < motion.length; i++) {
    const index = connect + 1 + i;
    while (script.length <= index) script.push([0, 0]);
    const motionBit = motion[i] ?? 0;
    const current = script[index];
    if (!current) throw new Error("motion slot missing");
    script[index] = [current[0] | motionBit, current[1]];
  }

  const played = script.map((pair): [number, number] => [pair[0], pair[1]]);
  for (let i = 0; i < 90; i++) {
    const fight = replay(played);
    if (fight.fighters[1].phase === "knockdown" && fight.hitsLanded >= 3) return played;
    played.push([0, 0]);
  }
  const last = replay(played);
  throw new Error(comboMiss(last));
}

function comboMiss(fight: Fight): string {
  const holt = fight.fighters[1];
  const mara = fight.fighters[0];
  return `combo missed hits=${fight.hitsLanded} mara=${mara.phase}/${mara.move?.name ?? "-"} holt=${holt.phase} frame=${fight.frame}`;
}
