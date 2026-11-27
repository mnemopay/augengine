/** Absolute stick and six attack buttons. One bit per control. */
export const BUTTON = {
  LEFT: 1,
  RIGHT: 2,
  UP: 4,
  DOWN: 8,
  LP: 16,
  MP: 32,
  HP: 64,
  LK: 128,
  MK: 256,
  HK: 512,
} as const;

export type AttackButton = "LP" | "MP" | "HP" | "LK" | "MK" | "HK";

export const ATTACK_MASK =
  BUTTON.LP | BUTTON.MP | BUTTON.HP | BUTTON.LK | BUTTON.MK | BUTTON.HK;

export const DIR_MASK = BUTTON.LEFT | BUTTON.RIGHT | BUTTON.UP | BUTTON.DOWN;

/** Presses this many frames before the legal frame still come out. The legal frame itself is included, so the window is 5. */
export const BUFFER_EARLY = 4;

export type RelDir = "N" | "B" | "F" | "D" | "U" | "DF" | "DB" | "UF" | "UB";

export function relDir(facing: 1 | -1, held: number): RelDir {
  const left = (held & BUTTON.LEFT) !== 0;
  const right = (held & BUTTON.RIGHT) !== 0;
  const down = (held & BUTTON.DOWN) !== 0;
  const up = (held & BUTTON.UP) !== 0;
  let horizontal: -1 | 0 | 1 = 0;
  if (left && !right) horizontal = -1;
  if (right && !left) horizontal = 1;
  const forward = facing === 1 ? horizontal === 1 : horizontal === -1;
  const back = facing === 1 ? horizontal === -1 : horizontal === 1;
  if (down && forward) return "DF";
  if (down && back) return "DB";
  if (up && forward) return "UF";
  if (up && back) return "UB";
  if (down) return "D";
  if (up) return "U";
  if (forward) return "F";
  if (back) return "B";
  return "N";
}

export function holdingBack(facing: 1 | -1, held: number): boolean {
  return facing === 1 ? (held & BUTTON.LEFT) !== 0 : (held & BUTTON.RIGHT) !== 0;
}

export function holdingForward(facing: 1 | -1, held: number): boolean {
  return facing === 1 ? (held & BUTTON.RIGHT) !== 0 : (held & BUTTON.LEFT) !== 0;
}
