import type { AttackButton } from "./input.js";

export type Guard = "mid" | "low" | "overhead" | "throw";

export interface Box {
  /** Forward offset from the feet, in the fighter's facing. */
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface MoveDef {
  id: string;
  name: string;
  button: AttackButton | "special" | "throw";
  startup: number;
  active: number;
  recovery: number;
  hitstop: number;
  hitstun: number;
  blockstun: number;
  damage: number;
  guard: Guard;
  hitsAir: boolean;
  hitsGround: boolean;
  knockdown: boolean;
  /** A connected hit or block can be cancelled into the special. */
  cancel: boolean;
  hit: Box;
  /** Frames the defender stays down after a knockdown hit. */
  down: number;
}

/**
 * Advantage for a hit that connects on the first active frame.
 * Valid when `active` is 1. The sim ticks stun once per recovery frame
 * and keeps the defender locked on a stun value of 0, so a +6 move
 * links a 6-frame startup pressed on the first actionable frame.
 */
export function listedOnHit(move: MoveDef): number | null {
  if (move.knockdown || move.guard === "throw") return null;
  return move.hitstun - ((move.active - 1) + move.recovery);
}

/** Block advantage for a hit that connects on the first active frame. */
export function listedOnBlock(move: MoveDef): number | null {
  if (move.guard === "throw") return null;
  return move.blockstun - ((move.active - 1) + move.recovery);
}

export function totalFrames(move: MoveDef): number {
  return move.startup + move.active + move.recovery;
}

const PUNCH: Box = { x: 18, y: 58, w: 58, h: 20 };
const LOW: Box = { x: 16, y: 8, w: 64, h: 22 };
const OVERHEAD: Box = { x: 20, y: 28, w: 52, h: 44 };
const ANTIAIR: Box = { x: 8, y: 64, w: 56, h: 96 };
const SWEEP: Box = { x: 10, y: 6, w: 78, h: 24 };
const ARC: Box = { x: 22, y: 48, w: 78, h: 28 };
const THROW: Box = { x: 8, y: 36, w: 34, h: 36 };

export const MOVES: readonly MoveDef[] = [
  {
    id: "jab",
    name: "jab",
    button: "LP",
    startup: 6,
    active: 1,
    recovery: 16,
    hitstop: 3,
    hitstun: 20,
    blockstun: 14,
    damage: 30,
    guard: "mid",
    hitsAir: true,
    hitsGround: true,
    knockdown: false,
    cancel: true,
    hit: PUNCH,
    down: 0,
  },
  {
    id: "cross",
    name: "cross",
    button: "MP",
    startup: 8,
    active: 1,
    recovery: 12,
    hitstop: 4,
    hitstun: 18,
    blockstun: 10,
    damage: 50,
    guard: "mid",
    hitsAir: true,
    hitsGround: true,
    knockdown: false,
    cancel: true,
    hit: PUNCH,
    down: 0,
  },
  {
    id: "crest",
    name: "crest",
    button: "HP",
    startup: 18,
    active: 1,
    recovery: 18,
    hitstop: 5,
    hitstun: 22,
    blockstun: 14,
    damage: 80,
    guard: "overhead",
    hitsAir: false,
    hitsGround: true,
    knockdown: false,
    cancel: false,
    hit: OVERHEAD,
    down: 0,
  },
  {
    id: "shin",
    name: "shin",
    button: "LK",
    startup: 7,
    active: 1,
    recovery: 14,
    hitstop: 3,
    hitstun: 16,
    blockstun: 10,
    damage: 40,
    guard: "low",
    hitsAir: false,
    hitsGround: true,
    knockdown: false,
    cancel: false,
    hit: LOW,
    down: 0,
  },
  {
    id: "spire",
    name: "spire",
    button: "MK",
    startup: 4,
    active: 4,
    recovery: 14,
    hitstop: 4,
    hitstun: 24,
    blockstun: 12,
    damage: 60,
    guard: "mid",
    hitsAir: true,
    hitsGround: false,
    knockdown: false,
    cancel: false,
    hit: ANTIAIR,
    down: 0,
  },
  {
    id: "reaping",
    name: "reaping",
    button: "HK",
    startup: 10,
    active: 1,
    recovery: 20,
    hitstop: 5,
    hitstun: 0,
    blockstun: 14,
    damage: 70,
    guard: "low",
    hitsAir: false,
    hitsGround: true,
    knockdown: true,
    cancel: false,
    hit: SWEEP,
    down: 35,
  },
  {
    id: "arc",
    name: "arc",
    button: "special",
    startup: 10,
    active: 1,
    recovery: 18,
    hitstop: 8,
    hitstun: 0,
    blockstun: 8,
    damage: 90,
    guard: "mid",
    hitsAir: true,
    hitsGround: true,
    knockdown: true,
    cancel: false,
    hit: ARC,
    down: 40,
  },
  {
    id: "clasp",
    name: "clasp",
    button: "throw",
    startup: 5,
    active: 1,
    recovery: 22,
    hitstop: 0,
    hitstun: 0,
    blockstun: 0,
    damage: 80,
    guard: "throw",
    hitsAir: false,
    hitsGround: true,
    knockdown: true,
    cancel: false,
    hit: THROW,
    down: 30,
  },
];

export const THROW_LOCK = 18;
/** Defender can tech a clasp for this many frames after it connects, including the connect frame. */
export const THROW_TECH = 8;
export const CANCEL_WINDOW = 16;
export const MOTION_LIFE = 12;

export function moveById(id: string): MoveDef {
  const move = MOVES.find((item) => item.id === id);
  if (!move) throw new Error(`unknown move ${id}`);
  return move;
}

export function moveForButton(button: AttackButton): MoveDef {
  const move = MOVES.find((item) => item.button === button);
  if (!move) throw new Error(`no move for ${button}`);
  return move;
}
