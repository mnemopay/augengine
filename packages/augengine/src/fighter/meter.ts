import { samplePunch } from "./clip.js";
import { listedOnBlock, listedOnHit, totalFrames, type MoveDef } from "./moves.js";
import { ROUND_SECONDS, hurtRect, type Fight, type Fighter, type Rect, type RoundDecision } from "./sim.js";

export interface MeterSide {
  name: string;
  phase: string;
  move: string;
  startup: number | null;
  active: number | null;
  recovery: number | null;
  moveFrame: number;
  poseFrame: number;
  poseSeconds: number;
  stun: number;
  health: number;
  advantage: number | null;
  tableOnHit: number | null;
  tableOnBlock: number | null;
  hurt: Rect;
  hit: Rect | null;
}

export interface FrameMeter {
  frame: number;
  hitstop: number;
  hitsLanded: number;
  contact: "hit" | "block" | "throw" | null;
  secondsLeft: number;
  roundOver: boolean;
  decision: RoundDecision | null;
  winner: string | null;
  roundNumber: number;
  wins: [number, number];
  matchOver: boolean;
  matchWinner: string | null;
  p1: MeterSide;
  p2: MeterSide;
}

export function frameMeter(fight: Fight): FrameMeter {
  return {
    frame: fight.frame,
    hitstop: fight.hitstop,
    hitsLanded: fight.hitsLanded,
    contact: fight.contactKind,
    secondsLeft: Math.max(0, ROUND_SECONDS - Math.floor(fight.frame / 60)),
    roundOver: fight.roundOver,
    decision: fight.decision,
    winner: fight.winner === null ? null : fight.fighters[fight.winner].character.name,
    roundNumber: fight.roundNumber,
    wins: [fight.wins[0], fight.wins[1]],
    matchOver: fight.matchOver,
    matchWinner: fight.matchWinner === null ? null : fight.fighters[fight.matchWinner].character.name,
    p1: side(fight, 0),
    p2: side(fight, 1),
  };
}

function side(fight: Fight, index: 0 | 1): MeterSide {
  const fighter = fight.fighters[index];
  const move = fighter.move;
  const attacking = fighter.phase === "attack" && move !== null;
  return {
    name: fighter.character.name,
    phase: fighter.phase,
    move: move ? move.name : "—",
    startup: move?.startup ?? null,
    active: move?.active ?? null,
    recovery: move?.recovery ?? null,
    moveFrame: fighter.moveFrame,
    poseFrame: fighter.poseFrame,
    poseSeconds: fighter.poseFrame / 60,
    stun: fighter.stun,
    health: fighter.health,
    advantage: shownAdvantage(fight, index),
    tableOnHit: move ? listedOnHit(move) : null,
    tableOnBlock: move ? listedOnBlock(move) : null,
    hurt: hurtRect(fighter),
    hit: attacking && move && inActive(fighter, move) ? hitBox(fighter, move) : null,
  };
}

function shownAdvantage(fight: Fight, index: 0 | 1): number | null {
  const fighter = fight.fighters[index];
  if (fight.contactKind && fighter.phase === "attack" && fighter.connected) {
    return fight.contactAdvantage;
  }
  return fight.advantage[index];
}

function inActive(fighter: Fighter, move: MoveDef): boolean {
  return fighter.moveFrame >= move.startup && fighter.moveFrame < move.startup + move.active;
}

function hitBox(fighter: Fighter, move: MoveDef): Rect {
  const box = move.hit;
  const l = fighter.facing === 1 ? fighter.x + box.x : fighter.x - box.x - box.w;
  return { l, r: l + box.w, b: fighter.y + box.y, t: fighter.y + box.y + box.h };
}

export function poseAngle(poseFrame: number): number {
  const quat = samplePunch(poseFrame);
  const z = quat[2] ?? 0;
  const w = quat[3] ?? 1;
  return 2 * Math.atan2(z, w);
}

export function moveProgress(fighter: Fighter): string {
  if (!fighter.move || fighter.phase !== "attack") return fighter.phase;
  const move = fighter.move;
  if (fighter.moveFrame < move.startup) return "startup";
  if (fighter.moveFrame < move.startup + move.active) return "active";
  if (fighter.moveFrame < totalFrames(move)) return "recovery";
  return "attack";
}
