import { FixedStepClock } from "../runtime/fixed-step.js";
import {
  ATTACK_MASK,
  BUFFER_EARLY,
  BUTTON,
  holdingBack,
  holdingForward,
  relDir,
  type AttackButton,
  type RelDir,
} from "./input.js";
import {
  CANCEL_WINDOW,
  MOTION_LIFE,
  THROW_LOCK,
  THROW_TECH,
  listedOnBlock,
  listedOnHit,
  moveForButton,
  totalFrames,
  type MoveDef,
} from "./moves.js";
import { HOLT, MARA, characterMove, type CharacterDef } from "./roster.js";

export const STAGE_LEFT = 80;
export const STAGE_RIGHT = 920;
export const MIN_GAP = 76;
export const JUMP_VY = 16;
export const ROUND_SECONDS = 99;
export const ROUND_FRAMES = ROUND_SECONDS * 60;
export const WINS_TO_TAKE = 2;

export type Phase = "idle" | "attack" | "hitstun" | "blockstun" | "knockdown" | "throw" | "thrown";
export type RoundDecision = "ko" | "time" | "draw";

interface BufferedButton {
  bit: number;
  age: number;
}

interface DirMark {
  rel: RelDir;
  age: number;
}

export interface Fighter {
  character: CharacterDef;
  x: number;
  y: number;
  vy: number;
  facing: 1 | -1;
  phase: Phase;
  move: MoveDef | null;
  moveFrame: number;
  poseFrame: number;
  stun: number;
  down: number;
  lock: number;
  tech: number;
  health: number;
  held: number;
  prevHeld: number;
  crouching: boolean;
  connected: boolean;
  connectAge: number;
  /** Contact frame: hold the stun/lock counters so they start ticking on the next simulated frame. */
  freshContact: boolean;
  freshStun: boolean;
  cancelSpent: boolean;
  buffer: BufferedButton[];
  dirs: DirMark[];
}

export class Fight {
  readonly clock = new FixedStepClock(1 / 60);
  readonly fighters: [Fighter, Fighter];
  frame = 0;
  hitstop = 0;
  hitsLanded = 0;
  /** Advantage snapshot taken after state transitions and before this frame's timer tick. */
  advantage: [number | null, number | null] = [null, null];
  /** Table advantage of the contact that just happened, else null. */
  contactAdvantage: number | null = null;
  contactKind: "hit" | "block" | "throw" | null = null;
  roundOver = false;
  decision: RoundDecision | null = null;
  /** 0 is Mara, 1 is Holt. Null on a draw or before the round ends. */
  winner: 0 | 1 | null = null;
  /** Round wins. First to WINS_TO_TAKE takes the match. A draw round gives neither side a win. */
  wins: [number, number] = [0, 0];
  roundNumber = 1;
  matchOver = false;
  matchWinner: 0 | 1 | null = null;

  constructor() {
    this.fighters = [makeFighter(MARA, 400, 1), makeFighter(HOLT, 480, -1)];
  }

  /** Advance exactly one 60 Hz step. Throws if the clock does not take one step. */
  tick(p1: number, p2: number): void {
    let ran = 0;
    this.clock.advance(this.clock.dt, () => {
      ran += 1;
      this.step(p1, p2);
    });
    if (ran !== 1) throw new Error(`60 Hz clock took ${ran} steps`);
  }

  place(x1: number, x2: number): void {
    this.fighters[0].x = x1;
    this.fighters[1].x = x2;
    this.faceIdle();
  }

  /**
   * Score the locked round and either start the next one or end the match.
   * A draw adds no win. Calling this before the round locks, or after the match
   * is over, leaves the state unchanged.
   */
  nextRound(): void {
    if (!this.roundOver || this.matchOver) return;
    if (this.winner === 0) this.wins[0] += 1;
    else if (this.winner === 1) this.wins[1] += 1;
    if (this.wins[0] >= WINS_TO_TAKE || this.wins[1] >= WINS_TO_TAKE) {
      this.matchOver = true;
      this.matchWinner = this.wins[0] >= WINS_TO_TAKE ? 0 : 1;
      return;
    }
    this.roundNumber += 1;
    this.frame = 0;
    this.hitstop = 0;
    this.hitsLanded = 0;
    this.roundOver = false;
    this.decision = null;
    this.winner = null;
    this.advantage = [null, null];
    this.contactAdvantage = null;
    this.contactKind = null;
    restoreFighter(this.fighters[0], makeFighter(MARA, 400, 1));
    restoreFighter(this.fighters[1], makeFighter(HOLT, 480, -1));
  }

  private step(p1: number, p2: number): void {
    if (this.roundOver) return;
    this.frame += 1;
    this.contactAdvantage = null;
    this.contactKind = null;
    this.read(this.fighters[0], p1);
    this.read(this.fighters[1], p2);
    if (this.hitstop > 0) {
      this.hitstop -= 1;
      this.ageInputs();
      this.advantage = [liveAdvantage(this, 0), liveAdvantage(this, 1)];
      this.settleRound();
      return;
    }
    this.beginFrame();
    this.advantage = [liveAdvantage(this, 0), liveAdvantage(this, 1)];
    this.tryAct(this.fighters[0], this.fighters[1]);
    this.tryAct(this.fighters[1], this.fighters[0]);
    this.resolve(this.fighters[0], this.fighters[1]);
    this.resolve(this.fighters[1], this.fighters[0]);
    this.integrate();
    this.ageInputs();
    this.settleRound();
  }

  /**
   * A killing blow waits out hitstop, then the round locks.
   * The fallen fighter stays down. Further ticks do not change the state.
   */
  private settleRound(): void {
    if (this.roundOver) return;
    const dead0 = this.fighters[0].health <= 0;
    const dead1 = this.fighters[1].health <= 0;
    if (dead0 || dead1) {
      if (this.hitstop > 0) return;
      const winner = dead0 && dead1 ? null : dead1 ? 0 : 1;
      this.finishRound(dead0 && dead1 ? "draw" : "ko", winner);
      return;
    }
    if (this.frame < ROUND_FRAMES) return;
    const ahead = this.fighters[0].health - this.fighters[1].health;
    this.finishRound(ahead === 0 ? "draw" : "time", ahead === 0 ? null : ahead > 0 ? 0 : 1);
  }

  private finishRound(decision: RoundDecision, winner: 0 | 1 | null): void {
    this.roundOver = true;
    this.decision = decision;
    this.winner = winner;
    this.advantage = [null, null];
    this.contactAdvantage = null;
    this.contactKind = null;
    for (const fighter of this.fighters) {
      if (fighter.health > 0) continue;
      fighter.phase = "knockdown";
      fighter.stun = 0;
      fighter.y = 0;
      fighter.vy = 0;
      fighter.move = null;
    }
  }

  private read(fighter: Fighter, bits: number): void {
    const edges = bits & ~fighter.held & ATTACK_MASK;
    fighter.prevHeld = fighter.held;
    fighter.held = bits;
    for (const bit of [BUTTON.LP, BUTTON.MP, BUTTON.HP, BUTTON.LK, BUTTON.MK, BUTTON.HK]) {
      if ((edges & bit) !== 0) fighter.buffer.push({ bit, age: 0 });
    }
    fighter.dirs.push({ rel: relDir(fighter.facing, bits), age: 0 });
  }

  private beginFrame(): void {
    for (const fighter of this.fighters) {
      if (fighter.phase === "attack" && fighter.move && fighter.moveFrame >= totalFrames(fighter.move)) {
        clearMove(fighter);
      }
      if ((fighter.phase === "hitstun" || fighter.phase === "blockstun") && fighter.stun < 0) {
        fighter.phase = "idle";
        fighter.stun = 0;
      }
      if (fighter.phase === "knockdown" && fighter.down <= 0) {
        fighter.phase = "idle";
        fighter.y = 0;
        fighter.vy = 0;
      }
      if (fighter.phase === "throw" && fighter.lock <= 0) clearMove(fighter);
      if (fighter.phase === "thrown" && fighter.lock <= 0) {
        fighter.phase = "knockdown";
        fighter.down = fighter.move?.down ?? 30;
        fighter.move = null;
      }
      const upEdge = (fighter.held & BUTTON.UP) !== 0 && (fighter.prevHeld & BUTTON.UP) === 0;
      if (fighter.phase === "idle" && fighter.y === 0 && upEdge && (fighter.held & BUTTON.DOWN) === 0) {
        fighter.vy = JUMP_VY;
      }
      fighter.crouching =
        fighter.y === 0 && (fighter.held & BUTTON.DOWN) !== 0 && fighter.phase === "idle";
    }
    this.faceIdle();
    for (const fighter of this.fighters) {
      if (fighter.phase !== "idle" || fighter.y !== 0 || fighter.crouching) continue;
      const dir = holdingForward(fighter.facing, fighter.held)
        ? 1
        : holdingBack(fighter.facing, fighter.held)
          ? -1
          : 0;
      if (dir === 0) continue;
      const speed = dir === 1 ? fighter.character.walkForward : fighter.character.walkBack;
      this.slide(fighter, dir * fighter.facing * speed);
    }
  }

  private tryAct(fighter: Fighter, other: Fighter): void {
    fighter.buffer = fighter.buffer.filter((item) => item.age <= BUFFER_EARLY);
    if (this.hitstop > 0) return;
    if (fighter.phase === "attack" && fighter.move?.cancel && fighter.connected && !fighter.cancelSpent) {
      if (fighter.connectAge <= CANCEL_WINDOW && seesQuarterCircle(fighter) && hasBit(fighter, BUTTON.LP)) {
        consume(fighter, BUTTON.LP);
        startMove(fighter, characterMove(fighter.character, "arc"));
        return;
      }
    }
    if (fighter.phase !== "idle" || fighter.y > 0) return;
    const lp = fresh(fighter, BUTTON.LP);
    const lk = fresh(fighter, BUTTON.LK);
    if (lp && lk) {
      consume(fighter, BUTTON.LP);
      consume(fighter, BUTTON.LK);
      startMove(fighter, characterMove(fighter.character, "clasp"));
      return;
    }
    if (seesQuarterCircle(fighter) && hasBit(fighter, BUTTON.LP)) {
      consume(fighter, BUTTON.LP);
      startMove(fighter, characterMove(fighter.character, "arc"));
      return;
    }
    const pressed = oldestAttack(fighter);
    if (!pressed) return;
    consume(fighter, pressed);
    const name = attackName(pressed);
    startMove(fighter, characterMove(fighter.character, moveForButton(name).id));
    void other;
  }

  private resolve(attacker: Fighter, defender: Fighter): void {
    if (attacker.phase === "thrown" && attacker.tech > 0) {
      const tech = fresh(attacker, BUTTON.LP) && fresh(attacker, BUTTON.LK);
      if (tech) {
        consume(attacker, BUTTON.LP);
        consume(attacker, BUTTON.LK);
        this.techThrow(attacker, defender);
        return;
      }
    }
    if (attacker.phase !== "attack" || !attacker.move || attacker.connected) return;
    const move = attacker.move;
    if (attacker.moveFrame < move.startup || attacker.moveFrame >= move.startup + move.active) return;
    if (!rectsOverlap(hitRect(attacker, move), hurtRect(defender))) return;
    const airborne = defender.y > 0;
    if (airborne && !move.hitsAir) return;
    if (!airborne && !move.hitsGround) return;
    if (move.guard === "throw") {
      if (airborne || defender.phase === "hitstun" || defender.phase === "knockdown") return;
      this.applyThrow(attacker, defender, move);
      return;
    }
    if (this.blocked(defender, move)) {
      this.applyBlock(attacker, defender, move);
      return;
    }
    this.applyHit(attacker, defender, move);
  }

  private blocked(defender: Fighter, move: MoveDef): boolean {
    if (defender.y > 0) return false;
    if (
      defender.phase === "hitstun" ||
      defender.phase === "attack" ||
      defender.phase === "knockdown" ||
      defender.phase === "throw" ||
      defender.phase === "thrown"
    ) {
      return false;
    }
    if (!holdingBack(defender.facing, defender.held)) return false;
    const crouching = (defender.held & BUTTON.DOWN) !== 0;
    if (move.guard === "low") return crouching;
    if (move.guard === "overhead") return !crouching;
    return true;
  }

  private applyHit(attacker: Fighter, defender: Fighter, move: MoveDef): void {
    attacker.connected = true;
    attacker.connectAge = 0;
    attacker.freshContact = true;
    defender.freshStun = true;
    defender.health = Math.max(0, defender.health - move.damage);
    this.hitsLanded += 1;
    this.hitstop = move.hitstop;
    this.contactKind = "hit";
    this.contactAdvantage = listedOnHit(move);
    if (move.knockdown) {
      defender.phase = "knockdown";
      defender.down = move.down;
      defender.stun = 0;
      defender.move = null;
      defender.vy = 0;
      defender.y = 0;
    } else {
      defender.phase = "hitstun";
      defender.stun = move.hitstun;
      defender.move = null;
    }
    this.slide(defender, attacker.facing * 4);
  }

  private applyBlock(attacker: Fighter, defender: Fighter, move: MoveDef): void {
    attacker.connected = true;
    attacker.connectAge = 0;
    attacker.freshContact = true;
    defender.freshStun = true;
    defender.phase = "blockstun";
    defender.stun = move.blockstun;
    defender.move = null;
    this.hitstop = move.hitstop;
    this.contactKind = "block";
    this.contactAdvantage = listedOnBlock(move);
    this.slide(defender, attacker.facing * 6);
  }

  private applyThrow(attacker: Fighter, defender: Fighter, move: MoveDef): void {
    attacker.connected = true;
    attacker.freshContact = true;
    attacker.phase = "throw";
    attacker.lock = THROW_LOCK;
    attacker.move = move;
    defender.phase = "thrown";
    defender.freshContact = true;
    defender.lock = THROW_LOCK;
    defender.tech = THROW_TECH;
    defender.move = move;
    defender.health = Math.max(0, defender.health - move.damage);
    defender.stun = 0;
    this.hitsLanded += 1;
    this.contactKind = "throw";
    this.contactAdvantage = null;
  }

  private techThrow(thrown: Fighter, thrower: Fighter): void {
    const refund = thrown.move?.damage ?? 0;
    thrown.health = Math.min(thrown.character.maxHealth, thrown.health + refund);
    this.hitsLanded = Math.max(0, this.hitsLanded - 1);
    clearMove(thrown);
    clearMove(thrower);
    this.slide(thrown, thrower.facing * 48);
  }

  private integrate(): void {
    for (const fighter of this.fighters) {
      if (fighter.phase === "attack" && fighter.move) {
        fighter.moveFrame += 1;
        fighter.poseFrame += 1;
        if (fighter.connected && !fighter.freshContact) fighter.connectAge += 1;
        fighter.freshContact = false;
      } else if (fighter.phase === "idle") {
        fighter.poseFrame += 1;
      }
      if (fighter.phase === "hitstun" || fighter.phase === "blockstun") {
        if (fighter.freshStun) fighter.freshStun = false;
        else fighter.stun -= 1;
      }
      if (fighter.phase === "knockdown") {
        if (fighter.freshStun) fighter.freshStun = false;
        else fighter.down -= 1;
      }
      if (fighter.phase === "throw" || fighter.phase === "thrown") {
        if (fighter.freshContact) fighter.freshContact = false;
        else {
          fighter.lock -= 1;
          if (fighter.phase === "thrown") fighter.tech -= 1;
        }
      }
      if (fighter.y > 0 || fighter.vy > 0) {
        fighter.y += fighter.vy;
        fighter.vy -= 1;
        if (fighter.y <= 0) {
          fighter.y = 0;
          fighter.vy = 0;
        }
      }
    }
  }

  private ageInputs(): void {
    for (const fighter of this.fighters) {
      for (const item of fighter.buffer) item.age += 1;
      for (const item of fighter.dirs) item.age += 1;
      fighter.dirs = fighter.dirs.filter((item) => item.age <= MOTION_LIFE);
    }
  }

  private slide(fighter: Fighter, dx: number): void {
    fighter.x += dx;
    const other = this.fighters[0] === fighter ? this.fighters[1] : this.fighters[0];
    if (Math.abs(fighter.x - other.x) < MIN_GAP) {
      fighter.x = other.x + (fighter.x < other.x ? -MIN_GAP : MIN_GAP);
    }
    if (fighter.x < STAGE_LEFT) fighter.x = STAGE_LEFT;
    if (fighter.x > STAGE_RIGHT) fighter.x = STAGE_RIGHT;
  }

  private faceIdle(): void {
    const [a, b] = this.fighters;
    if (a.phase !== "idle" || b.phase !== "idle") return;
    if (a.y > 0 || b.y > 0) return;
    if (a.x <= b.x) {
      a.facing = 1;
      b.facing = -1;
    } else {
      a.facing = -1;
      b.facing = 1;
    }
  }
}

function makeFighter(character: CharacterDef, x: number, facing: 1 | -1): Fighter {
  return {
    character,
    x,
    y: 0,
    vy: 0,
    facing,
    phase: "idle",
    move: null,
    moveFrame: 0,
    poseFrame: 0,
    stun: 0,
    down: 0,
    lock: 0,
    tech: 0,
    health: character.maxHealth,
    held: 0,
    prevHeld: 0,
    crouching: false,
    connected: false,
    connectAge: -1,
    freshContact: false,
    freshStun: false,
    cancelSpent: false,
    buffer: [],
    dirs: [],
  };
}

function restoreFighter(into: Fighter, source: Fighter): void {
  into.character = source.character;
  into.x = source.x;
  into.y = source.y;
  into.vy = source.vy;
  into.facing = source.facing;
  into.phase = source.phase;
  into.move = source.move;
  into.moveFrame = source.moveFrame;
  into.poseFrame = source.poseFrame;
  into.stun = source.stun;
  into.down = source.down;
  into.lock = source.lock;
  into.tech = source.tech;
  into.health = source.health;
  into.held = source.held;
  into.prevHeld = source.prevHeld;
  into.crouching = source.crouching;
  into.connected = source.connected;
  into.connectAge = source.connectAge;
  into.freshContact = source.freshContact;
  into.freshStun = source.freshStun;
  into.cancelSpent = source.cancelSpent;
  into.buffer = [];
  into.dirs = [];
}

function clearMove(fighter: Fighter): void {
  fighter.phase = "idle";
  fighter.move = null;
  fighter.moveFrame = 0;
  fighter.connected = false;
  fighter.connectAge = -1;
  fighter.freshContact = false;
  fighter.freshStun = false;
  fighter.cancelSpent = false;
  fighter.lock = 0;
  fighter.tech = 0;
}

function startMove(fighter: Fighter, move: MoveDef): void {
  fighter.phase = "attack";
  fighter.move = move;
  fighter.moveFrame = 0;
  fighter.poseFrame = 0;
  fighter.connected = false;
  fighter.connectAge = -1;
  fighter.freshContact = false;
  fighter.freshStun = false;
  fighter.cancelSpent = false;
  fighter.crouching = false;
}

function hasBit(fighter: Fighter, bit: number): boolean {
  return fighter.buffer.some((item) => item.bit === bit && item.age <= BUFFER_EARLY);
}

function fresh(fighter: Fighter, bit: number): boolean {
  return fighter.buffer.some((item) => item.bit === bit && item.age === 0);
}

function consume(fighter: Fighter, bit: number): void {
  const index = fighter.buffer.findIndex((item) => item.bit === bit);
  if (index >= 0) fighter.buffer.splice(index, 1);
}

function oldestAttack(fighter: Fighter): number | null {
  let best: BufferedButton | null = null;
  for (const item of fighter.buffer) {
    if (item.age > BUFFER_EARLY) continue;
    if (!best || item.age < best.age) best = item;
  }
  return best ? best.bit : null;
}

function attackName(bit: number): AttackButton {
  if (bit === BUTTON.LP) return "LP";
  if (bit === BUTTON.MP) return "MP";
  if (bit === BUTTON.HP) return "HP";
  if (bit === BUTTON.LK) return "LK";
  if (bit === BUTTON.MK) return "MK";
  return "HK";
}

function seesQuarterCircle(fighter: Fighter): boolean {
  let stage = 0;
  for (const mark of fighter.dirs) {
    if (mark.age > MOTION_LIFE || mark.rel === "N") continue;
    if (stage === 0 && mark.rel === "D") stage = 1;
    else if (stage === 1 && mark.rel === "DF") stage = 2;
    else if (stage === 1 && mark.rel === "D") stage = 1;
    else if (stage === 2 && mark.rel === "F") stage = 3;
    else if (stage === 2 && (mark.rel === "DF" || mark.rel === "D")) continue;
    else if (stage < 3 && mark.rel === "D") stage = 1;
    else if (stage < 3) stage = 0;
  }
  return stage === 3;
}

export interface Rect {
  l: number;
  r: number;
  b: number;
  t: number;
}

export function hurtRect(fighter: Fighter): Rect {
  const height = fighter.y > 0 ? 70 : fighter.crouching ? 58 : 96;
  const y = fighter.y > 0 ? fighter.y + 8 : fighter.y;
  return { l: fighter.x - 22, r: fighter.x + 22, b: y, t: y + height };
}

export function hitRect(fighter: Fighter, move: MoveDef): Rect {
  const box = move.hit;
  const l = fighter.facing === 1 ? fighter.x + box.x : fighter.x - box.x - box.w;
  return { l, r: l + box.w, b: fighter.y + box.y, t: fighter.y + box.y + box.h };
}

function rectsOverlap(a: Rect, b: Rect): boolean {
  return a.l < b.r && a.r > b.l && a.b < b.t && a.t > b.b;
}

/**
 * Defender stun minus the frames before the attacker can press.
 * A +6 reading on the first idle frame is the 1-frame link into a 6-frame startup.
 */
export function liveAdvantage(fight: Fight, index: 0 | 1): number | null {
  const attacker = fight.fighters[index];
  const defender = fight.fighters[index === 0 ? 1 : 0];
  if (defender.phase !== "hitstun" && defender.phase !== "blockstun") return null;
  let wait = 0;
  if (attacker.phase === "attack" && attacker.move) {
    wait = Math.max(0, totalFrames(attacker.move) - attacker.moveFrame);
  } else if (attacker.phase !== "idle") {
    return null;
  }
  return defender.stun - wait;
}

export function hashFight(fight: Fight): string {
  const parts: number[] = [
    fight.frame,
    fight.hitstop,
    fight.hitsLanded,
    decisionCode(fight.decision),
    fight.winner === null ? -1 : fight.winner,
    fight.wins[0],
    fight.wins[1],
    fight.roundNumber,
    fight.matchOver ? 1 : 0,
    fight.matchWinner === null ? -1 : fight.matchWinner,
  ];
  for (const fighter of fight.fighters) {
    parts.push(
      fighter.x,
      fighter.y,
      fighter.vy,
      fighter.facing,
      phaseCode(fighter.phase),
      fighter.moveFrame,
      fighter.poseFrame,
      fighter.stun,
      fighter.down,
      fighter.lock,
      fighter.health,
      fighter.held,
      fighter.crouching ? 1 : 0,
      fighter.buffer.length,
      fighter.dirs.length,
      moveCode(fighter.move?.id),
    );
  }
  let hash = 0x811c9dc5;
  const text = parts.join(",");
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

function moveCode(id: string | undefined): number {
  if (!id) return 0;
  let code = 0;
  for (let i = 0; i < id.length; i++) code = Math.imul(code, 33) + id.charCodeAt(i);
  return code;
}

function decisionCode(decision: RoundDecision | null): number {
  if (decision === "ko") return 1;
  if (decision === "time") return 2;
  if (decision === "draw") return 3;
  return 0;
}

function phaseCode(phase: Phase): number {
  switch (phase) {
    case "idle":
      return 0;
    case "attack":
      return 1;
    case "hitstun":
      return 2;
    case "blockstun":
      return 3;
    case "knockdown":
      return 4;
    case "throw":
      return 5;
    case "thrown":
      return 6;
  }
}

export function createFight(): Fight {
  return new Fight();
}

export function replay(inputs: ReadonlyArray<readonly [number, number]>): Fight {
  const fight = createFight();
  for (const pair of inputs) fight.tick(pair[0] ?? 0, pair[1] ?? 0);
  return fight;
}
