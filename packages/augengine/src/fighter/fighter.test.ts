import { describe, expect, it } from "vitest";
import { BUTTON } from "./input.js";
import { listedOnBlock, listedOnHit, moveById } from "./moves.js";
import { samplePunch } from "./clip.js";
import { frameMeter } from "./meter.js";
import { teachCombo } from "./combo.js";
import { cpuBits } from "./cpu.js";
import { ROUND_FRAMES, WINS_TO_TAKE, createFight, hashFight, replay } from "./sim.js";

const cross = moveById("cross");
const jab = moveById("jab");

function findLinkFrame(): { inputs: Array<[number, number]>; frame: number } {
  const inputs: Array<[number, number]> = [[BUTTON.MP, 0]];
  for (let i = 0; i < 120; i++) {
    const fight = replay(inputs);
    const mara = fight.fighters[0];
    const holt = fight.fighters[1];
    if (mara.phase === "idle" && holt.phase === "hitstun" && fight.advantage[0] === 6) {
      return { inputs, frame: inputs.length - 1 };
    }
    inputs.push([0, 0]);
  }
  throw new Error("cross never reached a +6 actionable frame");
}

describe("step 1 — deterministic 60 Hz tick", () => {
  it("uses its own 60 Hz clock and repeats a byte-identical hash", () => {
    const inputs: Array<[number, number]> = [
      [BUTTON.RIGHT, BUTTON.LEFT],
      [BUTTON.RIGHT, BUTTON.LEFT],
      [BUTTON.MP, 0],
      [0, 0],
      [0, BUTTON.RIGHT],
    ];
    const first = replay(inputs);
    const second = replay(inputs);
    expect(first.clock.dt).toBe(1 / 60);
    expect(hashFight(first)).toBe(hashFight(second));
    expect(first.frame).toBe(inputs.length);
    const changed = replay(inputs.map((pair, index) => (index === 2 ? [BUTTON.LP, pair[1]] : pair)));
    expect(hashFight(changed)).not.toBe(hashFight(first));
  });
});

describe("step 2 — move table and frame meter", () => {
  it("prints the table advantage from the tick", () => {
    expect(cross.active).toBe(1);
    expect(listedOnHit(cross)).toBe(6);
    expect(jab.startup).toBe(6);
    const { inputs } = findLinkFrame();
    const fight = replay(inputs);
    const meter = frameMeter(fight);
    expect(meter.p1.advantage).toBe(6);
    expect(meter.p1.advantage).toBe(listedOnHit(cross));
    expect(meter.p1.poseSeconds).toBe(fight.fighters[0].poseFrame / 60);
    expect(meter.frame).toBe(fight.frame);
  });
});

describe("step 3 — whiff, hit, block, hitstop", () => {
  it("whiffs outside range, hits in range, and blocks when Holt holds back", () => {
    const far = createFight();
    far.place(200, 700);
    far.tick(BUTTON.MP, 0);
    for (let i = 0; i < 40; i++) far.tick(0, 0);
    expect(far.hitsLanded).toBe(0);
    expect(far.fighters[1].health).toBe(1000);

    let hitAdvantage: number | null = null;
    const hit = createFight();
    hit.tick(BUTTON.MP, 0);
    for (let i = 0; i < 40 && hitAdvantage === null; i++) {
      if (hit.contactKind === "hit") hitAdvantage = hit.contactAdvantage;
      hit.tick(0, 0);
    }
    expect(hit.hitsLanded).toBe(1);
    expect(hit.fighters[1].health).toBe(1000 - cross.damage);
    expect(hitAdvantage).toBe(listedOnHit(cross));

    let blockAdvantage: number | null = null;
    const blocked = createFight();
    blocked.tick(BUTTON.MP, 0);
    for (let i = 0; i < 40 && blockAdvantage === null; i++) {
      const mara = blocked.fighters[0];
      const guard = mara.move && mara.moveFrame >= mara.move.startup ? BUTTON.RIGHT : 0;
      if (blocked.contactKind === "block") blockAdvantage = blocked.contactAdvantage;
      blocked.tick(0, guard);
    }
    expect(blocked.fighters[1].health).toBe(1000);
    expect(blockAdvantage).toBe(listedOnBlock(cross));
    expect(blockAdvantage).toBe(-2);
  });

  it("holds the pose for the hitstop frames while the frame counter advances", () => {
    const fight = createFight();
    fight.tick(BUTTON.MP, 0);
    let seen = false;
    for (let i = 0; i < 40 && !seen; i++) {
      if (fight.contactKind === "hit") seen = true;
      else fight.tick(0, 0);
    }
    expect(seen).toBe(true);
    const pose = fight.fighters[0].poseFrame;
    const holtPose = fight.fighters[1].poseFrame;
    const frame = fight.frame;
    for (let i = 0; i < cross.hitstop; i++) {
      fight.tick(0, 0);
      expect(fight.fighters[0].poseFrame).toBe(pose);
      expect(fight.fighters[1].poseFrame).toBe(holtPose);
      expect(fight.frame).toBe(frame + i + 1);
    }
    expect(fight.hitstop).toBe(0);
  });
});

describe("step 4 — four-frame input buffer", () => {
  it("links a 6-frame jab from 4 frames early and drops the press one frame before that", () => {
    const { inputs, frame } = findLinkFrame();
    const onTime = inputs.map((pair): [number, number] => [pair[0], pair[1]]);
    const timed = onTime[frame];
    if (!timed) throw new Error("missing link frame");
    timed[0] |= BUTTON.LP;
    while (onTime.length < frame + 20) onTime.push([0, 0]);
    expect(replay(onTime).hitsLanded).toBe(2);

    const early = inputs.map((pair): [number, number] => [pair[0], pair[1]]);
    const earlySlot = early[frame - 4];
    if (!earlySlot) throw new Error("missing early frame");
    earlySlot[0] |= BUTTON.LP;
    while (early.length < frame + 20) early.push([0, 0]);
    expect(replay(early).hitsLanded).toBe(2);

    const tooEarly = inputs.map((pair): [number, number] => [pair[0], pair[1]]);
    const dead = tooEarly[frame - 5];
    if (!dead) throw new Error("missing expired frame");
    dead[0] |= BUTTON.LP;
    while (tooEarly.length < frame + 20) tooEarly.push([0, 0]);
    expect(replay(tooEarly).hitsLanded).toBe(1);

    const late = inputs.map((pair): [number, number] => [pair[0], pair[1]]);
    while (late.length < frame + 16) late.push([0, BUTTON.RIGHT]);
    for (let i = frame; i < late.length; i++) {
      const row = late[i];
      if (row) row[1] |= BUTTON.RIGHT;
    }
    const lateSlot = late[frame + 1];
    if (!lateSlot) throw new Error("missing late frame");
    lateSlot[0] |= BUTTON.LP;
    const lateFight = replay(late);
    expect(lateFight.hitsLanded).toBe(1);
    expect(lateFight.fighters[1].health).toBe(1000 - cross.damage);
  });
});

describe("step 5 — clip sampled at frame/60", () => {
  it("moves the punch sample with the pose frame and holds it in hitstop", () => {
    const rest = samplePunch(0);
    const swung = samplePunch(8);
    expect(swung[2]).not.toBe(rest[2]);
    expect(samplePunch(8)).toEqual(swung);

    const fight = createFight();
    fight.tick(BUTTON.MP, 0);
    const first = fight.fighters[0].poseFrame;
    expect(frameMeter(fight).p1.poseSeconds).toBe(first / 60);
    while (fight.contactKind !== "hit") fight.tick(0, 0);
    const held = fight.fighters[0].poseFrame;
    expect(held).not.toBe(first);
    const heldSample = samplePunch(held);
    for (let i = 0; i < cross.hitstop; i++) {
      fight.tick(0, 0);
      expect(fight.fighters[0].poseFrame).toBe(held);
      expect(samplePunch(fight.fighters[0].poseFrame)).toEqual(heldSample);
      expect(frameMeter(fight).p1.poseSeconds).toBe(held / 60);
    }
  });
});

describe("step 6 — second character and a round", () => {
  it("walks, checks low and overhead, anti-airs a jump, and sweeps", () => {
    const maraWalk = createFight();
    maraWalk.place(300, 700);
    const before = maraWalk.fighters[0].x;
    maraWalk.tick(BUTTON.RIGHT, 0);
    expect(maraWalk.fighters[0].x).toBe(before + maraWalk.fighters[0].character.walkForward);

    const holtWalk = createFight();
    holtWalk.place(300, 700);
    const holtBefore = holtWalk.fighters[1].x;
    holtWalk.tick(0, BUTTON.LEFT);
    expect(holtWalk.fighters[1].x).toBe(holtBefore - holtWalk.fighters[1].character.walkForward);
    expect(holtWalk.fighters[1].character.walkForward).not.toBe(maraWalk.fighters[0].character.walkForward);

    const low = createFight();
    low.tick(0, BUTTON.LK);
    for (let i = 0; i < 24; i++) low.tick(0, 0);
    expect(low.fighters[0].health).toBeLessThan(1000);

    const lowBlocked = createFight();
    let lowWasBlocked = false;
    lowBlocked.tick(BUTTON.DOWN | BUTTON.LEFT, BUTTON.LK);
    for (let i = 0; i < 24; i++) {
      lowBlocked.tick(BUTTON.DOWN | BUTTON.LEFT, 0);
      if (lowBlocked.contactKind === "block") lowWasBlocked = true;
    }
    expect(lowWasBlocked).toBe(true);
    expect(lowBlocked.fighters[0].health).toBe(1000);

    const overhead = createFight();
    let overheadHit = false;
    overhead.tick(BUTTON.DOWN | BUTTON.LEFT, BUTTON.HP);
    for (let i = 0; i < 60; i++) {
      overhead.tick(BUTTON.DOWN | BUTTON.LEFT, 0);
      if (overhead.contactKind === "hit") overheadHit = true;
    }
    expect(overheadHit).toBe(true);
    expect(overhead.fighters[0].health).toBeLessThan(1000);

    const standBlock = createFight();
    let stood = false;
    standBlock.tick(0, BUTTON.HP);
    for (let i = 0; i < 60; i++) {
      const holt = standBlock.fighters[1];
      const guard = holt.move && holt.moveFrame >= holt.move.startup - 1 ? BUTTON.LEFT : 0;
      standBlock.tick(guard, 0);
      if (standBlock.contactKind === "block") stood = true;
    }
    expect(stood).toBe(true);
    expect(standBlock.fighters[0].health).toBe(1000);

    const air = createFight();
    air.tick(0, BUTTON.UP);
    let antiAir = false;
    for (let i = 0; i < 40; i++) {
      const holt = air.fighters[1];
      const press = holt.y > 0 && air.fighters[0].phase === "idle" ? BUTTON.MK : 0;
      air.tick(press, 0);
      if (air.contactKind === "hit" && holt.y > 0) antiAir = true;
    }
    expect(antiAir).toBe(true);

    const sweep = createFight();
    sweep.tick(0, BUTTON.HK);
    let down = false;
    for (let i = 0; i < 40; i++) {
      sweep.tick(0, 0);
      if (sweep.fighters[0].phase === "knockdown") down = true;
    }
    expect(down).toBe(true);
  });

  it("cancels the link into arc and knocks Holt down", () => {
    const inputs = teachCombo();
    const first = replay(inputs);
    const second = replay(inputs);
    expect(hashFight(first)).toBe(hashFight(second));
    expect(first.hitsLanded).toBeGreaterThanOrEqual(3);
    expect(first.fighters[1].phase).toBe("knockdown");
    const meter = frameMeter(first);
    expect(meter.p2.phase).toBe("knockdown");
    expect(meter.p1.name).toBe("Mara");
    expect(meter.p2.name).toBe("Holt");
    expect(first.roundOver).toBe(false);
    expect(meter.decision).toBeNull();

    const contact = createFight();
    for (const pair of inputs) {
      contact.tick(pair[0], pair[1]);
      if (contact.contactKind === "hit" && contact.fighters[0].move?.id === "arc") {
        expect(frameMeter(contact).p1.advantage).toBeNull();
        return;
      }
    }
    throw new Error("arc never connected");
  });

  it("techs a clasp inside the window and misses the tech after it", () => {
    const fight = createFight();
    fight.place(400, 450);
    fight.tick(BUTTON.LP | BUTTON.LK, 0);
    let grabbed = false;
    for (let i = 0; i < 20 && !grabbed; i++) {
      fight.tick(0, 0);
      if (fight.contactKind === "throw") grabbed = true;
    }
    expect(grabbed).toBe(true);
    const health = fight.fighters[1].health;
    fight.tick(0, BUTTON.LP | BUTTON.LK);
    expect(fight.fighters[1].health).toBe(1000);
    expect(health).toBeLessThan(1000);
    expect(fight.fighters[1].phase).toBe("idle");

    const late = createFight();
    late.place(400, 450);
    late.tick(BUTTON.LP | BUTTON.LK, 0);
    for (let i = 0; i < 30; i++) late.tick(0, 0);
    expect(late.fighters[1].health).toBeLessThan(1000);
    expect(late.fighters[1].phase === "thrown" || late.fighters[1].phase === "knockdown").toBe(true);
  });
});

describe("round — KO, time, and draw", () => {
  it("locks a KO after hitstop and ignores later input", () => {
    const fight = createFight();
    let killingHitstop = 0;
    for (let i = 0; i < 8000 && !fight.roundOver; i++) {
      const mara = fight.fighters[0];
      const holt = fight.fighters[1];
      const gap = Math.abs(holt.x - mara.x);
      const forward = mara.facing === 1 ? BUTTON.RIGHT : BUTTON.LEFT;
      const bits = mara.phase === "idle" && mara.y === 0 ? (gap > 96 ? forward : BUTTON.MP) : 0;
      fight.tick(bits, 0);
      if (killingHitstop === 0 && holt.health === 0 && fight.hitstop > 0) killingHitstop = fight.hitstop;
    }
    expect(fight.roundOver).toBe(true);
    expect(fight.decision).toBe("ko");
    expect(fight.winner).toBe(0);
    expect(fight.fighters[1].health).toBe(0);
    expect(fight.fighters[1].phase).toBe("knockdown");
    expect(killingHitstop).toBe(cross.hitstop);
    const meter = frameMeter(fight);
    expect(meter.winner).toBe("Mara");
    expect(meter.roundOver).toBe(true);
    expect(meter.p1.advantage).toBeNull();
    expect(meter.p2.stun).toBe(0);
    const frozen = hashFight(fight);
    const frame = fight.frame;
    const x = fight.fighters[0].x;
    fight.tick(BUTTON.HK, BUTTON.LP | BUTTON.LK);
    expect(fight.frame).toBe(frame);
    expect(fight.fighters[0].x).toBe(x);
    expect(fight.fighters[1].health).toBe(0);
    expect(hashFight(fight)).toBe(frozen);
  });

  it("calls a draw at 99 seconds and a time win when one side is ahead", () => {
    const draw = createFight();
    expect(frameMeter(draw).secondsLeft).toBe(99);
    for (let i = 0; i < ROUND_FRAMES; i++) draw.tick(0, 0);
    expect(draw.frame).toBe(ROUND_FRAMES);
    expect(draw.roundOver).toBe(true);
    expect(draw.decision).toBe("draw");
    expect(draw.winner).toBeNull();
    expect(frameMeter(draw).secondsLeft).toBe(0);
    expect(draw.fighters[0].health).toBe(1000);
    const frozen = hashFight(draw);
    draw.tick(BUTTON.MP, 0);
    expect(hashFight(draw)).toBe(frozen);

    const timed = createFight();
    timed.tick(BUTTON.MP, 0);
    while (timed.frame < ROUND_FRAMES) timed.tick(0, 0);
    expect(timed.decision).toBe("time");
    expect(timed.winner).toBe(0);
    expect(timed.fighters[1].health).toBe(1000 - cross.damage);
    expect(frameMeter(timed).winner).toBe("Mara");
  });
});

function koHolt(fight: ReturnType<typeof createFight>): void {
  for (let i = 0; i < 8000 && !fight.roundOver; i++) {
    const mara = fight.fighters[0];
    const holt = fight.fighters[1];
    const gap = Math.abs(holt.x - mara.x);
    const forward = mara.facing === 1 ? BUTTON.RIGHT : BUTTON.LEFT;
    const bits = mara.phase === "idle" && mara.y === 0 ? (gap > 96 ? forward : BUTTON.MP) : 0;
    fight.tick(bits, 0);
  }
}

describe("match — first to two", () => {
  it("scores two KOs as a match and then ignores nextRound and input", () => {
    expect(WINS_TO_TAKE).toBe(2);
    const fight = createFight();
    koHolt(fight);
    expect(fight.decision).toBe("ko");
    expect(fight.winner).toBe(0);
    fight.nextRound();
    expect(fight.roundOver).toBe(false);
    expect(fight.matchOver).toBe(false);
    expect(fight.wins).toEqual([1, 0]);
    expect(fight.roundNumber).toBe(2);
    expect(fight.frame).toBe(0);
    expect(fight.fighters[1].health).toBe(1000);
    expect(fight.fighters[0].x).toBe(400);
    koHolt(fight);
    expect(fight.winner).toBe(0);
    fight.nextRound();
    expect(fight.matchOver).toBe(true);
    expect(fight.matchWinner).toBe(0);
    expect(fight.wins).toEqual([2, 0]);
    expect(fight.roundOver).toBe(true);
    expect(fight.roundNumber).toBe(2);
    const frozen = hashFight(fight);
    fight.nextRound();
    fight.tick(BUTTON.HK, BUTTON.LP | BUTTON.LK);
    expect(hashFight(fight)).toBe(frozen);
    const meter = frameMeter(fight);
    expect(meter.matchOver).toBe(true);
    expect(meter.matchWinner).toBe("Mara");
    expect(meter.wins).toEqual([2, 0]);
    expect(meter.roundNumber).toBe(2);
  });

  it("opens the next round after a draw without giving either side a win", () => {
    const fight = createFight();
    for (let i = 0; i < ROUND_FRAMES; i++) fight.tick(0, 0);
    expect(fight.decision).toBe("draw");
    fight.nextRound();
    expect(fight.wins).toEqual([0, 0]);
    expect(fight.roundNumber).toBe(2);
    expect(fight.roundOver).toBe(false);
    expect(fight.matchOver).toBe(false);
    expect(fight.frame).toBe(0);
    expect(fight.fighters[0].health).toBe(1000);
    expect(fight.fighters[1].health).toBe(1000);
  });

  it("does nothing when the round is still running", () => {
    const fight = createFight();
    fight.tick(BUTTON.MP, 0);
    fight.nextRound();
    expect(fight.roundNumber).toBe(1);
    expect(fight.wins).toEqual([0, 0]);
    expect(fight.frame).toBe(1);
  });
});

describe("cpu — idle side", () => {
  it("walks Holt in from a wide gap by his forward speed", () => {
    const fight = createFight();
    fight.place(300, 700);
    expect(cpuBits(fight, 1)).toBe(BUTTON.LEFT);
    const before = fight.fighters[1].x;
    fight.tick(0, cpuBits(fight, 1));
    expect(fight.fighters[1].x).toBe(before - fight.fighters[1].character.walkForward);
  });

  it("blocks Mara's cross and stays at full health", () => {
    const fight = createFight();
    fight.tick(BUTTON.MP, cpuBits(fight, 1));
    let blocked = false;
    for (let i = 0; i < 30 && !blocked; i++) {
      fight.tick(0, cpuBits(fight, 1));
      if (fight.contactKind === "block") blocked = true;
    }
    expect(blocked).toBe(true);
    expect(fight.fighters[1].health).toBe(1000);
    expect(fight.fighters[1].phase).toBe("blockstun");
  });

  it("repeats the same presses from the same state", () => {
    const left = createFight();
    const right = createFight();
    for (let i = 0; i < 180; i++) {
      const press = cpuBits(left, 1);
      expect(press).toBe(cpuBits(right, 1));
      left.tick(0, press);
      right.tick(0, press);
    }
    expect(hashFight(left)).toBe(hashFight(right));
    expect(left.hitsLanded).toBeGreaterThan(0);
  });
});
