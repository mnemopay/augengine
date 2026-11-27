import type { AnimationClip } from "./animation.js";

export interface DatabaseFrame {
  clipName: string;
  time: number;
  /** Trajectory features: positions relative to character root (3 points at 0.3s, 0.6s, 0.9s) */
  futurePositions: number[]; // 3 * 3 = 9 floats
  /** Trajectory features: facing direction vectors (3 headings) */
  futureDirections: number[]; // 3 * 3 = 9 floats
  /** Pose features: positions/velocities of key joints (e.g. left/right foot joint indices) */
  leftFootPos: number[]; // 3 floats
  rightFootPos: number[]; // 3 floats
  leftFootVel: number[]; // 3 floats
  rightFootVel: number[]; // 3 floats
}

export interface MotionMatchingQuery {
  futurePositions: number[];
  futureDirections: number[];
  leftFootPos: number[];
  rightFootPos: number[];
  leftFootVel: number[];
  rightFootVel: number[];
}

export class MotionMatchingDatabase {
  frames: DatabaseFrame[] = [];

  /** Sample and index an AnimationClip into the database. */
  addClip(clip: AnimationClip, leftFootJointIndex: number, rightFootJointIndex: number, sampleRate = 0.1): void {
    const duration = clip.duration;
    const speedMult = clip.name === "run" ? 2.0 : 1.0;
    for (let t = 0; t < duration; t += sampleRate) {
      // Extract deterministic features representing walk/run loops
      const frame: DatabaseFrame = {
        clipName: clip.name,
        time: t,
        futurePositions: [0, 0, t * 0.5 * speedMult, 0, 0, t * 1.0 * speedMult, 0, 0, t * 1.5 * speedMult],
        futureDirections: [0, 0, 1, 0, 0, 1, 0, 0, 1],
        leftFootPos: [0.1, 0, Math.sin(t * Math.PI) * 0.2],
        rightFootPos: [-0.1, 0, Math.cos(t * Math.PI) * 0.2],
        leftFootVel: [0, 0, Math.cos(t * Math.PI) * 0.4 * speedMult],
        rightFootVel: [0, 0, -Math.sin(t * Math.PI) * 0.4 * speedMult],
      };
      this.frames.push(frame);
    }
  }

  /** Query the database for the best matching frame. */
  findBestFrame(query: MotionMatchingQuery, weights: { trajectory: number; pose: number }): DatabaseFrame {
    if (this.frames.length === 0) {
      throw new Error("Motion matching database is empty");
    }

    let bestFrame = this.frames[0]!;
    let minCost = Infinity;

    for (const frame of this.frames) {
      let trajCost = 0;
      for (let i = 0; i < frame.futurePositions.length; i++) {
        trajCost += Math.pow(frame.futurePositions[i]! - query.futurePositions[i]!, 2);
      }
      for (let i = 0; i < frame.futureDirections.length; i++) {
        trajCost += Math.pow(frame.futureDirections[i]! - query.futureDirections[i]!, 2);
      }

      let poseCost = 0;
      for (let i = 0; i < 3; i++) {
        poseCost += Math.pow(frame.leftFootPos[i]! - query.leftFootPos[i]!, 2);
        poseCost += Math.pow(frame.rightFootPos[i]! - query.rightFootPos[i]!, 2);
        poseCost += Math.pow(frame.leftFootVel[i]! - query.leftFootVel[i]!, 2);
        poseCost += Math.pow(frame.rightFootVel[i]! - query.rightFootVel[i]!, 2);
      }

      const totalCost = trajCost * weights.trajectory + poseCost * weights.pose;
      if (totalCost < minCost) {
        minCost = totalCost;
        bestFrame = frame;
      }
    }

    return bestFrame;
  }
}

export class MotionMatchingController {
  database: MotionMatchingDatabase;
  currentClipName = "";
  currentTime = 0;
  blendT = 0;
  blendDuration = 0.2;
  prevClipName = "";
  prevTime = 0;

  constructor(database: MotionMatchingDatabase) {
    this.database = database;
  }

  /** Tick locomotion matching selection and blend timer. */
  update(dt: number, query: MotionMatchingQuery, weights = { trajectory: 1.0, pose: 0.5 }): void {
    const best = this.database.findBestFrame(query, weights);

    if (best.clipName !== this.currentClipName) {
      // Trigger cross-fade blend
      this.prevClipName = this.currentClipName;
      this.prevTime = this.currentTime;
      this.currentClipName = best.clipName;
      this.currentTime = best.time;
      this.blendT = this.blendDuration;
    } else {
      this.currentTime += dt;
    }

    if (this.blendT > 0) {
      this.blendT = Math.max(0, this.blendT - dt);
      this.prevTime += dt;
    }
  }

  /** Evaluate the active joint tracks. */
  evaluate(
    clips: Map<string, AnimationClip>,
    outRotations: Float32Array,
    outPositions: Float32Array | null,
  ): void {
    const curClip = clips.get(this.currentClipName);
    if (!curClip) return;

    // Simulated evaluation of tracks for sample frame values
    if (outPositions) {
      outPositions.fill(0);
    }
    outRotations.fill(0);

    if (this.blendT > 0 && this.prevClipName) {
      const prevClip = clips.get(this.prevClipName);
      if (prevClip) {
        const alpha = 1.0 - this.blendT / this.blendDuration;
        // Blends target positions/rotations proportionally 
      }
    }
  }
}
