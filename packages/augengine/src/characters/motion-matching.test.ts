import { describe, it, expect } from "vitest";
import {
  MotionMatchingDatabase,
  MotionMatchingController,
  type MotionMatchingQuery,
} from "./motion-matching.js";
import type { AnimationClip } from "./animation.js";

describe("Motion Matching Engine", () => {
  it("samples and indexes AnimationClips into DatabaseFrames", () => {
    const database = new MotionMatchingDatabase();
    
    const mockClip: AnimationClip = {
      name: "walk",
      duration: 1.0,
      loopable: true,
      tracks: [],
    };

    database.addClip(mockClip, 1, 2, 0.2); // sample every 0.2 seconds

    // 1.0 seconds / 0.2 sample rate = 5 frames
    expect(database.frames.length).toBe(5);
    expect(database.frames[0]!.clipName).toBe("walk");
    expect(database.frames[0]!.time).toBe(0);
    expect(database.frames[4]!.time).toBeCloseTo(0.8, 5);
  });

  it("finds the best database frame matching the query trajectory and pose", () => {
    const database = new MotionMatchingDatabase();

    const walkClip: AnimationClip = {
      name: "walk",
      duration: 1.0,
      loopable: true,
      tracks: [],
    };

    const runClip: AnimationClip = {
      name: "run",
      duration: 1.0,
      loopable: true,
      tracks: [],
    };

    database.addClip(walkClip, 1, 2, 0.2);
    database.addClip(runClip, 1, 2, 0.2);

    // Create a query that closely matches the "run" clip speed/trajectory at t=0.4
    const query: MotionMatchingQuery = {
      // run: higher speed trajectory offset representation (speedMult = 2.0)
      futurePositions: [0, 0, 0.4 * 0.5 * 2.0, 0, 0, 0.4 * 1.0 * 2.0, 0, 0, 0.4 * 1.5 * 2.0],
      futureDirections: [0, 0, 1, 0, 0, 1, 0, 0, 1],
      leftFootPos: [0.1, 0, Math.sin(0.4 * Math.PI) * 0.2],
      rightFootPos: [-0.1, 0, Math.cos(0.4 * Math.PI) * 0.2],
      leftFootVel: [0, 0, Math.cos(0.4 * Math.PI) * 0.4 * 2.0],
      rightFootVel: [0, 0, -Math.sin(0.4 * Math.PI) * 0.4 * 2.0],
    };

    const bestFrame = database.findBestFrame(query, { trajectory: 1.0, pose: 0.5 });
    expect(bestFrame.clipName).toBe("run");
    expect(bestFrame.time).toBeCloseTo(0.4, 5);
  });

  it("controller updates active clips and transitions with cross-fade blend timers", () => {
    const database = new MotionMatchingDatabase();
    
    const walkClip: AnimationClip = {
      name: "walk",
      duration: 1.0,
      loopable: true,
      tracks: [],
    };

    const runClip: AnimationClip = {
      name: "run",
      duration: 1.0,
      loopable: true,
      tracks: [],
    };

    database.addClip(walkClip, 1, 2, 0.5);
    database.addClip(runClip, 1, 2, 0.5);

    const controller = new MotionMatchingController(database);
    controller.currentClipName = "walk";

    // Query that points to "run"
    const query: MotionMatchingQuery = {
      futurePositions: [0, 0, 0.5, 0, 0, 1.0, 0, 0, 1.5],
      futureDirections: [0, 0, 1, 0, 0, 1, 0, 0, 1],
      leftFootPos: [0.1, 0, 0],
      rightFootPos: [-0.1, 0, 0],
      leftFootVel: [0, 0, 0],
      rightFootVel: [0, 0, 0],
    };

    // Transition triggered (blendT initialized to 0.2, then decays by dt=0.016 to 0.184)
    controller.update(0.016, query);
    expect(controller.currentClipName).toBe("run");
    expect(controller.prevClipName).toBe("walk");
    expect(controller.blendT).toBeCloseTo(0.184, 5);

    // Further update decays the blend timer by dt=0.05 to 0.134, and increments currentTime to 0.55
    controller.update(0.05, query);
    expect(controller.blendT).toBeCloseTo(0.134, 5);
    expect(controller.currentTime).toBeCloseTo(0.55, 5);
  });
});
