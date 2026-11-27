/**
 * SpatialMixer against a minimal fake BaseAudioContext. Web Audio does the
 * actual HRTF panning and distance attenuation; the mixer's job is to
 * configure the PannerNode and keep source + listener positions in sync.
 */
import { describe, expect, it } from "vitest";
import { SpatialMixer } from "./spatial.ts";

function param() {
  const p = { value: 0, setValueAtTime(v: number) { p.value = v; return p; } };
  return p;
}

function fakeMixer() {
  const panners: Record<string, unknown>[] = [];
  const sources: { started: boolean; stopped: number }[] = [];
  const node = () => ({ connect() {}, disconnect() {} });
  const ctx = {
    currentTime: 0,
    listener: {
      positionX: param(), positionY: param(), positionZ: param(),
      forwardX: param(), forwardY: param(), forwardZ: param(),
      upX: param(), upY: param(), upZ: param(),
    },
    createBufferSource() {
      const s = { ...node(), buffer: null, loop: false, started: false, stopped: 0,
        start() { s.started = true; }, stop() { s.stopped++; } };
      sources.push(s);
      return s;
    },
    createPanner() {
      const p = { ...node(), positionX: param(), positionY: param(), positionZ: param(),
        orientationX: param(), orientationY: param(), orientationZ: param() } as Record<string, unknown>;
      panners.push(p);
      return p;
    },
    createGain() { return { ...node(), gain: { value: 1 } }; },
    createConvolver() { return { ...node(), buffer: null }; },
  };
  const buses = { master: node(), music: node(), sfx: node(), voice: node(), ui: node() };
  const mixer = new SpatialMixer({ ctx, buses } as never);
  return { mixer, ctx, panners, sources };
}

const pos = (p: Record<string, unknown>) =>
  ["positionX", "positionY", "positionZ"].map((k) => (p[k] as { value: number }).value);

describe("SpatialMixer", () => {
  it("configures an HRTF panner with inverse distance attenuation by default", () => {
    const f = fakeMixer();
    f.mixer.spawn({ buffer: {} as AudioBuffer, position: [1, 2, 3] });
    const p = f.panners[0]!;
    expect(p.panningModel).toBe("HRTF");
    expect(p.distanceModel).toBe("inverse");
    expect(p.refDistance).toBe(1);
    expect(p.maxDistance).toBe(100);
    expect(pos(p)).toEqual([1, 2, 3]);
    expect(f.sources[0]!.started).toBe(true);
  });

  it("follows the source position and stops idempotently", () => {
    const f = fakeMixer();
    const h = f.mixer.spawn({ buffer: {} as AudioBuffer, position: [0, 0, 0], distanceModel: "linear", maxDistance: 20 });
    h.setPosition([4, 0, -2]);
    expect(pos(f.panners[0]!)).toEqual([4, 0, -2]);
    expect(f.panners[0]!.distanceModel).toBe("linear");
    h.stop();
    h.stop();
    expect(f.sources[0]!.stopped).toBe(1);
    h.setPosition([9, 9, 9]); // ignored after stop
    expect(pos(f.panners[0]!)).toEqual([4, 0, -2]);
  });

  it("syncs the listener to the camera pose", () => {
    const f = fakeMixer();
    f.mixer.setListener({ position: [0, 1.7, 5.6], forward: [0, 0, -1], up: [0, 1, 0] });
    const l = f.ctx.listener;
    expect([l.positionX.value, l.positionY.value, l.positionZ.value]).toEqual([0, 1.7, 5.6]);
    expect([l.forwardX.value, l.forwardY.value, l.forwardZ.value]).toEqual([0, 0, -1]);
    expect([l.upX.value, l.upY.value, l.upZ.value]).toEqual([0, 1, 0]);
  });
});
