/**
 * Smoke-cube example — validates Phase 1 of @mnemopay/augengine.
 *
 * Loads scene.json → builds Scene → live runtime spins the cube + plays
 * a sine tone. Render button runs the same scene through the render
 * runtime (WebCodecs encoder) and reports byte size + frames encoded.
 */

import { Scene, qFromAxisAngle, type MutQuat } from "@mnemopay/augengine/scene";
import { LiveRuntime } from "@mnemopay/augengine/runtime/live";
import { RenderRuntime } from "@mnemopay/augengine/runtime/render";
import { LiveMixer, OfflineMixer } from "@mnemopay/augengine/audio";
import sceneJson from "./scene.json";

const canvas = document.getElementById("stage") as HTMLCanvasElement;
const playBtn = document.getElementById("play") as HTMLButtonElement;
const renderBtn = document.getElementById("render") as HTMLButtonElement;
const metricEl = document.getElementById("metric") as HTMLSpanElement;
const statusEl = document.getElementById("status") as HTMLSpanElement;
const errEl = document.getElementById("err") as HTMLSpanElement;

function setStatus(s: string): void {
  statusEl.textContent = s;
}
function setErr(e: unknown): void {
  errEl.textContent = e instanceof Error ? e.message : String(e);
}

resizeCanvas();
window.addEventListener("resize", resizeCanvas);

const scene = Scene.fromJson(sceneJson);
const liveMixer = new LiveMixer();

let runtime: LiveRuntime | null = null;
let cubeRotationState: MutQuat = [0, 0, 0, 1];

playBtn.addEventListener("click", async () => {
  try {
    setErr("");
    if (runtime) {
      runtime.stop();
      runtime.destroy();
      runtime = null;
      playBtn.textContent = "▶ Play (live)";
      return;
    }
    runtime = new LiveRuntime(canvas, scene, {
      simHz: 60,
      onSimTick: (dt) => {
        // Spin the cube around Y at 0.5 rad/s.
        const angle = 0.5 * dt;
        const delta: MutQuat = [0, 0, 0, 1];
        qFromAxisAngle(delta, [0, 1, 0], angle);
        // Multiply: cubeRotationState = delta * cubeRotationState.
        const x = cubeRotationState[0],
          y = cubeRotationState[1],
          z = cubeRotationState[2],
          w = cubeRotationState[3];
        const dx = delta[0],
          dy = delta[1],
          dz = delta[2],
          dw = delta[3];
        cubeRotationState = [
          dw * x + dx * w + dy * z - dz * y,
          dw * y - dx * z + dy * w + dz * x,
          dw * z + dx * y - dy * x + dz * w,
          dw * w - dx * x - dy * y - dz * z,
        ];
        scene.setLocalTransform(
          "cube-hero" as never,
          [0, 0, 0],
          cubeRotationState,
          [1, 1, 1],
        );
      },
      onRenderTick: (m) => {
        metricEl.textContent = `${m.fps.toFixed(0)} fps · ${m.frameTimeMs.toFixed(2)} ms · frame ${m.frameIndex}`;
      },
    });
    await runtime.init();
    await liveMixer.resume();
    liveMixer.playTone({ frequency: 220, durationSec: 0.8, waveform: "sine", bus: "music" });
    runtime.start();
    playBtn.textContent = "⏸ Stop";
    setStatus("live");
  } catch (e) {
    setErr(e);
  }
});

renderBtn.addEventListener("click", async () => {
  try {
    setErr("");
    setStatus("rendering…");
    renderBtn.disabled = true;
    if (runtime) {
      runtime.stop();
    }
    const off = new OffscreenCanvas(canvas.width || 960, canvas.height || 540);
    const r = new RenderRuntime(off);
    await r.init();
    let renderRotState: MutQuat = [0, 0, 0, 1];
    const result = await r.run({
      scene,
      encoder: {
        codec: "h264",
        width: off.width,
        height: off.height,
        fps: 30,
        bitrate: 4_000_000,
        bitrateMode: "variable",
        keyframeInterval: 30,
      },
      fps: 30,
      frames: 60,
      seed: 42,
      onSimTick: (i, t) => {
        const angle = 0.5 * (1 / 30);
        const delta: MutQuat = [0, 0, 0, 1];
        qFromAxisAngle(delta, [0, 1, 0], angle);
        const x = renderRotState[0],
          y = renderRotState[1],
          z = renderRotState[2],
          w = renderRotState[3];
        const dx = delta[0],
          dy = delta[1],
          dz = delta[2],
          dw = delta[3];
        renderRotState = [
          dw * x + dx * w + dy * z - dz * y,
          dw * y - dx * z + dy * w + dz * x,
          dw * z + dx * y - dy * x + dz * w,
          dw * w - dx * x - dy * y - dz * z,
        ];
        scene.setLocalTransform(
          "cube-hero" as never,
          [0, 0, 0],
          renderRotState,
          [1, 1, 1],
        );
        void i;
        void t;
      },
      onProgress: (p) => setStatus(`rendering ${(p * 100).toFixed(0)}%`),
    });
    // Render audio in parallel using OfflineMixer.
    const offMix = new OfflineMixer({ sampleRate: 48000, channels: 2, durationSec: 2 });
    offMix.scheduleTone({ frequency: 220, startSec: 0, durationSec: 0.8, bus: "music" });
    await offMix.render();
    setStatus(
      `done · ${result.framesEncoded} frames · ${(result.mp4.byteLength / 1024).toFixed(1)} KB · ${(
        result.elapsedMs / 1000
      ).toFixed(2)}s`,
    );
    r.destroy();
  } catch (e) {
    setErr(e);
  } finally {
    renderBtn.disabled = false;
  }
});

function resizeCanvas(): void {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const w = Math.max(640, document.querySelector("main")!.clientWidth);
  const h = Math.max(360, document.querySelector("main")!.clientHeight);
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;
}
