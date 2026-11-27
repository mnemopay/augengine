import { afterEach, it, expect, vi } from "vitest";
import { EngineRunner } from "./engine.js";
import { Scene } from "../scene/scene.js";
import type { WebGpuLiveRenderer } from "./live/webgpu/renderer.js";
afterEach(() => vi.unstubAllGlobals());
it("does not start duplicate RAF loops across async start/stop/start", async () => {
  const raf = vi.fn(() => 1);
  vi.stubGlobal("requestAnimationFrame", raf);
  vi.stubGlobal("cancelAnimationFrame", vi.fn());
  const scene = new Scene({
    schemaVersion: 1,
    source: { kind: "authored", path: "test" },
    entities: [],
  });
  const runner = new EngineRunner({
    scene,
    renderer: { render: vi.fn() } as unknown as WebGpuLiveRenderer,
  });
  let resolve!: () => void;
  const pending = new Promise<void>((r) => {
    resolve = r;
  });
  vi.spyOn(runner.simulation, "init").mockReturnValue(pending);
  const first = runner.start();
  runner.stop();
  const second = runner.start();
  resolve();
  await Promise.all([first, second]);
  expect(raf).toHaveBeenCalledTimes(1);
  runner.destroy();
});
it("accepts the first RAF timestamp preceding the scheduling performance.now", async () => {
  let callback!: FrameRequestCallback;
  vi.stubGlobal(
    "requestAnimationFrame",
    vi.fn((cb) => {
      callback = cb;
      return 1;
    }),
  );
  vi.stubGlobal("cancelAnimationFrame", vi.fn());
  const scene = new Scene({
    schemaVersion: 1,
    source: { kind: "authored", path: "test" },
    entities: [],
  });
  const runner = new EngineRunner({
    scene,
    renderer: { render: vi.fn() } as unknown as WebGpuLiveRenderer,
  });
  await runner.start();
  expect(() => callback(0)).not.toThrow();
  runner.destroy();
});
it("does not init or step physics when simulate is false", async () => {
  let callback!: FrameRequestCallback;
  vi.stubGlobal(
    "requestAnimationFrame",
    vi.fn((cb) => {
      callback = cb;
      return 1;
    }),
  );
  vi.stubGlobal("cancelAnimationFrame", vi.fn());
  const scene = new Scene({
    schemaVersion: 1,
    source: { kind: "authored", path: "test" },
    entities: [],
  });
  const render = vi.fn();
  const runner = new EngineRunner({
    scene,
    renderer: { render } as unknown as WebGpuLiveRenderer,
    simulate: false,
  });
  const init = vi.spyOn(runner.simulation, "init");
  const advance = vi.spyOn(runner.simulation, "advance");
  await runner.start();
  callback(16);
  expect(init).not.toHaveBeenCalled();
  expect(advance).not.toHaveBeenCalled();
  expect(render).toHaveBeenCalledTimes(1);
  runner.destroy();
});
