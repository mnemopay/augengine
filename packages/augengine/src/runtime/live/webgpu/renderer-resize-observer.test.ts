/**
 * Regression: destroy() must disconnect the canvas ResizeObserver. Hosts reuse
 * one canvas across renderers (React remounts, Studio play/edit toggles); a
 * stale observer used to call autosizeCanvas → createTargets on a destroyed
 * renderer and throw "Cannot read properties of null (reading 'updateBindGroup')".
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { WebGpuLiveRenderer } from "./renderer.js";

class FakeResizeObserver {
  static instances: FakeResizeObserver[] = [];
  disconnected = false;
  constructor(public cb: () => void) {
    FakeResizeObserver.instances.push(this);
  }
  observe(): void {}
  disconnect(): void {
    this.disconnected = true;
  }
}

describe("WebGpuLiveRenderer resize observer lifecycle", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    FakeResizeObserver.instances = [];
  });

  it("disconnects on destroy and ignores late resize callbacks", async () => {
    vi.stubGlobal("ResizeObserver", FakeResizeObserver);
    // No adapter: init() throws after wiring the observer, which is all this test needs.
    vi.stubGlobal("navigator", { gpu: { requestAdapter: async () => null } });
    const canvas = { clientWidth: 100, clientHeight: 80, width: 100, height: 80 } as unknown as HTMLCanvasElement;
    const renderer = new WebGpuLiveRenderer(canvas);
    await expect(renderer.init()).rejects.toThrow(/no adapter/);
    expect(FakeResizeObserver.instances).toHaveLength(1);
    const ro = FakeResizeObserver.instances[0]!;

    renderer.destroy();
    expect(ro.disconnected).toBe(true);

    // A reflow delivered after destroy must not resize the (dead) canvas targets.
    (canvas as unknown as { clientWidth: number }).clientWidth = 300;
    expect(() => ro.cb()).not.toThrow();
    expect(canvas.width).toBe(100);
  });

  it("re-init replaces the observer instead of stacking another one", async () => {
    vi.stubGlobal("ResizeObserver", FakeResizeObserver);
    vi.stubGlobal("navigator", { gpu: { requestAdapter: async () => null } });
    const canvas = { clientWidth: 100, clientHeight: 80, width: 100, height: 80 } as unknown as HTMLCanvasElement;
    const renderer = new WebGpuLiveRenderer(canvas);
    await expect(renderer.init()).rejects.toThrow();
    await expect(renderer.init()).rejects.toThrow();
    expect(FakeResizeObserver.instances).toHaveLength(2);
    expect(FakeResizeObserver.instances[0]!.disconnected).toBe(true);
    expect(FakeResizeObserver.instances[1]!.disconnected).toBe(false);
  });
});
