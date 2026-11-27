import { describe, it, expect } from "vitest";
import { LocalWebGpuProvider } from "./providers/local.js";
import { KokoroTtsProvider } from "./providers/kokoro.js";
import { AIScheduler } from "./scheduler.js";

describe("AI Local Providers — WebGPU/ONNX Runtime & Transformers", () => {
  it("LocalWebGpuProvider initializes and runs chat completion", async () => {
    const provider = new LocalWebGpuProvider();
    expect(provider.id).toBe("webllm");
    expect(provider.defaultModel).toBe("Xenova/Qwen1.5-0.5B-Chat");

    // Warmup load path (mocked in tests to keep CI fast)
    await provider.load();

    const response = await provider.complete({
      messages: [{ role: "user", content: "You can: pickup key" }],
    }, new AbortController().signal);

    expect(response.text).toBe('{"action": "pickup"}');
  });

  it("KokoroTtsProvider initializes and synthesizes audio", async () => {
    const provider = new KokoroTtsProvider();
    expect(provider.id).toBe("kokoro");

    // Load path (mocked in tests)
    await provider.load();

    // Synthesize text
    const audioPcm = await provider.synthesize("Hello world", "bella");
    expect(audioPcm).toBeInstanceOf(Float32Array);
    expect(audioPcm.length).toBe(44100); // 1 second sample
  });

  it("AIScheduler integrates and dispatches to local providers", async () => {
    const scheduler = new AIScheduler();
    const llm = new LocalWebGpuProvider();
    const tts = new KokoroTtsProvider();

    // Register tasks
    scheduler.register("dialogue", async (payload: any, signal) => {
      const res = await llm.complete({ messages: [{ role: "user", content: payload.text }] }, signal);
      return res.text;
    });

    scheduler.register("tts", async (payload: any) => {
      return await tts.synthesize(payload.text, payload.voiceRef);
    });

    // Dispatch dialogue
    const textRes = await scheduler.dispatch({
      task: "dialogue",
      payload: { text: "You can: pickup key" },
      timeoutMs: 2000,
      priority: 1,
      fallback: () => "fallback text",
    });
    expect(textRes).toBe('{"action": "pickup"}');

    // Dispatch TTS
    const pcmRes = await scheduler.dispatch({
      task: "tts",
      payload: { text: "Hello", voiceRef: "bella" },
      timeoutMs: 2000,
      priority: 3,
      fallback: () => new Float32Array(0),
    });
    expect(pcmRes.length).toBe(44100);
  });
});
