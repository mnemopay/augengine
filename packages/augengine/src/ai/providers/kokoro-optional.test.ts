import { afterEach, describe, expect, it } from "vitest";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { KokoroTtsProvider, KOKORO_MISSING_ONNX, __setKokoroOnnxLoader } from "./kokoro.js";
import { LocalWebGpuProvider, LOCAL_LLM_MISSING_TRANSFORMERS } from "./local.js";

const SRC = join(dirname(fileURLToPath(import.meta.url)), "..", "..");

function walk(dir: string, out: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) walk(p, out);
    else if (p.endsWith(".ts") && !p.endsWith(".test.ts")) out.push(p);
  }
  return out;
}

describe("Kokoro is optional (@mnemopay/augengine-onnx)", () => {
  afterEach(() => {
    delete (globalThis as any).__KOKORO_REAL_LOAD__;
  });

  it("no source file statically imports an optional peer (type-only imports are erased)", () => {
    const offenders = walk(SRC).filter((f) =>
      /^\s*import\s+(?!type\b)[^;]*from\s+["'](@mnemopay\/(augengine-onnx|sdk)|@xenova\/transformers)(\/[^"']*)?["']/m.test(readFileSync(f, "utf8")),
    );
    expect(offenders).toEqual([]);
  });

  it("an injected loadOnnx is used instead of the runtime import", async () => {
    (globalThis as any).__KOKORO_REAL_LOAD__ = true;
    let used = false;
    const tts = new KokoroTtsProvider("/k.onnx", {
      loadOnnx: async () => {
        used = true;
        return {
          createOrtSession: async () => ({ run: async () => ({ audio: { data: new Float32Array([0.5]) } }) }),
          createFloat32Tensor: async (data: Float32Array, dims: number[]) => ({ data, dims }),
        };
      },
    });
    await tts.load();
    expect(used).toBe(true);
  });

  it("a failing injected loadOnnx surfaces the install hint", async () => {
    (globalThis as any).__KOKORO_REAL_LOAD__ = true;
    const tts = new KokoroTtsProvider("/k.onnx", { loadOnnx: async () => { throw new Error("nope"); } });
    await expect(tts.load()).rejects.toThrow(KOKORO_MISSING_ONNX);
  });

  it("load() without the package throws the install hint", async () => {
    (globalThis as any).__KOKORO_REAL_LOAD__ = true;
    __setKokoroOnnxLoader(async () => {
      throw new Error(KOKORO_MISSING_ONNX);
    });
    await expect(new KokoroTtsProvider().load()).rejects.toThrow(
      "Install @mnemopay/augengine-onnx to use Kokoro voices",
    );
  });

  it("load() with the package present creates a session", async () => {
    (globalThis as any).__KOKORO_REAL_LOAD__ = true;
    let created = "";
    __setKokoroOnnxLoader(async () => ({
      createOrtSession: async (url: string) => {
        created = url;
        return { run: async () => ({ audio: { data: new Float32Array([0.1, 0.2]) } }) };
      },
      createFloat32Tensor: async (data: Float32Array, dims: number[]) => ({ data, dims }),
    }));
    const tts = new KokoroTtsProvider("/m.onnx");
    const pcm = await tts.synthesize("hi");
    expect(created).toBe("/m.onnx");
    expect(Array.from(pcm)).toEqual([Float32Array.from([0.1])[0], Float32Array.from([0.2])[0]]);
  });
});

describe("LocalWebGpuProvider is optional (@xenova/transformers)", () => {
  afterEach(() => {
    delete (globalThis as any).__LOCAL_LLM_REAL_LOAD__;
  });

  it("load() without the package throws the install hint", async () => {
    (globalThis as any).__LOCAL_LLM_REAL_LOAD__ = true;
    const llm = new LocalWebGpuProvider({ loadTransformers: async () => { throw new Error("Cannot find package"); } });
    await expect(llm.load()).rejects.toThrow(LOCAL_LLM_MISSING_TRANSFORMERS);
    await expect(llm.load()).rejects.toThrow("Install @xenova/transformers to use LocalWebGpuProvider");
  });

  it("load() with an injected loader builds the pipeline from it", async () => {
    (globalThis as any).__LOCAL_LLM_REAL_LOAD__ = true;
    const calls: unknown[][] = [];
    const llm = new LocalWebGpuProvider({
      model: "m",
      loadTransformers: async () => ({
        pipeline: async (...args: unknown[]) => {
          calls.push(args);
          return async () => [{ generated_text: "hi" }];
        },
      }),
    });
    await llm.load();
    expect(calls[0]?.slice(0, 2)).toEqual(["text-generation", "m"]);
  });
});
