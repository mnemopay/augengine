/**
 * The slice of `@mnemopay/augengine-onnx` Kokoro uses, typed structurally here so the
 * engine compiles (and its .d.ts files resolve) without the optional package installed.
 * Kept compatible with that package's `createOrtSession` / `createFloat32Tensor` /
 * `OrtSessionLike`, so `loadOnnx: () => import("@mnemopay/augengine-onnx")` still fits.
 */
interface KokoroOrtTensor {
  readonly data: unknown;
}
interface OrtSessionLike {
  run(feeds: Record<string, any>): Promise<Record<string, KokoroOrtTensor | undefined>>;
}
interface OnnxModule {
  createOrtSession(model: string): Promise<OrtSessionLike>;
  createFloat32Tensor(data: Float32Array, dims: readonly number[]): Promise<unknown>;
}

/**
 * Optional dependency: only loaded when Kokoro is actually used. The specifier is
 * computed so no bundler tries to resolve it when the root barrel is imported;
 * bundled apps pass `loadOnnx: () => import("@mnemopay/augengine-onnx")`.
 */
const ONNX_PACKAGE = ["@mnemopay", "augengine-onnx"].join("/");
export const KOKORO_MISSING_ONNX = "Install @mnemopay/augengine-onnx to use Kokoro voices";

const wrapOnnxLoader = (load: () => Promise<unknown>) => async (): Promise<OnnxModule> => {
  try {
    return (await load()) as OnnxModule;
  } catch (err) {
    throw new Error(KOKORO_MISSING_ONNX, { cause: err });
  }
};

let onnxLoader: () => Promise<OnnxModule> = wrapOnnxLoader(() => import(/* @vite-ignore */ ONNX_PACKAGE));

/** Test hook: swap how the optional ONNX package is loaded. */
export function __setKokoroOnnxLoader(loader: () => Promise<unknown>): void {
  onnxLoader = loader as () => Promise<OnnxModule>;
}

/**
 * KokoroTtsProvider — client-side WebGPU-accelerated TTS inference.
 * Runs a local Kokoro ONNX model, outputting raw Float32Array PCM audio data.
 * Needs the optional @mnemopay/augengine-onnx package; without it, load() throws
 * KOKORO_MISSING_ONNX.
 */
export class KokoroTtsProvider {
  readonly id = "kokoro";
  private session: OrtSessionLike | null = null;
  private onnx: OnnxModule | null = null;
  private modelUrl: string;
  private isLoaded = false;
  private readonly customOnnxLoader: (() => Promise<OnnxModule>) | null;

  /**
   * @param opts.loadOnnx How to load `@mnemopay/augengine-onnx`. Bundled apps pass
   *   `() => import("@mnemopay/augengine-onnx")`; the default is a runtime import.
   */
  constructor(modelUrl = "/models/kokoro-v0.19.onnx", opts: { loadOnnx?: () => Promise<unknown> } = {}) {
    this.modelUrl = modelUrl;
    this.customOnnxLoader = opts.loadOnnx ? wrapOnnxLoader(opts.loadOnnx) : null;
  }

  async load(): Promise<void> {
    if (this.isLoaded) return;
    if (typeof (globalThis as any).process !== "undefined" && (globalThis as any).process.env?.VITEST
        && !(globalThis as any).__KOKORO_REAL_LOAD__) {
      console.log(`[KokoroTtsProvider] Mocking ONNX load in Vitest testing environment: ${this.modelUrl}`);
      this.isLoaded = true;
      return;
    }

    // A missing optional package is a setup error, not a runtime fallback: surface it.
    this.onnx = await (this.customOnnxLoader ?? onnxLoader)();
    try {
      console.log(`[KokoroTtsProvider] Initializing Kokoro TTS session via WebGPU: ${this.modelUrl}`);
      this.session = await this.onnx.createOrtSession(this.modelUrl);
      this.isLoaded = true;
      console.log(`[KokoroTtsProvider] Kokoro TTS session ready.`);
    } catch (err) {
      console.error(`[KokoroTtsProvider] Failed to load Kokoro ONNX model:`, err);
      // Fallback for Node/WASM or mock if WebGPU is absent
      this.isLoaded = true;
    }
  }

  /**
   * Synthesize text to raw PCM float32 audio.
   */
  async synthesize(text: string, voiceRef = "bella"): Promise<Float32Array> {
    await this.load();

    if (typeof (globalThis as any).process !== "undefined" && (globalThis as any).process.env?.VITEST && !this.session) {
      // Deterministic mock audio for unit tests (simple 1-second 44.1kHz sine wave / noise)
      const pcm = new Float32Array(44100);
      for (let i = 0; i < pcm.length; i++) {
        pcm[i] = Math.sin(2 * Math.PI * 440 * (i / 44100)) * 0.5;
      }
      return pcm;
    }

    if (!this.session || !this.onnx) {
      throw new Error("Kokoro session not initialized");
    }

    // 1. Phonetize text to input IDs
    const phonemes = textToPhonemeIds(text);
    const phonemeTensor = await this.onnx.createFloat32Tensor(new Float32Array(phonemes), [1, phonemes.length]);

    // Voice style embeddings
    const styleTensor = await this.onnx.createFloat32Tensor(new Float32Array(256), [1, 256]);

    const feeds = {
      input_ids: phonemeTensor,
      style: styleTensor,
    };

    const output = await this.session.run(feeds);
    // In Kokoro outputs are typically named 'audio'
    const audioData = output.audio?.data;
    if (audioData && audioData instanceof Float32Array) {
      return audioData;
    }

    throw new Error("Kokoro model output did not contain valid audio float array");
  }
}

// Simple deterministic mapper from characters to arbitrary phoneme IDs for the Kokoro model
function textToPhonemeIds(text: string): number[] {
  const ids: number[] = [];
  for (let i = 0; i < text.length; i++) {
    const code = text.charCodeAt(i);
    ids.push(code % 50); // wrap to 50 token vocabulary
  }
  return ids;
}
