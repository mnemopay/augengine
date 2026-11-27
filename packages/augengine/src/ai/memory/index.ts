/** Memory adapters — MnemoPay-backed + null fallback. */

export {
  createMnemoPayMemoryAdapter,
  type MnemoPayClientLike,
  type MnemoPayMemoryOptions,
} from "./mnemopay.js";

export {
  createInProcessBrain,
  BRAIN_MISSING_SDK,
  type BrainRecallModule,
  type InProcessBrain,
  type InProcessBrainOptions,
  type BrainMemoryRecord,
} from "./brain.js";

export {
  createBrainHttpClient,
  compositeBrain,
  type BrainHttpClientOptions,
} from "./brain-http.js";

export {
  startPersonaMemoryRecorder,
  type PersonaMemoryRecorder,
  type PersonaMemoryRecorderOptions,
} from "./recorder.js";

export { BrainMemoryProvider } from "./provider.js";

// NULL_PERSONA_MEMORY is exported via the parent ai/index.ts barrel.
