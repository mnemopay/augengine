/**
 * @mnemopay/augengine — Browser-native engine for games + UGC + films.
 *
 * Public surface. Most callers should import from a sub-path
 * (`@mnemopay/augengine/scene`, `@mnemopay/augengine/runtime/live`, etc) for tree-
 * shaking, but the root barrel is convenient for prototypes.
 *
 * See AUGENGINE-ARCHITECTURE.md at the repo root for design rationale.
 */

export * from "./scene/index.js";
export * from "./runtime/types.js";
export * from "./ai/index.js";
export * from "./audio/index.js";
export * from "./characters/index.js";
export * from "./authoring/index.js";
export * from "./persistence/index.js";

// Runtime modules export browser-API-dependent classes; gated so
// importing the root barrel in a Node test doesn't crash. Callers that
// need the runtimes should import from "@mnemopay/augengine/runtime/live"
// or "@mnemopay/augengine/runtime/render".

export { AUGENGINE_VERSION } from "./version.js";

export * from './runtime/engine.js';
export * from './runtime/cinematic/index.js';
export * from './physics/index.js';
export * from './chemistry/index.js';
export * from './runtime/fixed-step.js';
export * from './runtime/simulation.js';
export * from './i18n/index.js';
export * from './telemetry/index.js';
export * from './bundler/index.js';
export * from './runtime/networking/index.js';

// Track 4 — runtime decoders for packed glTF (meshopt + KTX2/Basis).
// Host shells register transcoders via these — the engine bundle stays
// free of third-party WASM bytes (see runtime/loaders/ktx2-loader.ts).
export * from './runtime/loaders/index.js';

// Track 5 — game-native ECS components + systems. Opt-in: importing
// the root barrel pulls them in, but the sub-path
// `@mnemopay/augengine/runtime/systems` keeps tree-shaking tight for
// shells that only need a subset.
export * from './runtime/systems/index.js';

export * from './fighter/index.js';

// One-call glTF loading into a live renderer (no browser API touched at import time).
export {
  loadGltfIntoRenderer,
  loadLodIntoRenderer,
  type LoadGltfOptions,
  type LoadedGltfIntoRenderer,
} from './runtime/live/gltf.js';
