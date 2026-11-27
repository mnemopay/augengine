/** Persistence barrel — save/load + AI artifact manifest. */

export {
  ArtifactManifest,
  artifactKey,
  fnv1a64,
} from "./manifest.js";

export {
  captureSnapshot,
  restoreSnapshot,
  canonicalize,
  worldStateHash,
  sha256Hex,
  diffEntities,
  type SceneSnapshot,
  type SnapshotOptions,
} from "./state.js";

export {
  save,
  load,
  serialize,
  deserialize,
  serializeGzip,
  deserializeGzip,
  type SaveFile,
  type SaveOptions,
} from "./save.js";
