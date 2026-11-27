/**
 * Scene barrel — types + Scene class + math + transform helpers.
 */

export * from "./types.js";
export * from "./math.js";
export * from "./transform.js";
export { Scene, validateSceneSpec } from "./scene.js";
export {
  Materials,
  MaterialRegistry,
  type MaterialKind,
  type UnlitMaterial,
  type PBRMaterial,
  type CelMaterial,
  type OutlineMaterial,
  type MToonMaterial,
  type SplatMaterial,
} from "./material.js";

export {
  parseGlb,
  loadGlb,
  readAccessor,
  extractPrimitive,
  extractSkin,
  extractAnimation,
  type GltfParsed,
  type LoadedGlb,
  type AugEnginePrimitiveFromGltf,
  type AugEngineMeshFromGltf,
  type AugEngineSkinFromGltf,
  type AugEngineAnimationFromGltf,
  type AugEngineAnimationTrack,
  type AugEngineMorphTarget,
  type AugEngineTextureSource,
  type AugEnginePbrMaterialFromGltf,
} from './loaders/gltf.js';

export {
  buildGltfSceneInstance,
  applyGltfAnimation,
  sampleGltfAnimationTrack,
  type GltfSceneInstance,
  type GltfSceneInstanceOptions,
} from "./loaders/gltf-instance.js";
