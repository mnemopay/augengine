/**
 * Runtime glTF compression decoders — wire-up hooks for assets produced
 * by `tools/augengine-asset-pack/` (Track 4).
 *
 * The engine bundle itself stays free of third-party WASM. Application
 * shells (e.g. `examples/smoke-game/`) load the meshoptimizer +
 * Basis Universal transcoders on demand, then register them via the
 * `setKtx2Transcoder()` + `decodeGltfMeshopt(parsed, decoder)` paths.
 */

export {
  decodeGltfMeshopt,
  isMeshoptCompressed,
  type MeshoptCodec,
  type MeshoptBufferViewExtension,
  type MeshoptDecode,
} from "./meshopt-loader.js";

export {
  setKtx2Transcoder,
  configureKtx2TargetFormat,
  hasKtx2Textures,
  decodeGltfKtx2Images,
  estimateKtx2GpuBytes,
  extractKtx2Bytes,
  stubKtx2Transcoder,
  type DecodedKtx2,
  type Ktx2Transcoder,
  type TargetFormat,
} from "./ktx2-loader.js";
