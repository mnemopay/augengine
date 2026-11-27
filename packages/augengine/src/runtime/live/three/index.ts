/**
 * Backend B — three.js WebGPURenderer behind the LiveRenderer adapter.
 * Separate subpath (`@mnemopay/augengine/runtime/live/three`) so apps that do
 * not opt in never bundle three.js. Requires `three` (optional peer).
 */
export { ThreeLiveRenderer, type ThreeLiveRendererOptions } from "./renderer.js";
