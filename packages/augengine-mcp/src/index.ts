/**
 * @mnemopay/augengine-mcp — public surface.
 *
 * `tools.ts` pins the typed tool contracts; `server.ts` is the runtime that
 * serves them over MCP stdio. Embed the server in another process with
 * `new AugEngineMcpServer()` + `handle()`, or let `startMcpServer()` wire it
 * to stdin/stdout the way the CLI does.
 */

export type * from "./tools.js";

export {
  AugEngineMcpServer,
  startMcpServer,
  TOOL_MANIFEST,
  SERVER_NAME,
  SERVER_VERSION,
  type McpServerOptions,
} from "./server.js";

export { SceneRegistry, DEFAULT_QUEUE_CAP, type SceneHandle } from "./registry.js";

export {
  synthesiseFromPrompt,
  hashPrompt,
  type SynthOptions,
  type SynthResult,
} from "./scene-synth.js";

export {
  buildDispatcher,
  BASE_VERBS,
  BUILTIN_PREDICATE_IDS,
  BUILTIN_EFFECT_IDS,
} from "./grammar.js";

export {
  exportGodotProject,
  renderTscn,
  type GodotExportResult,
} from "./godot-export.js";

export const AUGENGINE_MCP_VERSION = "0.1.0";
