# `@mnemopay/augengine-mcp`

Model Context Protocol server exposing AugEngine engine primitives. Lets any
MCP-aware agent (Claude Code, Claude Desktop, Cursor, custom) author and drive
AugEngine scenes via typed tool calls.

## Why this exists

Decision #7 of `AUGENGINE-ARCHITECTURE.md` says: *"Observation bus = engine's
public agent API. Bots subscribe to the same channels cameras and speakers do."*
Exposing that surface as MCP tools is mostly a routing layer — agents already
speak to AugEngine through structured channels. MCP just gives them a standard
wire protocol.

Practical effect: in Claude Desktop or Cursor you say *"author me a scene with
two guards arguing near a fountain at night"*, and the agent calls
`augengine_create_scene` → `augengine_attach_persona` →
`augengine_dispatch_action` → `augengine_poll_observation` to build it and watch
it react.

## Install

```bash
npm run build --workspace=@mnemopay/augengine-mcp
```

Claude Code:

```bash
claude mcp add augengine -- node /abs/path/to/packages/augengine-mcp/dist/cli.js serve --stdio
```

Claude Desktop / Cursor:

```json
{
  "mcpServers": {
    "augengine": {
      "command": "node",
      "args": ["/abs/path/to/packages/augengine-mcp/dist/cli.js", "serve", "--stdio"]
    }
  }
}
```

`augengine-mcp tools` prints the manifest without starting a server.

## Tool surface

Tools are advertised with underscores because several MCP clients validate names
against `[A-Za-z0-9_-]+` and reject dots. The dotted names in `src/tools.ts`
remain valid aliases, so that file is still the typed source of truth.

| Tool | Status | Purpose |
|---|---|---|
| `augengine_create_scene` | ✅ | Prompt → SceneSpec, deterministic per (prompt, seed) |
| `augengine_add_entity` | ✅ | Add an Entity to an existing scene |
| `augengine_attach_persona` | ✅ | Attach a Persona component (decision #2) |
| `augengine_list_entities` | ✅ | Query by component tag / name regex, with world positions |
| `augengine_dispatch_action` | ✅ | Run a unified-grammar Action (decision #5) |
| `augengine_subscribe_observation` | ✅ | Open a buffered channel subscription (decision #7) |
| `augengine_poll_observation` | ✅ | Drain queued observation events |
| `augengine_describe_scene` | ✅ | Component census, verb set, predicates, effects |
| `augengine_export_godot` | ✅ | SceneSpec → Godot 4.4 `project.godot` + `Main.tscn` |
| `augengine_render_clip` | ⛔ | Needs WebGPU + WebCodecs; returns an explicit error over stdio |

### Where the intelligence lives

The engine's `AITask` union declares a `scene_gen` task, but **no provider
implements it**. That turns out not to matter over MCP: the caller *is* the
language model. `augengine_create_scene` therefore does deterministic layout —
it parses counts (`"two guards"`, `"5 robots"`), props, adjectives→traits, and
time-of-day→lighting — and the calling agent refines the result with
`add_entity` / `attach_persona`.

Consequences worth keeping: no API key, no network, no credit burn, and the same
`(prompt, seed)` always yields a byte-identical spec — which is what makes the
`cacheHit` flag mean anything.

### Action grammar

`ActionDispatcher` rejects unregistered verbs and unresolvable
precondition/effect ids, so the host installs a small library that is genuinely
implemented against the scene graph and bus:

- **Verbs** — `say`, `move`, `look`, `emote`, `use`, `talk`, plus every
  `AffordanceSet.affordances` and `Agent.permittedVerbs` entry in the scene.
  Adding an entity re-registers the verb set.
- **Preconditions** — `actor_exists`, `target_exists`, `target_in_range`
  (uses world transforms and `params.range`).
- **Effects** — `move_to_target` (respects `params.standoff`), `set_mood`,
  `emit_dialogue` (publishes to the `Dialogue` channel), `set_visible`.

Gameplay code that wants a stricter grammar builds its own dispatcher instead.

## Transport

Newline-delimited JSON-RPC 2.0 over stdio, implementing `initialize`,
`notifications/initialized`, `ping`, `tools/list` and `tools/call`. No
`@modelcontextprotocol/sdk` dependency — the package depends only on the engine.

Note that `@mnemopay/augengine-eval-server`'s `mcp-server.ts` exposes its
`eval.*` tools as bare JSON-RPC methods with no handshake and no `tools/call`;
real MCP clients cannot drive it. This server does the full handshake.

Observation subscriptions buffer rather than stream: MCP stdio has no
server-initiated event channel, so `subscribe_observation` opens a bounded queue
(1000 events, oldest dropped) and `poll_observation` drains it, reporting both
`remaining` and `dropped`.

## Tests

```bash
npm test --workspace=@mnemopay/augengine-mcp
```

34 specs covering the handshake, tool routing, dotted-name aliases, synthesis
determinism, prompt parsing, the action grammar, bus delivery, and the Godot
writer.

## Known gaps

- `augengine_render_clip` needs a browser host. Run the scene through
  `examples/smoke-game`, or export to Godot.
- The Godot `.tscn` writer has **not** been opened in Godot 4.4 — the format is
  written to spec and round-trips structurally in tests, but no import has been
  confirmed on a real editor.
- Scene state is per-process and in-memory. Restarting the server loses scenes.

## License

UNLICENSED while in incubation.
