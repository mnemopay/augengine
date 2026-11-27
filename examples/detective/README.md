# The Wrenfield Observatory — AugEngine gate 4

A small, finished detective game (15–25 minutes) that proves AugEngine's pitch:
**AI characters with memory and guardrails, acting only through deterministic
game rules, drivable by agents over MCP.**

Professor Aurel Vance is dead at his telescope. Four people were in the house.
Search five rooms, question the household, show them what you find — some are
lying, and the right evidence cracks them — then make **one** accusation: who,
how, why, backed by evidence.

## Play locally

```bash
# from the repo root, once
npm install
npm run build -w @mnemopay/augengine -w @mnemopay/augengine-detective

cd examples/detective
npx vite --port 5193        # http://localhost:5193
```

- `?offline=1` forces scripted dialogue (no model, no network). Same facts.
- Model dialogue needs `OPENROUTER_API_KEY` in the environment (or, for local
  dev only, `~/openrouter-key.txt`). The key is only ever read server-side.
- Keys 1–5 jump between rooms. Chrome/Edge with WebGPU renders the house; without
  WebGPU the case is still fully playable from the panels.

## How it's built

| Layer | Where | What it guarantees |
|---|---|---|
| Case + truth | `packages/augengine-detective/src/case.ts` | Original setting. Killer, method, motive, timeline, 8 clues, 4 testimonies are fixed data. |
| Rule engine | `rules.ts` | Every action (`move`, `examine`, `ask`, `present`, `accuse`) is validated and decides the *gist* a suspect conveys. Lies crack only when specific evidence is presented. |
| Dialogue | `dialogue.ts` | The model may only restate the gist. `validateLine` rejects confessions, naming a killer, meta/AI talk, invented clock times, a suspect's forbidden terms, and any clue the player hasn't found. Rejected lines are replaced by the gist. |
| Server proxy | `examples/detective/server/chat-api.ts` | Accepts only gists the rules can produce (`checkPayload`) — it can't be used as a general LLM proxy. Per-session and global cost caps, rate limit, 15 s timeout, streamed text. |
| Memory | `memory.ts` + `src/memory-store.ts` | Each suspect remembers per player across sessions (IndexedDB → localStorage → RAM): what you asked, showed, and whom you accused. |
| MCP | `packages/augengine-mcp/src/detective-tools.ts` | `detective_new_session`, `detective_status`, `detective_act`. Agents play the same rules; truth stays hidden until the accusation (or `AUGENGINE_DETECTIVE_DEBUG=1`). Memory persists with `AUGENGINE_DETECTIVE_MEMORY=/path/memory.json`. |

## Tests

```bash
cd packages/augengine-detective && npx vitest run      # rules, lies, guardrails, memory, payload check
cd packages/augengine-mcp && npx vitest run            # includes an agent playing the case over MCP
cd examples/detective && npx playwright test --config playwright.config.ts   # headed Chrome: win, loss + memory, perf, model mode
cd examples/detective && node scripts/adversarial.ts   # 10 attacks against the real model, cost-capped
```

Results and screenshots land in `results/detective/`.
