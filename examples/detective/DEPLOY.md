# Deploying the detective game

Deployed 2026-11-23 at https://augengine.com/detective (Option B, on the shared key). These are the steps used.

## Option A (recommended): its own Vercel project, linked from augengine.com

```bash
cd examples/detective
node scripts/build-vercel.mjs                      # → .vercel-deploy/ (static site + bundled API functions)
cd .vercel-deploy
vercel link --yes --project augengine-detective --scope <your-vercel-team>
vercel env add OPENROUTER_API_KEY production       # paste the key; never commit it
vercel deploy --prod --yes --scope <your-vercel-team>
```

Then check `https://augengine-detective.vercel.app/api/detective/health` returns `{"llm":true,...}`
and play once in model mode and once with `?offline=1`.

## Option B: serve at augengine.com/detective

Build with the sub-path base and deploy as in A:

```bash
DETECTIVE_BASE=/detective/ node scripts/build-vercel.mjs
```

Then add two rewrites to the augengine.com site's `vercel.json` (augengine-site repo) and redeploy that site:

```json
{ "source": "/detective/:path*", "destination": "https://augengine-detective.vercel.app/detective/:path*" },
{ "source": "/api/detective/:path*", "destination": "https://augengine-detective.vercel.app/api/detective/:path*" }
```

## Cost controls

- Model: `openai/gpt-4o-mini` (`DETECTIVE_MODEL` to change). Measured about $0.00006 per line.
- The live deployment uses an OpenRouter key shared with other apps. `server/key-guard.ts` reads
  OpenRouter's own usage for that key (`GET /api/v1/key`, cached 30 s) and closes the model path when the key
  **as a whole** reaches any of these. They include the other apps' spend, so they can only close early:
  - `DETECTIVE_KEY_DAILY_USD` (default $3.00 spent today)
  - `DETECTIVE_KEY_MONTHLY_USD` (default $12.00 spent this month)
  - `DETECTIVE_KEY_FLOOR_USD` (default: stop when $10.00 or less remains on the key's limit)
- Each server instance also stops after `DETECTIVE_INSTANCE_DAILY_USD` (default $1.00) of game spend per UTC day,
  plus the per-session cap `DETECTIVE_SESSION_USD` (default $0.05) and 30 requests/min per IP.
- If the key status can't be read, the gate closes. Closed → `/api/detective/chat` answers 429 `resting`,
  `/api/detective/health` reports `resting` and the budget, and the player gets scripted dialogue with a
  "storyteller is resting" note for the rest of the session.
- `DETECTIVE_TEST_SECRET` (16+ chars) enables the `x-dtv-force-offline` header for live checks; without it the
  header is ignored.
