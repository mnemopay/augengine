# Suspect portraits and busts — sources, licences, spend

All characters are original (The Wrenfield Observatory case). No real person's
likeness and no existing franchise characters were requested or used.

## Portraits (`portraits/<id>/<mood>.webp`)

- **Model:** Google Gemini 3.1 Flash Image (`google/gemini-3.1-flash-image`) via
  OpenRouter, 2026-11-23. fal.ai (FLUX) was the plan, but the fal account had no
  balance (HTTP 403 "Exhausted balance"), so OpenRouter's image model was used.
- **Calm portraits** (one per character, plus Professor Vance): text-to-image from
  the character brief + the shared style below. Briefs are derived from the case
  file personas (`packages/augengine-detective/src/case.ts`). Exact prompts are in
  `_source/prompts.json`.
- **Moods** (evasive, angry, cracked): image editing of the character's own calm
  portrait ("keep exactly the same person, face, clothing, framing, lighting and
  style; change only the expression to …"), which keeps identity consistent.
  The first angry set added heavy red blush; it was regenerated with "natural skin
  tone, no added redness".
- **Style:** painted noir — oil brushwork, 1930s, single warm key light upper left,
  deep shadow, muted umber/slate/ochre palette, plain dark background, no text.
- **Files:** resized to 576×768, WebP q80 (17 files, ~350 KB total).
- **Terms:** generated content via OpenRouter / Google's Gemini API terms. Google's
  generative AI terms let the user use outputs they generate, subject to the
  prohibited-use policy; check the current terms before commercial distribution.
  Images carry Google's invisible SynthID watermark.

## Busts (`busts/<id>.glb`)

- **Model:** Microsoft TRELLIS (MIT licence, github.com/microsoft/TRELLIS) via
  Replicate `firtoz/trellis` (version `e8f6c452…`), image-to-3D from each calm
  portrait, 2026-11-23. Example prediction: `z7efys794xrna0d21d6vcden6g` (Desmond).
- **Raw outputs:** `examples/detective/assets-src/busts/<id>_bust.glb`
  (14–30k triangles, 1 MB PNG texture each).
- **Game assets:** `node scripts/prepare-busts.mjs` simplifies each to 4,000
  triangles with meshoptimizer (UV seams locked, so no holes), recomputes smooth
  normals and re-encodes the texture as JPEG (~200–240 KB per bust).
  Measured on the Intel UHD laptop: 8k-triangle busts dropped the parlour from 60
  to 30 fps; at 4k the parlour holds 60 fps (16.7 ms median, same as no busts).
  At 3k without locked seams the mesh developed visible holes.
- The busts are static: TRELLIS meshes have no face rig, so expressions live in
  the dialogue-panel portraits.

## Spend (hard cap $5)

| Item | Calls | Cost |
|---|---|---|
| Portrait generation + edits (OpenRouter, exact from account usage) | 22 | $1.4886 |
| TRELLIS busts (Replicate, estimated from GPU seconds × A100 rate) | 4 | ~$0.13 |
| **Total** | | **~$1.62** |

The OpenRouter key's usage rose by $1.4887 over the whole task (the extra
$0.0001 is the game's own model-mode e2e test). `_source/ledger.json` lists every call.
