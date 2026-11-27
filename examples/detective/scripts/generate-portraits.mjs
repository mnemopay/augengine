// Generate noir portraits for a case on Replicate (google/nano-banana), in the
// same painted style as the first case's portraits, with a hard spend cap.
//
//   REPLICATE_API_TOKEN=... node scripts/generate-portraits.mjs calm            # one calm portrait per character
//   REPLICATE_API_TOKEN=... node scripts/generate-portraits.mjs moods           # evasive/angry/cracked edits of each calm
//   REPLICATE_API_TOKEN=... node scripts/generate-portraits.mjs one ebun angry  # redo a single image
//
// Prompts live in public/portraits/_source/deadair-prompts.json. Every call is
// appended to public/portraits/_source/deadair-ledger.json; the script refuses
// to start a call that could take the ledger past SPEND_CAP_USD.
import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import sharp from "sharp";

const here = dirname(fileURLToPath(import.meta.url));
const PUB = join(here, "..", "public", "portraits");
const SRC = join(PUB, "_source");
const RAW = join(here, "..", "assets-src", "portraits", "deadair");
const OUT = join(PUB, "deadair");
const LEDGER = join(SRC, "deadair-ledger.json");
const PROMPTS = JSON.parse(readFileSync(join(SRC, "deadair-prompts.json"), "utf8"));
const SPEND_CAP_USD = Number(process.env.SPEND_CAP_USD ?? 3);
/** Shared with the busts: the whole case-2 art budget. */
const PRICE_PER_IMAGE = 0.039;
const MODEL = "google/nano-banana";
const token = process.env.REPLICATE_API_TOKEN;
if (!token) throw new Error("REPLICATE_API_TOKEN not set");

const ledger = existsSync(LEDGER) ? JSON.parse(readFileSync(LEDGER, "utf8")) : { capUsd: SPEND_CAP_USD, totalUsd: 0, calls: [] };
const saveLedger = () => {
  ledger.totalUsd = Number(ledger.calls.reduce((n, c) => n + (c.costUsd ?? 0), 0).toFixed(4));
  writeFileSync(LEDGER, JSON.stringify(ledger, null, 1));
};

const dataUri = (file, mime) => `data:${mime};base64,${readFileSync(file).toString("base64")}`;

async function predict(input, label) {
  if (ledger.totalUsd + PRICE_PER_IMAGE > SPEND_CAP_USD) throw new Error(`spend cap: ledger at $${ledger.totalUsd}, cap $${SPEND_CAP_USD}`);
  let res;
  for (let retry = 0; retry < 6; retry++) {
    res = await fetch(`https://api.replicate.com/v1/models/${MODEL}/predictions`, {
    method: "POST",
    headers: { authorization: `Bearer ${token}`, "content-type": "application/json", prefer: "wait=60" },
    body: JSON.stringify({ input }),
  });
    if (res.status !== 429) break;
    await new Promise(resolve => setTimeout(resolve, Math.max(12, Number(res.headers.get("retry-after")) || 12) * 1000));
  }
  let p = await res.json();
  if (!res.ok) throw new Error(`${label}: HTTP ${res.status} ${JSON.stringify(p).slice(0, 200)}`);
  while (p.status === "starting" || p.status === "processing") {
    await new Promise((r) => setTimeout(r, 2000));
    p = await (await fetch(p.urls.get, { headers: { authorization: `Bearer ${token}` } })).json();
  }
  const ok = p.status === "succeeded";
  ledger.calls.push({ at: new Date().toISOString(), label, model: MODEL, prediction: p.id, status: p.status, costUsd: ok ? PRICE_PER_IMAGE : 0, predictTime: p.metrics?.predict_time ?? null });
  saveLedger();
  if (!ok) throw new Error(`${label}: ${p.status} ${p.error ?? ""}`);
  const url = Array.isArray(p.output) ? p.output[0] : p.output;
  return Buffer.from(await (await fetch(url)).arrayBuffer());
}

async function store(id, mood, buf) {
  mkdirSync(RAW, { recursive: true });
  mkdirSync(join(OUT, id), { recursive: true });
  writeFileSync(join(RAW, `${id}_${mood}.png`), await sharp(buf).png().toBuffer());
  await sharp(buf).resize(576, 768, { fit: "cover", position: "attention" }).webp({ quality: 80 }).toFile(join(OUT, id, `${mood}.webp`));
  console.log(`  -> portraits/deadair/${id}/${mood}.webp   (ledger $${ledger.totalUsd})`);
}

async function calm(id) {
  if (existsSync(join(OUT, id, "calm.webp")) && existsSync(join(RAW, id + "_calm.png"))) return;
  const subj = PROMPTS.subjects[id];
  const ref = join(PUB, PROMPTS.styleRefs[subj.ref]);
  const prompt = `${PROMPTS.calmInstruction}\n\nThe new person: ${subj.text}.\n\nStyle: ${PROMPTS.style}`;
  console.log(`calm ${id}`);
  await store(id, "calm", await predict({ prompt, image_input: [dataUri(ref, "image/webp")], aspect_ratio: "3:4", output_format: "png" }, `${id}_calm`));
}

async function mood(id, m) {
  if (cmd !== "one" && existsSync(join(OUT, id, m + ".webp"))) return;
  const src = join(RAW, `${id}_calm.png`);
  const prompt = `${PROMPTS.moodInstruction} ${PROMPTS.moods[m]}`;
  console.log(`${m} ${id}`);
  await store(id, m, await predict({ prompt, image_input: [dataUri(src, "image/png")], aspect_ratio: "3:4", output_format: "png" }, `${id}_${m}`));
}

const [cmd, a, b] = process.argv.slice(2);
const ids = Object.keys(PROMPTS.subjects);
if (cmd === "calm") for (const id of a ? [a] : ids) await calm(id);
else if (cmd === "moods") for (const id of (a ? [a] : ids).filter((i) => i !== PROMPTS.victim)) for (const m of Object.keys(PROMPTS.moods)) await mood(id, m);
else if (cmd === "one") await (b === "calm" ? calm(a) : mood(a, b));
else console.log("usage: calm [id] | moods [id] | one <id> <mood>");
console.log(`ledger total $${ledger.totalUsd}`);
