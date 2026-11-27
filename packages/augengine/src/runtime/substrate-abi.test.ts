import { describe, it, expect } from "vitest";
import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const SYMBOLS = [
  "ae_substrate_info",
  "ae_npc_create",
  "ae_npc_think",
  "ae_npc_destroy",
  "ae_replay_begin",
  "ae_replay_push",
  "ae_replay_seal",
  "ae_splat_load_spz",
  "ae_splat_draw",
  "ae_splat_destroy",
] as const;

const here = dirname(fileURLToPath(import.meta.url));
const headerPath = resolve(here, "../../../augengine-substrate/include/augengine_substrate.h");
// The native substrate package is not part of the public mirror (mnemopay/augengine);
// skip there rather than fail on a missing sibling package.
const header = existsSync(headerPath) ? readFileSync(headerPath, "utf8") : "";

describe.skipIf(!header)("Substrate native ABI", () => {
  it("matches the C header symbols", () => {
    for (const symbol of SYMBOLS) {
      expect(header).toContain(symbol);
    }
    expect(header).toContain("AUGENGINE_SUBSTRATE_ABI_VERSION 1");
    expect(header).toContain("AE_KERNEL_NPC");
    expect(header).toContain("AE_KERNEL_REPLAY");
    expect(header).toContain("AE_KERNEL_SPLAT");
  });
});
