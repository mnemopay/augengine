// Writes src/version.ts from package.json so AUGENGINE_VERSION always matches
// the published version. Runs before every build (see "build" in package.json).
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const { version } = JSON.parse(readFileSync(join(root, "package.json"), "utf8"));
const out = join(root, "src", "version.ts");
const body =
  "// Generated from package.json by scripts/write-version.mjs. Do not edit.\n" +
  `export const AUGENGINE_VERSION = ${JSON.stringify(version)};\n`;
let current = "";
try { current = readFileSync(out, "utf8"); } catch {}
if (current !== body) writeFileSync(out, body);
console.log(`[write-version] AUGENGINE_VERSION = ${version}`);
