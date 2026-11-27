/**
 * Guards issue #26: `import … from "@mnemopay/augengine"` failed under Vite
 * because a dist file contained a literal `import("@xenova/transformers")`.
 * Bundlers (Vite's dep optimizer, esbuild, Rollup) resolve every literal
 * import — static or dynamic — at build time, so an optional package must never
 * appear as a literal specifier in anything the published entry points reach.
 *
 * Scans the BUILT output (dist/) with es-module-lexer, the same lexer Vite
 * uses, so comments and strings that merely mention a package do not count.
 * Run `npm run build` first.
 */
import { beforeAll, describe, expect, it } from "vitest";
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { dirname, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { init, parse } from "es-module-lexer";

const PKG_DIR = join(dirname(fileURLToPath(import.meta.url)), "..");
const DIST = join(PKG_DIR, "dist");
const pkg = JSON.parse(readFileSync(join(PKG_DIR, "package.json"), "utf8")) as {
  dependencies: Record<string, string>;
  exports: Record<string, { import: string }>;
};

/** Packages that are optional (peer or not declared at all). */
const OPTIONAL = [
  "@xenova/transformers",
  "@huggingface/transformers",
  "@mnemopay/augengine-onnx",
  "@mnemopay/sdk",
  "three",
  "onnxruntime-web",
  "onnxruntime-node",
  "onnxruntime-common",
  "sharp",
];
const THREE_SUBPATH_DIR = join(DIST, "runtime", "live", "three") + sep;

const pkgName = (spec: string) =>
  spec.startsWith("@") ? spec.split("/").slice(0, 2).join("/") : spec.split("/")[0]!;
const isBare = (spec: string) => !spec.startsWith(".") && !spec.startsWith("/") && !spec.startsWith("node:");

function walk(dir: string, out: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) walk(p, out);
    else if (p.endsWith(".js")) out.push(p);
  }
  return out;
}

interface Spec { spec: string; dynamic: boolean }
const cache = new Map<string, Spec[]>();
function importsOf(file: string): Spec[] {
  let hit = cache.get(file);
  if (!hit) {
    const [imports] = parse(readFileSync(file, "utf8"), file);
    // n is undefined for non-literal dynamic imports (computed specifiers):
    // bundlers cannot resolve those, which is exactly the point.
    hit = imports.filter((i) => typeof i.n === "string").map((i) => ({ spec: i.n!, dynamic: i.d > -1 }));
    cache.set(file, hit);
  }
  return hit;
}

/** Every file reachable from `entry` via relative imports, and the bare specifiers they use. */
function reach(entry: string): { files: Set<string>; bare: Map<string, string> } {
  const files = new Set<string>();
  const bare = new Map<string, string>();
  const stack = [entry];
  while (stack.length) {
    const f = stack.pop()!;
    if (files.has(f)) continue;
    files.add(f);
    for (const { spec } of importsOf(f)) {
      if (isBare(spec)) {
        if (!bare.has(spec)) bare.set(spec, relative(DIST, f));
      } else if (spec.startsWith(".")) {
        stack.push(resolve(dirname(f), spec));
      }
    }
  }
  return { files, bare };
}

describe("dist: optional packages are never literal imports (issue #26)", () => {
  beforeAll(async () => {
    if (!existsSync(join(DIST, "index.js"))) {
      throw new Error("dist/ is missing: run `npm run build` in packages/augengine before this test");
    }
    await init;
  });

  it("no dist file imports an optional package literally (three only inside the ./runtime/live/three sub-path)", () => {
    const offenders: string[] = [];
    for (const file of walk(DIST)) {
      for (const { spec, dynamic } of importsOf(file)) {
        if (!isBare(spec) || !OPTIONAL.includes(pkgName(spec))) continue;
        if (pkgName(spec) === "three" && file.startsWith(THREE_SUBPATH_DIR)) continue;
        offenders.push(`${relative(DIST, file)} -> ${dynamic ? "import()" : "import"} "${spec}"`);
      }
    }
    expect(offenders).toEqual([]);
  });

  it("the root entry reaches only declared dependencies", () => {
    const { files, bare } = reach(join(DIST, "index.js"));
    expect(files.size).toBeGreaterThan(50); // sanity: the walk really followed the barrel
    const undeclared = [...bare].filter(([s]) => !(pkgName(s) in pkg.dependencies)).map(([s, f]) => `${f} -> ${s}`);
    expect(undeclared).toEqual([]);
    expect([...files].some((f) => f.startsWith(THREE_SUBPATH_DIR))).toBe(false);
  });

  it("every published sub-path reaches only declared dependencies (three allowed for ./runtime/live/three)", () => {
    const problems: string[] = [];
    for (const [subpath, target] of Object.entries(pkg.exports)) {
      const allowed = new Set(Object.keys(pkg.dependencies));
      if (subpath === "./runtime/live/three") allowed.add("three");
      const { bare } = reach(join(PKG_DIR, target.import));
      for (const [s, f] of bare) if (!allowed.has(pkgName(s))) problems.push(`${subpath}: ${f} -> ${s}`);
    }
    expect(problems).toEqual([]);
  });
});
