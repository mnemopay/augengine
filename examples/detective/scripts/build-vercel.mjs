// Build a self-contained Vercel deployment folder for the detective game:
//   .vercel-deploy/            static site (vite build) + bundled API functions
//   .vercel-deploy/api/detective/{chat,health}.js   (esbuild, no workspace imports)
//
//   node scripts/build-vercel.mjs              # site served at /
//   DETECTIVE_BASE=/detective/ node scripts/build-vercel.mjs   # served under augengine.com/detective
//
// Nothing here deploys. See DEPLOY.md for the deploy commands.
import { execSync } from "node:child_process";
import { cpSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { build } from "esbuild";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const out = join(root, ".vercel-deploy");
rmSync(out, { recursive: true, force: true });
mkdirSync(out, { recursive: true });

execSync("npx vite build", { cwd: root, stdio: "inherit", env: { ...process.env } });
// With a sub-path base (e.g. /detective/) the built HTML asks for /detective/assets/…, so the static
// files must live under that folder in the deployment too. The API stays at /api/detective/*.
const base = (process.env.DETECTIVE_BASE || "/").replace(/^\/+|\/+$/g, "");
if (base && !/^[a-z0-9-]+$/i.test(base)) throw new Error(`DETECTIVE_BASE must be a simple path like /detective/, got ${process.env.DETECTIVE_BASE}`);
cpSync(join(root, "dist"), base ? join(out, base) : out, { recursive: true });

for (const fn of ["chat", "health"]) {
  await build({
    entryPoints: [join(root, "api", "detective", `${fn}.ts`)],
    outfile: join(out, "api", "detective", `${fn}.js`),
    bundle: true,
    platform: "node",
    format: "esm",
    target: "node20",
    banner: { js: "import { createRequire as __cr } from 'node:module'; const require = __cr(import.meta.url);" },
  });
}

writeFileSync(join(out, "package.json"), JSON.stringify({ name: "augengine-detective", private: true, type: "module" }, null, 2));
writeFileSync(
  join(out, "vercel.json"),
  JSON.stringify(
    {
      functions: { "api/detective/*.js": { maxDuration: 20 } },
      ...(base ? { redirects: [{ source: "/", destination: `/${base}/`, permanent: false }] } : {}),
      headers: [{ source: "/(.*)", headers: [{ key: "X-Content-Type-Options", value: "nosniff" }] }],
    },
    null,
    2,
  ),
);
console.log(`\nDeploy folder ready: ${out}`);
