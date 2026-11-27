/**
 * Scene loader — one entry point for all three SceneSource kinds (decision #6).
 *
 * Authored: read JSON from disk/URL, validate, build a Scene.
 * Procgen: invoke a registered seeded generator → produces SceneSpec → same path.
 * Generated: prompt → AIScheduler dispatch → SceneSpec → same path. Cached by
 *            prompt hash; second call hits cache.
 *
 * Phase 1 implements `authored`. Procgen + generated stubs throw with a clear
 * message until phase 4.
 */

import type { SceneSpec } from "./types.js";
import { Scene } from "./scene.js";

export interface LoaderContext {
  /** Where authored scenes live. URL or file path prefix. */
  baseUrl?: string;
  /** Optional fetch override (for tests / non-browser envs). */
  fetch?: (url: string) => Promise<{ ok: boolean; json(): Promise<unknown> }>;
}

const DEFAULT_CTX: LoaderContext = {};

export async function loadScene(
  spec: SceneSpec | string,
  ctx: LoaderContext = DEFAULT_CTX,
): Promise<Scene> {
  if (typeof spec === "string") {
    const json = await fetchSpec(spec, ctx);
    validate(json);
    return new Scene(json);
  }
  validate(spec);
  return new Scene(spec);
}

async function fetchSpec(url: string, ctx: LoaderContext): Promise<SceneSpec> {
  const fetcher = ctx.fetch ?? globalThis.fetch;
  if (!fetcher) throw new Error("loadScene: no fetch implementation available");
  const fullUrl = ctx.baseUrl ? joinUrl(ctx.baseUrl, url) : url;
  const res = await fetcher(fullUrl);
  if (!res.ok) throw new Error(`loadScene: fetch ${fullUrl} failed`);
  const data = (await res.json()) as unknown;
  return data as SceneSpec;
}

function joinUrl(base: string, path: string): string {
  if (path.startsWith("http") || path.startsWith("/")) return path;
  return base.endsWith("/") ? base + path : `${base}/${path}`;
}

/** Minimal structural validation. Throws on malformed input. */
function validate(spec: unknown): asserts spec is SceneSpec {
  if (typeof spec !== "object" || spec === null) {
    throw new Error("SceneSpec: not an object");
  }
  const s = spec as Record<string, unknown>;
  if (s.schemaVersion !== 1) {
    throw new Error(`SceneSpec: unsupported schemaVersion (got ${String(s.schemaVersion)})`);
  }
  if (!Array.isArray(s.entities)) {
    throw new Error("SceneSpec: entities must be an array");
  }
  const seen = new Set<string>();
  for (const e of s.entities as Array<Record<string, unknown>>) {
    if (typeof e.id !== "string" || !e.id) {
      throw new Error("SceneSpec: entity missing id");
    }
    if (seen.has(e.id)) {
      throw new Error(`SceneSpec: duplicate entity id ${e.id}`);
    }
    seen.add(e.id);
    if (typeof e.components !== "object" || e.components === null) {
      throw new Error(`SceneSpec: entity ${e.id} missing components`);
    }
  }
}
