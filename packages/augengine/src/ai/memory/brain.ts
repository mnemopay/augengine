/**
 * In-process brain — concrete `MnemoPayClientLike` backed by
 * `@mnemopay/sdk/recall` primitives (`localEmbed` + `cosineSimilarity`).
 *
 * Dogfoods the MnemoPay brain inside the engine runtime. No hosted service
 * required; embeddings + cosine search run in the same process (works in
 * browser, Node, Bun). Optional `persist` hook lets the caller serialize
 * memories to localStorage / IndexedDB / a save file between sessions —
 * that's how NPCs remember players across reloads.
 *
 * Subpath import (`@mnemopay/sdk/recall`) is deliberate. Importing the root
 * `@mnemopay/sdk` would auto-start the SDK's stdio MCP server and leak
 * `[mnemopay-mcp]` lines to stderr — catastrophic when AugEngine runs alongside
 * other MCP servers.
 *
 * `@mnemopay/sdk` is an optional peer dependency, so it is loaded lazily on the
 * first remember()/recall(); importing AugEngine never requires it. In a bundled
 * browser app, pass `loadRecall: () => import("@mnemopay/sdk/recall")`.
 */

import type { MnemoPayClientLike } from "./mnemopay.js";

type RecallModule = typeof import("@mnemopay/sdk/recall");
/** The two recall primitives the brain needs from `@mnemopay/sdk/recall`. */
export type BrainRecallModule = Pick<RecallModule, "localEmbed" | "cosineSimilarity">;

export const BRAIN_MISSING_SDK =
  "Install @mnemopay/sdk (>=1.5.0) to use the in-process brain. Under a bundler (Vite, webpack, esbuild), " +
  "also pass loadRecall: () => import(\"@mnemopay/sdk/recall\") to createInProcessBrain()";

/**
 * The specifier is computed (and marked `@vite-ignore`) on purpose: a literal
 * `import("@mnemopay/sdk/recall")` makes bundlers resolve the optional peer at
 * build time, so merely importing the AugEngine root barrel failed for anyone
 * who had not installed it. This default works in Node, Bun and with import
 * maps; bundled apps pass their own literal import via `loadRecall`.
 */
const RECALL_PACKAGE = ["@mnemopay", "sdk", "recall"].join("/");

let recallModule: Promise<BrainRecallModule> | null = null;
function loadDefaultRecall(): Promise<BrainRecallModule> {
  recallModule ??= (import(/* @vite-ignore */ RECALL_PACKAGE) as Promise<BrainRecallModule>).catch((err: unknown) => {
    recallModule = null;
    throw new Error(BRAIN_MISSING_SDK, { cause: err });
  });
  return recallModule;
}

export interface BrainMemoryRecord {
  id: string;
  namespace: string;
  content: string;
  tags: string[];
  metadata: Record<string, unknown>;
  ts: string;
  embedding: number[];
}

export interface InProcessBrainOptions {
  dimensions?: number;
  /** Notified whenever a memory is written. Use to persist to disk / localStorage. */
  onPersist?: (record: BrainMemoryRecord) => void;
  /** Hydrate memories from a previous session. Called once at construction. */
  initial?: BrainMemoryRecord[];
  /**
   * How to load `@mnemopay/sdk/recall`. Bundled apps should pass
   * `() => import("@mnemopay/sdk/recall")` so their bundler resolves it.
   * Defaults to a runtime import that works in Node and Bun.
   */
  loadRecall?: () => Promise<BrainRecallModule>;
}

export interface InProcessBrain extends MnemoPayClientLike {
  /** Dump all memories — useful for save/load + replay manifests. */
  snapshot(): BrainMemoryRecord[];
  /** Wipe everything in a namespace. Returns how many memories were removed. */
  forget(namespace: string): number;
  /** @deprecated Misspelling of {@link InProcessBrain.forget}; kept for 0.4.0 callers. Removed in 0.5. */
  augenginet(namespace: string): number;
}

export function createInProcessBrain(opts: InProcessBrainOptions = {}): InProcessBrain {
  const dim = opts.dimensions ?? 384;
  const store: BrainMemoryRecord[] = opts.initial ? [...opts.initial] : [];
  let counter = store.length;
  const customLoader = opts.loadRecall;
  let customRecall: Promise<BrainRecallModule> | null = null;
  const loadRecall = (): Promise<BrainRecallModule> => {
    if (!customLoader) return loadDefaultRecall();
    customRecall ??= customLoader().catch((err: unknown) => {
      customRecall = null;
      throw new Error(BRAIN_MISSING_SDK, { cause: err });
    });
    return customRecall;
  };

  function forget(namespace: string): number {
    let removed = 0;
    for (let i = store.length - 1; i >= 0; i--) {
      if (store[i]!.namespace === namespace) {
        store.splice(i, 1);
        removed++;
      }
    }
    return removed;
  }

  function asFloat32(arr: number[]): Float32Array {
    const out = new Float32Array(arr.length);
    for (let i = 0; i < arr.length; i++) out[i] = arr[i] ?? 0;
    return out;
  }

  return {
    snapshot() {
      return store.map((r) => ({ ...r, tags: [...r.tags], embedding: [...r.embedding] }));
    },

    forget,
    augenginet: forget,

    async remember(input) {
      const namespace = input.namespace ?? input.memoryId ?? "default";
      const { localEmbed } = await loadRecall();
      const vec = localEmbed(input.content, dim);
      const record: BrainMemoryRecord = {
        id: `mem-${counter++}`,
        namespace,
        content: input.content,
        tags: input.tags ?? [],
        metadata: input.metadata ?? {},
        ts:
          typeof input.metadata?.ts === "string"
            ? (input.metadata.ts as string)
            : new Date().toISOString(),
        embedding: Array.from(vec),
      };
      store.push(record);
      opts.onPersist?.(record);
      return { id: record.id };
    },

    async recall(input) {
      const namespace = input.namespace ?? input.memoryId ?? "default";
      const limit = input.limit ?? 5;
      const candidates = store.filter((r) => r.namespace === namespace);
      if (candidates.length === 0) return { memories: [] };

      const { localEmbed, cosineSimilarity } = await loadRecall();
      const queryVec = localEmbed(input.query, dim);
      const scored = candidates.map((r) => ({
        content: r.content,
        text: r.content,
        score: cosineSimilarity(queryVec, asFloat32(r.embedding)),
      }));
      scored.sort((a, b) => b.score - a.score);
      return { memories: scored.slice(0, limit) };
    },

    async summarize(input) {
      const lines = input.docs.slice(0, 5);
      const joined = lines.map((line, i) => `${i + 1}. ${line}`).join("\n");
      const budget = input.maxTokens ?? 1500;
      const sliced = joined.length > budget * 4 ? joined.slice(0, budget * 4) : joined;
      return { summary: sliced };
    },
  };
}
