/**
 * AI Artifact Manifest — content-addressable side-cache for AI-generated
 * artifacts (text, images, audio, scenes, models).
 *
 * Decision #4 of AUGENGINE-ARCHITECTURE.md: AI artifacts are stored OUT-OF-BAND
 * from the deterministic save state, keyed by `(persona_id, prompt_hash,
 * world_state_hash)`. The save file references manifest entries by hash;
 * the actual artifact bytes live in a content-addressable blob store.
 *
 * Why: enables save/load + multiplayer without freezing AI outputs into the
 * core state. On reload, hashes hit cache and the same text/image/audio
 * comes back deterministically.
 */

import type { AIArtifactRef, AIArtifactManifest } from "../ai/types.js";

export class ArtifactManifest {
  private byHash = new Map<string, AIArtifactRef>();
  private worldStateHash: string;

  constructor(worldStateHash = "") {
    this.worldStateHash = worldStateHash;
  }

  /** Set the world-state hash this manifest snapshot was captured against. */
  setWorldStateHash(hash: string): void {
    this.worldStateHash = hash;
  }

  /** Add an artifact reference. Returns the existing ref if hash already cached. */
  add(ref: AIArtifactRef): AIArtifactRef {
    const existing = this.byHash.get(ref.hash);
    if (existing) return existing;
    this.byHash.set(ref.hash, ref);
    return ref;
  }

  /** Look up by content-addressable hash. */
  get(hash: string): AIArtifactRef | undefined {
    return this.byHash.get(hash);
  }

  has(hash: string): boolean {
    return this.byHash.has(hash);
  }

  /** Iterate all artifacts. */
  *[Symbol.iterator](): IterableIterator<AIArtifactRef> {
    yield* this.byHash.values();
  }

  size(): number {
    return this.byHash.size;
  }

  /** Filter by artifact kind. */
  *byKind(kind: AIArtifactRef["kind"]): IterableIterator<AIArtifactRef> {
    for (const ref of this.byHash.values()) {
      if (ref.kind === kind) yield ref;
    }
  }

  /** Drop an artifact by hash. Returns true if it existed. */
  remove(hash: string): boolean {
    return this.byHash.delete(hash);
  }

  /** Wipe everything. */
  clear(): void {
    this.byHash.clear();
  }

  /** Serialize to the wire-format JSON shape. */
  toJson(): AIArtifactManifest {
    return {
      worldStateHash: this.worldStateHash,
      artifacts: [...this.byHash.values()],
    };
  }

  /** Rehydrate from JSON. Last entry wins on duplicate hashes (defensive). */
  static fromJson(j: AIArtifactManifest): ArtifactManifest {
    const m = new ArtifactManifest(j.worldStateHash);
    for (const a of j.artifacts ?? []) {
      m.byHash.set(a.hash, a);
    }
    return m;
  }
}

/**
 * Helper — compute a stable artifact hash for `(persona_id, prompt_hash, world_state_hash)`.
 * Uses sync FNV-1a-64 (no async, no crypto.subtle) so it works inside frame
 * loops if needed. For high-collision-resistance use cases (multiplayer
 * authority), upgrade to SHA-256 via the async helper in `state.ts`.
 */
export function artifactKey(input: {
  personaId: string;
  promptHash: string;
  worldStateHash: string;
}): string {
  const s = `${input.personaId}|${input.promptHash}|${input.worldStateHash}`;
  return fnv1a64(s);
}

/** FNV-1a 64-bit hash, hex output. Pure JS, sync, deterministic across platforms. */
export function fnv1a64(s: string): string {
  // 64-bit FNV-1a using two 32-bit halves so we don't need BigInt in hot paths.
  // Constants: offset basis 0xcbf29ce484222325, prime 0x100000001b3
  const PRIME_LO = 0x1b3;
  const PRIME_HI = 0x100;
  let lo = 0x84222325 >>> 0;
  let hi = 0xcbf29ce4 >>> 0;
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    lo = (lo ^ c) >>> 0;
    // 64-bit multiply: split each operand into 16-bit halves.
    const a0 = lo & 0xffff;
    const a1 = lo >>> 16;
    const a2 = hi & 0xffff;
    const a3 = hi >>> 16;
    const b0 = PRIME_LO & 0xffff;
    const b1 = PRIME_LO >>> 16;
    const b2 = PRIME_HI & 0xffff;
    const b3 = PRIME_HI >>> 16;

    let c0 = a0 * b0;
    let c1 = (c0 >>> 16) + a1 * b0 + a0 * b1;
    let c2 = (c1 >>> 16) + a2 * b0 + a1 * b1 + a0 * b2;
    let c3 = (c2 >>> 16) + a3 * b0 + a2 * b1 + a1 * b2 + a0 * b3;

    lo = (((c1 & 0xffff) << 16) | (c0 & 0xffff)) >>> 0;
    hi = (((c3 & 0xffff) << 16) | (c2 & 0xffff)) >>> 0;
  }
  return hi.toString(16).padStart(8, "0") + lo.toString(16).padStart(8, "0");
}
