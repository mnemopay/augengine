/**
 * AIScheduler — async, frame-budgeted, fallback-always inference dispatcher.
 *
 * Decision #1 of AUGENGINE-ARCHITECTURE.md. Game logic NEVER blocks on inference.
 * Every AI call has a hard deadline. If the deadline passes, the fallback
 * fires synchronously and the result is returned. If/when the inference
 * completes after the deadline, the result is dropped (or surfaced as a
 * stale-result event for opportunistic caching).
 *
 * Phase 1 ships the scheduler infrastructure. Phase 4 wires real LLM/TTS/
 * STT/etc. providers via registered handlers. Phase 5 ships in-engine usage.
 */

import type { AIRequest, AITask } from "./types.js";

/** Provider that actually runs an AI task. Registered per AITask kind. */
export type AIProvider<T = unknown> = (
  payload: unknown,
  signal: AbortSignal,
) => Promise<T>;

interface CacheEntry {
  result: unknown;
  storedAt: number;
}

export interface AISchedulerOptions {
  /** Maximum cache entries (LRU). Set to 0 to disable caching. */
  cacheCap?: number;
  /** Cache TTL in ms. Default 5 minutes. */
  cacheTtlMs?: number;
  /** Hook called for telemetry on every dispatch outcome. */
  onMetric?: ((metric: AIMetric) => void) | undefined;
}

export interface AIMetric {
  task: AITask;
  outcome: "hit" | "miss" | "fallback" | "error";
  elapsedMs: number;
  cacheKey?: string | undefined;
}

export class AIScheduler {
  private providers = new Map<AITask, AIProvider>();
  private cache = new Map<string, CacheEntry>();
  private cacheCap: number;
  private cacheTtlMs: number;
  private onMetric?: ((m: AIMetric) => void) | undefined;
  private inflight = new Map<string, Promise<unknown>>();

  constructor(opts: AISchedulerOptions = {}) {
    this.cacheCap = opts.cacheCap ?? 256;
    this.cacheTtlMs = opts.cacheTtlMs ?? 5 * 60 * 1000;
    this.onMetric = opts.onMetric;
  }

  register<T>(task: AITask, provider: AIProvider<T>): void {
    this.providers.set(task, provider as AIProvider<unknown>);
  }

  /**
   * Dispatch a request. Resolves with either the real provider result OR
   * the request's deterministic fallback (never rejects from the caller's
   * perspective for missed deadlines or provider errors).
   */
  async dispatch<T>(req: AIRequest<T>): Promise<T> {
    const start = performance.now();

    // Cache hit short-circuits all timeouts.
    if (req.cacheKey) {
      const hit = this.cache.get(req.cacheKey);
      if (hit && performance.now() - hit.storedAt < this.cacheTtlMs) {
        this.metric({
          task: req.task,
          outcome: "hit",
          elapsedMs: performance.now() - start,
          cacheKey: req.cacheKey,
        });
        return hit.result as T;
      }
    }

    // Coalesce duplicate in-flight requests on the same cache key.
    if (req.cacheKey) {
      const existing = this.inflight.get(req.cacheKey);
      if (existing) {
        try {
          const r = (await this.raceTimeout(existing, req.timeoutMs)) as T;
          this.metric({
            task: req.task,
            outcome: "hit",
            elapsedMs: performance.now() - start,
            cacheKey: req.cacheKey,
          });
          return r;
        } catch {
          // Fall through to fallback below.
        }
      }
    }

    const provider = this.providers.get(req.task);
    if (!provider) {
      const fb = req.fallback();
      this.metric({ task: req.task, outcome: "fallback", elapsedMs: performance.now() - start });
      return fb;
    }

    const ac = new AbortController();
    const promise = provider(req.payload, ac.signal);
    if (req.cacheKey) this.inflight.set(req.cacheKey, promise);

    try {
      const result = (await this.raceTimeout(promise, req.timeoutMs, ac)) as T;
      if (req.cacheKey) {
        this.cacheStore(req.cacheKey, result);
      }
      this.metric({
        task: req.task,
        outcome: "miss",
        elapsedMs: performance.now() - start,
        cacheKey: req.cacheKey,
      });
      return result;
    } catch (err) {
      const fb = req.fallback();
      this.metric({
        task: req.task,
        outcome: err instanceof TimeoutError ? "fallback" : "error",
        elapsedMs: performance.now() - start,
        cacheKey: req.cacheKey,
      });
      return fb;
    } finally {
      if (req.cacheKey) this.inflight.delete(req.cacheKey);
    }
  }

  private async raceTimeout<T>(
    p: Promise<T>,
    timeoutMs: number,
    ac?: AbortController,
  ): Promise<T> {
    return await new Promise<T>((resolve, reject) => {
      const t = setTimeout(() => {
        ac?.abort();
        reject(new TimeoutError());
      }, timeoutMs);
      p.then(
        (r) => {
          clearTimeout(t);
          resolve(r);
        },
        (e) => {
          clearTimeout(t);
          reject(e instanceof Error ? e : new Error(String(e)));
        },
      );
    });
  }

  private cacheStore(key: string, result: unknown): void {
    if (this.cacheCap <= 0) return;
    if (this.cache.size >= this.cacheCap) {
      // Naive LRU — drop oldest insertion.
      const firstKey = this.cache.keys().next().value;
      if (firstKey !== undefined) this.cache.delete(firstKey);
    }
    this.cache.set(key, { result, storedAt: performance.now() });
  }

  private metric(m: AIMetric): void {
    this.onMetric?.(m);
  }
}

class TimeoutError extends Error {
  constructor() {
    super("AIRequest deadline exceeded");
    this.name = "TimeoutError";
  }
}
