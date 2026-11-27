/**
 * Global spend guard for the public detective game.
 *
 * The OpenRouter key is shared with other apps (Dele, MnemoPay, the social
 * manager), and Vercel functions keep no memory between cold instances, so an
 * in-process budget alone cannot cap total spend. This guard asks OpenRouter
 * itself (GET /api/v1/key) how much the key has spent and closes the model
 * path when the key as a whole crosses any of these lines:
 *
 *   - spent today        >= dailyCapUsd    (DETECTIVE_KEY_DAILY_USD,   default 3.00)
 *   - spent this month   >= monthlyCapUsd  (DETECTIVE_KEY_MONTHLY_USD, default 12.00)
 *   - limit remaining    <= floorUsd       (DETECTIVE_KEY_FLOOR_USD,   default 10.00)
 *
 * These are key-wide (they include the other apps' spend), so they can only
 * close early, never late: the game stops before it can eat into what the
 * other apps need. On top of that each instance keeps its own per-day game
 * budget (DETECTIVE_INSTANCE_DAILY_USD, default 1.00) from the usage
 * OpenRouter reports per call.
 *
 * Any failure to read the key status counts as closed (fail safe).
 */

export interface KeyStatus {
  usageDaily: number;
  usageMonthly: number;
  limitRemaining: number | null;
}

export interface GuardLimits {
  dailyCapUsd: number;
  monthlyCapUsd: number;
  floorUsd: number;
  instanceDailyUsd: number;
  cacheMs: number;
}

export type GuardDecision = { open: true } | { open: false; reason: string };

export type StatusFetcher = () => Promise<KeyStatus>;

export function limitsFromEnv(env: NodeJS.ProcessEnv = process.env): GuardLimits {
  const num = (v: string | undefined, d: number) => {
    const n = Number(v);
    return Number.isFinite(n) && n >= 0 ? n : d;
  };
  return {
    dailyCapUsd: num(env.DETECTIVE_KEY_DAILY_USD, 3),
    monthlyCapUsd: num(env.DETECTIVE_KEY_MONTHLY_USD, 12),
    floorUsd: num(env.DETECTIVE_KEY_FLOOR_USD, 10),
    instanceDailyUsd: num(env.DETECTIVE_INSTANCE_DAILY_USD, 1),
    cacheMs: num(env.DETECTIVE_KEY_CACHE_MS, 30_000),
  };
}

export function openRouterStatusFetcher(apiKey: string, fetchImpl: typeof fetch = fetch): StatusFetcher {
  return async () => {
    const r = await fetchImpl("https://openrouter.ai/api/v1/key", {
      headers: { authorization: `Bearer ${apiKey}` },
      signal: AbortSignal.timeout(5_000),
    });
    if (!r.ok) throw new Error(`key status ${r.status}`);
    const d = ((await r.json()) as { data?: Record<string, unknown> }).data ?? {};
    const n = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : NaN);
    const status: KeyStatus = {
      usageDaily: n(d.usage_daily),
      usageMonthly: n(d.usage_monthly),
      limitRemaining: d.limit_remaining === null || d.limit_remaining === undefined ? null : n(d.limit_remaining),
    };
    if (Number.isNaN(status.usageDaily) || Number.isNaN(status.usageMonthly) || Number.isNaN(status.limitRemaining ?? 0)) {
      throw new Error("key status malformed");
    }
    return status;
  };
}

const dayKey = (now: number) => new Date(now).toISOString().slice(0, 10);

export class KeyGuard {
  private cached: { at: number; decision: GuardDecision; status: KeyStatus | null } | null = null;
  private inflight: Promise<GuardDecision> | null = null;
  private spentToday = 0;
  private day: string;

  constructor(
    private readonly fetchStatus: StatusFetcher,
    readonly limits: GuardLimits = limitsFromEnv(),
    private readonly now: () => number = Date.now,
  ) {
    this.day = dayKey(now());
  }

  /** Game spend this instance has recorded today (UTC). */
  get instanceSpentToday(): number {
    this.rollDay();
    return this.spentToday;
  }

  get lastStatus(): KeyStatus | null {
    return this.cached?.status ?? null;
  }

  record(costUsd: number): void {
    this.rollDay();
    if (Number.isFinite(costUsd) && costUsd > 0) this.spentToday += costUsd;
  }

  async check(): Promise<GuardDecision> {
    this.rollDay();
    if (this.spentToday >= this.limits.instanceDailyUsd) {
      return { open: false, reason: "instance daily budget reached" };
    }
    const t = this.now();
    if (this.cached && t - this.cached.at < this.limits.cacheMs) return this.cached.decision;
    if (!this.inflight) {
      this.inflight = this.refresh().finally(() => {
        this.inflight = null;
      });
    }
    return this.inflight;
  }

  private async refresh(): Promise<GuardDecision> {
    let status: KeyStatus | null = null;
    let decision: GuardDecision;
    try {
      status = await this.fetchStatus();
      decision = this.decide(status);
    } catch (err) {
      decision = { open: false, reason: `key status unavailable: ${(err as Error).message.slice(0, 60)}` };
    }
    this.cached = { at: this.now(), decision, status };
    return decision;
  }

  decide(s: KeyStatus): GuardDecision {
    const L = this.limits;
    if (s.usageDaily >= L.dailyCapUsd) return { open: false, reason: "key daily cap reached" };
    if (s.usageMonthly >= L.monthlyCapUsd) return { open: false, reason: "key monthly cap reached" };
    if (s.limitRemaining !== null && s.limitRemaining <= L.floorUsd) return { open: false, reason: "key floor reached" };
    return { open: true };
  }

  private rollDay(): void {
    const d = dayKey(this.now());
    if (d !== this.day) {
      this.day = d;
      this.spentToday = 0;
    }
  }
}
