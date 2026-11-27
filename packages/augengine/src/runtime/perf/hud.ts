/**
 * CPU + GPU frame-budget HUD overlay — Track 6 (AUGENGINE-9-TRACK-PLAN.md).
 *
 * Top-right DOM panel showing live: FPS, CPU ms, GPU ms (timestamp-query
 * or CPU-fallback), face ms, asset bytes, plus p50/p95/p99 over the
 * latest published 600-frame window from `window.__augengineFrameBudget`.
 *
 * Toggled by `?hud=1` from the caller. Style is inlined to avoid a
 * build dependency and stay opt-in / zero-cost when disabled.
 *
 * Pure DOM — no Canvas/OffscreenCanvas paint loop. The overlay reads
 * the budget object on its own 250ms tick so it doesn't add per-frame
 * cost to the engine.
 */

interface BudgetLatest {
  fps?: number;
  p50Ms?: number;
  p95Ms?: number;
  p99Ms?: number;
  maxMs?: number;
  budgetMissCount?: number;
  count?: number;
  // Track 6
  gpu_ms?: number;
  gpu_ms_p95?: number;
  gpu_ms_source?: "timestamp-query" | "cpu-fallback";
  // Track 3
  face_ms?: number;
  face_ms_p95?: number;
  face_provider?: string;
  face_tick_mode?: "full" | "half";
}

interface BudgetGlobal {
  latest: BudgetLatest | null;
}

export interface HudHandle {
  /** Detach + remove DOM node. */
  stop: () => void;
  /** The DOM root, exposed for tests. */
  el: HTMLDivElement;
}

const HUD_ID = "augengine-perf-hud";
const HUD_TICK_MS = 250;

/**
 * Mount the HUD into the document body. Idempotent — calling twice
 * leaves a single overlay (the second call returns the existing handle).
 */
export function mountFrameBudgetHud(): HudHandle {
  if (typeof document === "undefined") {
    throw new Error("mountFrameBudgetHud: no document — call from browser only");
  }
  const existing = document.getElementById(HUD_ID) as HTMLDivElement | null;
  if (existing) {
    return {
      el: existing,
      stop: () => existing.remove(),
    };
  }

  const el = document.createElement("div");
  el.id = HUD_ID;
  el.setAttribute("data-augengine-perf-hud", "1");
  el.style.cssText = [
    "position:fixed",
    "top:8px",
    "right:8px",
    "z-index:2147483647",
    "background:rgba(0,0,0,0.78)",
    "color:#9efba8",
    "font:11px/1.35 ui-monospace,Menlo,Consolas,monospace",
    "padding:8px 10px",
    "border:1px solid rgba(158,251,168,0.25)",
    "border-radius:6px",
    "min-width:220px",
    "pointer-events:none",
    "white-space:pre",
    "box-shadow:0 4px 18px rgba(0,0,0,0.45)",
    "letter-spacing:0.2px",
  ].join(";");

  document.body.appendChild(el);

  const render = () => {
    const g = window as unknown as { __augengineFrameBudget?: BudgetGlobal };
    const b = g.__augengineFrameBudget?.latest ?? null;
    // Write only on change: the perf overlay must not itself cost frames
    // (an identical textContent write still re-lays-out and re-rasters it).
    const text = formatHud(b);
    if (el.textContent !== text) el.textContent = text;
  };

  render();
  const timer = window.setInterval(render, HUD_TICK_MS);

  return {
    el,
    stop: () => {
      window.clearInterval(timer);
      el.remove();
    },
  };
}

/**
 * Pure formatter — exported so the unit test can assert on the layout
 * without mounting the DOM.
 */
export function formatHud(b: BudgetLatest | null): string {
  if (!b) return "augengine perf\nwaiting for first window…";
  const fps = fmt(b.fps, 1);
  const p50 = fmt(b.p50Ms, 2);
  const p95 = fmt(b.p95Ms, 2);
  const p99 = fmt(b.p99Ms, 2);
  const maxV = fmt(b.maxMs, 2);
  const gpu = fmt(b.gpu_ms, 2);
  const gpuP95 = fmt(b.gpu_ms_p95, 2);
  const gpuSrc = b.gpu_ms_source === "timestamp-query"
    ? "ts"
    : b.gpu_ms_source === "cpu-fallback"
      ? "cpu"
      : "—";
  const face = b.face_ms !== undefined ? fmt(b.face_ms, 2) : "—";
  const faceProv = b.face_provider ?? "—";
  const miss = `${b.budgetMissCount ?? 0}/${b.count ?? 0}`;
  return [
    `augengine perf`,
    `fps     ${fps}`,
    `cpu p50 ${p50} ms`,
    `cpu p95 ${p95} ms`,
    `cpu p99 ${p99} ms`,
    `cpu max ${maxV} ms`,
    `gpu     ${gpu} ms (${gpuSrc})`,
    `gpu p95 ${gpuP95} ms`,
    `face    ${face} ms (${faceProv})`,
    `miss    ${miss}`,
  ].join("\n");
}

function fmt(v: number | undefined, digits: number): string {
  if (v === undefined || v === null || Number.isNaN(v)) return "—";
  return v.toFixed(digits);
}
