/**
 * Structured proof-room perf sample. Headed capture fills these fields;
 * PERFORMANCE.md stays `pending` until a real adapter writes a sample.
 */

export interface ProofCaptureSample {
  capturedAt: string;
  adapter: {
    vendor: string;
    device: string;
    description: string;
    isFallbackAdapter: boolean;
  };
  resolution: { width: number; height: number };
  fps: number;
  p50Ms: number;
  p99Ms: number;
  budgetMissCount: number;
  host: { platform: string; userAgent: string };
}

export function isHardwareAdapter(sample: ProofCaptureSample): boolean {
  const blob = `${sample.adapter.vendor} ${sample.adapter.device} ${sample.adapter.description}`.toLowerCase();
  if (sample.adapter.isFallbackAdapter) return false;
  return !blob.includes("software") && !blob.includes("basic render") && !blob.includes("swiftshader");
}

export function formatPerformanceRow(sample: ProofCaptureSample): string {
  const gpu = sample.adapter.description || sample.adapter.device || sample.adapter.vendor;
  return `| ${gpu} | ${sample.resolution.width}×${sample.resolution.height} | ${sample.fps.toFixed(0)} | ${sample.p50Ms.toFixed(2)} | ${sample.p99Ms.toFixed(2)} | ${sample.capturedAt} |`;
}
