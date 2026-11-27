/**
 * GPU-side home of the streamed irradiance probes (world/irradiance-probes.ts).
 *
 * One fixed storage buffer holds a toroidal ring of PROBE_RING × PROBE_RING
 * probe columns × PROBE_LAYERS: global column (i, j) lives in slot
 * (i mod R, j mod R). Each probe is 4 vec4s (r, g, b L1 ambient + meta
 * (gi, gj, groundY, valid)); the shader only trusts a slot whose meta names
 * the column it asked for, so a stale or aliased slot (512 m away) is ignored
 * rather than read. 32² × 2 × 64 B = 128 KB, allocated once — the binding
 * never changes, so frame bind groups never rebuild for probes.
 *
 * `sampleProbeRing` is the CPU mirror of the WGSL sampler
 * (pipeline-factory.ts `augengine_probe_ambient`): bilinear ground base from
 * the valid layer-0 columns, then trilinear over the 8 surrounding probes,
 * renormalised over the valid ones. Tests use it to check tile seams.
 */

export const PROBE_RING = 32;
export const PROBE_LAYERS = 2;
const FLOATS = 16;
export const PROBE_RING_FLOATS = PROBE_RING * PROBE_RING * PROBE_LAYERS * FLOATS;

export interface ProbeGridParams {
  spacing: number;
  heightAboveGround: number;
  layerSpacing: number;
}

export interface ProbeTileLike {
  gi0: number;
  gj0: number;
  perAxis: number;
  layers: number;
  probes: Float32Array;
}

/** First float of global probe (i, j, layer k) in the ring. */
export function ringOffset(i: number, j: number, k: number): number {
  const si = ((i % PROBE_RING) + PROBE_RING) % PROBE_RING;
  const sj = ((j % PROBE_RING) + PROBE_RING) % PROBE_RING;
  return ((k * PROBE_RING + sj) * PROBE_RING + si) * FLOATS;
}

/** A contiguous run of the ring to upload: [firstFloat, floatCount). */
export type RingRange = [number, number];

export class ProbeRing {
  readonly data = new Float32Array(PROBE_RING_FLOATS);

  /** Copy a tile's probes into its slots. Returns the ranges that changed. */
  setTile(t: ProbeTileLike): RingRange[] {
    const ranges: RingRange[] = [];
    for (let k = 0; k < t.layers; k++) {
      for (let b = 0; b < t.perAxis; b++) {
        for (let a = 0; a < t.perAxis; a++) {
          const src = ((k * t.perAxis + b) * t.perAxis + a) * FLOATS;
          const dst = ringOffset(t.gi0 + a, t.gj0 + b, k);
          this.data.set(t.probes.subarray(src, src + FLOATS), dst);
          pushRange(ranges, dst, FLOATS);
        }
      }
    }
    return ranges;
  }

  /** Invalidate a tile's slots — only those still holding that tile's columns. */
  clearTile(gi0: number, gj0: number, perAxis: number, layers: number): RingRange[] {
    const ranges: RingRange[] = [];
    for (let k = 0; k < layers; k++) {
      for (let b = 0; b < perAxis; b++) {
        for (let a = 0; a < perAxis; a++) {
          const o = ringOffset(gi0 + a, gj0 + b, k);
          if (this.data[o + 15] === 0 || this.data[o + 12] !== gi0 + a || this.data[o + 13] !== gj0 + b) continue;
          this.data.fill(0, o, o + FLOATS);
          pushRange(ranges, o, FLOATS);
        }
      }
    }
    return ranges;
  }

  clear(): void {
    this.data.fill(0);
  }
}

function pushRange(ranges: RingRange[], start: number, count: number): void {
  const last = ranges[ranges.length - 1];
  if (last && last[0] + last[1] === start) last[1] += count;
  else ranges.push([start, count]);
}

function okAt(d: Float32Array, o: number, i: number, j: number): boolean {
  return d[o + 15]! > 0.5 && Math.round(d[o + 12]!) === i && Math.round(d[o + 13]!) === j;
}

/**
 * CPU mirror of the shader's probe sampler. Returns the relative ambient
 * (rgb, multiply by the frame ambient) and a coverage in [0, 1] (0 = no
 * probes here: the shader falls back to the sky/flat ambient).
 */
export function sampleProbeRing(
  data: Float32Array,
  p: readonly [number, number, number],
  n: readonly [number, number, number],
  params: ProbeGridParams,
): { rgb: [number, number, number]; coverage: number } {
  const gx = p[0] / params.spacing - 0.5;
  const gz = p[2] / params.spacing - 0.5;
  const fx0 = Math.floor(gx), fz0 = Math.floor(gz);
  const fx = gx - fx0, fz = gz - fz0;
  let baseSum = 0, baseW = 0;
  for (let c = 0; c < 4; c++) {
    const di = c & 1, dj = c >> 1;
    const i = fx0 + di, j = fz0 + dj;
    const o = ringOffset(i, j, 0);
    if (!okAt(data, o, i, j)) continue;
    const w = (di ? fx : 1 - fx) * (dj ? fz : 1 - fz);
    baseSum += w * data[o + 14]!;
    baseW += w;
  }
  if (baseW < 1e-4) return { rgb: [0, 0, 0], coverage: 0 };
  const base = baseSum / baseW;
  const gy = Math.min(1, Math.max(0, (p[1] - base - params.heightAboveGround) / params.layerSpacing));
  const acc = [0, 0, 0];
  let wsum = 0;
  for (let c = 0; c < 8; c++) {
    const di = c & 1, dj = (c >> 1) & 1, k = c >> 2;
    const i = fx0 + di, j = fz0 + dj;
    const o = ringOffset(i, j, k);
    if (!okAt(data, o, i, j)) continue;
    const w = (di ? fx : 1 - fx) * (dj ? fz : 1 - fz) * (k ? gy : 1 - gy);
    for (let ch = 0; ch < 3; ch++) {
      const q = o + ch * 4;
      acc[ch]! += w * (data[q]! + data[q + 1]! * n[0] + data[q + 2]! * n[1] + data[q + 3]! * n[2]);
    }
    wsum += w;
  }
  if (wsum < 1e-4) return { rgb: [0, 0, 0], coverage: 0 };
  return {
    rgb: [Math.max(0, acc[0]! / wsum), Math.max(0, acc[1]! / wsum), Math.max(0, acc[2]! / wsum)],
    coverage: Math.min(1, baseW * 2),
  };
}
