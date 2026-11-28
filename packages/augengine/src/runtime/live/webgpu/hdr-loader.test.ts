import { describe, expect, it } from "vitest";
import { extractSun, floatToRgbe, parseRadianceHdr, rgbeToFloat } from "./hdr-loader.js";
import { equirectDirection } from "./sh9.js";

/** Minimal Radiance writer for the tests: new-style RLE (runs + literals) or flat. */
function encodeHdr(width: number, height: number, px: (x: number, y: number) => [number, number, number], opts: { rle?: boolean; header?: string[]; orientation?: string } = {}): Uint8Array {
  const head = ["#?RADIANCE", "# test image", "FORMAT=32-bit_rle_rgbe", ...(opts.header ?? []), "", opts.orientation ?? `-Y ${height} +X ${width}`, ""].join("\n");
  const out: number[] = [...new TextEncoder().encode(head)];
  for (let y = 0; y < height; y++) {
    const row = Array.from({ length: width }, (_, x) => floatToRgbe(...px(x, y)));
    if (!opts.rle) {
      for (const p of row) out.push(...p);
      continue;
    }
    out.push(2, 2, width >> 8, width & 255);
    for (let ch = 0; ch < 4; ch++) {
      const vals = row.map((p) => p[ch]!);
      let x = 0;
      while (x < width) {
        let run = 1;
        while (x + run < width && vals[x + run] === vals[x] && run < 127) run++;
        if (run >= 3) {
          out.push(128 + run, vals[x]!);
          x += run;
        } else {
          let lit = 0;
          while (x + lit < width && lit < 128) {
            if (x + lit + 2 < width && vals[x + lit] === vals[x + lit + 1] && vals[x + lit] === vals[x + lit + 2]) break;
            lit++;
          }
          lit = Math.max(lit, 1);
          out.push(lit, ...vals.slice(x, x + lit));
          x += lit;
        }
      }
    }
  }
  return new Uint8Array(out);
}

describe("RGBE decode", () => {
  it("mantissa · 2^(E−136), E = 0 is black", () => {
    const o = [0, 0, 0];
    rgbeToFloat(128, 64, 255, 129, o);
    expect(o[0]).toBeCloseTo(1, 9);
    expect(o[1]).toBeCloseTo(0.5, 9);
    expect(o[2]).toBeCloseTo(255 / 128, 9);
    rgbeToFloat(200, 200, 200, 0, o);
    expect(o).toEqual([0, 0, 0]);
  });

  it("round-trips through floatToRgbe within 8-bit mantissa precision", () => {
    for (const v of [[0.001, 0.5, 2], [1234.5, 10, 0.25], [65000, 3, 1e-4]] as const) {
      const [r, g, b, e] = floatToRgbe(v[0], v[1], v[2]);
      const o = [0, 0, 0];
      rgbeToFloat(r, g, b, e, o);
      const peak = Math.max(...v);
      for (let i = 0; i < 3; i++) expect(Math.abs(o[i]! - v[i]!)).toBeLessThanOrEqual(peak / 128 + 1e-12);
    }
  });
});

describe("Radiance .hdr parsing", () => {
  const W = 40, H = 6;
  const px = (x: number, y: number): [number, number, number] => [x < 20 ? 0.25 : 4, (y + 1) * 0.5, x === 7 ? 1000 : 0.1];

  for (const rle of [true, false]) {
    it(`decodes a ${rle ? "new-RLE" : "flat"} image pixel-exactly vs the encoder`, () => {
      const img = parseRadianceHdr(encodeHdr(W, H, px, { rle }));
      expect(img.width).toBe(W);
      expect(img.height).toBe(H);
      for (const [x, y] of [[0, 0], [7, 2], [39, 5], [20, 3]] as const) {
        const want = px(x, y);
        const [r, g, b, e] = floatToRgbe(...want);
        const exact = [0, 0, 0];
        rgbeToFloat(r, g, b, e, exact);
        const o = (y * W + x) * 3;
        for (let c = 0; c < 3; c++) expect(img.data[o + c]).toBeCloseTo(exact[c]!, 6);
      }
    });
  }

  it("honours +Y (bottom-up) orientation and divides out EXPOSURE", () => {
    const img = parseRadianceHdr(encodeHdr(W, H, px, { rle: true, orientation: `+Y ${H} +X ${W}`, header: ["EXPOSURE=2"] }));
    // File row 0 is the BOTTOM row → lands at image row H-1.
    expect(img.data[((H - 1) * W) * 3 + 1]).toBeCloseTo(0.5 / 2, 6);
    expect(img.exposure).toBe(2);
  });

  it("rejects non-Radiance data and truncated files", () => {
    expect(() => parseRadianceHdr(new TextEncoder().encode("P6\n1 1\n255\n"))).toThrow(/not a Radiance/);
    const good = encodeHdr(W, H, px, { rle: true });
    expect(() => parseRadianceHdr(good.slice(0, good.length - 20))).toThrow(/truncated/);
    expect(() => parseRadianceHdr(new TextEncoder().encode("#?RADIANCE\nFORMAT=32-bit_rle_xyze\n\n-Y 1 +X 1\n"))).toThrow(/FORMAT/);
  });
});

describe("sun extraction", () => {
  it("finds the sun's direction and illuminance in an equirect and removes it", () => {
    const W = 512, H = 256;
    const sun = [0.4, 0.7, -0.59];
    const l = Math.hypot(...sun);
    const s = sun.map((v) => v / l);
    const radius = (2 * Math.PI) / 180; // 2° disc → plenty of pixels at 0.7°/px
    const Lsun = 20000;
    const data = new Float32Array(W * H * 3);
    let expectedE = 0;
    for (let y = 0; y < H; y++) {
      const omega = Math.sin(((y + 0.5) / H) * Math.PI) * (Math.PI / H) * ((2 * Math.PI) / W);
      for (let x = 0; x < W; x++) {
        const d = equirectDirection((x + 0.5) / W, (y + 0.5) / H);
        const inSun = d[0] * s[0]! + d[1] * s[1]! + d[2] * s[2]! > Math.cos(radius);
        const sky = d[1] > 0 ? 1 : 0.2;
        const v = inSun ? Lsun : sky;
        if (inSun) expectedE += (Lsun - 20 /* cut = p99(=1) × 20 */) * omega;
        data.set([v, v, v], (y * W + x) * 3);
      }
    }
    const found = extractSun({ width: W, height: H, data, channels: 3 })!;
    expect(found).not.toBeNull();
    const dot = found.dir[0] * s[0]! + found.dir[1] * s[1]! + found.dir[2] * s[2]!;
    expect(Math.acos(Math.min(1, dot)) * 180 / Math.PI).toBeLessThan(0.5);
    expect(found.illuminance[1] / expectedE).toBeCloseTo(1, 2);
    // The sun is gone from the image (clamped to the cut level).
    let max = 0;
    for (let i = 0; i < data.length; i++) max = Math.max(max, data[i]!);
    expect(max).toBeLessThanOrEqual(20 + 1e-3);
  });

  it("returns null for an overcast image with no compact bright region", () => {
    const W = 64, H = 32;
    const data = new Float32Array(W * H * 3).fill(3);
    expect(extractSun({ width: W, height: H, data, channels: 3 })).toBeNull();
  });
});
