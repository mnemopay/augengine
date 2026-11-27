/**
 * world/tile-nav — walkability that streams with the tiles.
 *
 * Each resident tile owns a small occupancy grid built from its own
 * colliders when it loads, and drops it when it is evicted, so navigation
 * never covers ground the world has not loaded (and never costs memory for
 * ground it has unloaded). Paths are A* over the union of resident tiles;
 * a cell in an unloaded tile is not walkable.
 */

export interface NavBlocker {
  /** World-space XZ footprint. */
  minX: number;
  minZ: number;
  maxX: number;
  maxZ: number;
}

export interface TileNavOptions {
  originX: number;
  originZ: number;
  tileSize: number;
  /** Cell size in metres. */
  cell: number;
  /** Blockers are inflated by this (agent radius). */
  agentRadius: number;
}

interface TileGrid {
  cells: Uint8Array; // 1 = blocked
}

export class TileNav {
  readonly opts: TileNavOptions;
  private readonly perSide: number;
  private readonly tiles = new Map<string, TileGrid>();

  constructor(opts: TileNavOptions) {
    this.opts = opts;
    this.perSide = Math.round(opts.tileSize / opts.cell);
  }

  get tileCount(): number {
    return this.tiles.size;
  }

  /** Bytes held by resident grids (for the memory budget / diagnostics). */
  get bytes(): number {
    return this.tiles.size * this.perSide * this.perSide;
  }

  addTile(tx: number, tz: number, blockers: readonly NavBlocker[]): void {
    const { originX, originZ, tileSize, cell, agentRadius } = this.opts;
    const n = this.perSide;
    const cells = new Uint8Array(n * n);
    const x0 = originX + tx * tileSize;
    const z0 = originZ + tz * tileSize;
    for (const b of blockers) {
      // Clip to this tile; only touch cells the inflated footprint covers.
      const i0 = Math.max(0, Math.floor((b.minX - agentRadius - x0) / cell));
      const i1 = Math.min(n - 1, Math.floor((b.maxX + agentRadius - x0) / cell));
      const j0 = Math.max(0, Math.floor((b.minZ - agentRadius - z0) / cell));
      const j1 = Math.min(n - 1, Math.floor((b.maxZ + agentRadius - z0) / cell));
      for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) cells[j * n + i] = 1;
    }
    this.tiles.set(`${tx}|${tz}`, { cells });
  }

  removeTile(tx: number, tz: number): void {
    this.tiles.delete(`${tx}|${tz}`);
  }

  hasTile(tx: number, tz: number): boolean {
    return this.tiles.has(`${tx}|${tz}`);
  }

  /** Global cell coordinate of a world point. */
  private cellOf(x: number, z: number): [number, number] {
    return [Math.floor((x - this.opts.originX) / this.opts.cell), Math.floor((z - this.opts.originZ) / this.opts.cell)];
  }

  private walkableCell(ci: number, cj: number): boolean {
    const n = this.perSide;
    const tx = Math.floor(ci / n);
    const tz = Math.floor(cj / n);
    const grid = this.tiles.get(`${tx}|${tz}`);
    if (!grid) return false;
    const li = ci - tx * n;
    const lj = cj - tz * n;
    return grid.cells[lj * n + li] === 0;
  }

  isWalkable(x: number, z: number): boolean {
    const [ci, cj] = this.cellOf(x, z);
    return this.walkableCell(ci, cj);
  }

  /**
   * A* from `from` to `to` (world XZ) across resident tiles, 8-connected,
   * no corner cutting. Returns world-space cell centres, or null when the
   * goal is unloaded, blocked, or unreachable within `maxExpanded`.
   */
  findPath(from: [number, number], to: [number, number], maxExpanded = 60_000): [number, number][] | null {
    const [si, sj] = this.cellOf(from[0], from[1]);
    const [gi, gj] = this.cellOf(to[0], to[1]);
    if (!this.walkableCell(si, sj) || !this.walkableCell(gi, gj)) return null;
    const key = (i: number, j: number) => `${i},${j}`;
    const h = (i: number, j: number) => {
      const dx = Math.abs(i - gi), dz = Math.abs(j - gj);
      return Math.max(dx, dz) + (Math.SQRT2 - 1) * Math.min(dx, dz);
    };
    const open = new MinHeap();
    const g = new Map<string, number>([[key(si, sj), 0]]);
    const came = new Map<string, string>();
    open.push(h(si, sj), si, sj);
    let expanded = 0;
    while (open.size > 0) {
      const [ci, cj] = open.pop()!;
      const ck = key(ci, cj);
      if (ci === gi && cj === gj) return this.reconstruct(came, ck);
      if (++expanded > maxExpanded) return null;
      const gc = g.get(ck)!;
      for (let dj = -1; dj <= 1; dj++) {
        for (let di = -1; di <= 1; di++) {
          if (di === 0 && dj === 0) continue;
          const ni = ci + di, nj = cj + dj;
          if (!this.walkableCell(ni, nj)) continue;
          if (di !== 0 && dj !== 0 && (!this.walkableCell(ci + di, cj) || !this.walkableCell(ci, cj + dj))) continue;
          const nk = key(ni, nj);
          const ng = gc + (di !== 0 && dj !== 0 ? Math.SQRT2 : 1);
          if (ng < (g.get(nk) ?? Infinity)) {
            g.set(nk, ng);
            came.set(nk, ck);
            open.push(ng + h(ni, nj), ni, nj);
          }
        }
      }
    }
    return null;
  }

  private reconstruct(came: Map<string, string>, end: string): [number, number][] {
    const out: [number, number][] = [];
    let k: string | undefined = end;
    const c = this.opts.cell;
    while (k) {
      const [i, j] = k.split(",").map(Number) as [number, number];
      out.push([this.opts.originX + (i + 0.5) * c, this.opts.originZ + (j + 0.5) * c]);
      k = came.get(k);
    }
    return out.reverse();
  }
}

/** Binary min-heap of (priority, i, j). */
class MinHeap {
  private p: number[] = [];
  private a: number[] = [];
  private b: number[] = [];
  get size(): number {
    return this.p.length;
  }
  push(pri: number, i: number, j: number): void {
    this.p.push(pri); this.a.push(i); this.b.push(j);
    let n = this.p.length - 1;
    while (n > 0) {
      const parent = (n - 1) >> 1;
      if (this.p[parent]! <= this.p[n]!) break;
      this.swap(n, parent);
      n = parent;
    }
  }
  pop(): [number, number] | undefined {
    if (this.p.length === 0) return undefined;
    const out: [number, number] = [this.a[0]!, this.b[0]!];
    const last = this.p.length - 1;
    this.swap(0, last);
    this.p.pop(); this.a.pop(); this.b.pop();
    let n = 0;
    for (;;) {
      const l = 2 * n + 1, r = l + 1;
      let m = n;
      if (l < this.p.length && this.p[l]! < this.p[m]!) m = l;
      if (r < this.p.length && this.p[r]! < this.p[m]!) m = r;
      if (m === n) break;
      this.swap(n, m);
      n = m;
    }
    return out;
  }
  private swap(x: number, y: number): void {
    [this.p[x], this.p[y]] = [this.p[y]!, this.p[x]!];
    [this.a[x], this.a[y]] = [this.a[y]!, this.a[x]!];
    [this.b[x], this.b[y]] = [this.b[y]!, this.b[x]!];
  }
}
