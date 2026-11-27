/**
 * Click picking for the 3D view, done on the CPU against oriented boxes.
 * The camera convention matches the engine: local −Z is forward, +Y up,
 * +X right, vertical field of view `fovY`.
 */
export type V3 = [number, number, number];
export type Q4 = [number, number, number, number];

export interface PickCamera {
  position: V3;
  rotation: Q4;
  fovY: number;
  aspect: number;
}

/** An oriented box: centre, rotation, and half-extents along its local axes. */
export interface PickBox {
  id: string;
  center: V3;
  rotation: Q4;
  half: V3;
}

export function rotate(q: Q4, v: V3): V3 {
  const [x, y, z, w] = q;
  const ix = w * v[0] + y * v[2] - z * v[1];
  const iy = w * v[1] + z * v[0] - x * v[2];
  const iz = w * v[2] + x * v[1] - y * v[0];
  const iw = -x * v[0] - y * v[1] - z * v[2];
  return [
    ix * w + iw * -x + iy * -z - iz * -y,
    iy * w + iw * -y + iz * -x - ix * -z,
    iz * w + iw * -z + ix * -y - iy * -x,
  ];
}

const conj = (q: Q4): Q4 => [-q[0], -q[1], -q[2], q[3]];

/** World-space ray through a point given in normalised device coordinates (x, y in [−1, 1], y up). */
export function rayFromNdc(cam: PickCamera, ndcX: number, ndcY: number): { origin: V3; dir: V3 } {
  const t = Math.tan(cam.fovY / 2);
  const local: V3 = [ndcX * t * cam.aspect, ndcY * t, -1];
  const d = rotate(cam.rotation, local);
  const l = Math.hypot(d[0], d[1], d[2]);
  return { origin: cam.position, dir: [d[0] / l, d[1] / l, d[2] / l] };
}

/** Distance along the ray to the box, or null if it misses (slab test in the box's frame). */
export function rayBox(origin: V3, dir: V3, box: PickBox): number | null {
  const inv = conj(box.rotation);
  const o = rotate(inv, [origin[0] - box.center[0], origin[1] - box.center[1], origin[2] - box.center[2]]);
  const d = rotate(inv, dir);
  let tmin = -Infinity;
  let tmax = Infinity;
  for (let i = 0; i < 3; i++) {
    const h = box.half[i]!;
    if (Math.abs(d[i]!) < 1e-9) {
      if (o[i]! < -h || o[i]! > h) return null;
      continue;
    }
    let t1 = (-h - o[i]!) / d[i]!;
    let t2 = (h - o[i]!) / d[i]!;
    if (t1 > t2) [t1, t2] = [t2, t1];
    tmin = Math.max(tmin, t1);
    tmax = Math.min(tmax, t2);
    if (tmin > tmax) return null;
  }
  if (tmax < 0) return null;
  return tmin >= 0 ? tmin : tmax;
}

/** The nearest box under a screen point, or null. */
export function pick(cam: PickCamera, ndcX: number, ndcY: number, boxes: readonly PickBox[]): { id: string; distance: number } | null {
  const { origin, dir } = rayFromNdc(cam, ndcX, ndcY);
  let best: { id: string; distance: number } | null = null;
  for (const b of boxes) {
    const t = rayBox(origin, dir, b);
    if (t !== null && (!best || t < best.distance)) best = { id: b.id, distance: t };
  }
  return best;
}
