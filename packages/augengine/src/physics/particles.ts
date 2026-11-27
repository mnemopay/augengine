/**
 * GPU compute particle system — Phase 11.
 *
 * Pure data + WGSL source. The compute shader steps a structured-buffer
 * particle array per frame: gravity + drag + alive-test + spawn from emitter.
 *
 * The vertex/fragment pair (next session, alongside the renderer refactor)
 * reads the same buffer and renders point sprites or instanced quads.
 *
 * Phase 11 ships:
 *   - Particle struct + ParticleBuffer (CPU-side ring + emit queue)
 *   - WGSL compute step source
 *   - Emitter shapes (point, sphere, cone, box)
 */

export interface Particle {
  position: [number, number, number];
  velocity: [number, number, number];
  /** Seconds remaining; ≤0 = dead. */
  age: number;
  /** Current size in world units. */
  size: number;
  /** Linear color per component (rgb). */
  color: [number, number, number];
  /** Linear opacity 0..1. */
  alpha: number;
}

export type EmitterShape =
  | { kind: "point" }
  | { kind: "sphere"; radius: number; surfaceOnly?: boolean }
  | { kind: "cone"; halfAngle: number; height: number }
  | { kind: "box"; size: [number, number, number] };

export interface EmitterDef {
  /** Emitter world-space origin. */
  origin: [number, number, number];
  /** Direction (unit vector) the emitter "fires" toward. */
  direction: [number, number, number];
  /** Particles emitted per second. */
  rate: number;
  /** Initial speed of new particles. */
  initialSpeed: number;
  /** Random spread in radians around `direction`. */
  spread: number;
  /** Particle lifetime in seconds. */
  lifetime: number;
  /** Initial size. */
  size: number;
  /** Initial color (rgb 0..1). */
  color: [number, number, number];
  /** Initial alpha. */
  alpha: number;
  /** Shape determining where in the volume particles spawn. */
  shape: EmitterShape;
}

/**
 * CPU-side particle ring buffer. Caller-supplied capacity. Emit pushes new
 * particles; integrate steps live ones; the WGSL shader reads from the
 * upload of this data each frame.
 */
export class ParticleBuffer {
  particles: Particle[];
  /** RNG seed for deterministic spawn. */
  seed: number;

  constructor(capacity: number, seed = 1) {
    this.particles = new Array(capacity);
    for (let i = 0; i < capacity; i++) {
      this.particles[i] = {
        position: [0, 0, 0], velocity: [0, 0, 0],
        age: -1, size: 0, color: [1, 1, 1], alpha: 0,
      };
    }
    this.seed = seed;
  }

  /** Step gravity + drag for live particles. Returns count alive. */
  step(dt: number, gravity: [number, number, number] = [0, -9.81, 0], drag = 0.05): number {
    let alive = 0;
    for (const p of this.particles) {
      if (p.age <= 0) continue;
      const k = 1 - drag * dt;
      p.velocity[0] = p.velocity[0] * k + gravity[0] * dt;
      p.velocity[1] = p.velocity[1] * k + gravity[1] * dt;
      p.velocity[2] = p.velocity[2] * k + gravity[2] * dt;
      p.position[0] += p.velocity[0] * dt;
      p.position[1] += p.velocity[1] * dt;
      p.position[2] += p.velocity[2] * dt;
      p.age -= dt;
      if (p.age > 0) alive++;
    }
    return alive;
  }

  /** Emit `count` new particles per emitter spec into available slots. */
  emit(emitter: EmitterDef, count: number): number {
    let emitted = 0;
    for (let i = 0; i < this.particles.length && emitted < count; i++) {
      const p = this.particles[i]!;
      if (p.age > 0) continue;
      const spawn = sampleEmitter(emitter, () => this.rand());
      p.position = spawn.position;
      p.velocity = spawn.velocity;
      p.age = emitter.lifetime;
      p.size = emitter.size;
      p.color = [...emitter.color];
      p.alpha = emitter.alpha;
      emitted++;
    }
    return emitted;
  }

  /** xorshift RNG for deterministic spawn. */
  private rand(): number {
    let x = this.seed | 0;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    this.seed = x | 0;
    return ((x >>> 0) / 0xffffffff);
  }
}

function sampleEmitter(e: EmitterDef, rand: () => number): {
  position: [number, number, number];
  velocity: [number, number, number];
} {
  // Position
  let pos: [number, number, number];
  switch (e.shape.kind) {
    case "point":
      pos = [...e.origin];
      break;
    case "sphere": {
      const r = e.shape.surfaceOnly ? e.shape.radius : e.shape.radius * Math.cbrt(rand());
      const theta = rand() * Math.PI * 2;
      const phi = Math.acos(2 * rand() - 1);
      pos = [
        e.origin[0] + r * Math.sin(phi) * Math.cos(theta),
        e.origin[1] + r * Math.sin(phi) * Math.sin(theta),
        e.origin[2] + r * Math.cos(phi),
      ];
      break;
    }
    case "cone": {
      // sample within a cone aligned to `direction`, then translate to origin.
      const u = rand();
      const v = rand();
      const r = u * Math.tan(e.shape.halfAngle) * e.shape.height;
      const theta = v * Math.PI * 2;
      // local axes
      const fwd = e.direction;
      const right = orthogonal(fwd);
      const up = cross(fwd, right);
      const h = u * e.shape.height;
      const lx = r * Math.cos(theta);
      const ly = r * Math.sin(theta);
      pos = [
        e.origin[0] + fwd[0] * h + right[0] * lx + up[0] * ly,
        e.origin[1] + fwd[1] * h + right[1] * lx + up[1] * ly,
        e.origin[2] + fwd[2] * h + right[2] * lx + up[2] * ly,
      ];
      break;
    }
    case "box": {
      pos = [
        e.origin[0] + (rand() - 0.5) * e.shape.size[0],
        e.origin[1] + (rand() - 0.5) * e.shape.size[1],
        e.origin[2] + (rand() - 0.5) * e.shape.size[2],
      ];
      break;
    }
  }
  // Velocity: direction with random cone spread.
  const theta = rand() * Math.PI * 2;
  const phi = rand() * e.spread;
  const fwd = e.direction;
  const right = orthogonal(fwd);
  const up = cross(fwd, right);
  const cosP = Math.cos(phi);
  const sinP = Math.sin(phi);
  const dirX = fwd[0] * cosP + (right[0] * Math.cos(theta) + up[0] * Math.sin(theta)) * sinP;
  const dirY = fwd[1] * cosP + (right[1] * Math.cos(theta) + up[1] * Math.sin(theta)) * sinP;
  const dirZ = fwd[2] * cosP + (right[2] * Math.cos(theta) + up[2] * Math.sin(theta)) * sinP;
  return {
    position: pos,
    velocity: [dirX * e.initialSpeed, dirY * e.initialSpeed, dirZ * e.initialSpeed],
  };
}

function orthogonal(v: [number, number, number]): [number, number, number] {
  // Pick the smallest axis-aligned vector that's not parallel.
  if (Math.abs(v[0]) < 0.9) return normalize([0, -v[2], v[1]]);
  return normalize([-v[2], 0, v[0]]);
}
function cross(a: [number, number, number], b: [number, number, number]): [number, number, number] {
  return [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]];
}
function normalize(v: [number, number, number]): [number, number, number] {
  const len = Math.hypot(v[0], v[1], v[2]);
  return len > 0 ? [v[0]/len, v[1]/len, v[2]/len] : [0, 0, 0];
}

// ──────────────────────────────────────────────────────────────────────
// WGSL — compute shader source. Phase 11 ships the source; binding into the
// renderer happens with the Phase 9 GPU refactor.
// ──────────────────────────────────────────────────────────────────────

export const PARTICLE_STEP_WGSL = /* wgsl */ `
struct Particle {
  position: vec3<f32>,
  age:      f32,
  velocity: vec3<f32>,
  size:     f32,
  color:    vec3<f32>,
  alpha:    f32,
};

struct StepUniforms {
  dt: f32,
  drag: f32,
  gravity: vec3<f32>,
  count: u32,
};

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform>             u: StepUniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u.count) { return; }
  var p = particles[i];
  if (p.age <= 0.0) { return; }
  let k = 1.0 - u.drag * u.dt;
  p.velocity = p.velocity * k + u.gravity * u.dt;
  p.position = p.position + p.velocity * u.dt;
  p.age = p.age - u.dt;
  particles[i] = p;
}
`;
