import { describe, it, expect, beforeEach } from "vitest";
import {
  BlendshapeUniformPool,
  BLENDSHAPE_UNIFORM_SIZE_BYTES,
  BLENDSHAPE_WGSL_HELPER,
  BLENDSHAPE_BIND_GROUP_ENTRY,
  type BlendshapeBufferLike,
  type BlendshapeDeviceLike,
} from "./blendshapes.js";
import { ARKIT_BLENDSHAPE_COUNT } from "./blendshape-constants.js";

interface BufferCall {
  size: number;
  usage: number;
}

interface WriteCall {
  buf: BlendshapeBufferLike;
  offset: number;
  bytes: number;
}

function makeFakeDevice(): {
  device: BlendshapeDeviceLike;
  buffers: BufferCall[];
  writes: WriteCall[];
  destroys: number;
} {
  const buffers: BufferCall[] = [];
  const writes: WriteCall[] = [];
  let destroys = 0;
  const device: BlendshapeDeviceLike = {
    createBuffer(desc) {
      buffers.push({ size: desc.size, usage: desc.usage });
      return {
        destroy() {
          destroys++;
        },
      };
    },
    queue: {
      writeBuffer(buf, offset, data) {
        const bytes = data instanceof ArrayBuffer ? data.byteLength : data.byteLength;
        writes.push({ buf, offset, bytes });
      },
    },
  };
  return {
    device,
    buffers,
    writes,
    get destroys() {
      return destroys;
    },
  };
}

describe("BLENDSHAPE_UNIFORM_SIZE_BYTES", () => {
  it("matches the ARKit shape count × 4 bytes", () => {
    expect(BLENDSHAPE_UNIFORM_SIZE_BYTES).toBe(ARKIT_BLENDSHAPE_COUNT * 4);
    expect(BLENDSHAPE_UNIFORM_SIZE_BYTES).toBe(208);
  });
});

describe("BlendshapeUniformPool", () => {
  let fake: ReturnType<typeof makeFakeDevice>;
  let pool: BlendshapeUniformPool;

  beforeEach(() => {
    fake = makeFakeDevice();
    pool = new BlendshapeUniformPool(fake.device);
  });

  it("allocates a 208-byte uniform on first setWeights", () => {
    const weights = new Float32Array(ARKIT_BLENDSHAPE_COUNT);
    pool.setWeights("maya", weights);
    expect(fake.buffers).toHaveLength(1);
    expect(fake.buffers[0]!.size).toBe(BLENDSHAPE_UNIFORM_SIZE_BYTES);
    // UNIFORM | COPY_DST = 0x40 | 0x8 = 0x48
    expect(fake.buffers[0]!.usage).toBe(0x48);
  });

  it("reuses the buffer on subsequent setWeights for the same entity", () => {
    const weights = new Float32Array(ARKIT_BLENDSHAPE_COUNT);
    pool.setWeights("maya", weights);
    pool.setWeights("maya", weights);
    pool.setWeights("maya", weights);
    expect(fake.buffers).toHaveLength(1);
    expect(fake.writes).toHaveLength(3);
  });

  it("rejects mis-sized weight buffers", () => {
    expect(() => pool.setWeights("maya", new Float32Array(10))).toThrow(/52/);
  });

  it("ensureBuffer lazy-allocates if the entity hasn't called setWeights yet", () => {
    pool.ensureBuffer("maya");
    expect(fake.buffers).toHaveLength(1);
    // Second call returns the same buffer without re-allocating.
    pool.ensureBuffer("maya");
    expect(fake.buffers).toHaveLength(1);
  });

  it("peekWeights returns a snapshot of the last push", () => {
    const w = new Float32Array(ARKIT_BLENDSHAPE_COUNT);
    w[0] = 0.5;
    w[17] = 1.0;
    pool.setWeights("maya", w);
    const back = pool.peekWeights("maya");
    expect(back).not.toBeNull();
    expect(back![0]).toBeCloseTo(0.5, 6);
    expect(back![17]).toBeCloseTo(1.0, 6);
  });

  it("peekWeights returns null for unknown entities", () => {
    expect(pool.peekWeights("nope")).toBeNull();
  });

  it("clear releases one entity + calls destroy on its buffer", () => {
    pool.setWeights("maya", new Float32Array(ARKIT_BLENDSHAPE_COUNT));
    pool.setWeights("recruit", new Float32Array(ARKIT_BLENDSHAPE_COUNT));
    expect(pool.size()).toBe(2);
    pool.clear("maya");
    expect(pool.size()).toBe(1);
    expect(fake.destroys).toBe(1);
    expect(pool.peekWeights("maya")).toBeNull();
  });

  it("clearAll empties the pool", () => {
    pool.setWeights("a", new Float32Array(ARKIT_BLENDSHAPE_COUNT));
    pool.setWeights("b", new Float32Array(ARKIT_BLENDSHAPE_COUNT));
    pool.setWeights("c", new Float32Array(ARKIT_BLENDSHAPE_COUNT));
    pool.clearAll();
    expect(pool.size()).toBe(0);
    expect(fake.destroys).toBe(3);
  });
});

describe("BLENDSHAPE_WGSL_HELPER", () => {
  it("defines augengine_morph_position + augengine_morph_normal", () => {
    expect(BLENDSHAPE_WGSL_HELPER).toContain("fn augengine_morph_position");
    expect(BLENDSHAPE_WGSL_HELPER).toContain("fn augengine_morph_normal");
  });

  it("hardcodes the ARKit blendshape count constant", () => {
    expect(BLENDSHAPE_WGSL_HELPER).toContain("AUGENGINE_BLENDSHAPE_COUNT: u32 = 52u");
  });

  it("skips zero-weight channels for early-exit in the inner loop", () => {
    expect(BLENDSHAPE_WGSL_HELPER).toContain("if (w == 0.0) { continue; }");
  });
});

describe("BLENDSHAPE_BIND_GROUP_ENTRY", () => {
  it("declares binding 0, vertex-stage, uniform-type", () => {
    expect(BLENDSHAPE_BIND_GROUP_ENTRY.binding).toBe(0);
    expect(BLENDSHAPE_BIND_GROUP_ENTRY.visibility).toBe(1); // VERTEX = 1
    expect(BLENDSHAPE_BIND_GROUP_ENTRY.buffer.type).toBe("uniform");
  });
});
