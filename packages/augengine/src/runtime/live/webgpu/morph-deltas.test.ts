import { describe, it, expect, beforeEach } from "vitest";
import {
  MorphDeltaStorage,
  MORPH_POSITION_DELTAS_BINDING,
  MORPH_NORMAL_DELTAS_BINDING,
  MORPH_DELTA_STRIDE_BYTES,
} from "./morph-deltas.js";
import type { BlendshapeBufferLike, BlendshapeDeviceLike } from "./blendshapes.js";
import { ARKIT_BLENDSHAPE_COUNT } from "./blendshape-constants.js";

interface BufferCall {
  size: number;
  usage: number;
}
interface WriteCall {
  size: number;
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
      const handle: BlendshapeBufferLike = {
        destroy() {
          destroys++;
        },
      };
      return handle;
    },
    queue: {
      writeBuffer(_buf, _offset, data) {
        const bytes = data instanceof ArrayBuffer ? data.byteLength : data.byteLength;
        writes.push({ size: bytes });
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

describe("MorphDeltaStorage — uploads", () => {
  let fake: ReturnType<typeof makeFakeDevice>;
  let storage: MorphDeltaStorage;

  beforeEach(() => {
    fake = makeFakeDevice();
    storage = new MorphDeltaStorage(fake.device);
  });

  it("emits position and zero-normal buffers when only positions are supplied", () => {
    storage.upload("primitive:cube", {
      vertexCount: 8,
      positionDeltas: new Float32Array(8 * ARKIT_BLENDSHAPE_COUNT * 3),
    });
    expect(fake.buffers).toHaveLength(2);
    // STORAGE | COPY_DST = 0x80 | 0x8 = 0x88
    expect(fake.buffers[0]!.usage).toBe(0x88);
    // 8 verts × 52 shapes × vec4 stride (16 B) = 6656 B
    expect(fake.buffers[0]!.size).toBe(8 * ARKIT_BLENDSHAPE_COUNT * MORPH_DELTA_STRIDE_BYTES);
    expect(fake.buffers[1]!.size).toBe(8 * ARKIT_BLENDSHAPE_COUNT * MORPH_DELTA_STRIDE_BYTES);
  });

  it("emits two buffers when normals are supplied", () => {
    storage.upload("gltf:foo#0", {
      vertexCount: 4,
      positionDeltas: new Float32Array(4 * ARKIT_BLENDSHAPE_COUNT * 3),
      normalDeltas: new Float32Array(4 * ARKIT_BLENDSHAPE_COUNT * 3),
    });
    expect(fake.buffers).toHaveLength(2);
  });

  it("rejects vertexCount <= 0", () => {
    expect(() =>
      storage.upload("bad", { vertexCount: 0, positionDeltas: new Float32Array(0) }),
    ).toThrow(/vertexCount must be positive/);
  });

  it("rejects positionDeltas that exceed vertexCount × 52 × 3", () => {
    expect(() =>
      storage.upload("overlong", {
        vertexCount: 1,
        // vertexCount=1 → max 52 × 3 = 156 floats; supply 200 to trigger.
        positionDeltas: new Float32Array(200),
      }),
    ).toThrow(/exceeds/);
  });

  it("pads vec3 → vec4 stride at upload time", () => {
    const verts = 2;
    const shapes = ARKIT_BLENDSHAPE_COUNT;
    const src = new Float32Array(verts * shapes * 3);
    // Set non-zero markers we can verify post-pad.
    src[0] = 1;
    src[1] = 2;
    src[2] = 3;
    src[3] = 4; // second vec3
    src[4] = 5;
    src[5] = 6;
    storage.upload("test", { vertexCount: verts, positionDeltas: src });
    // Expected upload: verts × shapes × 4 floats × 4 bytes
    const expectedBytes = verts * shapes * 4 * 4;
    expect(fake.writes).toHaveLength(2);
    expect(fake.writes[0]!.size).toBe(expectedBytes);
  });

  it("upload is idempotent on duplicate meshKey", () => {
    const input = {
      vertexCount: 4,
      positionDeltas: new Float32Array(4 * ARKIT_BLENDSHAPE_COUNT * 3),
    };
    storage.upload("primitive:cube", input);
    storage.upload("primitive:cube", input);
    expect(fake.buffers).toHaveLength(2);
    expect(storage.size()).toBe(1);
  });
});

describe("MorphDeltaStorage — lookup + lifecycle", () => {
  let fake: ReturnType<typeof makeFakeDevice>;
  let storage: MorphDeltaStorage;

  beforeEach(() => {
    fake = makeFakeDevice();
    storage = new MorphDeltaStorage(fake.device);
  });

  it("has() flips after upload", () => {
    expect(storage.has("primitive:cube")).toBe(false);
    storage.upload("primitive:cube", {
      vertexCount: 8,
      positionDeltas: new Float32Array(8 * ARKIT_BLENDSHAPE_COUNT * 3),
    });
    expect(storage.has("primitive:cube")).toBe(true);
  });

  it("get returns the uploaded handle", () => {
    storage.upload("primitive:cube", {
      vertexCount: 8,
      positionDeltas: new Float32Array(8 * ARKIT_BLENDSHAPE_COUNT * 3),
    });
    const handle = storage.get("primitive:cube");
    expect(handle).not.toBeNull();
    expect(handle!.vertexCount).toBe(8);
    expect(handle!.normalBuffer).toBeDefined();
  });

  it("get returns null for an unknown meshKey", () => {
    expect(storage.get("nope")).toBeNull();
  });

  it("clear releases one mesh's buffers", () => {
    storage.upload("primitive:cube", {
      vertexCount: 4,
      positionDeltas: new Float32Array(4 * ARKIT_BLENDSHAPE_COUNT * 3),
      normalDeltas: new Float32Array(4 * ARKIT_BLENDSHAPE_COUNT * 3),
    });
    storage.upload("primitive:sphere", {
      vertexCount: 4,
      positionDeltas: new Float32Array(4 * ARKIT_BLENDSHAPE_COUNT * 3),
    });
    storage.clear("primitive:cube");
    // 2 buffers from the cube (pos + normal) destroyed; sphere left alone
    expect(fake.destroys).toBe(2);
    expect(storage.size()).toBe(1);
  });

  it("clearAll empties everything", () => {
    storage.upload("a", { vertexCount: 2, positionDeltas: new Float32Array(2 * ARKIT_BLENDSHAPE_COUNT * 3) });
    storage.upload("b", { vertexCount: 2, positionDeltas: new Float32Array(2 * ARKIT_BLENDSHAPE_COUNT * 3) });
    storage.upload("c", {
      vertexCount: 2,
      positionDeltas: new Float32Array(2 * ARKIT_BLENDSHAPE_COUNT * 3),
      normalDeltas: new Float32Array(2 * ARKIT_BLENDSHAPE_COUNT * 3),
    });
    storage.clearAll();
    expect(storage.size()).toBe(0);
    // Each uploaded mesh owns both position and normal storage.
    expect(fake.destroys).toBe(6);
  });

  it("totalUploadedBytes tallies across meshes", () => {
    storage.upload("a", { vertexCount: 4, positionDeltas: new Float32Array(4 * ARKIT_BLENDSHAPE_COUNT * 3) });
    storage.upload("b", { vertexCount: 8, positionDeltas: new Float32Array(8 * ARKIT_BLENDSHAPE_COUNT * 3) });
    // (4 + 8) verts x 52 shapes x 16 B/vec4 x position+normal storage.
    expect(storage.totalUploadedBytes()).toBe((4 + 8) * ARKIT_BLENDSHAPE_COUNT * 16 * 2);
  });
});

describe("Bind-group constants", () => {
  it("MORPH_POSITION_DELTAS_BINDING + MORPH_NORMAL_DELTAS_BINDING are distinct + slot >= 1", () => {
    expect(MORPH_POSITION_DELTAS_BINDING).toBe(1);
    expect(MORPH_NORMAL_DELTAS_BINDING).toBe(2);
  });
});
