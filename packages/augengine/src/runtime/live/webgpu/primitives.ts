/** Mesh primitives — pure data, no GPU dependency. */

import type { MorphTargetSource } from "./morph-deltas.js";
import type { AugEnginePbrMaterialFromGltf } from "../../../scene/loaders/gltf.js";

export type PrimitiveShape = "box" | "plane" | "sphere" | "cylinder";

export interface MeshData {
  /** Interleaved positions + normals + UVs (8 floats per vertex). */
  vertices: Float32Array;
  indices: Uint32Array;
  vertexCount: number;
  indexCount: number;
  joints?: Uint16Array;
  weights?: Float32Array;
  colors?: Float32Array;
  /** Optional glTF morph targets uploaded for native facial deformation. */
  morphTargets?: ReadonlyArray<MorphTargetSource>;
  /** Optional glTF source material factors and embedded base-color texture. */
  pbrMaterial?: AugEnginePbrMaterialFromGltf;
}

/** Pick a primitive mesh by name. Used by the renderer's geometry dispatch. */
export function primitiveMesh(shape: PrimitiveShape): MeshData {
  switch (shape) {
    case "box": return cubeMesh();
    case "plane": return planeMesh();
    case "sphere": return sphereMesh();
    case "cylinder": return cylinderMesh();
  }
}

/** Unit cube centered at origin, 24 verts (per-face normals + UVs), 36 indices. */
export function cubeMesh(size = 1): MeshData {
  const s = size / 2;
  // 6 faces × 4 verts = 24 verts. positions + normals + UVs interleaved.
  const v = new Float32Array(24 * 8);
  const idx = new Uint32Array(36);

  const faces: Array<[number[], number[]]> = [
    // [normal, [v0..v3 positions]]
    [[0, 0, 1], [-s, -s, s, s, -s, s, s, s, s, -s, s, s]], // +Z
    [[0, 0, -1], [s, -s, -s, -s, -s, -s, -s, s, -s, s, s, -s]], // -Z
    [[1, 0, 0], [s, -s, s, s, -s, -s, s, s, -s, s, s, s]], // +X
    [[-1, 0, 0], [-s, -s, -s, -s, -s, s, -s, s, s, -s, s, -s]], // -X
    [[0, 1, 0], [-s, s, s, s, s, s, s, s, -s, -s, s, -s]], // +Y
    [[0, -1, 0], [-s, -s, -s, s, -s, -s, s, -s, s, -s, -s, s]], // -Y
  ];

  const faceUvs = [
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1],
  ];

  let vi = 0;
  let ii = 0;
  let baseVert = 0;
  for (const [n, pos] of faces) {
    for (let q = 0; q < 4; q++) {
      v[vi++] = pos[q * 3]!;
      v[vi++] = pos[q * 3 + 1]!;
      v[vi++] = pos[q * 3 + 2]!;
      v[vi++] = n[0]!;
      v[vi++] = n[1]!;
      v[vi++] = n[2]!;
      v[vi++] = faceUvs[q]![0]!;
      v[vi++] = faceUvs[q]![1]!;
    }
    idx[ii++] = baseVert + 0;
    idx[ii++] = baseVert + 1;
    idx[ii++] = baseVert + 2;
    idx[ii++] = baseVert + 0;
    idx[ii++] = baseVert + 2;
    idx[ii++] = baseVert + 3;
    baseVert += 4;
  }
  return { vertices: v, indices: idx, vertexCount: 24, indexCount: 36 };
}

/**
 * Unit plane on the XZ axis, normal +Y. 4 verts, 2 triangles.
 * Designed to be scaled by the entity transform.
 */
export function planeMesh(size = 1): MeshData {
  const s = size / 2;
  const v = new Float32Array([
    // position           normal     uv
    -s, 0,  s,   0, 1, 0,  0, 0,
     s, 0,  s,   0, 1, 0,  1, 0,
     s, 0, -s,   0, 1, 0,  1, 1,
    -s, 0, -s,   0, 1, 0,  0, 1,
  ]);
  const idx = new Uint32Array([0, 1, 2, 0, 2, 3]);
  return { vertices: v, indices: idx, vertexCount: 4, indexCount: 6 };
}

/** UV sphere centered at origin. radial = longitude rings, height = latitude rings. */
export function sphereMesh(radius = 1, radial = 24, height = 16): MeshData {
  const vCount = (radial + 1) * (height + 1);
  const v = new Float32Array(vCount * 8);
  const idx = new Uint32Array(radial * height * 6);

  let vi = 0;
  for (let y = 0; y <= height; y++) {
    const theta = (y / height) * Math.PI;
    const sinT = Math.sin(theta), cosT = Math.cos(theta);
    for (let x = 0; x <= radial; x++) {
      const phi = (x / radial) * Math.PI * 2;
      const sinP = Math.sin(phi), cosP = Math.cos(phi);
      const nx = sinT * cosP, ny = cosT, nz = sinT * sinP;
      v[vi++] = nx * radius;
      v[vi++] = ny * radius;
      v[vi++] = nz * radius;
      v[vi++] = nx;
      v[vi++] = ny;
      v[vi++] = nz;
      v[vi++] = x / radial;
      v[vi++] = y / height;
    }
  }

  let ii = 0;
  const stride = radial + 1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < radial; x++) {
      const a = y * stride + x;
      const b = a + stride;
      idx[ii++] = a;
      idx[ii++] = a + 1;
      idx[ii++] = b;
      idx[ii++] = a + 1;
      idx[ii++] = b + 1;
      idx[ii++] = b;
    }
  }

  return { vertices: v, indices: idx, vertexCount: vCount, indexCount: idx.length };
}

/** Cylinder with end caps. Y-axis aligned. radial = side segments. */
export function cylinderMesh(radius = 1, height = 2, radial = 24): MeshData {
  const halfH = height / 2;
  // 2 rings × (radial+1) for sides, plus 2 × radial+1 + 2 centers for caps.
  const sideV = (radial + 1) * 2;
  const capV = (radial + 1) * 2 + 2;
  const vCount = sideV + capV;
  const v = new Float32Array(vCount * 8);
  const idx = new Uint32Array(radial * 6 + radial * 6);

  let vi = 0;
  // Sides
  for (let ring = 0; ring < 2; ring++) {
    const y = ring === 0 ? -halfH : halfH;
    const uvV = ring === 0 ? 0 : 1;
    for (let i = 0; i <= radial; i++) {
      const t = (i / radial) * Math.PI * 2;
      const cx = Math.cos(t), cz = Math.sin(t);
      v[vi++] = cx * radius;
      v[vi++] = y;
      v[vi++] = cz * radius;
      v[vi++] = cx;
      v[vi++] = 0;
      v[vi++] = cz;
      v[vi++] = i / radial;
      v[vi++] = uvV;
    }
  }
  // Top cap
  const topCenterIdx = sideV;
  v[vi++] = 0; v[vi++] = halfH; v[vi++] = 0; v[vi++] = 0; v[vi++] = 1; v[vi++] = 0; v[vi++] = 0.5; v[vi++] = 0.5;
  for (let i = 0; i <= radial; i++) {
    const t = (i / radial) * Math.PI * 2;
    const cx = Math.cos(t), cz = Math.sin(t);
    v[vi++] = cx * radius;
    v[vi++] = halfH;
    v[vi++] = cz * radius;
    v[vi++] = 0;
    v[vi++] = 1;
    v[vi++] = 0;
    v[vi++] = 0.5 + 0.5 * cx;
    v[vi++] = 0.5 + 0.5 * cz;
  }
  // Bottom cap
  const bottomCenterIdx = sideV + radial + 2;
  v[vi++] = 0; v[vi++] = -halfH; v[vi++] = 0; v[vi++] = 0; v[vi++] = -1; v[vi++] = 0; v[vi++] = 0.5; v[vi++] = 0.5;
  for (let i = 0; i <= radial; i++) {
    const t = (i / radial) * Math.PI * 2;
    const cx = Math.cos(t), cz = Math.sin(t);
    v[vi++] = cx * radius;
    v[vi++] = -halfH;
    v[vi++] = cz * radius;
    v[vi++] = 0;
    v[vi++] = -1;
    v[vi++] = 0;
    v[vi++] = 0.5 + 0.5 * cx;
    v[vi++] = 0.5 + 0.5 * cz;
  }

  let ii = 0;
  // Sides — bottom ring is i, top ring is i + (radial+1). CCW from outside.
  for (let i = 0; i < radial; i++) {
    const a = i;
    const b = a + (radial + 1);
    idx[ii++] = a;
    idx[ii++] = b;
    idx[ii++] = a + 1;
    idx[ii++] = a + 1;
    idx[ii++] = b;
    idx[ii++] = b + 1;
  }
  // Top cap fan — CCW when viewed from +Y.
  for (let i = 0; i < radial; i++) {
    idx[ii++] = topCenterIdx;
    idx[ii++] = topCenterIdx + 2 + i;
    idx[ii++] = topCenterIdx + 1 + i;
  }
  // Bottom cap fan — CCW when viewed from -Y.
  for (let i = 0; i < radial; i++) {
    idx[ii++] = bottomCenterIdx;
    idx[ii++] = bottomCenterIdx + 1 + i;
    idx[ii++] = bottomCenterIdx + 2 + i;
  }

  // The cap-fan loop above runs radial*2 = 2*radial extra triangles past the
  // sides allocation. Reallocate to the exact final length.
  const sliced = idx.slice(0, ii);
  return { vertices: v, indices: sliced, vertexCount: vCount, indexCount: ii };
}
