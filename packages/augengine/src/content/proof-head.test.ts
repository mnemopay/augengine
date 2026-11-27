import { describe, expect, it } from "vitest";
import { Tag } from "../scene/types.js";
import { ARKIT_BLENDSHAPE_COUNT } from "../runtime/live/webgpu/blendshape-constants.js";
import { packMorphTargetsForGpu } from "../runtime/live/webgpu/morph-deltas.js";
import {
  PROOF_HEAD_ENTITY_ID,
  PROOF_HEAD_LINE_BLINK_S,
  PROOF_HEAD_LINE_BROW_S,
  PROOF_HEAD_LINE_DURATION,
  PROOF_HEAD_LINE_SPEAK_S,
  PROOF_HEAD_LINE_SWALLOW_S,
  PROOF_HEAD_LINE_TEXT,
  PROOF_HEAD_MESH_KEY,
  PROOF_HEAD_VERTEX_STRIDE,
  buildProofHeadMesh,
  deformProofHead,
  proofHeadEntity,
  proofHeadLineWeights,
} from "./proof-head.js";

const mesh = buildProofHeadMesh();

function meanAxis(positions: Float32Array, indices: readonly number[], axis: 0 | 1 | 2): number {
  let sum = 0;
  for (const index of indices) sum += positions[index * 3 + axis] ?? 0;
  return sum / indices.length;
}

function rest(): Float32Array {
  return deformProofHead(mesh, new Float32Array(ARKIT_BLENDSHAPE_COUNT));
}

describe("proof head", () => {
  it("faces +Z, with the character's left lid on +X", () => {
    let maxZ = -Infinity;
    let minZ = Infinity;
    for (let i = 0; i < mesh.vertexCount; i++) {
      const z = mesh.vertices[i * PROOF_HEAD_VERTEX_STRIDE + 2] ?? 0;
      if (z > maxZ) maxZ = z;
      if (z < minZ) minZ = z;
    }
    expect(maxZ).toBeGreaterThan(0.1);
    expect(minZ).toBeLessThan(-0.05);
    const posed = rest();
    expect(meanAxis(posed, mesh.regions.lidL, 0)).toBeGreaterThan(0.01);
    expect(meanAxis(posed, mesh.regions.lidR, 0)).toBeLessThan(-0.01);
  });

  it("winds the nose toward the camera", () => {
    let bestZ = -Infinity;
    let bestNormalZ = 0;
    for (let t = 0; t < mesh.indexCount; t += 3) {
      const ia = mesh.indices[t]!;
      const ib = mesh.indices[t + 1]!;
      const ic = mesh.indices[t + 2]!;
      const p = (index: number) => [
        mesh.vertices[index * PROOF_HEAD_VERTEX_STRIDE] ?? 0,
        mesh.vertices[index * PROOF_HEAD_VERTEX_STRIDE + 1] ?? 0,
        mesh.vertices[index * PROOF_HEAD_VERTEX_STRIDE + 2] ?? 0,
      ];
      const a = p(ia);
      const b = p(ib);
      const c = p(ic);
      const cx = (a[0]! + b[0]! + c[0]!) / 3;
      const cz = (a[2]! + b[2]! + c[2]!) / 3;
      if (Math.abs(cx) > 0.02 || cz < 0.09) continue;
      const abx = b[0]! - a[0]!;
      const aby = b[1]! - a[1]!;
      const acx = c[0]! - a[0]!;
      const acy = c[1]! - a[1]!;
      const nz = abx * acy - aby * acx;
      if (cz > bestZ) {
        bestZ = cz;
        bestNormalZ = nz;
      }
    }
    expect(bestZ).toBeGreaterThan(0.09);
    expect(bestNormalZ).toBeGreaterThan(0);
  });

  it("drops the lower lip on the slot Cherry writes for an open vowel", () => {
    expect(mesh.morphTargets).toHaveLength(ARKIT_BLENDSHAPE_COUNT);
    expect(mesh.slots.jawOpen).toBe(23);
    expect(mesh.slots.jawOpen).not.toBe(17);
    expect(mesh.slots.blinkL).toBe(8);
    expect(mesh.slots.browInnerUp).toBe(1);
    expect(mesh.slots.tongueOut).toBe(51);
    const weights = new Float32Array(ARKIT_BLENDSHAPE_COUNT);
    weights[mesh.slots.jawOpen] = 1;
    const open = deformProofHead(mesh, weights);
    const base = rest();
    const lipDrop = meanAxis(base, mesh.regions.lowerLip, 1) - meanAxis(open, mesh.regions.lowerLip, 1);
    expect(lipDrop).toBeGreaterThan(0.01);
    const browMove = Math.abs(
      meanAxis(base, mesh.regions.browInner, 1) - meanAxis(open, mesh.regions.browInner, 1),
    );
    expect(browMove).toBeLessThan(0.001);

    const packed = packMorphTargetsForGpu(mesh.vertexCount, mesh.morphTargets ?? []);
    const lip = mesh.regions.lowerLip[0]!;
    const jawY = packed.positionDeltas[(lip * ARKIT_BLENDSHAPE_COUNT + mesh.slots.jawOpen) * 3 + 1] ?? 0;
    const canonicalJawY = packed.positionDeltas[(lip * ARKIT_BLENDSHAPE_COUNT + 17) * 3 + 1] ?? 0;
    expect(jawY).toBeLessThan(-0.01);
    expect(canonicalJawY).toBe(0);
    const unused = mesh.morphTargets?.[0]?.positions ?? new Float32Array();
    let unusedSum = 0;
    for (let i = 0; i < unused.length; i++) unusedSum += Math.abs(unused[i] ?? 0);
    expect(unusedSum).toBe(0);
  });

  it("closes the left lid without closing the right one the same amount", () => {
    const weights = new Float32Array(ARKIT_BLENDSHAPE_COUNT);
    weights[mesh.slots.blinkL] = 1;
    const shut = deformProofHead(mesh, weights);
    const base = rest();
    const left = meanAxis(base, mesh.regions.lidL, 1) - meanAxis(shut, mesh.regions.lidL, 1);
    const right = Math.abs(meanAxis(base, mesh.regions.lidR, 1) - meanAxis(shut, mesh.regions.lidR, 1));
    expect(left).toBeGreaterThan(0.008);
    expect(right).toBeLessThan(0.001);
  });

  it("pushes the tongue forward on Cherry's tongue slot", () => {
    const weights = new Float32Array(ARKIT_BLENDSHAPE_COUNT);
    weights[mesh.slots.tongueOut] = 1;
    const out = deformProofHead(mesh, weights);
    const base = rest();
    const forward = meanAxis(out, mesh.regions.tongue, 2) - meanAxis(base, mesh.regions.tongue, 2);
    expect(forward).toBeGreaterThan(0.015);
  });

  it("speaks one line: a swallowed consonant, then the vowel, then a late uneven brow and blink", () => {
    expect(PROOF_HEAD_LINE_TEXT.length).toBeGreaterThan(0);
    const speak = proofHeadLineWeights(PROOF_HEAD_LINE_SPEAK_S);
    const swallow = proofHeadLineWeights(PROOF_HEAD_LINE_SWALLOW_S);
    const brow = proofHeadLineWeights(PROOF_HEAD_LINE_BROW_S);
    const blink = proofHeadLineWeights(PROOF_HEAD_LINE_BLINK_S);
    expect(speak).toHaveLength(ARKIT_BLENDSHAPE_COUNT);
    expect(speak[mesh.slots.jawOpen] ?? 0).toBeGreaterThan(0.5);
    expect(swallow[mesh.slots.jawOpen] ?? 0).toBeGreaterThan(0);
    expect(swallow[mesh.slots.jawOpen] ?? 0).toBeLessThan(0.5);

    const base = rest();
    const spoken = deformProofHead(mesh, speak);
    const swallowed = deformProofHead(mesh, swallow);
    const speakDrop = meanAxis(base, mesh.regions.lowerLip, 1) - meanAxis(spoken, mesh.regions.lowerLip, 1);
    const swallowDrop = meanAxis(base, mesh.regions.lowerLip, 1) - meanAxis(swallowed, mesh.regions.lowerLip, 1);
    expect(speakDrop).toBeGreaterThan(swallowDrop * 1.5);

    expect(brow[mesh.slots.jawOpen] ?? 0).toBe(0);
    expect(brow[mesh.slots.browInnerUp] ?? 0).toBeGreaterThan(0.5);
    expect(brow[mesh.slots.browOuterUpL] ?? 0).toBeGreaterThan(brow[mesh.slots.browOuterUpR] ?? 0);
    const raised = deformProofHead(mesh, brow);
    expect(meanAxis(raised, mesh.regions.browInner, 1) - meanAxis(base, mesh.regions.browInner, 1)).toBeGreaterThan(0.004);
    expect(Math.abs(meanAxis(raised, mesh.regions.lowerLip, 1) - meanAxis(base, mesh.regions.lowerLip, 1))).toBeLessThan(0.001);

    expect((blink[mesh.slots.blinkL] ?? 0) - (blink[mesh.slots.blinkR] ?? 0)).toBeGreaterThan(0.5);
    const winked = deformProofHead(mesh, blink);
    const leftDrop = meanAxis(base, mesh.regions.lidL, 1) - meanAxis(winked, mesh.regions.lidL, 1);
    const rightDrop = meanAxis(base, mesh.regions.lidR, 1) - meanAxis(winked, mesh.regions.lidR, 1);
    expect(leftDrop - rightDrop).toBeGreaterThan(0.005);

    const looped = proofHeadLineWeights(PROOF_HEAD_LINE_SPEAK_S + PROOF_HEAD_LINE_DURATION);
    expect(looped[mesh.slots.jawOpen]).toBeCloseTo(speak[mesh.slots.jawOpen] ?? 0);
  });

  it("registers the procedural mesh key the renderer already resolves", () => {
    const entity = proofHeadEntity();
    expect(entity.id).toBe(PROOF_HEAD_ENTITY_ID);
    const geometry = entity.components[Tag.Mesh]?.geometry;
    expect(geometry).toEqual({ kind: "procedural", generator: "proof-head", seed: 1 });
    expect(PROOF_HEAD_MESH_KEY).toBe("proc:proof-head:1");
    expect(mesh.vertices.length).toBe(mesh.vertexCount * PROOF_HEAD_VERTEX_STRIDE);
    expect(mesh.colors?.length).toBe(mesh.vertexCount * 4);
    expect(mesh.indexCount % 3).toBe(0);
  });
});
