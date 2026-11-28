/**
 * Split-sum environment BRDF for the natural-light tier.
 *
 * `integrateDfg` is the reference: the split-sum DFG terms [A, B] (specular =
 * F0·A + B) by GGX importance sampling with Smith–Schlick visibility
 * (k = α/2, Karis 2013). `DFG_LUT_WGSL` runs the SAME integral on the GPU once
 * per device into a 64×64 rgba16float LUT (u = N·V, v = perceptual roughness)
 * that the material shader samples.
 *
 * Why a LUT and not Karis's analytic mobile fit (ibl.ts envBrdfApprox, used
 * by the legacy gradient-sky path): measured against this integral the fit is
 * off by up to 0.16 (roughness 0.5, N·V 0.95: A 0.72 vs 0.88) — a visibly
 * dimmer mid-rough specular — and a 19-term polynomial fitted here still
 * missed by 0.07 near grazing angles. ibl-brdf.test.ts records both.
 *
 * `envIblResponse` mirrors pipeline-factory.ts `augengine_env_ibl` for a
 * uniform white environment (white furnace).
 */

export const DFG_LUT_SIZE = 64;
const DFG_SAMPLES_I = 256;
const DFG_SAMPLES_J = 4;

/** Numerically integrated split-sum DFG terms [A, B]. Same sampling as DFG_LUT_WGSL. */
export function integrateDfg(roughness: number, nDotV: number, samplesI = DFG_SAMPLES_I, samplesJ = DFG_SAMPLES_J): [number, number] {
  const a = Math.max(roughness * roughness, 1e-4);
  const a2 = a * a;
  const k = a / 2;
  const nv = Math.max(nDotV, 1e-4);
  const V: [number, number, number] = [Math.sqrt(Math.max(0, 1 - nv * nv)), 0, nv];
  let A = 0, B = 0;
  for (let i = 0; i < samplesI; i++) {
    for (let j = 0; j < samplesJ; j++) {
      const u1 = (i + 0.5) / samplesI;
      const phi = 2 * Math.PI * ((j + 0.5) / samplesJ + i * 0.618034);
      const cosT = Math.sqrt((1 - u1) / (1 + (a2 - 1) * u1));
      const sinT = Math.sqrt(Math.max(0, 1 - cosT * cosT));
      const H: [number, number, number] = [sinT * Math.cos(phi), sinT * Math.sin(phi), cosT];
      const vDotH = V[0] * H[0] + V[1] * H[1] + V[2] * H[2];
      const nDotL = 2 * vDotH * H[2] - V[2];
      if (nDotL <= 0 || vDotH <= 0) continue;
      const gV = nv / (nv * (1 - k) + k);
      const gL = nDotL / (nDotL * (1 - k) + k);
      const gVis = (gV * gL * vDotH) / (cosT * nv);
      const fc = Math.pow(1 - vDotH, 5);
      A += (1 - fc) * gVis;
      B += fc * gVis;
    }
  }
  const n = samplesI * samplesJ;
  return [A / n, B / n];
}

/** Texel-centre coordinates of the LUT: [N·V, roughness] for texel (x, y). */
export function dfgLutCoords(x: number, y: number): [number, number] {
  return [(x + 0.5) / DFG_LUT_SIZE, (y + 0.5) / DFG_LUT_SIZE];
}

export const DFG_LUT_WGSL = /* wgsl */ `
@group(0) @binding(0) var dst: texture_storage_2d<rgba16float, write>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let size = textureDimensions(dst);
  if (id.x >= size.x || id.y >= size.y) { return; }
  let nv = max((f32(id.x) + 0.5) / f32(size.x), 1e-4);
  let roughness = (f32(id.y) + 0.5) / f32(size.y);
  let a = max(roughness * roughness, 1e-4);
  let a2 = a * a;
  let k = a / 2.0;
  let V = vec3<f32>(sqrt(max(0.0, 1.0 - nv * nv)), 0.0, nv);
  var A = 0.0;
  var B = 0.0;
  for (var i = 0u; i < ${DFG_SAMPLES_I}u; i = i + 1u) {
    for (var j = 0u; j < ${DFG_SAMPLES_J}u; j = j + 1u) {
      let u1 = (f32(i) + 0.5) / ${DFG_SAMPLES_I}.0;
      let phi = 6.28318530718 * ((f32(j) + 0.5) / ${DFG_SAMPLES_J}.0 + f32(i) * 0.618034);
      let cosT = sqrt((1.0 - u1) / (1.0 + (a2 - 1.0) * u1));
      let sinT = sqrt(max(0.0, 1.0 - cosT * cosT));
      let H = vec3<f32>(sinT * cos(phi), sinT * sin(phi), cosT);
      let vDotH = dot(V, H);
      let nDotL = 2.0 * vDotH * H.z - V.z;
      if (nDotL <= 0.0 || vDotH <= 0.0) { continue; }
      let gV = nv / (nv * (1.0 - k) + k);
      let gL = nDotL / (nDotL * (1.0 - k) + k);
      let gVis = gV * gL * vDotH / (cosT * nv);
      let fc = pow(1.0 - vDotH, 5.0);
      A = A + (1.0 - fc) * gVis;
      B = B + fc * gVis;
    }
  }
  let n = f32(${DFG_SAMPLES_I * DFG_SAMPLES_J}u);
  textureStore(dst, vec2<i32>(id.xy), vec4<f32>(A / n, B / n, 0.0, 1.0));
}
`;

/** Create + fill the DFG LUT (one dispatch, submitted immediately). */
export function createDfgLut(device: GPUDevice): GPUTexture {
  const tex = device.createTexture({
    label: "env/dfg-lut",
    size: [DFG_LUT_SIZE, DFG_LUT_SIZE],
    format: "rgba16float",
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });
  const pipeline = device.createComputePipeline({
    label: "env/dfg-lut",
    layout: "auto",
    compute: { module: device.createShaderModule({ label: "env/dfg-lut", code: DFG_LUT_WGSL }), entryPoint: "main" },
  });
  const encoder = device.createCommandEncoder({ label: "env/dfg-lut" });
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: tex.createView() }] }));
  pass.dispatchWorkgroups(DFG_LUT_SIZE / 8, DFG_LUT_SIZE / 8);
  pass.end();
  device.queue.submit([encoder.finish()]);
  return tex;
}

/**
 * Reflected radiance of the shader's IBL under a uniform environment of
 * radiance 1 (prefiltered radiance = irradiance/π = 1), for a grey material.
 */
export function envIblResponse(
  albedo: number,
  metallic: number,
  roughness: number,
  nDotV: number,
  multiScatter = true,
  dfg: (r: number, nv: number) => [number, number] = (r, nv) => integrateDfg(r, nv, 128, 4),
): number {
  const F0 = 0.04 * (1 - metallic) + albedo * metallic;
  const [Ab, Bb] = dfg(roughness, Math.max(nDotV, 1e-4));
  const FssEss = F0 * Ab + Bb;
  const Ems = Math.min(1, Math.max(0, 1 - (Ab + Bb)));
  const Favg = F0 + (1 - F0) / 21;
  const multi = multiScatter ? ((FssEss * Favg) / (1 - Ems * Favg)) * Ems : 0;
  const kd = albedo * (1 - metallic) * Math.max(0, 1 - FssEss - multi);
  const pre = 1, irr = 1;
  return pre * FssEss + irr * (multi + kd);
}
