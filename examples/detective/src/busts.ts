/**
 * Load the suspects' 3D busts: fetch each .glb, parse it with the engine's
 * glTF loader and upload it under the key a `{ kind: "gltf", url }` mesh
 * resolves to, plus its base-colour texture. Same pattern as the smoke-game.
 */
import { loadGlb } from "@mnemopay/augengine";
import type { LiveRenderer } from "@mnemopay/augengine/runtime/live";

export async function loadBust(renderer: LiveRenderer, url: string): Promise<void> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`bust ${url}: ${res.status}`);
  const loaded = loadGlb(new Uint8Array(await res.arrayBuffer()));
  for (const p of loaded.primitives) {
    const key = `gltf:${url}#${p.primitiveIndex}`;
    const m = p.mesh;
    renderer.uploadMesh(key, { vertices: m.vertices, indices: m.indices, vertexCount: m.vertexCount, indexCount: m.indexCount, ...(m.pbrMaterial ? { pbrMaterial: m.pbrMaterial } : {}) }, m.vertexStride);
    if (m.pbrMaterial?.baseColorTexture) await renderer.uploadPbrTextures(key, { baseColor: m.pbrMaterial.baseColorTexture });
  }
}
