/**
 * One-call glTF loading for the live renderer: fetch a .glb, upload every
 * primitive (and its PBR textures) under the `gltf:<url>#<primitiveIndex>` keys
 * that `{ kind: "gltf", url }` mesh components resolve to, and optionally add the
 * file's node hierarchy to a scene. A thin wrapper over `loadGlb`,
 * `buildGltfSceneInstance` and `LiveRenderer.uploadMesh/uploadPbrTextures`.
 */

import { loadGlb, type LoadedGlb } from "../../scene/loaders/gltf.js";
import {
  buildGltfSceneInstance,
  type GltfSceneInstance,
  type GltfSceneInstanceOptions,
} from "../../scene/loaders/gltf-instance.js";
import type { Scene } from "../../scene/scene.js";
import type { LiveRenderer } from "./renderer-api.js";

export interface LoadGltfOptions extends Partial<Omit<GltfSceneInstanceOptions, "url">> {
  /**
   * Base for resolving the .glb URL and any external texture URIs. Defaults to
   * `location.href` in a browser.
   */
  baseUrl?: string;
  /** Custom fetch (tests, auth headers). Defaults to the global `fetch`. */
  fetch?: (url: string) => Promise<{ ok: boolean; status: number; statusText: string; arrayBuffer(): Promise<ArrayBuffer> }>;
  /** Skip PBR texture upload (untextured preview). Default false. */
  skipTextures?: boolean;
}

export interface LoadedGltfIntoRenderer {
  /** The parsed file (primitives, skins, animations). */
  loaded: LoadedGlb;
  /** Mesh keys uploaded, in primitive order (`gltf:<url>#<n>`). */
  meshKeys: string[];
  /** The scene entities created, when a scene was passed. */
  instance: GltfSceneInstance | null;
}

let instanceCounter = 0;

function resolveUri(uri: string, glbUrl: string, baseUrl: string | undefined): string {
  try {
    return baseUrl ? new URL(uri, new URL(glbUrl, baseUrl)).href : new URL(uri, glbUrl).href;
  } catch {
    return uri; // relative URL with no base available: leave it for the renderer
  }
}

/**
 * Load a .glb into a live renderer. Pass `scene` to also add the model's
 * entities (root at `opts.position`); pass `null` to upload meshes only.
 *
 * ```ts
 * const { instance } = await loadGltfIntoRenderer(renderer, scene, "/character.glb", { position: [0, 0, 0] });
 * ```
 */
export async function loadGltfIntoRenderer(
  renderer: Pick<LiveRenderer, "uploadMesh" | "uploadPbrTextures">,
  scene: Pick<Scene, "add"> | null,
  url: string,
  opts: LoadGltfOptions = {},
): Promise<LoadedGltfIntoRenderer> {
  const doFetch = opts.fetch ?? ((u: string) => fetch(u));
  const baseUrl = opts.baseUrl ?? (globalThis as { location?: { href: string } }).location?.href;
  const res = await doFetch(url);
  if (!res.ok) throw new Error(`loadGltfIntoRenderer(${url}): ${res.status} ${res.statusText}`);
  const loaded = loadGlb(new Uint8Array(await res.arrayBuffer()));
  if (loaded.primitives.length === 0) throw new Error(`loadGltfIntoRenderer(${url}): the file has no mesh primitive`);

  const meshKeys: string[] = [];
  for (const primitive of loaded.primitives) {
    const key = `gltf:${url}#${primitive.primitiveIndex}`;
    const mesh = primitive.mesh;
    renderer.uploadMesh(key, mesh, mesh.vertexStride);
    meshKeys.push(key);
    const pbr = mesh.pbrMaterial;
    if (pbr && !opts.skipTextures) {
      const src = <T extends { bytes?: Uint8Array; uri?: string }>(t: T): T =>
        t.uri && !t.bytes ? { ...t, uri: resolveUri(t.uri, url, baseUrl) } : t;
      const sources = {
        ...(pbr.baseColorTexture ? { baseColor: src(pbr.baseColorTexture) } : {}),
        ...(pbr.metallicRoughnessTexture ? { metallicRoughness: src(pbr.metallicRoughnessTexture) } : {}),
        ...(pbr.normalTexture ? { normal: src(pbr.normalTexture) } : {}),
      };
      if (Object.keys(sources).length > 0) await renderer.uploadPbrTextures(key, sources);
    }
  }

  let instance: GltfSceneInstance | null = null;
  if (scene) {
    const { baseUrl: _b, fetch: _f, skipTextures: _s, idPrefix, ...placement } = opts;
    instance = buildGltfSceneInstance(loaded, {
      ...placement,
      idPrefix: idPrefix ?? `gltf${instanceCounter++}`,
      url,
    });
    for (const e of instance.entities) scene.add(e);
  }
  return { loaded, meshKeys, instance };
}

/** Upload a baked GLB once and resolve the renderable primitive keys for each
 * detail level. Both manifest versions name their meshes in the GLB. V2
 * can have several primitives (and therefore several materials) per level. */
export async function loadLodIntoRenderer(
  renderer: Pick<LiveRenderer, "uploadMesh" | "uploadPbrTextures">,
  url: string,
  manifest: { format: string; lods: ReadonlyArray<{ level: number; mesh: string }> },
  opts: LoadGltfOptions = {},
): Promise<Map<number, string[]>> {
  if (!["augengine-lod/1", "augengine-lod/2"].includes(manifest.format)) throw new Error("Unsupported LOD manifest format");
  const { loaded, meshKeys } = await loadGltfIntoRenderer(renderer, null, url, opts);
  const levels = new Map<number, string[]>();
  for (const level of manifest.lods) {
    const meshIndex = loaded.parsed.root.meshes?.findIndex(mesh => mesh.name === level.mesh);
    const keys = loaded.primitives.filter(p => p.meshIndex === meshIndex).map(p => meshKeys[p.primitiveIndex]!);
    if (!keys.length || levels.has(level.level)) throw new Error("Invalid LOD manifest entry: " + level.mesh);
    levels.set(level.level, keys);
  }
  return levels;
}
