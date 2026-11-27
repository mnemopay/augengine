import type { GltfRoot } from "../scene/loaders/gltf-types.js";

export interface ImportFile {
  name: string;
  webkitRelativePath?: string;
  arrayBuffer(): Promise<ArrayBuffer>;
}

function cleanPath(path: string): string {
  const parts: string[] = [];
  for (const part of path.replace(/\\/g, "/").split("/")) {
    if (!part || part === ".") continue;
    if (part === "..") {
      if (!parts.length) throw new Error("glTF dependency escapes the selected folder");
      parts.pop();
    } else parts.push(part);
  }
  return parts.join("/");
}

/** Pack a selected JSON glTF and its selected dependencies into one portable GLB. */
export async function bundleGltfFiles(source: ImportFile, files: readonly ImportFile[]): Promise<{ bytes: Uint8Array; dependencies: string[] }> {
  const root = JSON.parse(new TextDecoder().decode(await source.arrayBuffer())) as GltfRoot;
  if (root.asset?.version !== "2.0") throw new Error("Only glTF 2.0 assets are supported");
  const sourcePath = cleanPath(source.webkitRelativePath || source.name);
  const base = sourcePath.includes("/") ? sourcePath.slice(0, sourcePath.lastIndexOf("/") + 1) : "";
  const selected = new Map(files.map((file) => [cleanPath(file.webkitRelativePath || file.name), file]));
  const dependencies = new Set<string>();
  const chunks: Uint8Array[] = [];
  let length = 0;
  function append(bytes: Uint8Array): number {
    const offset = length;
    const padded = new Uint8Array(Math.ceil(bytes.byteLength / 4) * 4);
    padded.set(bytes);
    chunks.push(padded);
    length += padded.byteLength;
    return offset;
  }
  async function resolve(uri: string): Promise<Uint8Array> {
    if (uri.startsWith("data:")) {
      const comma = uri.indexOf(",");
      if (comma < 0) throw new Error("Invalid glTF data URI");
      const raw = uri.slice(comma + 1);
      return uri.slice(0, comma).endsWith(";base64")
        ? Uint8Array.from(atob(raw), (c) => c.charCodeAt(0))
        : new TextEncoder().encode(decodeURIComponent(raw));
    }
    if (/^[a-z][a-z0-9+.-]*:|^\/\//i.test(uri)) throw new Error("Select local glTF dependencies instead of remote URLs");
    const path = cleanPath(base + decodeURIComponent(uri.split(/[?#]/)[0]!));
    let file = selected.get(path);
    if (!file) {
      // A multi-file picker does not retain directory paths. Accept a
      // basename only when it identifies exactly one selected dependency.
      const name = path.split("/").pop();
      const matches = files.filter((candidate) => candidate.name === name);
      if (matches.length === 1) file = matches[0];
    }
    if (!file) throw new Error(`Missing glTF dependency: ${uri}. Select the model folder or include its companion files.`);
    dependencies.add(file.webkitRelativePath || file.name);
    return new Uint8Array(await file.arrayBuffer());
  }
  const offsets: number[] = [];
  const sizes: number[] = [];
  for (const buffer of root.buffers ?? []) {
    if (!buffer.uri) throw new Error("JSON glTF buffer has no URI");
    const bytes = await resolve(buffer.uri);
    if (!Number.isSafeInteger(buffer.byteLength) || buffer.byteLength < 0 || bytes.byteLength < buffer.byteLength) throw new Error("glTF buffer is shorter than its declared length");
    sizes.push(buffer.byteLength);
    offsets.push(append(bytes.subarray(0, buffer.byteLength)));
  }
  for (const view of root.bufferViews ?? []) {
    const start = view.byteOffset ?? 0;
    const size = sizes[view.buffer];
    if (size === undefined || !Number.isSafeInteger(start) || start < 0 || !Number.isSafeInteger(view.byteLength) || view.byteLength < 0 || start + view.byteLength > size) throw new Error("glTF bufferView is outside its buffer");
    if (view.extensions?.EXT_meshopt_compression) throw new Error("Import a packed GLB for meshopt-compressed JSON glTF");
    view.byteOffset = offsets[view.buffer]! + start;
    view.buffer = 0;
  }
  root.bufferViews ??= [];
  for (const image of root.images ?? []) {
    if (!image.uri) continue;
    const uri = image.uri;
    const bytes = await resolve(uri);
    const extension = uri.split(/[?#]/)[0]!.split(".").pop()?.toLowerCase();
    const mimeType = image.mimeType ?? (uri.startsWith("data:") ? uri.slice(5, uri.indexOf(";") > 0 ? uri.indexOf(";") : uri.indexOf(",")) : ({ png: "image/png", jpg: "image/jpeg", jpeg: "image/jpeg", webp: "image/webp", ktx2: "image/ktx2" } as Record<string, string>)[extension ?? ""]);
    if (!mimeType) throw new Error(`Unknown glTF image type: ${uri}`);
    image.mimeType = mimeType;
    image.bufferView = root.bufferViews.length;
    root.bufferViews.push({ buffer: 0, byteOffset: append(bytes), byteLength: bytes.byteLength });
    delete image.uri;
  }
  root.buffers = [{ byteLength: length }];
  const encoded = new TextEncoder().encode(JSON.stringify(root));
  const jsonLength = Math.ceil(encoded.byteLength / 4) * 4;
  const bytes = new Uint8Array(12 + 8 + jsonLength + 8 + length);
  const view = new DataView(bytes.buffer);
  view.setUint32(0, 0x46546c67, true);
  view.setUint32(4, 2, true);
  view.setUint32(8, bytes.length, true);
  view.setUint32(12, jsonLength, true);
  view.setUint32(16, 0x4e4f534a, true);
  bytes.fill(0x20, 20, 20 + jsonLength);
  bytes.set(encoded, 20);
  view.setUint32(20 + jsonLength, length, true);
  view.setUint32(24 + jsonLength, 0x004e4942, true);
  let cursor = 28 + jsonLength;
  for (const chunk of chunks) { bytes.set(chunk, cursor); cursor += chunk.byteLength; }
  return { bytes, dependencies: [...dependencies] };
}
