import {
  m4DecomposeTRS,
  qSlerp,
  v3Lerp,
  type MutQuat,
  type MutVec3,
} from "../math.js";
import {
  Tag,
  type Entity,
  type EntityId,
  type MaterialRef,
  type Quat,
  type Vec3,
} from "../types.js";
import type { Scene } from "../scene.js";
import type {
  AugEngineAnimationFromGltf,
  AugEngineAnimationTrack,
  LoadedGlb,
} from "./gltf.js";

export interface GltfSceneInstanceOptions {
  idPrefix: string;
  url: string;
  position?: Vec3;
  rotation?: Quat;
  scale?: Vec3;
  material?: MaterialRef;
  visible?: boolean;
}

export interface GltfSceneInstance {
  rootEntityId: EntityId;
  nodeToEntity: Map<number, EntityId>;
  meshEntityIds: EntityId[];
  jointEntityIds: EntityId[];
  entities: Entity[];
  animations: AugEngineAnimationFromGltf[];
}

const DEFAULT_MATERIAL: MaterialRef = {
  kind: "pbr",
  baseColor: [0.85, 0.85, 0.85],
  metallic: 0,
  roughness: 0.7,
};

export function buildGltfSceneInstance(
  loaded: LoadedGlb,
  opts: GltfSceneInstanceOptions,
): GltfSceneInstance {
  const rootEntityId = `${opts.idPrefix}_root` as EntityId;
  const entities: Entity[] = [{
    id: rootEntityId,
    name: `${opts.idPrefix} glTF root`,
    components: {
      [Tag.Transform]: {
        position: opts.position ?? [0, 0, 0],
        rotation: opts.rotation ?? [0, 0, 0, 1],
        scale: opts.scale ?? [1, 1, 1],
        parent: null,
      },
    },
  }];
  const nodeToEntity = new Map<number, EntityId>();
  const root = loaded.parsed.root;
  const nodes = root.nodes ?? [];
  const parentByNode = new Map<number, number>();
  for (const [nodeIndex, node] of nodes.entries()) {
    for (const child of node.children ?? []) parentByNode.set(child, nodeIndex);
  }

  for (const [nodeIndex, node] of nodes.entries()) {
    const entityId = `${opts.idPrefix}_node_${nodeIndex}` as EntityId;
    nodeToEntity.set(nodeIndex, entityId);
    const tx = localTransformOfNode(node);
    const parentNode = parentByNode.get(nodeIndex);
    entities.push({
      id: entityId,
      name: node.name ?? `glTF node ${nodeIndex}`,
      components: {
        [Tag.Transform]: {
          ...tx,
          parent: parentNode === undefined
            ? rootEntityId
            : (`${opts.idPrefix}_node_${parentNode}` as EntityId),
        },
      },
    });
  }

  const meshEntityIds: EntityId[] = [];
  for (const [nodeIndex, node] of nodes.entries()) {
    if (node.mesh === undefined) continue;
    const primitiveCount = loaded.primitives.filter((p) => p.meshIndex === node.mesh).length;
    for (let primitiveOffset = 0; primitiveOffset < primitiveCount; primitiveOffset++) {
      const primitive = loaded.primitives.find((p) =>
        p.meshIndex === node.mesh && p.meshPrimitiveIndex === primitiveOffset
      );
      if (!primitive) continue;
      const id = primitiveCount === 1
        ? nodeToEntity.get(nodeIndex)!
        : (`${opts.idPrefix}_node_${nodeIndex}_primitive_${primitiveOffset}` as EntityId);
      if (primitiveCount > 1) {
        entities.push({
          id,
          name: `${node.name ?? `glTF node ${nodeIndex}`} primitive ${primitiveOffset}`,
          components: {
            [Tag.Transform]: {
              position: [0, 0, 0],
              rotation: [0, 0, 0, 1],
              scale: [1, 1, 1],
              parent: nodeToEntity.get(nodeIndex)!,
            },
          },
        });
      }
      const entity = entities.find((e) => e.id === id)!;
      entity.components[Tag.Mesh] = {
        geometry: { kind: "gltf", url: opts.url, primitiveIndex: primitive.primitiveIndex },
        material: opts.material ?? DEFAULT_MATERIAL,
        visible: opts.visible ?? true,
      };
      if (node.skin !== undefined) {
        // Each skinned mesh node binds its OWN skin (multi-skin characters
        // share one skeleton but carry per-mesh inverse-bind matrices). Fall
        // back to the first skin for legacy single-skin assets.
        const nodeSkin = loaded.skins[node.skin] ?? loaded.skin;
        if (nodeSkin) {
          entity.components[Tag.Skin] = {
            jointEntities: nodeSkin.joints.map((jointNode) => nodeToEntity.get(jointNode)!).filter(Boolean),
            inverseBindMatrices: nodeSkin.inverseBindMatrices,
            rootEntity: rootEntityId,
          };
        }
      }
      meshEntityIds.push(id);
    }
  }

  const jointEntityIds = loaded.skin
    ? loaded.skin.joints.map((jointNode) => nodeToEntity.get(jointNode)!).filter(Boolean)
    : [];
  return {
    rootEntityId,
    nodeToEntity,
    meshEntityIds,
    jointEntityIds,
    entities,
    animations: loaded.animations,
  };
}

export function applyGltfAnimation(
  scene: Scene,
  instance: Pick<GltfSceneInstance, "nodeToEntity">,
  clip: AugEngineAnimationFromGltf,
  timeSec: number,
  opts: { loop?: boolean } = {},
): void {
  const t = opts.loop === false || clip.duration <= 0
    ? Math.max(0, Math.min(clip.duration, timeSec))
    : ((timeSec % clip.duration) + clip.duration) % clip.duration;
  for (const track of clip.tracks) {
    const id = instance.nodeToEntity.get(track.node);
    if (!id) continue;
    const entity = scene.get(id);
    const tx = entity?.components[Tag.Transform];
    if (!entity || !tx) continue;
    const sampled = sampleGltfAnimationTrack(track, t);
    if (!sampled) continue;
    if (track.path === "translation") {
      scene.setLocalTransform(id, sampled as Vec3, tx.rotation, tx.scale);
    } else if (track.path === "rotation") {
      scene.setLocalTransform(id, tx.position, sampled as Quat, tx.scale);
    } else {
      scene.setLocalTransform(id, tx.position, tx.rotation, sampled as Vec3);
    }
  }
}

export function sampleGltfAnimationTrack(
  track: AugEngineAnimationTrack,
  timeSec: number,
): Vec3 | Quat | null {
  const times = track.times;
  if (times.length === 0) return null;
  const componentCount = track.path === "rotation" ? 4 : 3;
  let lo = 0;
  let hi = times.length - 1;
  if (timeSec <= times[0]!) {
    hi = 0;
  } else if (timeSec >= times[hi]!) {
    lo = hi;
  } else {
    while (lo + 1 < hi) {
      const mid = (lo + hi) >> 1;
      if (times[mid]! <= timeSec) lo = mid;
      else hi = mid;
    }
  }
  const alpha = track.interpolation === "STEP" || hi === lo
    ? 0
    : (timeSec - times[lo]!) / (times[hi]! - times[lo]!);
  if (track.path === "rotation") {
    const a = readTrackQuat(track, lo);
    const b = readTrackQuat(track, hi);
    const out: MutQuat = [0, 0, 0, 1];
    return qSlerp(out, a, b, alpha);
  }
  const a = readTrackVec3(track, lo, componentCount);
  const b = readTrackVec3(track, hi, componentCount);
  const out: MutVec3 = [0, 0, 0];
  return v3Lerp(out, a, b, alpha);
}

function localTransformOfNode(node: {
  translation?: Vec3;
  rotation?: Quat;
  scale?: Vec3;
  matrix?: number[];
}): { position: Vec3; rotation: Quat; scale: Vec3 } {
  if (node.matrix && node.matrix.length >= 16) {
    const t: MutVec3 = [0, 0, 0];
    const r: MutQuat = [0, 0, 0, 1];
    const s: MutVec3 = [1, 1, 1];
    m4DecomposeTRS(new Float32Array(node.matrix), t, r, s);
    return { position: t, rotation: r, scale: s };
  }
  return {
    position: node.translation ?? [0, 0, 0],
    rotation: node.rotation ?? [0, 0, 0, 1],
    scale: node.scale ?? [1, 1, 1],
  };
}

function readTrackVec3(track: AugEngineAnimationTrack, key: number, componentCount: number): Vec3 {
  const base = track.interpolation === "CUBICSPLINE"
    ? (key * 3 + 1) * componentCount
    : key * componentCount;
  return [
    track.values[base] ?? 0,
    track.values[base + 1] ?? 0,
    track.values[base + 2] ?? 0,
  ];
}

function readTrackQuat(track: AugEngineAnimationTrack, key: number): Quat {
  const base = track.interpolation === "CUBICSPLINE" ? (key * 3 + 1) * 4 : key * 4;
  return [
    track.values[base] ?? 0,
    track.values[base + 1] ?? 0,
    track.values[base + 2] ?? 0,
    track.values[base + 3] ?? 1,
  ];
}
