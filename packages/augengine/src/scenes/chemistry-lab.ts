import {
  Tag,
  type Entity,
  type EntityId,
  type SceneSpec,
  type Vec3,
} from "../scene/types.js";
import { createReactiveBody } from "../chemistry/materials.js";
const id = (s: string) => s as EntityId;
function object(
  name: string,
  position: Vec3,
  scale: Vec3,
  color: Vec3,
): Entity {
  return {
    id: id(name),
    name,
    components: {
      [Tag.Transform]: {
        position,
        scale,
        rotation: [0, 0, 0, 1],
        parent: null,
      },
      [Tag.Mesh]: {
        geometry: { kind: "primitive", shape: "box" },
        material: {
          kind: "pbr",
          baseColor: color,
          metallic: 0.1,
          roughness: 0.45,
        },
        visible: true,
      },
    },
  };
}
/** Independent fresh instances: simulations never mutate a shared template. */
export function createChemistryLab(): SceneSpec {
  const floor = object(
    "lab.floor",
    [0, -0.1, 0],
    [12, 0.2, 10],
    [0.12, 0.16, 0.2],
  );
  floor.components[Tag.Collider] = { shape: "box", extents: [6, 0.1, 5] };
  const reactor = object(
    "lab.reactor",
    [-1.8, 0.65, 0],
    [1.1, 1.3, 1.1],
    [0.3, 0.08, 0.025],
  );
  reactor.components[Tag.Reactive] = createReactiveBody(
    { carbon: 0.02, oxygen: 0.02, nitrogen: 1 },
    720,
    {
      reactionIds: ["carbonOxidation"],
      volumeM3: 0.05,
      maxPressurePa: 250000,
      thermalLinks: [{ to: "lab.water", conductanceWPerK: 0.02 }],
    },
  );
  const water = object(
    "lab.water",
    [0, 0.55, 0],
    [1, 1.1, 1],
    [0.02, 0.35, 0.65],
  );
  water.components[Tag.Reactive] = createReactiveBody({ water: 0.4 }, 293.15, {
    waterPhaseEquilibrium: true,
  });
  const bodies: Entity[] = [];
  for (let i = 0; i < 4; i++) {
    const b = object(
      `lab.block.${i}`,
      [2, 0.5 + i * 0.85, 0],
      [0.65, 0.65, 0.65],
      [0.15 + i * 0.1, 0.5, 0.4],
    );
    b.components[Tag.Collider] = { shape: "box", extents: [0.325, 0.325, 0.325] };
    b.components[Tag.RigidBody] = {
      velocity: [0, 0, 0],
      angularVelocity: [0, 0, 0],
      mass: 1,
      isKinematic: false,
      friction: 0.7,
      restitution: 0.05,
      ccd: true,
    };
    bodies.push(b);
  }
  return {
    schemaVersion: 1,
    source: { kind: "authored", path: "chemistry-lab" },
    defaultCamera: id("lab.camera"),
    entities: [
      floor,
      reactor,
      water,
      ...bodies,
      {
        id: id("lab.camera"),
        name: "Lab camera",
        components: {
          [Tag.Transform]: {
            position: [0, 3.6, 8],
            rotation: [-Math.sin(0.12), 0, 0, Math.cos(0.12)],
            scale: [1, 1, 1],
            parent: null,
          },
          [Tag.Camera]: { fovY: 0.85, near: 0.05, far: 100, active: true },
        },
      },
      {
        id: id("lab.light"),
        name: "Lab light",
        components: {
          [Tag.Transform]: {
            position: [2, 6, 4],
            rotation: [0, 0, 0, 1],
            scale: [1, 1, 1],
            parent: null,
          },
          [Tag.Light]: {
            kind: "directional",
            color: [1, 0.95, 0.9],
            intensity: 2,
          },
        },
      },
    ],
  };
}
