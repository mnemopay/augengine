/**
 * Built-in action grammar for MCP-authored scenes.
 *
 * `ActionDispatcher` refuses any verb that has not been registered and any
 * precondition/effect id it cannot resolve — correct, but it means a freshly
 * synthesised scene would reject every action an agent tried to dispatch.
 *
 * So the MCP host installs a small library of predicates and effects that are
 * genuinely implemented against the scene graph and the observation bus. They
 * are deliberately generic (move, speak, emote, look, set visibility) — enough
 * for an agent to drive a scene and *observe the consequence* through
 * `poll_observation`, which is the whole point of the observation bus being the
 * public agent API.
 *
 * Verbs are registered from the scene itself: every `AffordanceSet.affordances`
 * entry and every `Agent.permittedVerbs` entry becomes dispatchable. Gameplay
 * code that wants a stricter grammar registers its own dispatcher instead.
 */

import {
  Tag,
  type Action,
  type EntityId,
  type Scene,
  type Vec3,
  ActionDispatcher,
  type ObservationBus,
} from "@mnemopay/augengine";

/** Verbs always available, independent of what the scene declares. */
export const BASE_VERBS = ["say", "move", "look", "emote", "use", "talk"] as const;

export const BUILTIN_PREDICATE_IDS = [
  "actor_exists",
  "target_exists",
  "target_in_range",
] as const;

export const BUILTIN_EFFECT_IDS = [
  "move_to_target",
  "set_mood",
  "emit_dialogue",
  "set_visible",
] as const;

function worldPosition(scene: Scene, id: EntityId): Vec3 | null {
  const cache = scene.worldTransforms();
  const idx = cache.indexById.get(id);
  if (idx === undefined) return null;
  const base = idx * 16;
  const m = cache.worldMatrices;
  const x = m[base + 12];
  const y = m[base + 13];
  const z = m[base + 14];
  if (x === undefined || y === undefined || z === undefined) return null;
  return [x, y, z];
}

function targetPosition(scene: Scene, action: Action): Vec3 | null {
  const t = action.target;
  if (t === undefined) return null;
  if (Array.isArray(t)) return t as unknown as Vec3;
  return worldPosition(scene, t as EntityId);
}

function distance(a: Vec3, b: Vec3): number {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  const dz = a[2] - b[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * Build a dispatcher wired to this scene + bus, with the built-in grammar
 * installed and every verb the scene declares registered.
 */
export function buildDispatcher(scene: Scene, bus: ObservationBus): ActionDispatcher {
  const dispatcher = new ActionDispatcher({ scene, bus });

  for (const verb of BASE_VERBS) dispatcher.registerVerb(verb);
  for (const entity of scene) {
    const affordances = entity.components[Tag.AffordanceSet];
    if (affordances) {
      for (const verb of affordances.affordances) dispatcher.registerVerb(verb);
    }
    const agent = entity.components[Tag.Agent];
    if (agent) {
      for (const verb of agent.permittedVerbs) dispatcher.registerVerb(verb);
    }
  }

  dispatcher.registerPredicate("actor_exists", (s, action) => s.has(action.actor));
  dispatcher.registerPredicate("target_exists", (s, action) => {
    const t = action.target;
    if (t === undefined) return false;
    if (Array.isArray(t)) return true;
    return s.has(t as EntityId);
  });
  dispatcher.registerPredicate("target_in_range", (s, action) => {
    const actorPos = worldPosition(s, action.actor);
    const targetPos = targetPosition(s, action);
    if (!actorPos || !targetPos) return false;
    const max = typeof action.params.range === "number" ? action.params.range : 3;
    return distance(actorPos, targetPos) <= max;
  });

  dispatcher.registerEffect("move_to_target", (s, action) => {
    const dest = targetPosition(s, action);
    if (!dest) return;
    const entity = s.get(action.actor);
    const tx = entity?.components[Tag.Transform];
    if (!entity || !tx) return;
    // Stop short of the target so bodies do not interpenetrate.
    const from = worldPosition(s, action.actor) ?? tx.position;
    const gap = typeof action.params.standoff === "number" ? action.params.standoff : 1.2;
    const d = distance(from, dest);
    const next: Vec3 =
      d <= gap
        ? [from[0], tx.position[1], from[2]]
        : [
            dest[0] + ((from[0] - dest[0]) / d) * gap,
            tx.position[1],
            dest[2] + ((from[2] - dest[2]) / d) * gap,
          ];
    s.setLocalTransform(action.actor, next, tx.rotation, tx.scale);
  });

  dispatcher.registerEffect("set_mood", (s, action) => {
    const persona = s.get(action.actor)?.components[Tag.Persona];
    if (!persona) return;
    const mood = typeof action.params.mood === "number" ? action.params.mood : 0;
    persona.mood = Math.max(-1, Math.min(1, mood));
  });

  dispatcher.registerEffect("emit_dialogue", (_s, action) => {
    const text = typeof action.params.text === "string" ? action.params.text : "";
    bus.publish({
      channel: "Dialogue",
      ts: new Date().toISOString(),
      payload: { kind: "dialogue.line", text, verb: action.verb },
      source: action.actor,
    });
  });

  dispatcher.registerEffect("set_visible", (s, action) => {
    const mesh = s.get(action.actor)?.components[Tag.Mesh];
    if (!mesh) return;
    mesh.visible = action.params.visible !== false;
  });

  return dispatcher;
}
