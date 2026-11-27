import type { Scene } from "../../scene/scene.js";
import type { Entity } from "../../scene/types.js";
import { Tag } from "../../scene/types.js";

/**
 * Converts the immediate physical surroundings of an agent into a compact
 * text prompt suitable for an LLM to read and reason about.
 */
export function serializeAgentObservation(scene: Scene, agentEntity: Entity, radius = 10): string {
  const myTx = agentEntity.components[Tag.Transform];
  if (!myTx) return "You are floating in an empty void.";

  let prompt = `You are ${agentEntity.name}. Your position is [${myTx.position.map(n => n.toFixed(1)).join(", ")}].\n`;
  prompt += "Entities and objects nearby:\n";

  let nearbyCount = 0;
  for (const other of scene.entities.values()) {
    if (other.id === agentEntity.id) continue;
    
    const tx = other.components[Tag.Transform];
    if (!tx) continue;

    // Fast distance check (squared)
    const dx = tx.position[0] - myTx.position[0];
    const dy = tx.position[1] - myTx.position[1];
    const dz = tx.position[2] - myTx.position[2];
    const distSq = dx * dx + dy * dy + dz * dz;

    if (distSq <= radius * radius) {
      nearbyCount++;
      const dist = Math.sqrt(distSq).toFixed(1);
      
      const isStatic = other.components[Tag.RigidBody]?.isKinematic === true || !other.components[Tag.RigidBody];
      const typeLabel = isStatic ? "Static Prop/Building" : "Active Entity";
      
      prompt += `- [${other.id}] ${other.name} (${typeLabel}) is ${dist} units away at [${tx.position.map(n => n.toFixed(1)).join(", ")}].\n`;
      
      const affordances = other.components[Tag.AffordanceSet];
      if (affordances && affordances.affordances.length > 0) {
        prompt += `  You can: ${affordances.affordances.join(", ")}\n`;
      }
    }
  }

  if (nearbyCount === 0) {
    prompt += "There is nothing nearby.\n";
  }

  prompt += `\nDecide your next action using the Action IR format: {"action": "verb", "target": "entity_id"}`;
  return prompt;
}
