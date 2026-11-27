/**
 * PersonaMemoryRecorder — wires the ObservationBus into persona memory.
 *
 * Decision #7 of AUGENGINE-ARCHITECTURE.md says the ObservationBus is the
 * canonical event surface for cameras, speakers, AI agents. This recorder
 * subscribes to the channels persona memory cares about (Dialogue + Social
 * by default) and persists each event into the configured memory adapter,
 * namespaced by the target persona's `memoryId`.
 *
 * Without this, every call site that says/does something interesting to an
 * NPC has to remember to also write it into the brain. Wiring it through
 * the bus means *anything* published goes into memory automatically — same
 * source of truth for human players, AI bots, and replay/training corpora.
 */

import type { ObservationBus } from "../bus.js";
import type { ObservationEvent } from "../types.js";
import type {
  Entity,
  EntityId,
  ObservationChannel,
  PersonaComponent,
} from "../../scene/types.js";
import { Tag } from "../../scene/types.js";
import type { PersonaMemoryAdapter, PersonaObservation } from "../persona.js";

export interface PersonaMemoryRecorderOptions {
  bus: ObservationBus;
  memory: PersonaMemoryAdapter;
  /** Resolve an entity by id so we can look up its Persona. */
  resolveEntity: (id: EntityId) => Entity | undefined;
  /**
   * Pick the persona-bearing entity to record this event against. Default:
   * the event's `source`. For dialogue, you usually want to record on the
   * speaker. For player→NPC events, override to point at the listening NPC
   * so the memory lands under their persona namespace.
   */
  resolveTargetPersona?: (event: ObservationEvent) => EntityId | null;
  /** Channels to subscribe to. Default: Dialogue + Social. */
  channels?: ObservationChannel[];
  /** Turn an event into a memory string. Default: stringify payload. */
  renderContent?: (event: ObservationEvent) => string | null;
  /** Extra tags to attach to every observation. */
  tags?: string[];
}

export interface PersonaMemoryRecorder {
  stop(): void;
  /** How many events have been written since construction (success path). */
  count(): number;
  /** Drain the pending-write promises — test helper. */
  flush(): Promise<void>;
}

const DEFAULT_CHANNELS: ObservationChannel[] = ["Dialogue", "Social"];

export function startPersonaMemoryRecorder(
  opts: PersonaMemoryRecorderOptions,
): PersonaMemoryRecorder {
  const channels = opts.channels ?? DEFAULT_CHANNELS;
  const resolveTarget = opts.resolveTargetPersona ?? ((e) => e.source);
  const render = opts.renderContent ?? defaultRender;
  const unsubs: Array<() => void> = [];
  const pending = new Set<Promise<unknown>>();
  let written = 0;
  let stopped = false;

  for (const ch of channels) {
    const unsub = opts.bus.subscribe(ch, null, (event) => {
      if (stopped) return;
      const targetId = resolveTarget(event);
      if (!targetId) return;
      const target = opts.resolveEntity(targetId);
      if (!target) return;
      const persona = getPersona(target);
      if (!persona?.memoryId) return;

      const content = render(event);
      if (!content) return;

      const observation: PersonaObservation = {
        content,
        channel: event.channel,
        source: event.source,
        ts: event.ts,
        ...(opts.tags ? { tags: [...opts.tags] } : {}),
        meta: extractMeta(event),
      };

      const task = writeObservation(opts.memory, persona.memoryId, observation).then(
        (ok) => {
          if (ok) written++;
        },
      );
      pending.add(task);
      task.finally(() => pending.delete(task));
    });
    unsubs.push(unsub);
  }

  return {
    stop() {
      stopped = true;
      for (const u of unsubs) u();
      unsubs.length = 0;
    },
    count() {
      return written;
    },
    async flush() {
      while (pending.size > 0) {
        await Promise.allSettled([...pending]);
      }
    },
  };
}

async function writeObservation(
  memory: PersonaMemoryAdapter,
  memoryId: string,
  obs: PersonaObservation,
): Promise<boolean> {
  try {
    if (memory.rememberObservation) {
      await memory.rememberObservation(memoryId, obs);
    } else {
      await memory.remember(memoryId, obs.content, obs.tags);
    }
    return true;
  } catch {
    return false;
  }
}

function defaultRender(event: ObservationEvent): string | null {
  const payload = event.payload;
  if (payload == null) return null;
  if (typeof payload === "string") return payload;
  if (typeof payload === "object") {
    const obj = payload as Record<string, unknown>;
    if (typeof obj.text === "string") return obj.text;
    if (typeof obj.message === "string") return obj.message;
    if (typeof obj.line === "string") return obj.line;
    try {
      return JSON.stringify(obj);
    } catch {
      return null;
    }
  }
  return String(payload);
}

function extractMeta(event: ObservationEvent): Record<string, unknown> {
  const meta: Record<string, unknown> = { channel: event.channel };
  if (event.origin) meta.origin = event.origin;
  if (event.payload && typeof event.payload === "object") {
    const obj = event.payload as Record<string, unknown>;
    if (obj.meta && typeof obj.meta === "object") {
      Object.assign(meta, obj.meta as Record<string, unknown>);
    }
  }
  return meta;
}

function getPersona(entity: Entity): PersonaComponent | undefined {
  return entity.components[Tag.Persona] as PersonaComponent | undefined;
}
