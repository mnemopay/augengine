/**
 * Live scene state for the MCP host.
 *
 * MCP tool calls are stateless request/response, but scene authoring is not —
 * `add_entity` has to land in the scene `create_scene` just made. The registry
 * is that continuity: one Scene + ObservationBus + ActionDispatcher per
 * sceneId, for the lifetime of the server process.
 *
 * Observation subscriptions buffer into a bounded queue rather than streaming.
 * MCP stdio has no server-initiated event channel, so `subscribe_observation`
 * opens a queue and `poll_observation` drains it. The queue is capped so a
 * chatty scene cannot grow memory without bound while nobody polls.
 */

import {
  ObservationBus,
  Scene,
  type ActionDispatcher,
  type ObservationChannel,
  type ObservationEvent,
  type SceneSpec,
  type ViewFilter,
} from "@mnemopay/augengine";
import { buildDispatcher } from "./grammar.js";

export const DEFAULT_QUEUE_CAP = 1000;

interface Subscription {
  id: string;
  sceneId: string;
  channel: ObservationChannel;
  queue: ObservationEvent[];
  dropped: number;
  unsubscribe: () => void;
}

export interface SceneHandle {
  id: string;
  scene: Scene;
  bus: ObservationBus;
  dispatcher: ActionDispatcher;
  createdAt: string;
  promptHash: string | null;
}

export class SceneRegistry {
  private scenes = new Map<string, SceneHandle>();
  private subscriptions = new Map<string, Subscription>();
  /** promptHash+seed -> sceneId, so identical create_scene calls report a cache hit. */
  private byPromptKey = new Map<string, string>();
  private counter = 0;
  private subCounter = 0;
  private queueCap: number;

  constructor(opts: { queueCap?: number } = {}) {
    this.queueCap = opts.queueCap ?? DEFAULT_QUEUE_CAP;
  }

  create(spec: SceneSpec, promptKey?: string): SceneHandle {
    const id = `scene_${++this.counter}`;
    const scene = new Scene(spec);
    const bus = new ObservationBus();
    const handle: SceneHandle = {
      id,
      scene,
      bus,
      dispatcher: buildDispatcher(scene, bus),
      createdAt: new Date().toISOString(),
      promptHash: spec.source.kind === "generated" ? spec.source.promptHash : null,
    };
    this.scenes.set(id, handle);
    if (promptKey !== undefined) this.byPromptKey.set(promptKey, id);
    return handle;
  }

  /** Previously-created scene for an identical (prompt, seed), if any. */
  findByPromptKey(promptKey: string): SceneHandle | undefined {
    const id = this.byPromptKey.get(promptKey);
    return id === undefined ? undefined : this.scenes.get(id);
  }

  get(sceneId: string): SceneHandle | undefined {
    return this.scenes.get(sceneId);
  }

  /** Re-register verbs after entities are added post-creation. */
  rebuildDispatcher(sceneId: string): void {
    const handle = this.scenes.get(sceneId);
    if (!handle) return;
    handle.dispatcher = buildDispatcher(handle.scene, handle.bus);
  }

  list(): SceneHandle[] {
    return Array.from(this.scenes.values());
  }

  subscribe(
    sceneId: string,
    channel: ObservationChannel,
    filter: ViewFilter | null,
  ): Subscription {
    const handle = this.scenes.get(sceneId);
    if (!handle) throw new Error(`invalid params: unknown scene ${sceneId}`);
    const id = `sub_${++this.subCounter}`;
    const sub: Subscription = {
      id,
      sceneId,
      channel,
      queue: [],
      dropped: 0,
      unsubscribe: () => undefined,
    };
    sub.unsubscribe = handle.bus.subscribe(channel, filter, (event) => {
      if (sub.queue.length >= this.queueCap) {
        sub.queue.shift();
        sub.dropped++;
      }
      sub.queue.push(event);
    });
    this.subscriptions.set(id, sub);
    return sub;
  }

  poll(
    subscriptionId: string,
    maxEvents: number,
  ): { events: ObservationEvent[]; remaining: number; dropped: number } {
    const sub = this.subscriptions.get(subscriptionId);
    if (!sub) throw new Error(`invalid params: unknown subscription ${subscriptionId}`);
    const events = sub.queue.splice(0, Math.max(0, maxEvents));
    const dropped = sub.dropped;
    sub.dropped = 0;
    return { events, remaining: sub.queue.length, dropped };
  }

  unsubscribe(subscriptionId: string): boolean {
    const sub = this.subscriptions.get(subscriptionId);
    if (!sub) return false;
    sub.unsubscribe();
    this.subscriptions.delete(subscriptionId);
    return true;
  }
}
