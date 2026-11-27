/**
 * ObservationBus — typed pub/sub bus shared by cameras, speakers, AI agents.
 *
 * Decision #7 of AUGENGINE-ARCHITECTURE.md: every game event publishes here. Cameras
 * subscribe to Visual, speakers to Audible, UI to Inventory, agents to whichever
 * channels they care about. Same source of truth for human players, AI bots,
 * and replay/training corpora.
 *
 * In-process pub/sub. No broker dependency. Serializable event format means the
 * bus log can also be the replay log.
 */

import type {
  ObservationBus as IObservationBus,
  ObservationEvent,
  Unsubscribe,
  ViewFilter,
} from "./types.js";
import type { ObservationChannel, Vec3 } from "../scene/types.js";

interface Subscription {
  channel: ObservationChannel;
  filter: ViewFilter | null;
  cb: (event: ObservationEvent) => void;
}

export class ObservationBus implements IObservationBus {
  private subs: Subscription[] = [];
  private log: ObservationEvent[] = [];
  private logCap: number;

  constructor(opts: { logCap?: number } = {}) {
    this.logCap = opts.logCap ?? 5000;
  }

  publish(event: ObservationEvent): void {
    if (this.log.length >= this.logCap) {
      this.log.shift();
    }
    this.log.push(event);
    for (const s of this.subs) {
      if (s.channel !== event.channel) continue;
      if (s.filter && !passesFilter(event, s.filter)) continue;
      s.cb(event);
    }
  }

  subscribe(
    channel: ObservationChannel,
    filter: ViewFilter | null,
    cb: (event: ObservationEvent) => void,
  ): Unsubscribe {
    const sub: Subscription = { channel, filter, cb };
    this.subs.push(sub);
    return () => {
      const idx = this.subs.indexOf(sub);
      if (idx >= 0) this.subs.splice(idx, 1);
    };
  }

  /** Drain the rolling event log. Useful for replay export + agent training. */
  drainLog(): ObservationEvent[] {
    const out = this.log;
    this.log = [];
    return out;
  }

  /** Read the rolling log without clearing. */
  peekLog(): readonly ObservationEvent[] {
    return this.log;
  }
}

function passesFilter(event: ObservationEvent, filter: ViewFilter): boolean {
  if (filter.rangeMax !== undefined && filter.rangeMax > 0 && event.origin) {
    // Range check from origin to listener — but we don't know the listener's
    // position here. Phase 1 keeps this as a simple "events with origin pass
    // when range is unbounded; events without origin always pass." Phase 5
    // wires actual listener position via the subscriber identity.
    // Conservative: always pass unless we have explicit listener pos.
    void event.origin;
  }
  if (filter.allowedSourceTags && filter.allowedSourceTags.length > 0) {
    // Source tags live on entity components, looked up by the publisher
    // before publish. Phase 1 stub: always pass; the publisher is expected
    // to pre-filter where it matters.
  }
  return true;
}

/** Convenience publisher for spatial events that need an origin. */
export function spatial<T>(
  bus: IObservationBus,
  channel: ObservationChannel,
  payload: T,
  source: ObservationEvent["source"],
  origin: Vec3,
): void {
  bus.publish({
    channel,
    ts: new Date().toISOString(),
    payload,
    source,
    origin,
  });
}
