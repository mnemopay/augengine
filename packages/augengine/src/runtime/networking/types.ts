/**
 * Networking types — Phase 7 skeleton.
 *
 * WebTransport server-authoritative state sync (primary) + WebRTC fallback
 * (Safari until WebTransport lands). Snapshot/delta replication, lag
 * compensation, voice via SFU.
 *
 * Phase 7 ships type contracts + a single-file local "loopback" transport
 * for unit tests. Real WebTransport server + protocol implementation lands
 * in a separate package (`@mnemopay/augengine-net`) since it requires Node-side
 * code that's outside the browser-only AugEngine core.
 */

import type { Action } from "../../ai/types.js";
import type { EntityId } from "../../scene/types.js";

export type ClientId = string & { readonly __brand: "ClientId" };
export type Tick = number;

export interface NetSnapshot {
  tick: Tick;
  /** Per-entity authoritative state (transform + key components). */
  entities: NetEntity[];
  /** Optional deltas-only payload — if present, applied on top of last snapshot. */
  delta?: boolean;
  /** Server clock (ms) the snapshot describes, for client interpolation. */
  serverTime?: number;
  /** Game events since the previous snapshot (hits, spawns), schema per game. */
  events?: Array<Record<string, unknown>>;
}

export interface NetEntity {
  id: EntityId;
  /** Position + rotation + key state. Schema TBD per game; serialized as JSON
   *  for phase 7 prototype, swap to a binary codec when latency matters. */
  state: Record<string, unknown>;
}

export type ClientToServer =
  | { kind: "join"; clientId: ClientId; nick?: string }
  | { kind: "leave"; clientId: ClientId }
  | { kind: "input"; tick: Tick; clientId: ClientId; actions: Action[] }
  | { kind: "ping"; t: number };

export type ServerToClient =
  | { kind: "welcome"; clientId: ClientId; tick: Tick }
  | { kind: "snapshot"; snapshot: NetSnapshot }
  | { kind: "actionResult"; tick: Tick; result: { ok: boolean; reason?: string } }
  | { kind: "pong"; t: number; serverT: number };

export interface NetTransport {
  /** Send a message to the server. */
  send(msg: ClientToServer): void;
  /** Subscribe to messages from the server. Returns unsubscribe. */
  onMessage(cb: (msg: ServerToClient) => void): () => void;
  /** Tear down the transport. */
  close(): void;
}

export interface NetServerTransport {
  /** Subscribe to messages from any client. */
  onMessage(cb: (clientId: ClientId, msg: ClientToServer) => void): () => void;
  /** Send to one client. */
  sendTo(clientId: ClientId, msg: ServerToClient): void;
  /** Broadcast to all connected clients. */
  broadcast(msg: ServerToClient): void;
  /** List currently connected clients. */
  clients(): ClientId[];
  close(): void;
}
