/**
 * Loopback transport — in-process pub/sub between a single client and a
 * single server, useful for unit tests + single-player builds that want
 * the same code path as multiplayer.
 *
 * Phase 7 prototype. Real WebTransport implementation lives in a separate
 * `@mnemopay/augengine-net` package because the server side is Node-only.
 */

import type {
  ClientId,
  ClientToServer,
  NetServerTransport,
  NetTransport,
  ServerToClient,
} from "./types.js";

export class LoopbackPair {
  private serverListeners: Array<(clientId: ClientId, msg: ClientToServer) => void> = [];
  private clientListenersById = new Map<ClientId, Array<(msg: ServerToClient) => void>>();

  client(clientId: ClientId): NetTransport {
    if (!this.clientListenersById.has(clientId)) {
      this.clientListenersById.set(clientId, []);
    }
    const self = this;
    return {
      send(msg) {
        for (const cb of self.serverListeners) cb(clientId, msg);
      },
      onMessage(cb) {
        const arr = self.clientListenersById.get(clientId)!;
        arr.push(cb);
        return () => {
          const i = arr.indexOf(cb);
          if (i >= 0) arr.splice(i, 1);
        };
      },
      close() {
        self.clientListenersById.delete(clientId);
      },
    };
  }

  server(): NetServerTransport {
    const self = this;
    return {
      onMessage(cb) {
        self.serverListeners.push(cb);
        return () => {
          const i = self.serverListeners.indexOf(cb);
          if (i >= 0) self.serverListeners.splice(i, 1);
        };
      },
      sendTo(clientId, msg) {
        const arr = self.clientListenersById.get(clientId);
        if (!arr) return;
        for (const cb of arr) cb(msg);
      },
      broadcast(msg) {
        for (const arr of self.clientListenersById.values()) {
          for (const cb of arr) cb(msg);
        }
      },
      clients() {
        return [...self.clientListenersById.keys()];
      },
      close() {
        self.clientListenersById.clear();
        self.serverListeners.length = 0;
      },
    };
  }
}
