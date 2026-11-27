/**
 * MCP tools for the gate-4 detective game. An agent plays (or inspects) a
 * session through the same rule engine as the browser game: every fact comes
 * from `@mnemopay/augengine-detective`'s deterministic case, dialogue is the
 * rule gist (offline mode), and suspect memory persists per playerId.
 *
 * Ground truth stays hidden until an accusation is made, unless the server is
 * started with AUGENGINE_DETECTIVE_DEBUG=1.
 */
import { randomUUID } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import {
  CASE_INTRO,
  CASE_TITLE,
  act,
  blankMemory,
  coerceMemory,
  newGame,
  publicView,
  recordAccusation,
  recordAsk,
  recordPresent,
  rememberLine,
  startSession,
  type Action,
  type GameState,
  type MemoryStore,
  type PlayerMemory,
  type SuspectId,
} from "@mnemopay/augengine-detective";

const obj = { type: "object" } as const;

export const DETECTIVE_TOOLS = [
  {
    name: "detective_new_session",
    description:
      `Start a new session of the AugEngine detective game "${CASE_TITLE}". Returns a sessionId, the case intro and the public view (rooms, objects here, suspects, notebook). Pass the same playerId across sessions and the suspects remember you.`,
    inputSchema: { ...obj, properties: { playerId: { type: "string", description: "Stable id for this player; suspects remember them across sessions." } } },
  },
  {
    name: "detective_status",
    description: "Read the public view of a detective session: current room, objects here, suspects, notebook, what was presented to whom, which stories cracked, and the accusation result if any.",
    inputSchema: { ...obj, properties: { sessionId: { type: "string" } }, required: ["sessionId"] },
  },
  {
    name: "detective_act",
    description:
      "Take one game action through the rule engine. action is one of: {type:'move', room}, {type:'examine', objectId}, {type:'ask', suspect, text}, {type:'present', suspect, evidence}, {type:'accuse', suspect, method, motive, evidence:[...]}. Suspects are mara|desmond|ines|teodor; questioning happens in the parlour. Rooms: parlour|dome|library|kitchen|conservatory. Methods: poison_tea|struck|fall|smothered. Motives: inheritance|stolen_research|credit|grudge. One accusation ends the case.",
    inputSchema: {
      ...obj,
      properties: { sessionId: { type: "string" }, action: { type: "object" } },
      required: ["sessionId", "action"],
    },
  },
];

interface Session {
  id: string;
  playerId: string;
  state: GameState;
  greeted: Set<SuspectId>;
}

class JsonFileMemoryStore implements MemoryStore {
  constructor(private readonly path: string) {}
  private async all(): Promise<Record<string, unknown>> {
    try {
      return JSON.parse(await readFile(this.path, "utf8")) as Record<string, unknown>;
    } catch {
      return {};
    }
  }
  async load(playerId: string): Promise<PlayerMemory | null> {
    const raw = (await this.all())[playerId];
    return raw ? coerceMemory(playerId, raw) : null;
  }
  async save(memory: PlayerMemory): Promise<void> {
    const all = await this.all();
    all[memory.playerId] = memory;
    await writeFile(this.path, JSON.stringify(all, null, 2));
  }
}

class RamMemoryStore implements MemoryStore {
  private m = new Map<string, PlayerMemory>();
  async load(playerId: string) {
    return this.m.get(playerId) ?? null;
  }
  async save(memory: PlayerMemory) {
    this.m.set(memory.playerId, structuredClone(memory));
  }
}

export class DetectiveSessions {
  private sessions = new Map<string, Session>();
  readonly store: MemoryStore;
  readonly debug: boolean;

  constructor(opts: { memoryPath?: string; debug?: boolean } = {}) {
    const path = opts.memoryPath ?? process.env.AUGENGINE_DETECTIVE_MEMORY;
    this.store = path ? new JsonFileMemoryStore(path) : new RamMemoryStore();
    this.debug = opts.debug ?? process.env.AUGENGINE_DETECTIVE_DEBUG === "1";
  }

  private get(id: unknown): Session {
    if (typeof id !== "string") throw new Error("invalid params: sessionId must be a string");
    const s = this.sessions.get(id);
    if (!s) throw new Error(`unknown sessionId: ${id}`);
    return s;
  }

  async run(name: string, args: Record<string, unknown>): Promise<unknown> {
    switch (name) {
      case "detective_new_session": {
        const playerId = typeof args.playerId === "string" && args.playerId.trim() ? args.playerId.trim().slice(0, 80) : "agent";
        const id = randomUUID();
        const mem = startSession((await this.store.load(playerId)) ?? blankMemory(playerId), id);
        await this.store.save(mem);
        const session: Session = { id, playerId, state: newGame(), greeted: new Set() };
        this.sessions.set(id, session);
        return {
          sessionId: id,
          title: CASE_TITLE,
          intro: CASE_INTRO,
          previousSessions: mem.sessions - 1,
          view: publicView(session.state, { revealTruth: this.debug }),
        };
      }
      case "detective_status": {
        const s = this.get(args.sessionId);
        return publicView(s.state, { revealTruth: this.debug });
      }
      case "detective_act": {
        const s = this.get(args.sessionId);
        const action = args.action as Action | undefined;
        if (!action || typeof action !== "object" || typeof (action as { type?: unknown }).type !== "string") {
          throw new Error("invalid params: action must be an object with a type");
        }
        const r = act(s.state, action);
        if (!r.ok) throw new Error(r.error);
        s.state = r.state;
        let mem = (await this.store.load(s.playerId)) ?? blankMemory(s.playerId);
        let memoryLine: string | null = null;
        if (r.utterance) {
          const sid = r.utterance.suspect;
          if (!s.greeted.has(sid)) {
            memoryLine = rememberLine(mem, sid);
            s.greeted.add(sid);
          }
          mem =
            action.type === "present"
              ? recordPresent(mem, sid, action.evidence, r.utterance.cracked)
              : recordAsk(mem, sid, r.utterance.topic, r.utterance.cracked);
        }
        if (r.accusation) mem = recordAccusation(mem, r.accusation);
        await this.store.save(mem);
        return {
          message: r.message,
          ...(r.clue ? { clueFound: r.clue } : {}),
          ...(r.utterance
            ? {
                suspect: r.utterance.suspect,
                says: memoryLine ? `${memoryLine} ${r.utterance.gist}` : r.utterance.gist,
                newTestimony: r.utterance.revealed,
                storyCracked: r.utterance.cracked,
              }
            : {}),
          ...(r.accusation ? { accusation: r.accusation } : {}),
          view: publicView(s.state, { revealTruth: this.debug }),
        };
      }
      default:
        throw new Error(`unknown tool: ${name}`);
    }
  }
}
