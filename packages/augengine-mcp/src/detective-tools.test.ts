import { describe, expect, it } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { AugEngineMcpServer, TOOL_MANIFEST } from "./server.js";
import { DetectiveSessions } from "./detective-tools.js";

async function call(server: AugEngineMcpServer, name: string, args: Record<string, unknown>) {
  const res = (await server.handle({ jsonrpc: "2.0", id: 1, method: "tools/call", params: { name, arguments: args } })) as {
    result: { content: { text: string }[]; isError: boolean };
  };
  const text = res.result.content[0]!.text;
  let payload: any = text;
  try {
    payload = JSON.parse(text);
  } catch {
    /* error string */
  }
  return { payload, isError: res.result.isError };
}

describe("detective MCP tools", () => {
  it("are listed in the manifest", () => {
    const names = TOOL_MANIFEST.map((t) => t.name);
    expect(names).toEqual(expect.arrayContaining(["detective_new_session", "detective_status", "detective_act"]));
  });

  it("an agent can play the case to a solved accusation; truth is hidden until then", async () => {
    const server = new AugEngineMcpServer();
    const { payload: s } = await call(server, "detective_new_session", { playerId: "agent-1" });
    expect(s.view.truth).toBeUndefined();
    const id = s.sessionId as string;
    const step = async (action: Record<string, unknown>) => {
      const r = await call(server, "detective_act", { sessionId: id, action });
      expect(r.isError, JSON.stringify(r.payload)).toBe(false);
      return r.payload;
    };
    for (const [room, objs] of [
      ["dome", ["dome.flask"]],
      ["library", ["library.drawer"]],
      ["kitchen", ["kitchen.cabinet"]],
      ["conservatory", ["conservatory.planter"]],
    ] as const) {
      await step({ type: "move", room });
      for (const o of objs) await step({ type: "examine", objectId: o });
    }
    await step({ type: "move", room: "parlour" });
    const lie = await step({ type: "ask", suspect: "desmond", text: "Where were you tonight?" });
    expect(lie.says).toMatch(/never set foot/);
    await step({ type: "present", suspect: "desmond", evidence: "C_GLOVE" });
    const crack = await step({ type: "present", suspect: "desmond", evidence: "C_CABINET" });
    expect(crack.storyCracked).toContain("alibi");
    const done = await step({ type: "accuse", suspect: "desmond", method: "poison_tea", motive: "inheritance", evidence: ["C_GLOVE", "C_CABINET", "C_WILL"] });
    expect(done.accusation.verdict).toBe("solved");
    expect(done.view.truth.killer).toBe("desmond");
  });

  it("rejects illegal actions as tool errors (the rules, not the agent, decide)", async () => {
    const server = new AugEngineMcpServer();
    const { payload: s } = await call(server, "detective_new_session", {});
    const r = await call(server, "detective_act", { sessionId: s.sessionId, action: { type: "present", suspect: "desmond", evidence: "C_GLOVE" } });
    expect(r.isError).toBe(true);
    expect(String(r.payload)).toMatch(/haven't found/);
  });

  it("suspects remember the player across sessions (file-backed memory)", async () => {
    const dir = await mkdtemp(join(tmpdir(), "dtv-"));
    try {
      const sessions = new DetectiveSessions({ memoryPath: join(dir, "memory.json"), debug: false });
      const a: any = await sessions.run("detective_new_session", { playerId: "p" });
      await sessions.run("detective_act", { sessionId: a.sessionId, action: { type: "move", room: "dome" } });
      await sessions.run("detective_act", { sessionId: a.sessionId, action: { type: "examine", objectId: "dome.desk" } });
      await sessions.run("detective_act", { sessionId: a.sessionId, action: { type: "accuse", suspect: "teodor", method: "poison_tea", motive: "stolen_research", evidence: ["C_LETTER"] } });
      // New server process, same memory file.
      const fresh = new DetectiveSessions({ memoryPath: join(dir, "memory.json"), debug: false });
      const b: any = await fresh.run("detective_new_session", { playerId: "p" });
      expect(b.previousSessions).toBe(1);
      const r: any = await fresh.run("detective_act", { sessionId: b.sessionId, action: { type: "ask", suspect: "teodor", text: "Hello again" } });
      expect(r.says).toMatch(/accused me last time/);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});
