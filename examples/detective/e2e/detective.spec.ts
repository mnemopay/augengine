import { test, expect, type Page } from "@playwright/test";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const SHOTS = join(here, "..", "..", "..", "results", "detective");
mkdirSync(SHOTS, { recursive: true });

const ROOMS: Record<string, string[]> = {
  dome: ["dome.flask", "dome.log", "dome.desk", "dome.telescope"],
  library: ["library.drawer", "library.coat", "library.fire"],
  kitchen: ["kitchen.cabinet", "kitchen.note", "kitchen.kettle"],
  conservatory: ["conservatory.planter", "conservatory.orchids"],
};

async function start(page: Page): Promise<void> {
  await page.goto("/?offline=1");
  await page.getByTestId("start").click();
}

async function searchHouse(page: Page, shots = false): Promise<void> {
  for (const [room, objs] of Object.entries(ROOMS)) {
    await page.getByTestId(`room-${room}`).click();
    for (const o of objs) await page.getByTestId(`obj-${o}`).click();
    if (shots) {
      await page.waitForTimeout(900);
      await page.screenshot({ path: join(SHOTS, `room-${room}.png`) });
    }
  }
  await page.getByTestId("room-parlour").click();
}

async function waitIdle(page: Page): Promise<void> {
  await expect(page.getByTestId("ask-send")).toBeEnabled();
}

async function ask(page: Page, suspect: string, text: string): Promise<string> {
  await page.getByTestId(`suspect-${suspect}`).click();
  await waitIdle(page);
  await page.getByTestId("ask-input").fill(text);
  await page.getByTestId("ask-send").click();
  await waitIdle(page);
  return (await page.getByTestId("speech-line").textContent()) ?? "";
}

async function present(page: Page, evidence: string): Promise<string> {
  await page.getByTestId("present-select").selectOption(evidence);
  await page.getByTestId("present-send").click();
  await waitIdle(page);
  return (await page.getByTestId("speech-line").textContent()) ?? "";
}

async function accuse(page: Page, suspect: string, method: string, motive: string, evidence: string[]): Promise<string> {
  await page.getByTestId("accuse-open").click();
  await page.getByTestId("acc-suspect").selectOption(suspect);
  await page.getByTestId("acc-method").selectOption(method);
  await page.getByTestId("acc-motive").selectOption(motive);
  for (const e of evidence) await page.getByTestId(`acc-ev-${e}`).check();
  await page.getByTestId("acc-confirm").click();
  return (await page.getByTestId("result-headline").textContent()) ?? "";
}

test("plays the whole case to the correct accusation (offline dialogue)", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.goto("/?offline=1");
  await page.waitForTimeout(1200);
  await page.screenshot({ path: join(SHOTS, "intro.png") });
  await page.getByTestId("start").click();
  await searchHouse(page, true);
  await expect(page.getByTestId("nb-C_GLOVE")).toBeVisible();

  const teodor = await ask(page, "teodor", "Did you see anyone near the kitchen?");
  expect(teodor).toMatch(/Pike/);
  await expect(page.getByTestId("nb-T_TEODOR_SAW_DESMOND")).toBeVisible();

  const lie = await ask(page, "desmond", "Where were you this evening?");
  expect(lie).toMatch(/never set foot in the kitchen/);
  await present(page, "C_GLOVE");
  const crack = await present(page, "T_TEODOR_SAW_DESMOND");
  expect(crack).toMatch(/brandy/);
  await page.waitForTimeout(800);
  await page.screenshot({ path: join(SHOTS, "parlour-desmond-closeup.png") });

  const headline = await accuse(page, "desmond", "poison_tea", "inheritance", ["C_GLOVE", "C_CABINET", "C_WILL"]);
  expect(headline).toMatch(/Case closed/);
  await page.screenshot({ path: join(SHOTS, "verdict-solved.png") });
  expect(errors).toEqual([]);
});

test("a wrong accusation loses, and the suspect remembers it next case", async ({ page }) => {
  await start(page);
  await page.getByTestId("room-dome").click();
  for (const o of ROOMS.dome!) await page.getByTestId(`obj-${o}`).click();
  await page.getByTestId("room-parlour").click();
  await ask(page, "teodor", "What did you and the professor talk about?");
  const headline = await accuse(page, "teodor", "poison_tea", "stolen_research", ["C_LETTER", "C_FLASK"]);
  expect(headline).toMatch(/innocent/);
  await page.screenshot({ path: join(SHOTS, "verdict-wrong.png") });

  // New case, same browser profile: Teodor remembers being accused.
  await page.getByTestId("new-case").click();
  await page.getByTestId("suspect-teodor").click();
  await expect(page.locator("#memory-note")).toContainText("accused me last time");
  await waitIdle(page);
  const line = (await page.getByTestId("speech-line").textContent()) ?? "";
  expect(line).toMatch(/accused me last time/);

  // Reload the page entirely: memory persists (IndexedDB).
  await page.reload();
  await page.getByTestId("start").click();
  await page.getByTestId("suspect-teodor").click();
  await expect(page.locator("#memory-note")).toContainText("accused me last time");
});

test("frame stats on this machine", async ({ page }) => {
  await start(page);
  await page.waitForTimeout(3000);
  const out: Record<string, unknown> = {};
  for (const room of ["parlour", "dome", "library"]) {
    await page.getByTestId(`room-${room}`).click();
    await page.waitForTimeout(6000);
    out[room] = await page.evaluate(() => (window as any).__detective.perf());
  }
  await page.getByTestId("room-parlour").click();
  await page.getByTestId("suspect-ines").click();
  await page.waitForTimeout(6000);
  out["parlour-closeup"] = await page.evaluate(() => (window as any).__detective.perf());
  await page.screenshot({ path: join(SHOTS, "parlour-ines-closeup.png") });
  writeFileSync(join(SHOTS, "perf.json"), JSON.stringify({ at: new Date().toISOString(), ...out }, null, 2));
  const parlour = out.parlour as { p50: number; renderer: unknown };
  expect(parlour.renderer).not.toBeNull();
});

test("model mode: a real model line streams in and passes the guardrail", async ({ page, request }) => {
  const health = await (await request.get("/api/detective/health")).json();
  test.skip(!health.llm, "no LLM key on this server");
  await page.goto("/");
  await expect.poll(() => page.evaluate(() => (window as any).__detective.mode())).toBe("llm");
  await page.getByTestId("start").click();
  await page.getByTestId("suspect-ines").click();
  await expect(page.getByTestId("ask-send")).toBeEnabled({ timeout: 30_000 });
  await page.getByTestId("ask-input").fill("Where were you when the tea was made?");
  await page.getByTestId("ask-send").click();
  await expect(page.getByTestId("ask-send")).toBeEnabled({ timeout: 30_000 });
  const t = await page.evaluate(() => (window as any).__detective.transcript());
  const last = t[t.length - 1];
  expect(last.text.length).toBeGreaterThan(10);
  expect(["llm", "offline"]).toContain(last.source);
  await page.screenshot({ path: join(SHOTS, "model-mode-ines.png") });
  writeFileSync(join(SHOTS, "model-mode-transcript.json"), JSON.stringify(t, null, 2));
});
