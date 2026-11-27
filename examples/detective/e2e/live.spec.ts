import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test } from "@playwright/test";

/**
 * Live checks of the deployed game (augengine.com/detective). Skipped unless
 * DETECTIVE_LIVE_URL is set. The forced-offline check also needs
 * DETECTIVE_TEST_SECRET (the same value as the server's env var).
 */
const here = dirname(fileURLToPath(import.meta.url));
const SHOTS = join(here, "..", "..", "..", "results", "detective", "live");
const LIVE = process.env.DETECTIVE_LIVE_URL;
const SECRET = process.env.DETECTIVE_TEST_SECRET ?? "";

test.skip(!LIVE, "set DETECTIVE_LIVE_URL to check a deployment");

test("live: boots under /detective, renders, and a real model line comes back", async ({ page, request }) => {
  mkdirSync(SHOTS, { recursive: true });
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  const health = await (await request.get("/api/detective/health")).json();
  expect(health.llm).toBe(true);
  await page.goto("/detective");
  await expect.poll(() => page.evaluate(() => (window as any).__detective?.mode())).toBe("llm");
  await page.getByTestId("start").click();
  await page.getByTestId("suspect-ines").click();
  await expect(page.getByTestId("ask-send")).toBeEnabled({ timeout: 30_000 });
  await page.getByTestId("ask-input").fill("Where were you when the tea was made?");
  await page.getByTestId("ask-send").click();
  await expect(page.getByTestId("ask-send")).toBeEnabled({ timeout: 30_000 });
  const t = await page.evaluate(() => (window as any).__detective.transcript?.() ?? null);
  const perf = await page.evaluate(() => (window as any).__detective.perf());
  const portrait = await page.evaluate(() => [...document.images].filter((i) => i.src.includes("/detective/portraits/") && i.naturalWidth > 0).length);
  await page.waitForTimeout(1500);
  await page.screenshot({ path: join(SHOTS, "live-model-line.png") });
  writeFileSync(join(SHOTS, "live-model-line.json"), JSON.stringify({ transcript: t, perf, portraitsLoaded: portrait, errors }, null, 2));
  expect(errors).toEqual([]);
  expect(portrait).toBeGreaterThan(0);
  const last = t?.[t.length - 1];
  if (last) {
    expect(last.text.length).toBeGreaterThan(10);
    expect(last.source).toBe("llm");
  }
});

test("live: forced offline shows the resting note and scripted lines", async ({ page }) => {
  test.skip(SECRET.length < 16, "needs DETECTIVE_TEST_SECRET");
  await page.route("**/api/detective/**", (route) =>
    route.continue({ headers: { ...route.request().headers(), "x-dtv-force-offline": SECRET } }),
  );
  await page.goto("/detective");
  await expect.poll(() => page.evaluate(() => (window as any).__detective?.mode())).toBe("offline");
  await expect(page.locator("#dtv-resting-note")).toBeVisible();
  await page.getByTestId("start").click();
  await page.getByTestId("suspect-ines").click();
  await page.getByTestId("ask-input").fill("Where were you when the tea was made?");
  await page.getByTestId("ask-send").click();
  await expect(page.getByTestId("speech-line")).not.toBeEmpty();
  await page.screenshot({ path: join(SHOTS, "live-resting.png") });
});
