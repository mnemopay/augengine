import { test, expect, type Page } from "@playwright/test";
import { mkdirSync } from "node:fs";
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

async function searchHouse(page: Page): Promise<void> {
  for (const [room, objs] of Object.entries(ROOMS)) {
    await page.getByTestId(`room-${room}`).click();
    for (const o of objs) await page.getByTestId(`obj-${o}`).click();
  }
  await page.getByTestId("room-parlour").click();
}

const mood = (page: Page) => page.locator("#portrait").getAttribute("data-mood");

async function screenPoint(page: Page, entityId: string): Promise<{ x: number; y: number }> {
  const p = await page.evaluate((id) => (window as unknown as { __detective: { screenPointOf(id: string): { x: number; y: number } | null } }).__detective.screenPointOf(id), entityId);
  expect(p, `${entityId} should be in view`).not.toBeNull();
  return p!;
}

test("suspect portraits load and the mood follows the rules", async ({ page }) => {
  await start(page);
  // Every suspect button carries a loaded calm portrait thumbnail.
  for (const sid of ["mara", "desmond", "ines", "teodor"]) {
    const ok = await page.getByTestId(`suspect-${sid}`).locator("img").evaluate((img: HTMLImageElement) => img.complete && img.naturalWidth > 0);
    expect(ok, `${sid} thumbnail`).toBe(true);
  }
  await searchHouse(page);
  await page.getByTestId("suspect-desmond").click();
  await expect(page.getByTestId("ask-send")).toBeEnabled();
  await expect(page.locator("#portrait")).toHaveClass(/show/);
  expect(await page.locator("#portrait").evaluate((img: HTMLImageElement) => img.naturalWidth)).toBeGreaterThan(0);

  // A scripted lie shows the evasive face.
  await page.getByTestId("ask-input").fill("Where were you this evening?");
  await page.getByTestId("ask-send").click();
  await expect(page.getByTestId("ask-send")).toBeEnabled();
  expect(await mood(page)).toBe("evasive");

  // Shown his IOUs, the lie cracks: the cracked face.
  await page.getByTestId("present-select").selectOption("C_IOU");
  await page.getByTestId("present-send").click();
  await expect(page.getByTestId("ask-send")).toBeEnabled();
  await expect.poll(() => mood(page)).toBe("cracked");
  await page.waitForTimeout(600);
  await page.screenshot({ path: join(SHOTS, "faces-dialogue-desmond-cracked.png") });
});

test("clicking a clue in the 3D view examines it, and clicking a bust opens the suspect", async ({ page }) => {
  await start(page);
  await page.getByTestId("room-dome").click();
  const flask = await screenPoint(page, "obj.dome.flask");
  await page.mouse.move(flask.x, flask.y);
  await expect(page.getByTestId("hover-label")).toHaveText("Examine: Tea flask");
  await page.mouse.click(flask.x, flask.y);
  await expect(page.getByTestId("nb-C_FLASK")).toBeVisible();

  await page.getByTestId("room-parlour").click();
  await page.waitForTimeout(1500); // busts stream in
  await page.screenshot({ path: join(SHOTS, "faces-parlour-busts.png") });
  const ines = await screenPoint(page, "head.ines");
  await page.mouse.move(ines.x, ines.y);
  await expect(page.getByTestId("hover-label")).toHaveText("Question Ines Calloway");
  await page.mouse.click(ines.x, ines.y);
  await expect(page.getByTestId("suspect-ines")).toHaveClass(/active/);
  await expect(page.getByTestId("ask-send")).toBeEnabled();
  await page.waitForTimeout(800);
  await page.screenshot({ path: join(SHOTS, "faces-closeup-ines.png") });
});
