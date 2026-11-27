import { describe, expect, it } from "vitest";
import { assetUrl } from "./asset-url.js";

describe("assetUrl", () => {
  it("keeps root paths at the site root", () => {
    expect(assetUrl("/portraits/mara/calm.webp", "/")).toBe("/portraits/mara/calm.webp");
  });
  it("prefixes the sub-path base (augengine.com/detective)", () => {
    expect(assetUrl("/portraits/mara/calm.webp", "/detective/")).toBe("/detective/portraits/mara/calm.webp");
    expect(assetUrl("/busts/ines.glb", "/detective")).toBe("/detective/busts/ines.glb");
  });
});
