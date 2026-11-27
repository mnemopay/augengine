import { defineConfig, devices } from "@playwright/test";

/** Checks the deployed game (no local server). DETECTIVE_LIVE_URL, e.g. https://augengine.com */
export default defineConfig({
  testDir: "./e2e",
  testMatch: /live\.spec\.ts/,
  timeout: 180_000,
  workers: 1,
  reporter: [["list"]],
  use: { baseURL: process.env.DETECTIVE_LIVE_URL, viewport: { width: 1440, height: 900 } },
  projects: [
    {
      name: "chrome-headed",
      use: {
        ...devices["Desktop Chrome"],
        viewport: { width: 1440, height: 900 },
        channel: "chrome",
        headless: false,
        launchOptions: { args: ["--enable-unsafe-webgpu", "--enable-features=WebGPU"] },
      },
    },
  ],
});
