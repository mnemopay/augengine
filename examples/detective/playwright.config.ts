import { defineConfig, devices } from "@playwright/test";

/** Port for the dev server under test (override with DETECTIVE_PORT when 5193 is taken). */
const PORT = Number(process.env.DETECTIVE_PORT ?? 5193);

/**
 * Detective game e2e. Headed system Chrome so WebGPU uses the real adapter
 * (same reasoning as the repo's chromium-webgpu-headed project). The game is
 * fully playable without WebGPU too; the tests drive the DOM.
 */
export default defineConfig({
  testDir: "./e2e",
  timeout: 180_000,
  fullyParallel: false,
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL: `http://localhost:${PORT}`,
    viewport: { width: 1440, height: 900 },
    screenshot: "only-on-failure",
  },
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
  webServer: {
    command: `npx vite --port ${PORT} --strictPort`,
    url: `http://localhost:${PORT}`,
    reuseExistingServer: true,
    timeout: 60_000,
  },
});
