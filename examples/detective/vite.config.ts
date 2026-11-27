import { defineConfig, type Plugin } from "vite";
import { handleDetectiveApi } from "./server/chat-api.js";

/** Serves the dialogue API from Vite (dev and preview) so the key stays server-side. */
function detectiveApi(): Plugin {
  const mount = (server: { middlewares: { use: (fn: (req: any, res: any, next: () => void) => void) => void } }) => {
    server.middlewares.use((req, res, next) => {
      handleDetectiveApi(req, res)
        .then((handled) => {
          if (!handled) next();
        })
        .catch(next);
    });
  };
  return {
    name: "detective-api",
    configureServer: mount,
    configurePreviewServer: mount,
  };
}

export default defineConfig({
  base: process.env.DETECTIVE_BASE || "/",
  plugins: [detectiveApi()],
  build: { target: "es2022", chunkSizeWarningLimit: 2000 },
});
