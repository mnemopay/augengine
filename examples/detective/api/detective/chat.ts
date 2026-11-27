/**
 * Vercel Node function: POST /api/detective/chat. Same handler as the Vite
 * dev middleware. Set OPENROUTER_API_KEY (and optionally DETECTIVE_MODEL,
 * DETECTIVE_SESSION_USD, DETECTIVE_GLOBAL_USD) in the Vercel project.
 */
import type { IncomingMessage, ServerResponse } from "node:http";
import { handleDetectiveApi } from "../../server/chat-api.js";

export default async function handler(req: IncomingMessage, res: ServerResponse): Promise<void> {
  process.env.DETECTIVE_KEY_FILE = "0";
  const handled = await handleDetectiveApi(req, res);
  if (!handled) {
    res.writeHead(404);
    res.end();
  }
}
