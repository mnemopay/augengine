/** Vercel Node function: GET /api/detective/health. */
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
