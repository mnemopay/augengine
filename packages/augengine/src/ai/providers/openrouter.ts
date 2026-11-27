/**
 * OpenRouter provider — wraps OpenAI's chat-completions shape against
 * OpenRouter's aggregator. Useful for routing to free-tier `:free` slugs
 * during dev (e.g. `meta-llama/llama-3.3-70b-instruct:free`) and for
 * multi-provider failover via OpenRouter's pricing routing.
 *
 * Pass the key in from your own environment; never commit it.
 */

import { OpenAIProvider } from "./openai.js";
import type { ProviderConfig } from "./types.js";

const OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions";

export class OpenRouterProvider extends OpenAIProvider {
  override readonly id = "openrouter";

  constructor(config: ProviderConfig) {
    super({
      ...config,
      model: config.model ?? "xiaomi/mimo:free",
      baseUrl: config.baseUrl ?? OPENROUTER_API_URL,
    });
  }
}
