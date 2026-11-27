/**
 * Anthropic Claude provider — direct REST adapter, no SDK dep.
 *
 * Uses the Messages API. Supports system prompts + multi-turn messages.
 * Price hint: Claude Haiku ~$1/1M input, ~$5/1M output (rough — confirm
 * at runtime via /v1/models/{id}/pricing if costing matters).
 */

import type { LLMProvider, LLMRequest, LLMResponse, ProviderConfig } from "./types.js";

const ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_API_VERSION = "2023-06-01";

export class AnthropicProvider implements LLMProvider {
  readonly id = "anthropic";
  readonly defaultModel: string;
  private apiKey: string;
  private baseUrl: string;

  constructor(config: ProviderConfig) {
    this.apiKey = config.apiKey;
    this.defaultModel = config.model ?? "claude-haiku-4-5-20251001";
    this.baseUrl = config.baseUrl ?? ANTHROPIC_API_URL;
  }

  async complete(req: LLMRequest, signal: AbortSignal, model?: string): Promise<LLMResponse> {
    const body = {
      model: model ?? this.defaultModel,
      max_tokens: req.maxTokens ?? 1024,
      temperature: req.temperature ?? 0.7,
      ...(req.system ? { system: req.system } : {}),
      messages: req.messages.filter((m) => m.role !== "system"),
      ...(req.stopSequences ? { stop_sequences: req.stopSequences } : {}),
    };

    const r = await fetch(this.baseUrl, {
      method: "POST",
      headers: {
        "x-api-key": this.apiKey,
        "anthropic-version": ANTHROPIC_API_VERSION,
        "content-type": "application/json",
      },
      body: JSON.stringify(body),
      signal,
    });

    if (!r.ok) {
      const errText = await r.text().catch(() => "");
      throw new Error(`Anthropic ${r.status}: ${errText.slice(0, 200)}`);
    }
    const data = (await r.json()) as {
      content: Array<{ type: string; text?: string }>;
      stop_reason: string;
      usage?: { input_tokens: number; output_tokens: number };
    };

    const text = (data.content ?? [])
      .filter((c) => c.type === "text" && typeof c.text === "string")
      .map((c) => c.text!)
      .join("");

    const out: LLMResponse = { text };
    const stop = data.stop_reason;
    if (stop) out.finishReason = stop as NonNullable<LLMResponse["finishReason"]>;
    if (data.usage) out.usage = { input: data.usage.input_tokens, output: data.usage.output_tokens };
    return out;
  }
}
