/**
 * OpenAI Chat Completions provider — direct REST, no SDK dep.
 *
 * Compatible with any OpenAI-API-shaped service: Groq, Together, Fireworks,
 * Cerebras, vLLM-served endpoints. Just override `baseUrl`.
 */

import type { LLMProvider, LLMRequest, LLMResponse, ProviderConfig } from "./types.js";

const OPENAI_API_URL = "https://api.openai.com/v1/chat/completions";

export class OpenAIProvider implements LLMProvider {
  readonly id: string = "openai";
  readonly defaultModel: string;
  private apiKey: string;
  private baseUrl: string;

  constructor(config: ProviderConfig) {
    this.apiKey = config.apiKey;
    this.defaultModel = config.model ?? "gpt-4o-mini";
    this.baseUrl = config.baseUrl ?? OPENAI_API_URL;
  }

  async complete(req: LLMRequest, signal: AbortSignal, model?: string): Promise<LLMResponse> {
    const messages: { role: string; content: string }[] = [];
    if (req.system) messages.push({ role: "system", content: req.system });
    for (const m of req.messages) messages.push({ role: m.role, content: m.content });

    const body = {
      model: model ?? this.defaultModel,
      messages,
      max_tokens: req.maxTokens ?? 1024,
      temperature: req.temperature ?? 0.7,
      ...(req.stopSequences ? { stop: req.stopSequences } : {}),
    };

    const r = await fetch(this.baseUrl, {
      method: "POST",
      headers: {
        authorization: `Bearer ${this.apiKey}`,
        "content-type": "application/json",
      },
      body: JSON.stringify(body),
      signal,
    });

    if (!r.ok) {
      const errText = await r.text().catch(() => "");
      throw new Error(`OpenAI ${r.status}: ${errText.slice(0, 200)}`);
    }
    const data = (await r.json()) as {
      choices: Array<{ message: { content: string }; finish_reason: string }>;
      usage?: { prompt_tokens: number; completion_tokens: number };
    };

    const choice = data.choices?.[0];
    const out: LLMResponse = { text: choice?.message?.content ?? "" };
    const fin = choice?.finish_reason;
    if (fin) out.finishReason = fin as NonNullable<LLMResponse["finishReason"]>;
    if (data.usage) out.usage = { input: data.usage.prompt_tokens, output: data.usage.completion_tokens };
    return out;
  }
}
