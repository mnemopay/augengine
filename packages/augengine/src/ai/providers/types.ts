/**
 * LLM provider interface — every concrete provider (Anthropic, OpenAI,
 * OpenRouter, WebLLM) implements this so the AIScheduler can route
 * dialogue/ambient/action_plan tasks to whichever provider the app
 * registered for that task.
 *
 * Phase 4 ships concrete adapters for Anthropic, OpenAI, and OpenRouter.
 * WebLLM (in-browser local inference) is a phase-6 stretch.
 */

export interface LLMRequest {
  /** System prompt — persona + world context. */
  system?: string;
  /** Conversation messages. Last must be user role. */
  messages: LLMMessage[];
  /** Generation knobs. */
  maxTokens?: number;
  temperature?: number;
  stopSequences?: string[];
}

export interface LLMMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

export interface LLMResponse {
  text: string;
  /** Provider-specific finish reason. */
  finishReason?: "end_turn" | "max_tokens" | "stop_sequence" | "tool_use" | string;
  /** Cost estimate in USD if the provider returns usage. */
  costUsd?: number;
  /** Raw token usage if available. */
  usage?: { input: number; output: number };
}

export interface LLMProvider {
  /** Stable id — "anthropic", "openai", "openrouter", "webllm". */
  readonly id: string;
  /** Default model the provider uses unless the request overrides. */
  readonly defaultModel: string;
  /** Run an LLM request. Must respect the abort signal. */
  complete(req: LLMRequest, signal: AbortSignal, model?: string): Promise<LLMResponse>;
}

export interface ProviderConfig {
  apiKey: string;
  /** Override the default model. */
  model?: string;
  /** Override the API base URL (useful for self-hosted / proxy setups). */
  baseUrl?: string;
}
