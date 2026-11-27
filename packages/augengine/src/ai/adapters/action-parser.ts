export interface ActionIR {
  action: string;
  target?: string;
  payload?: Record<string, unknown>;
}

/**
 * Parses an LLM's raw string output and extracts a valid Action IR.
 * Designed to gracefully handle LLM hallucinations, markdown wrappers,
 * and conversational padding.
 */
export function parseActionIR(llmOutput: string): ActionIR | null {
  // Try to find the first JSON-like block
  const jsonMatch = llmOutput.match(/\{[\s\S]*\}/);
  if (!jsonMatch) return null;

  try {
    const parsed = JSON.parse(jsonMatch[0]);
    if (typeof parsed !== "object" || parsed === null) return null;
    
    // Validate Action IR schema
    if (typeof parsed.action !== "string") return null;

    const ir: ActionIR = { action: parsed.action };
    if (typeof parsed.target === "string") {
      ir.target = parsed.target;
    }
    if (typeof parsed.payload === "object" && parsed.payload !== null) {
      ir.payload = parsed.payload;
    }

    return ir;
  } catch (e) {
    // Malformed JSON
    return null;
  }
}
