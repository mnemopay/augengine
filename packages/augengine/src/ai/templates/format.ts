/**
 * `.prompt` file format — decision #3 of AUGENGINE-ARCHITECTURE.md.
 *
 * Every authored line in the engine is a template (zero-slot for static
 * lines, slotted for dynamic). Template files have YAML-style frontmatter
 * + a body with `{{slot}}` placeholders.
 *
 * Example file `dialogue/intro.prompt`:
 *
 *   ---
 *   slots: [persona_name, scene_mood, user_message]
 *   examples:
 *     - input: { persona_name: "Maya", scene_mood: "tense", user_message: "Where are we?" }
 *       output: "Quiet — listen."
 *   ---
 *   You are {{persona_name}}. The scene mood is {{scene_mood}}.
 *   The user just said: {{user_message}}
 *   Reply in one short line.
 *
 * The frontmatter is parsed via the same charter YAML parser we already
 * ship (no new dep). Body is the rest of the file.
 */

import { parseYaml } from "../../authoring/charter.js";
import type { PromptTemplate, PromptExample } from "../types.js";

const FRONTMATTER_RE = /^---\s*\n([\s\S]*?)\n---\s*\n([\s\S]*)$/;

export interface ParsedPromptFile {
  /** Path of origin — informational only. */
  path: string;
  template: PromptTemplate;
}

export function parsePromptFile(path: string, raw: string): ParsedPromptFile {
  const m = FRONTMATTER_RE.exec(raw);
  let frontmatter: Record<string, unknown> = {};
  let body = raw;
  if (m) {
    const fmRaw = m[1] ?? "";
    body = (m[2] ?? "").replace(/^\n+/, "");
    const parsed = parseYaml(fmRaw);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      frontmatter = parsed as Record<string, unknown>;
    }
  }

  // Auto-detect slots from {{...}} occurrences if not declared.
  const declaredSlots = Array.isArray(frontmatter.slots)
    ? (frontmatter.slots as unknown[]).filter((s): s is string => typeof s === "string")
    : detectSlots(body);

  const examples = Array.isArray(frontmatter.examples)
    ? (frontmatter.examples as Array<Record<string, unknown>>)
        .filter((e) => typeof e === "object" && e !== null)
        .map((e) => ({
          input: (e.input ?? {}) as Record<string, string>,
          output: typeof e.output === "string" ? e.output : "",
        }) as PromptExample)
    : undefined;

  const expectedOutputs = Array.isArray(frontmatter.expected_outputs)
    ? (frontmatter.expected_outputs as Array<Record<string, unknown>>)
        .filter((e) => typeof e === "object" && e !== null)
        .map((e) => ({
          input: (e.input ?? {}) as Record<string, string>,
          output: typeof e.output === "string" ? e.output : "",
        }) as PromptExample)
    : undefined;

  return {
    path,
    template: {
      path,
      body,
      slots: declaredSlots,
      ...(examples ? { examples } : {}),
      ...(expectedOutputs ? { expectedOutputs } : {}),
    },
  };
}

function detectSlots(body: string): string[] {
  const slotRe = /\{\{\s*([a-zA-Z_][\w]*)\s*\}\}/g;
  const out = new Set<string>();
  let m;
  while ((m = slotRe.exec(body)) !== null) {
    out.add(m[1]!);
  }
  return [...out];
}

/**
 * Render a template against a slot map. Unknown slots leave their
 * `{{name}}` literal in the output (so missing data shows in the prompt
 * rather than silently corrupting it). Caller can pass `strict: true` to
 * throw on missing slots instead.
 */
export function renderTemplate(
  template: PromptTemplate,
  slots: Record<string, string>,
  opts: { strict?: boolean } = {},
): string {
  return template.body.replace(/\{\{\s*([a-zA-Z_][\w]*)\s*\}\}/g, (full, name) => {
    if (Object.prototype.hasOwnProperty.call(slots, name)) {
      return slots[name] ?? "";
    }
    if (opts.strict) throw new Error(`renderTemplate: missing slot "${name}"`);
    return full;
  });
}

/**
 * Compose few-shot examples into a string block prepended to the prompt
 * body. Returned format is `Input: {...}\nOutput: ...` per example, blank
 * line between, suitable for system-prompt or assistant-shot styling.
 */
export function renderFewShot(template: PromptTemplate): string {
  if (!template.examples || template.examples.length === 0) return "";
  const blocks = template.examples.map((ex) => {
    const inputJson = JSON.stringify(ex.input);
    return `Input: ${inputJson}\nOutput: ${ex.output}`;
  });
  return blocks.join("\n\n");
}
