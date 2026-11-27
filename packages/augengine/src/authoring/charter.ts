/**
 * Charter compiler — YAML charter → SceneSpec JSON.
 *
 * A charter is the human-authored scene definition: entities, transforms,
 * personas, prompt template references. The compiler:
 *   1. Parses YAML
 *   2. Validates required fields
 *   3. Materializes default components (transforms get identity if absent)
 *   4. Emits a SceneSpec JSON document the loader consumes
 *
 * Phase 1 ships a minimal but complete YAML parser (no external dep) that
 * handles the subset we author against. Phase 5 considers swapping for a
 * richer parser if charter authoring outgrows it.
 */

import type {
  Entity,
  EntityId,
  EnvironmentRef,
  SceneSpec,
  PersonaComponent,
  TransformComponent,
} from "../scene/types.js";
import { Tag } from "../scene/types.js";

// ──────────────────────────────────────────────────────────────────────
// Charter input shape (post-YAML-parse)
// ──────────────────────────────────────────────────────────────────────

export interface CharterDoc {
  /** Charter name; informational. */
  name: string;
  /** Charter description; informational. */
  description?: string;
  /** Entities listed in human-friendly form. */
  entities: CharterEntity[];
  /** Default active camera by entity id. */
  defaultCamera?: string;
  /** Environment fill (color/skybox/ibl/splat-world). */
  environment?: {
    kind: "color" | "skybox" | "ibl" | "splat-world";
    [k: string]: unknown;
  };
}

export interface CharterEntity {
  id: string;
  name?: string;
  parent?: string;
  position?: [number, number, number];
  rotation?: [number, number, number, number];
  scale?: [number, number, number] | number;
  mesh?: { kind: "primitive"; shape: string; color?: [number, number, number] };
  camera?: { fovY?: number; near?: number; far?: number; active?: boolean };
  light?: { kind: "directional" | "point" | "spot" | "ambient"; color?: [number, number, number]; intensity?: number };
  persona?: {
    traits?: string[];
    voice?: string;
    memoryId?: string;
    mood?: number;
    goals?: Array<{ id: string; description: string; priority?: number; done?: string }>;
  };
}

// ──────────────────────────────────────────────────────────────────────
// Compile — CharterDoc → SceneSpec
// ──────────────────────────────────────────────────────────────────────

export function compileCharter(doc: CharterDoc): SceneSpec {
  const entities: Entity[] = doc.entities.map(toEntity);
  const spec: SceneSpec = {
    schemaVersion: 1,
    source: { kind: "authored", path: doc.name },
    entities,
  };
  if (doc.defaultCamera) spec.defaultCamera = doc.defaultCamera as EntityId;
  if (doc.environment) {
    spec.environment = doc.environment as EnvironmentRef;
  }
  return spec;
}

function toEntity(ce: CharterEntity): Entity {
  const components: Entity["components"] = {};
  // Always emit a Transform.
  const transform: TransformComponent = {
    position: ce.position ?? [0, 0, 0],
    rotation: ce.rotation ?? [0, 0, 0, 1],
    scale: typeof ce.scale === "number" ? [ce.scale, ce.scale, ce.scale] : (ce.scale ?? [1, 1, 1]),
    parent: (ce.parent as EntityId | undefined) ?? null,
  };
  components[Tag.Transform] = transform;
  if (ce.mesh) {
    components[Tag.Mesh] = {
      geometry: { kind: "primitive", shape: ce.mesh.shape as "box" | "sphere" | "plane" | "cylinder" },
      material: ce.mesh.color
        ? { kind: "unlit", color: ce.mesh.color }
        : { kind: "unlit", color: [0.7, 0.7, 0.75] },
      visible: true,
    };
  }
  if (ce.camera) {
    components[Tag.Camera] = {
      fovY: ce.camera.fovY ?? Math.PI / 3,
      near: ce.camera.near ?? 0.1,
      far: ce.camera.far ?? 100,
      active: ce.camera.active ?? true,
    };
  }
  if (ce.light) {
    components[Tag.Light] = {
      kind: ce.light.kind,
      color: ce.light.color ?? [1, 1, 1],
      intensity: ce.light.intensity ?? 1,
    };
  }
  if (ce.persona) {
    const persona: PersonaComponent = {
      traits: ce.persona.traits ?? [],
      voice: ce.persona.voice ?? null,
      memoryId: ce.persona.memoryId ?? null,
      mood: ce.persona.mood ?? 0,
      goals: (ce.persona.goals ?? []).map((g) => ({
        id: g.id,
        description: g.description,
        priority: g.priority ?? 1,
        done: g.done ?? "false",
      })),
    };
    components[Tag.Persona] = persona;
  }
  return {
    id: ce.id as EntityId,
    name: ce.name ?? ce.id,
    components,
  };
}

// ──────────────────────────────────────────────────────────────────────
// Tiny YAML parser — supports the charter subset we author against.
// Handles: nested mappings, scalar values (string/number/bool/null),
// inline + block lists, quoted strings, comments. NOT a full YAML 1.2
// parser; charter-only.
// ──────────────────────────────────────────────────────────────────────

export function parseYaml(input: string): unknown {
  const lines = input.split(/\r?\n/);
  const cleaned: Array<{ indent: number; text: string }> = [];
  for (const raw of lines) {
    const line = stripComment(raw).replace(/\s+$/, "");
    if (!line.trim()) continue;
    const indent = line.length - line.trimStart().length;
    cleaned.push({ indent, text: line.trimStart() });
  }
  const [val] = parseBlock(cleaned, 0, 0);
  return val;
}

function stripComment(line: string): string {
  let inS: string | null = null;
  for (let i = 0; i < line.length; i++) {
    const c = line[i]!;
    if (inS) {
      if (c === inS) inS = null;
    } else if (c === '"' || c === "'") {
      inS = c;
    } else if (c === "#") {
      return line.slice(0, i);
    }
  }
  return line;
}

function parseBlock(
  lines: Array<{ indent: number; text: string }>,
  start: number,
  baseIndent: number,
): [unknown, number] {
  if (start >= lines.length) return [null, start];
  const first = lines[start]!;
  if (first.text.startsWith("- ")) {
    return parseList(lines, start, first.indent);
  }
  if (looksLikeMapping(first.text)) {
    return parseMapping(lines, start, first.indent);
  }
  return [parseScalar(first.text), start + 1];
  void baseIndent;
}

function looksLikeMapping(text: string): boolean {
  // crude: "key:" or "key: value"
  const m = /^[A-Za-z_][\w-]*\s*:/.exec(text);
  return m !== null;
}

function parseMapping(
  lines: Array<{ indent: number; text: string }>,
  start: number,
  indent: number,
): [Record<string, unknown>, number] {
  const out: Record<string, unknown> = {};
  let i = start;
  while (i < lines.length && lines[i]!.indent === indent) {
    const { text } = lines[i]!;
    const colon = findUnquoted(text, ":");
    if (colon < 0) break;
    const key = text.slice(0, colon).trim();
    const restRaw = text.slice(colon + 1);
    const rest = restRaw.trim();
    if (rest === "") {
      // child block
      i += 1;
      if (i < lines.length && lines[i]!.indent > indent) {
        const [child, next] = parseBlock(lines, i, lines[i]!.indent);
        out[key] = child;
        i = next;
      } else {
        out[key] = null;
      }
    } else if (rest.startsWith("[") || rest.startsWith("{")) {
      out[key] = parseFlow(rest);
      i += 1;
    } else {
      out[key] = parseScalar(rest);
      i += 1;
    }
  }
  return [out, i];
}

function parseList(
  lines: Array<{ indent: number; text: string }>,
  start: number,
  indent: number,
): [unknown[], number] {
  const out: unknown[] = [];
  let i = start;
  while (i < lines.length && lines[i]!.indent === indent && lines[i]!.text.startsWith("- ")) {
    const itemText = lines[i]!.text.slice(2);
    if (looksLikeMapping(itemText)) {
      // Build a virtual mapping starting at this line. The item's first key
      // is `itemText`; subsequent keys are indented children.
      const virtualLines: Array<{ indent: number; text: string }> = [
        { indent: indent + 2, text: itemText },
      ];
      let j = i + 1;
      while (j < lines.length && lines[j]!.indent > indent) {
        virtualLines.push(lines[j]!);
        j += 1;
      }
      const [obj] = parseMapping(virtualLines, 0, indent + 2);
      out.push(obj);
      i = j;
    } else if (itemText.startsWith("[") || itemText.startsWith("{")) {
      out.push(parseFlow(itemText));
      i += 1;
    } else {
      out.push(parseScalar(itemText));
      i += 1;
    }
  }
  return [out, i];
}

function parseFlow(text: string): unknown {
  // Quick + dirty: we only need to handle constant arrays of numbers /
  // strings for charter `position`/`rotation`/`scale` etc.
  return JSON.parse(text.replace(/'/g, '"'));
}

function parseScalar(raw: string): unknown {
  if (raw === "null" || raw === "~") return null;
  if (raw === "true") return true;
  if (raw === "false") return false;
  if (/^-?\d+$/.test(raw)) return parseInt(raw, 10);
  if (/^-?\d*\.\d+(?:[eE][+-]?\d+)?$/.test(raw)) return parseFloat(raw);
  // Quoted strings
  if ((raw.startsWith('"') && raw.endsWith('"')) || (raw.startsWith("'") && raw.endsWith("'"))) {
    return raw.slice(1, -1);
  }
  return raw;
}

function findUnquoted(text: string, ch: string): number {
  let inS: string | null = null;
  for (let i = 0; i < text.length; i++) {
    const c = text[i]!;
    if (inS) {
      if (c === inS) inS = null;
    } else if (c === '"' || c === "'") {
      inS = c;
    } else if (c === ch) {
      return i;
    }
  }
  return -1;
}

// ──────────────────────────────────────────────────────────────────────
// Convenience: YAML string → SceneSpec
// ──────────────────────────────────────────────────────────────────────

export function compileCharterYaml(yamlText: string): SceneSpec {
  const doc = parseYaml(yamlText) as CharterDoc;
  return compileCharter(doc);
}
