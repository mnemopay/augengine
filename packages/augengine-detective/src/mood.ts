/**
 * Which portrait a suspect shows for a line. Decided by the rules, never by
 * the model: the same utterance always gives the same mood.
 *
 *   cracked — a lie broke on this turn
 *   angry   — they are pressed with evidence that points at them, or accused
 *   evasive — the line is a scripted lie, or they deflect a question
 *   calm    — everything else
 */
import { EVIDENCE, type EvidenceId, type SuspectId } from "./case.js";
import type { Utterance } from "./rules.js";

export type Mood = "calm" | "evasive" | "angry" | "cracked";
export const MOODS: readonly Mood[] = ["calm", "evasive", "angry", "cracked"];

export interface MoodInput {
  utterance: Utterance;
  /** Evidence shown on this turn, if any. */
  presented?: EvidenceId;
  /** What the player typed, for ask turns. */
  playerText?: string;
}

const ACCUSING = /\b(you (killed|murdered|poisoned|did it)|murderer|killer|liar|you(?:'re| are) lying|confess)\b/i;

export function moodFor(input: MoodInput): Mood {
  const u = input.utterance;
  if (u.cracked.length > 0) return "cracked";
  const pointsAtThem = (id: EvidenceId | undefined) => !!id && EVIDENCE[id]?.pointsAt === u.suspect;
  if (pointsAtThem(input.presented)) return "angry";
  if (input.playerText && ACCUSING.test(input.playerText)) return "angry";
  if (u.lying || u.topic === "unknown") return "evasive";
  return "calm";
}

/** Public path of a portrait image (served from the app's /portraits). */
export function portraitPath(suspect: SuspectId | "vance", mood: Mood = "calm"): string {
  return `/portraits/${suspect}/${suspect === "vance" ? "calm" : mood}.webp`;
}
