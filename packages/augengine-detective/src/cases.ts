/** Registry of hand-authored cases. Add a case here and every layer picks it up. */
import { WRENFIELD, type CaseDef } from "./case.js";
import { DEAD_AIR } from "./case-dead-air.js";

export const CASES: readonly CaseDef[] = [WRENFIELD, DEAD_AIR];
export const DEFAULT_CASE_ID = WRENFIELD.id;

export function isCaseId(v: unknown): v is string {
  return typeof v === "string" && CASES.some((c) => c.id === v);
}

/** The case with this id; unknown or missing ids fall back to the first case. */
export function caseById(id: string | null | undefined): CaseDef {
  return CASES.find((c) => c.id === id) ?? WRENFIELD;
}
