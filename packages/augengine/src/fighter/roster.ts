import { MOVES, type MoveDef } from "./moves.js";

export interface CharacterDef {
  id: "mara" | "holt";
  name: string;
  maxHealth: number;
  walkForward: number;
  walkBack: number;
  moves: readonly MoveDef[];
}

function withCrest(startup: number): MoveDef[] {
  return MOVES.map((move) => (move.id === "crest" ? { ...move, startup } : move));
}

export const MARA: CharacterDef = {
  id: "mara",
  name: "Mara",
  maxHealth: 1000,
  walkForward: 5,
  walkBack: 3,
  moves: withCrest(18),
};

export const HOLT: CharacterDef = {
  id: "holt",
  name: "Holt",
  maxHealth: 1000,
  walkForward: 4,
  walkBack: 2,
  moves: withCrest(22),
};

export function characterMove(character: CharacterDef, id: string): MoveDef {
  const move = character.moves.find((item) => item.id === id);
  if (!move) throw new Error(`${character.id} has no ${id}`);
  return move;
}
