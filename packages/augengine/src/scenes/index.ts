/** Authored scene specs shared by Studio, proof-room, and the flagship slice. */
export { createChemistryLab } from "./chemistry-lab.js";

export {
  CHARACTER_PROOF_ROOM,
  PROOF_LEAD_ID,
  PROOF_NPC_ID,
  PROOF_PLAYER_ID,
  PROOF_ROOM_SKY,
  PROOF_ROOM_IBL,
  buildCharacterProofRoomEntities,
} from "./character-proof-room.js";

export {
  LEDGERFALL_BEATS,
  LEDGERFALL_SCENES,
  ledgerfallGoldenPathSeconds,
  buildMezzanineArrival,
  buildInformantBooth,
  buildSubstationExtract,
  type LedgerfallBeat,
  type LedgerfallBeatId,
} from "./ledgerfall.js";
