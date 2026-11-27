/**
 * Track 5 — Game-native ECS systems barrel.
 *
 * Public surface for the engine-side game systems. Import from
 * `@mnemopay/augengine/runtime/systems`.
 *
 * Bundle cost (gzip, measured after Track 5 lands): ~10 KB.
 */

export {
  GameTag,
  MISSION_ENTITY_ID,
  getGameComponent,
  setGameComponent,
  removeGameComponent,
  trustTierForScore,
  type GameTagKey,
  type HealthComponent,
  type ArmorComponent,
  type WeaponComponent,
  type ObjectiveComponent,
  type MissionComponent,
  type RecruitComponent,
  type RecruitFsmState,
  type SquadTrustComponent,
  type TrustTier,
  type MayaMemoryComponent,
  type VehicleComponent,
  type RacerComponent,
  type RaceStateComponent,
  type UIComponent,
} from "./components.js";

export {
  buildSystemContext,
  type GameEvent,
  type GameSystem,
  type SystemContext,
} from "./system.js";

export { HealthSystem, type ApplyDamageOptions } from "./health-system.js";
export {
  WeaponSystem,
  type FireOptions,
  type FireResult,
} from "./weapon-system.js";
export {
  ObjectiveSystem,
  type ObjectiveSystemOptions,
} from "./objective-system.js";
export {
  MissionSystem,
  createMissionEntity,
  type MissionCheckpoint,
  type MissionSystemOptions,
} from "./mission-system.js";
export {
  RecruitSystem,
  type RecruitSystemOptions,
} from "./recruit-system.js";
export { SquadTrustSystem } from "./squad-trust-system.js";
export {
  MayaMemorySystem,
  ensureMayaMemoryComponent,
  type MayaMemorySystemOptions,
} from "./maya-memory-system.js";
export {
  SaveSystem,
  browserSaveIO,
  memorySaveIO,
  type SaveIO,
} from "./save-system.js";
export { VehicleSystem } from "./vehicle-system.js";
export { RaceSystem } from "./race-system.js";
export { ChaseCameraSystem } from "./chase-camera-system.js";
export { UISystem } from "./ui-system.js";
export { VfxSystem } from "./vfx-system.js";
export { AudioSystem } from "./audio-system.js";

