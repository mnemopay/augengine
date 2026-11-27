/** AI subsystem barrel. */

export * from "./types.js";
export { ObservationBus, spatial } from "./bus.js";
export {
  AIScheduler,
  type AIProvider,
  type AISchedulerOptions,
  type AIMetric,
} from "./scheduler.js";
export {
  NpcThinkScheduler,
  type ThinkTier,
  type NpcThinkContext,
  type NpcCombatState,
  type ScheduledNpc,
  type NpcThinkSchedulerOptions,
} from "./npc-think-scheduler.js";
export {
  ActionDispatcher,
  type DispatcherOptions,
  type Predicate,
  type Effect,
} from "./dispatcher.js";
export {
  renderPersonaSystem,
  activeGoals,
  getPersona,
  ensurePersona,
  ensurePersonaAxes,
  composeTurnContext,
  composeTurnPrompt,
  NULL_PERSONA_MEMORY,
  type PersonaMemoryAdapter,
  type PersonaMemoryContext,
  type PersonaObservation,
  type PersonaPromptInput,
  type PersonaTurnContext,
  type PersonaTurnInput,
} from "./persona.js";

// Phase 4 — providers, templates, runner, memory adapters.
export * from "./adapters/index.js";
export * from "./providers/index.js";
export * from "./templates/index.js";
export {
  PersonaRunner,
  type PersonaRunnerOptions,
  type DialogueRequest,
} from "./runner.js";
export * from "./memory/index.js";
