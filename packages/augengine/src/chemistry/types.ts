/** SI units: mol, J, K, m³, Pa, W. Enthalpy reference is 298.15 K. */
export interface Substance {
  id: string;
  phase: "solid" | "liquid" | "gas";
  elements: Readonly<Record<string, number>>;
  molarMassKg: number;
  heatCapacityJPerMolK: number;
  formationEnthalpyJPerMol: number;
}
export interface Reaction {
  id: string;
  reactants: Readonly<Record<string, number>>;
  products: Readonly<Record<string, number>>;
  /** Authored first-order limiting-reactant model, NOT measured kinetics. */
  ratePerSecond: number;
  activationTemperatureK: number;
}
export interface ReactiveComponent {
  amountsMol: Record<string, number>;
  /** U for fixed-volume bodies, H otherwise; includes chemical and sensible terms. */
  energyJ: number;
  reactionIds?: string[];
  heaterWatts?: number;
  /** Gas headspace only; solid/liquid displacement is not inferred. */
  volumeM3?: number;
  /** Ideal-gas vent, removes gas and its enthalpy into an explicit ledger. */
  maxPressurePa?: number;
  /** Water phase equilibrium is currently the fixed 1-atm approximation. */
  waterPhaseEquilibrium?: boolean;
  thermalLinks?: { to: string; conductanceWPerK: number }[];
  contactConductanceWPerK?: number;
  /** Explicit finite-volume interface to a local environmental field. */
  fieldExchange?: {
    fieldIndex: number;
    cell: [number, number, number];
    conductanceWPerK: number;
    gasConductanceM3PerSecond?: number;
  };
}
export interface ChemistryEvent {
  type:
    | "reaction"
    | "evaporation"
    | "condensation"
    | "melting"
    | "freezing"
    | "vent";
  body: string;
  process: string;
  moles: number;
}
