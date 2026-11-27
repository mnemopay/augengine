import type { Substance, Reaction, ReactiveComponent } from "./types.js";
export const REFERENCE_K = 298.15;
export const GAS_CONSTANT = 8.31446261815324;
export const WATER_BOILING_K = 373.15;
export const WATER_LATENT_J_PER_MOL = 40650;
export const WATER_FREEZING_K = 273.15;
export const WATER_FUSION_J_PER_MOL = 6010;
/** Constant-Cp engineering approximations; not a broad-temperature property database. */
export const SUBSTANCES: Readonly<Record<string, Substance>> = Object.freeze({
  carbon: {
    id: "carbon",
    phase: "solid",
    elements: { C: 1 },
    molarMassKg: 0.012011,
    heatCapacityJPerMolK: 8.53,
    formationEnthalpyJPerMol: 0,
  },
  oxygen: {
    id: "oxygen",
    phase: "gas",
    elements: { O: 2 },
    molarMassKg: 0.031998,
    heatCapacityJPerMolK: 29.4,
    formationEnthalpyJPerMol: 0,
  },
  nitrogen: {
    id: "nitrogen",
    phase: "gas",
    elements: { N: 2 },
    molarMassKg: 0.028014,
    heatCapacityJPerMolK: 29.1,
    formationEnthalpyJPerMol: 0,
  },
  carbonDioxide: {
    id: "carbonDioxide",
    phase: "gas",
    elements: { C: 1, O: 2 },
    molarMassKg: 0.044009,
    heatCapacityJPerMolK: 37.1,
    formationEnthalpyJPerMol: -393510,
  },
  water: {
    id: "water",
    phase: "liquid",
    elements: { H: 2, O: 1 },
    molarMassKg: 0.018015,
    heatCapacityJPerMolK: 75.3,
    formationEnthalpyJPerMol: -285830,
  },
  ice: {
    id: "ice",
    phase: "solid",
    elements: { H: 2, O: 1 },
    molarMassKg: 0.018015,
    heatCapacityJPerMolK: 37.6,
    formationEnthalpyJPerMol:
      -285830 -
      WATER_FUSION_J_PER_MOL +
      (75.3 - 37.6) * (WATER_FREEZING_K - REFERENCE_K),
  },
  steam: {
    id: "steam",
    phase: "gas",
    elements: { H: 2, O: 1 },
    molarMassKg: 0.018015,
    heatCapacityJPerMolK: 33.6,
    formationEnthalpyJPerMol:
      -285830 +
      WATER_LATENT_J_PER_MOL +
      (75.3 - 33.6) * (WATER_BOILING_K - REFERENCE_K),
  },
});
export const REACTIONS: Readonly<Record<string, Reaction>> = Object.freeze({
  carbonOxidation: {
    id: "carbonOxidation",
    reactants: { carbon: 1, oxygen: 1 },
    products: { carbonDioxide: 1 },
    ratePerSecond: 0.12,
    activationTemperatureK: 700,
  },
});
for (const s of Object.values(SUBSTANCES)) {
  Object.freeze(s.elements);
  Object.freeze(s);
}
for (const r of Object.values(REACTIONS)) {
  Object.freeze(r.reactants);
  Object.freeze(r.products);
  Object.freeze(r);
}
export function substance(id: string): Substance {
  const s = Object.hasOwn(SUBSTANCES, id) ? SUBSTANCES[id] : undefined;
  if (!s) throw new Error(`Unknown substance ${id}`);
  return s;
}
export function heatCapacity(body: ReactiveComponent): number {
  let sum = 0;
  for (const id in body.amountsMol)
    sum +=
      body.amountsMol[id]! *
      (substance(id).heatCapacityJPerMolK -
        (body.volumeM3 !== undefined && substance(id).phase === "gas"
          ? GAS_CONSTANT
          : 0));
  return sum;
}
export function enthalpyAt(
  body: ReactiveComponent,
  temperatureK: number,
): number {
  let sum = 0;
  for (const id in body.amountsMol) {
    const s = substance(id);
    sum +=
      body.amountsMol[id]! *
      (s.formationEnthalpyJPerMol +
        s.heatCapacityJPerMolK * (temperatureK - REFERENCE_K));
  }
  return sum;
}
export function temperature(body: ReactiveComponent): number {
  let cp = 0,
    formation = 0;
  for (const id in body.amountsMol) {
    const s = substance(id),
      n = body.amountsMol[id]!;
    const gasCorrection =
      body.volumeM3 !== undefined && s.phase === "gas" ? GAS_CONSTANT : 0;
    cp += n * (s.heatCapacityJPerMolK - gasCorrection);
    formation += n * (s.formationEnthalpyJPerMol - gasCorrection * REFERENCE_K);
  }
  if (cp <= 0) throw new Error("Reactive body must contain material");
  return REFERENCE_K + (body.energyJ - formation) / cp;
}
export function pressure(body: ReactiveComponent): number {
  if (!body.volumeM3) return 0;
  const moles = Object.entries(body.amountsMol).reduce(
    (sum, [id, n]) => sum + (substance(id).phase === "gas" ? n : 0),
    0,
  );
  return (moles * GAS_CONSTANT * temperature(body)) / body.volumeM3;
}
/** Fixed-volume bodies track U=H-nRT; open/fixed-pressure bodies track H. */
export function energyAt(
  body: ReactiveComponent,
  temperatureK: number,
): number {
  let energy = enthalpyAt(body, temperatureK);
  if (body.volumeM3 !== undefined)
    for (const id in body.amountsMol) {
      if (substance(id).phase === "gas")
        energy -= body.amountsMol[id]! * GAS_CONSTANT * temperatureK;
    }
  return energy;
}
export function createReactiveBody(
  amountsMol: Record<string, number>,
  temperatureK: number,
  options: Omit<ReactiveComponent, "amountsMol" | "energyJ"> = {},
): ReactiveComponent {
  const body: ReactiveComponent = {
    ...structuredClone(options),
    amountsMol: { ...amountsMol },
    energyJ: 0,
  };
  if (!Number.isFinite(temperatureK) || temperatureK < 1)
    throw new Error("Temperature must be at least 1 K");
  for (const [id, n] of Object.entries(amountsMol)) {
    substance(id);
    if (!Number.isFinite(n) || n < 0)
      throw new Error("Invalid substance amount");
  }
  if (heatCapacity(body) <= 0)
    throw new Error("Reactive body must contain material");
  if (
    body.volumeM3 !== undefined &&
    (!Number.isFinite(body.volumeM3) || body.volumeM3 <= 0)
  )
    throw new Error("Gas volume must be positive");
  body.energyJ = energyAt(body, temperatureK);
  return body;
}
export function inventory(body: ReactiveComponent): {
  massKg: number;
  elementsMol: Record<string, number>;
} {
  let massKg = 0;
  const elementsMol: Record<string, number> = {};
  for (const [id, n] of Object.entries(body.amountsMol)) {
    const s = substance(id);
    massKg += n * s.molarMassKg;
    for (const [element, count] of Object.entries(s.elements))
      elementsMol[element] = (elementsMol[element] ?? 0) + n * count;
  }
  return { massKg, elementsMol };
}
