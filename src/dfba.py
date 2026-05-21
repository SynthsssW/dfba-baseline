"""Direct Optimization Approach (DOA) dFBA solver.

Reference: Mahadevan, Edwards, Doyle (2002) "Dynamic Flux Balance Analysis of
Diauxic Growth in Escherichia coli". Biophysical Journal 83:1331-1340.

At each time step:
  1. Set substrate uptake bounds via Michaelis-Menten kinetics
  2. Solve FBA (max growth)
  3. Forward-Euler integrate biomass and metabolite concentrations
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import cobra


@dataclass
class KineticSubstrate:
    exchange_id: str
    vmax: float
    km: float
    initial_conc: float

    def uptake_bound(self, conc: float) -> float:
        if conc <= 0:
            return 0.0
        return self.vmax * conc / (self.km + conc)


@dataclass
class Species:
    model: cobra.Model
    name: str
    initial_biomass: float
    substrates: list[KineticSubstrate]
    # Exchange rxns whose flux we track but which we treat as unbounded sources
    # (e.g. O2 in aerobic batch, NH4) — bound is set once at the start.
    unbounded_sinks: dict[str, float] = field(default_factory=dict)
    # Secretion products to track (exchange IDs).
    products: list[str] = field(default_factory=list)


def _apply_bounds(species: Species, concentrations: dict[str, float]) -> None:
    for sub in species.substrates:
        bound = sub.uptake_bound(concentrations.get(sub.exchange_id, 0.0))
        species.model.reactions.get_by_id(sub.exchange_id).lower_bound = -bound
    for ex_id, lb in species.unbounded_sinks.items():
        species.model.reactions.get_by_id(ex_id).lower_bound = lb


def simulate(
    species_list: list[Species],
    t_end: float,
    dt: float,
    extra_initial: dict[str, float] | None = None,
) -> dict:
    """Run dFBA forward.

    Returns dict with keys: t, biomass[name], conc[exchange_id], flux[name][rxn_id].
    """
    # Pool all tracked metabolite concentrations (shared environment).
    concentrations: dict[str, float] = {}
    for sp in species_list:
        for sub in sp.substrates:
            concentrations.setdefault(sub.exchange_id, sub.initial_conc)
        for ex_id in sp.products:
            concentrations.setdefault(ex_id, 0.0)
    if extra_initial:
        concentrations.update(extra_initial)

    biomass = {sp.name: sp.initial_biomass for sp in species_list}

    n_steps = int(np.ceil(t_end / dt))
    times = np.linspace(0.0, n_steps * dt, n_steps + 1)

    hist_biomass = {sp.name: np.zeros(n_steps + 1) for sp in species_list}
    hist_conc = {ex: np.zeros(n_steps + 1) for ex in concentrations}
    for sp in species_list:
        hist_biomass[sp.name][0] = biomass[sp.name]
    for ex, c in concentrations.items():
        hist_conc[ex][0] = c

    for i in range(n_steps):
        # Per-species: apply bounds and solve FBA.
        fluxes_per_species = []
        mu_per_species = []
        for sp in species_list:
            _apply_bounds(sp, concentrations)
            sol = sp.model.optimize()
            if sol.status != "optimal" or sol.objective_value is None:
                mu = 0.0
                fluxes = {r.id: 0.0 for r in sp.model.reactions}
            else:
                mu = max(0.0, sol.objective_value)
                fluxes = sol.fluxes.to_dict()
            mu_per_species.append(mu)
            fluxes_per_species.append(fluxes)

        # Forward Euler: biomass and concentrations.
        for sp, mu, fluxes in zip(species_list, mu_per_species, fluxes_per_species):
            b = biomass[sp.name]
            new_b = b + dt * mu * b
            biomass[sp.name] = max(new_b, 0.0)
            for sub in sp.substrates:
                v = fluxes.get(sub.exchange_id, 0.0)  # < 0 = uptake
                concentrations[sub.exchange_id] += dt * v * b
            for ex_id in sp.products:
                v = fluxes.get(ex_id, 0.0)  # > 0 = secretion
                concentrations[ex_id] = concentrations.get(ex_id, 0.0) + dt * v * b

        # Clamp concentrations non-negative.
        for ex in concentrations:
            if concentrations[ex] < 0:
                concentrations[ex] = 0.0

        for sp in species_list:
            hist_biomass[sp.name][i + 1] = biomass[sp.name]
        for ex, c in concentrations.items():
            hist_conc[ex][i + 1] = c

    return {"t": times, "biomass": hist_biomass, "conc": hist_conc}
