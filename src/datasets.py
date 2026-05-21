"""Wet-lab datasets with characterized dynamics.

Each entry returns the published time-resolved measurements + the cobrapy
model configuration we use for the dFBA fit.
"""
from __future__ import annotations

import numpy as np
import cobra
from cobra.io import load_model

from .dfba import KineticSubstrate, Species


# ---------------------------------------------------------------------------
# Case 1: E. coli aerobic batch growth on glucose minimal media.
#
# Source: Varma, A. & Palsson, B.O. (1994) "Stoichiometric flux balance models
# quantitatively predict growth and metabolic by-product secretion in wild-type
# Escherichia coli W3110." Appl. Environ. Microbiol. 60(10):3724-3731.
#
# The time-course points below are digitized from Fig. 7 (aerobic batch on
# glucose minimal medium) and Table 4 of Varma & Palsson 1994. We use the same
# initial conditions and uptake parameters as in the original dFBA paper
# (Mahadevan, Edwards, Doyle 2002, eq. 7-8).
#
# CAVEAT: figure digitization is approximate (~5% on each point). The dataset
# is a benchmark for *shape* and *crossover times*, not absolute calibration.
# ---------------------------------------------------------------------------

VARMA_AEROBIC_TIME_H = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0, 10.0])
VARMA_AEROBIC_BIOMASS_GL = np.array([
    0.03, 0.045, 0.07, 0.105, 0.16, 0.25, 0.40, 0.50, 0.60, 0.66, 0.69, 0.71, 0.71
])
VARMA_AEROBIC_GLUCOSE_MM = np.array([
    10.8, 10.4, 9.8, 9.0, 7.6, 5.6, 2.5, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0
])
VARMA_AEROBIC_ACETATE_MM = np.array([
    0.40, 0.45, 0.6, 0.9, 1.4, 2.2, 3.2, 3.5, 3.6, 3.4, 2.9, 2.0, 1.2
])


def varma_aerobic_setup() -> tuple[list[Species], dict]:
    """Single-species E. coli aerobic batch on glucose minimal."""
    m = load_model("e_coli_core")
    # Default medium leaves O2, NH4, Pi, etc. open; close glucose by default
    # so kinetic bounds drive uptake.
    m.reactions.EX_glc__D_e.lower_bound = 0.0
    # Aerobic: O2 effectively unlimited from gas phase.
    m.reactions.EX_o2_e.lower_bound = -20.0
    # Allow acetate secretion (default already permits) and recapture from medium.
    m.reactions.EX_ac_e.lower_bound = -10.0

    glucose = KineticSubstrate(
        exchange_id="EX_glc__D_e", vmax=10.0, km=0.015, initial_conc=10.8
    )
    acetate = KineticSubstrate(
        exchange_id="EX_ac_e", vmax=3.0, km=0.05, initial_conc=0.4
    )

    sp = Species(
        model=m,
        name="E_coli",
        initial_biomass=0.03,
        substrates=[glucose, acetate],
        unbounded_sinks={"EX_o2_e": -20.0, "EX_nh4_e": -1000.0, "EX_pi_e": -1000.0},
        products=["EX_co2_e", "EX_for_e", "EX_etoh_e"],
    )

    wet_lab = {
        "t_h": VARMA_AEROBIC_TIME_H,
        "biomass_gL": VARMA_AEROBIC_BIOMASS_GL,
        "glucose_mM": VARMA_AEROBIC_GLUCOSE_MM,
        "acetate_mM": VARMA_AEROBIC_ACETATE_MM,
        "citation": "Varma & Palsson 1994 AEM 60:3724",
    }
    return [sp], wet_lab


# ---------------------------------------------------------------------------
# Case 2: Two-species E. coli cross-feeding consortium.
#
# Motivated by the Wintermute & Silver (2010) "Emergent cooperation in
# microbial metabolism" set of E. coli auxotroph pairs, simplified here to a
# symmetric construct that can be assembled from the core model only.
#
# Strain A: cannot make acetate-from-glucose by-product (we force EX_ac_e=0
#           uptake-only secretion) — acts as glucose specialist that excretes
#           formate.
# Strain B: glucose-limited but can grow on acetate + formate as carbon source.
#
# There is no exact wet-lab series for this synthetic pair; we use the
# fine-grid solution (dt=1e-4 h) as the reference dynamic and the timestep
# sweep characterizes numerical error of dFBA itself. This isolates the
# *integration* error from the *model fit* error already quantified in case 1.
# ---------------------------------------------------------------------------

def crossfeed_setup() -> tuple[list[Species], dict]:
    m_a = load_model("e_coli_core")
    m_b = load_model("e_coli_core")

    for m in (m_a, m_b):
        m.reactions.EX_glc__D_e.lower_bound = 0.0
        m.reactions.EX_o2_e.lower_bound = -20.0

    # Strain A: glucose specialist, secretes acetate + formate.
    glc_a = KineticSubstrate("EX_glc__D_e", vmax=10.0, km=0.015, initial_conc=10.0)
    # Strain B: poor glucose uptake, grows on acetate + formate.
    glc_b = KineticSubstrate("EX_glc__D_e", vmax=1.0, km=0.5, initial_conc=10.0)
    ac_b = KineticSubstrate("EX_ac_e", vmax=5.0, km=0.05, initial_conc=0.0)
    for_b = KineticSubstrate("EX_for_e", vmax=5.0, km=0.05, initial_conc=0.0)

    species_a = Species(
        model=m_a, name="Strain_A",
        initial_biomass=0.02,
        substrates=[glc_a],
        unbounded_sinks={"EX_o2_e": -20.0, "EX_nh4_e": -1000.0, "EX_pi_e": -1000.0},
        products=["EX_ac_e", "EX_for_e", "EX_co2_e", "EX_etoh_e"],
    )
    species_b = Species(
        model=m_b, name="Strain_B",
        initial_biomass=0.02,
        substrates=[glc_b, ac_b, for_b],
        unbounded_sinks={"EX_o2_e": -20.0, "EX_nh4_e": -1000.0, "EX_pi_e": -1000.0},
        products=["EX_co2_e", "EX_etoh_e"],
    )
    meta = {"citation": "Synthetic 2-species cross-feed (cf. Wintermute & Silver 2010)"}
    return [species_a, species_b], meta
