"""Timestep sweep on the Varma & Palsson 1994 E. coli aerobic batch dataset."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dfba import simulate
from src.datasets import varma_aerobic_setup
from src.metrics import all_metrics


def run_one(dt: float, t_end: float = 10.0):
    species, wet = varma_aerobic_setup()
    t0 = time.perf_counter()
    sim = simulate(species, t_end=t_end, dt=dt)
    wall = time.perf_counter() - t0
    return sim, wet, wall


def main():
    dts = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
    results = {}
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 9), sharex=True)

    species, wet = varma_aerobic_setup()
    # Plot wet-lab once.
    axes[0].scatter(wet["t_h"], wet["biomass_gL"], color="black", zorder=10, label="Varma 1994 wet-lab", s=30)
    axes[1].scatter(wet["t_h"], wet["glucose_mM"], color="black", zorder=10, s=30)
    axes[2].scatter(wet["t_h"], wet["acetate_mM"], color="black", zorder=10, s=30)

    cmap = plt.get_cmap("viridis")
    for i, dt in enumerate(dts):
        print(f"[varma] dt={dt} ...", flush=True)
        sim, _, wall = run_one(dt)
        m = all_metrics(wet, sim)
        results[dt] = {"metrics": m, "wall_s": wall, "n_steps": int(10.0 / dt)}
        c = cmap(i / max(1, len(dts) - 1))
        label = f"dt={dt} h  ({wall:.2f}s)"
        axes[0].plot(sim["t"], sim["biomass"]["E_coli"], color=c, label=label)
        axes[1].plot(sim["t"], sim["conc"]["EX_glc__D_e"], color=c)
        axes[2].plot(sim["t"], sim["conc"]["EX_ac_e"], color=c)

    axes[0].set_ylabel("Biomass [g/L]")
    axes[1].set_ylabel("Glucose [mM]")
    axes[2].set_ylabel("Acetate [mM]")
    axes[2].set_xlabel("Time [h]")
    axes[0].set_yscale("log")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("dFBA timestep sweep — E. coli aerobic batch (Varma & Palsson 1994)")
    fig.tight_layout()
    out = ROOT / "figures" / "varma_timestep_sweep.png"
    fig.savefig(out, dpi=140)
    print("wrote", out)

    # Convergence plot: metric vs dt (log-log).
    fig2, ax = plt.subplots(figsize=(6, 4.5))
    dts_arr = np.array(dts)
    for k in ["nRMSE_log_biomass", "AUC_relerr_glucose", "glucose_exhaust_time_err_h"]:
        ys = [results[dt]["metrics"][k] for dt in dts]
        ax.loglog(dts_arr, ys, marker="o", label=k)
    ax.set_xlabel("dt [h]")
    ax.set_ylabel("error (lower = better)")
    ax.set_title("Error vs. integration timestep")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig2.tight_layout()
    out2 = ROOT / "figures" / "varma_convergence.png"
    fig2.savefig(out2, dpi=140)
    print("wrote", out2)

    # Wall time vs dt.
    fig3, ax = plt.subplots(figsize=(6, 4.5))
    walls = [results[dt]["wall_s"] for dt in dts]
    ax.loglog(dts_arr, walls, marker="o", color="darkred")
    ax.set_xlabel("dt [h]")
    ax.set_ylabel("wall time [s]")
    ax.set_title("dFBA cost vs. timestep (single E. coli core model)")
    ax.grid(True, which="both", alpha=0.3)
    fig3.tight_layout()
    out3 = ROOT / "figures" / "varma_walltime.png"
    fig3.savefig(out3, dpi=140)
    print("wrote", out3)

    (ROOT / "data" / "varma_results.json").write_text(json.dumps(results, indent=2, default=float))
    print("wrote results.json")


if __name__ == "__main__":
    main()
