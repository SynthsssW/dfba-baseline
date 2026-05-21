"""Timestep convergence on the 2-species cross-feeding consortium.

There is no published wet-lab series for this synthetic pair, so the
'ground truth' here is the dFBA solution at the finest grid (dt=1e-4 h).
This isolates *integration* error from *model-fit* error and tells us
how small dt has to be before further refinement is wasted compute —
the practical question for surrogate training.
"""
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
from src.datasets import crossfeed_setup


def run_one(dt: float, t_end: float = 8.0):
    species, _ = crossfeed_setup()
    t0 = time.perf_counter()
    sim = simulate(species, t_end=t_end, dt=dt)
    return sim, time.perf_counter() - t0


def trajectory_l2(t_ref, y_ref, t, y):
    y_interp = np.interp(t_ref, t, y)
    denom = np.sqrt(np.trapezoid(y_ref ** 2, t_ref)) + 1e-9
    return float(np.sqrt(np.trapezoid((y_ref - y_interp) ** 2, t_ref)) / denom)


def main():
    dts = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
    ref_dt = 0.0005  # treat as ground truth
    print(f"[crossfeed] reference dt={ref_dt} ...", flush=True)
    sim_ref, wall_ref = run_one(ref_dt)
    print(f"reference done in {wall_ref:.1f}s")

    results = {}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    cmap = plt.get_cmap("viridis")

    # Plot reference dynamic in black behind everything.
    axes[0, 0].plot(sim_ref["t"], sim_ref["biomass"]["Strain_A"], color="black", lw=2, label="reference (dt=5e-4)")
    axes[0, 1].plot(sim_ref["t"], sim_ref["biomass"]["Strain_B"], color="black", lw=2)
    axes[1, 0].plot(sim_ref["t"], sim_ref["conc"]["EX_glc__D_e"], color="black", lw=2)
    axes[1, 1].plot(sim_ref["t"], sim_ref["conc"]["EX_ac_e"], color="black", lw=2)

    for i, dt in enumerate(dts):
        print(f"[crossfeed] dt={dt} ...", flush=True)
        sim, wall = run_one(dt)
        err = {
            "L2_Strain_A": trajectory_l2(sim_ref["t"], sim_ref["biomass"]["Strain_A"], sim["t"], sim["biomass"]["Strain_A"]),
            "L2_Strain_B": trajectory_l2(sim_ref["t"], sim_ref["biomass"]["Strain_B"], sim["t"], sim["biomass"]["Strain_B"]),
            "L2_glucose": trajectory_l2(sim_ref["t"], sim_ref["conc"]["EX_glc__D_e"], sim["t"], sim["conc"]["EX_glc__D_e"]),
            "L2_acetate": trajectory_l2(sim_ref["t"], sim_ref["conc"]["EX_ac_e"], sim["t"], sim["conc"]["EX_ac_e"]),
        }
        results[dt] = {"errors": err, "wall_s": wall, "n_steps": int(8.0 / dt)}
        c = cmap(i / max(1, len(dts) - 1))
        label = f"dt={dt}"
        axes[0, 0].plot(sim["t"], sim["biomass"]["Strain_A"], color=c, label=label, alpha=0.85)
        axes[0, 1].plot(sim["t"], sim["biomass"]["Strain_B"], color=c, alpha=0.85)
        axes[1, 0].plot(sim["t"], sim["conc"]["EX_glc__D_e"], color=c, alpha=0.85)
        axes[1, 1].plot(sim["t"], sim["conc"]["EX_ac_e"], color=c, alpha=0.85)

    axes[0, 0].set_ylabel("Strain A biomass [g/L]")
    axes[0, 1].set_ylabel("Strain B biomass [g/L]")
    axes[1, 0].set_ylabel("Glucose [mM]")
    axes[1, 1].set_ylabel("Acetate [mM]")
    axes[1, 0].set_xlabel("Time [h]")
    axes[1, 1].set_xlabel("Time [h]")
    axes[0, 0].set_yscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 0].legend(loc="lower right", fontsize=8)
    fig.suptitle("dFBA timestep convergence — 2-species E. coli cross-feed (ref = dt 5e-4 h)")
    fig.tight_layout()
    out = ROOT / "figures" / "crossfeed_timestep_sweep.png"
    fig.savefig(out, dpi=140)
    print("wrote", out)

    # Convergence plot.
    fig2, ax = plt.subplots(figsize=(6, 4.5))
    dts_arr = np.array(dts)
    for k in ["L2_Strain_A", "L2_Strain_B", "L2_glucose", "L2_acetate"]:
        ys = [results[dt]["errors"][k] for dt in dts]
        ax.loglog(dts_arr, ys, marker="o", label=k)
    ax.set_xlabel("dt [h]")
    ax.set_ylabel("relative L2 error vs. reference")
    ax.set_title("Numerical convergence (cross-feed consortium)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig2.tight_layout()
    out2 = ROOT / "figures" / "crossfeed_convergence.png"
    fig2.savefig(out2, dpi=140)
    print("wrote", out2)

    (ROOT / "data" / "crossfeed_results.json").write_text(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
