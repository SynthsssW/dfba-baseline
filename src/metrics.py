"""Error metrics for dFBA trajectories vs. wet-lab data.

We report three numbers and defend each in the README:

  nRMSE_log  — root-mean-square error in log-biomass (dimensionless, scale-free
               on a quantity that grows exponentially).
  AUC_diff   — relative absolute difference of trajectory area-under-curve,
               summed across observed species/metabolites. Captures total
               integrated flux through the system.
  cross_t    — absolute error in glucose-exhaustion time (h). The single most
               operationally important point in a batch culture: when does the
               culture switch carbon source?
"""
from __future__ import annotations

import numpy as np


def interpolate_to(t_query: np.ndarray, t_sim: np.ndarray, y_sim: np.ndarray) -> np.ndarray:
    return np.interp(t_query, t_sim, y_sim)


def nrmse_log(y_obs: np.ndarray, y_pred: np.ndarray, eps: float = 1e-3) -> float:
    a = np.log(np.maximum(y_obs, eps))
    b = np.log(np.maximum(y_pred, eps))
    return float(np.sqrt(np.mean((a - b) ** 2)) / (np.max(a) - np.min(a) + 1e-9))


def auc_relative_error(t: np.ndarray, y_obs: np.ndarray, y_pred: np.ndarray) -> float:
    auc_obs = np.trapezoid(y_obs, t)
    auc_pred = np.trapezoid(y_pred, t)
    if abs(auc_obs) < 1e-9:
        return float(abs(auc_pred))
    return float(abs(auc_pred - auc_obs) / abs(auc_obs))


def crossover_time_error(t: np.ndarray, sub_obs: np.ndarray, sub_pred: np.ndarray, threshold: float = 0.1) -> float:
    """|t_exhaust_obs - t_exhaust_pred| where exhaust = first crossing of threshold."""
    def first_below(y):
        idx = np.where(y < threshold)[0]
        return t[idx[0]] if idx.size else t[-1]
    return float(abs(first_below(sub_obs) - first_below(sub_pred)))


def all_metrics(wet, sim) -> dict:
    """wet: dict with t_h, biomass_gL, glucose_mM, acetate_mM.
    sim: dict from dfba.simulate (single-species case)."""
    t_obs = wet["t_h"]
    t_sim = sim["t"]

    sp_name = next(iter(sim["biomass"].keys()))
    b_pred = interpolate_to(t_obs, t_sim, sim["biomass"][sp_name])
    glc_pred = interpolate_to(t_obs, t_sim, sim["conc"]["EX_glc__D_e"])
    ac_pred = interpolate_to(t_obs, t_sim, sim["conc"]["EX_ac_e"])

    out = {
        "nRMSE_log_biomass": nrmse_log(wet["biomass_gL"], b_pred),
        "nRMSE_log_glucose": nrmse_log(wet["glucose_mM"], glc_pred),
        "nRMSE_log_acetate": nrmse_log(wet["acetate_mM"], ac_pred),
        "AUC_relerr_biomass": auc_relative_error(t_obs, wet["biomass_gL"], b_pred),
        "AUC_relerr_glucose": auc_relative_error(t_obs, wet["glucose_mM"], glc_pred),
        "AUC_relerr_acetate": auc_relative_error(t_obs, wet["acetate_mM"], ac_pred),
        "glucose_exhaust_time_err_h": crossover_time_error(
            t_obs, wet["glucose_mM"], glc_pred, threshold=0.5
        ),
    }
    return out
