"""Compact table-only analyses derived from the primary state-score output."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.validation import require_columns
from ..evaluation.metrics import safe_rank_correlation


def inflow_computability_summary(scores: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Summarize whether conditional inflow and R* are numerically usable."""

    require_columns(
        scores,
        ["state", "N_v", "L_v", "F_hat", "R_star", "count_eligible", "inflow_eligible", "eligible_relobstq"],
        "scores",
    )
    eligible = scores[scores["eligible_relobstq"].astype(bool)].copy()
    finite = np.isfinite(eligible[["L_v", "F_hat", "R_star"]].to_numpy(dtype=float)).all(axis=1)
    return pd.DataFrame(
        [
            {
                "observed_states": len(scores),
                "count_eligible_states": int(scores["count_eligible"].astype(bool).sum()),
                "positive_inflow_states": int(scores["inflow_eligible"].astype(bool).sum()),
                "rstar_eligible_states": len(eligible),
                "finite_eligible_states": int(finite.sum()),
                "finite_eligible_fraction": float(finite.mean()) if len(finite) else np.nan,
                "one_step_edges": len(edges),
                "total_conditional_inflow": float(pd.to_numeric(scores["F_hat"], errors="coerce").sum()),
                "median_positive_inflow": float(
                    pd.to_numeric(scores.loc[scores["F_hat"].gt(0), "F_hat"], errors="coerce").median()
                ),
            }
        ]
    )


def rstar_landscape_summary(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return an eligible-state R* landscape and a compact cohort summary."""

    require_columns(scores, ["state", "stage", "genotype", "N_v", "L_v", "F_hat", "R_star"], "scores")
    eligible = scores.copy()
    if "eligible_relobstq" in eligible:
        eligible = eligible[eligible["eligible_relobstq"].astype(bool)].copy()
    eligible = eligible.replace([np.inf, -np.inf], np.nan).dropna(subset=["R_star", "L_v", "F_hat"])
    eligible = eligible.sort_values(["R_star", "N_v"], ascending=[False, False]).reset_index(drop=True)
    eligible.insert(0, "rstar_rank", np.arange(1, len(eligible) + 1))
    summary = pd.DataFrame(
        [
            {
                "eligible_states": len(eligible),
                "median_R_star": float(eligible["R_star"].median()) if len(eligible) else np.nan,
                "q1_R_star": float(eligible["R_star"].quantile(0.25)) if len(eligible) else np.nan,
                "q3_R_star": float(eligible["R_star"].quantile(0.75)) if len(eligible) else np.nan,
                "maximum_R_star": float(eligible["R_star"].max()) if len(eligible) else np.nan,
                "states_R_star_gt_2": int(eligible["R_star"].gt(2).sum()),
                "states_R_star_lt_0_5": int(eligible["R_star"].lt(0.5).sum()),
            }
        ]
    )
    return eligible, summary


def information_gain_summary(scores: pd.DataFrame, *, top_k: int = 10) -> pd.DataFrame:
    """Quantify how R* reorders states relative to occupancy and inflow alone."""

    require_columns(scores, ["state", "L_v", "F_hat", "R_star"], "scores")
    work = scores.copy()
    if "eligible_relobstq" in work:
        work = work[work["eligible_relobstq"].astype(bool)]
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=["L_v", "F_hat", "R_star"])
    selected_k = min(top_k, len(work))
    rho_l, p_l = safe_rank_correlation(work["R_star"], work["L_v"])
    rho_f, p_f = safe_rank_correlation(work["R_star"], work["F_hat"])
    top_r = set(work.nlargest(selected_k, "R_star")["state"].astype(str))
    top_l = set(work.nlargest(selected_k, "L_v")["state"].astype(str))
    top_f = set(work.nlargest(selected_k, "F_hat")["state"].astype(str))
    return pd.DataFrame(
        [
            {
                "states": len(work),
                "top_k": selected_k,
                "spearman_R_vs_occupancy": rho_l,
                "spearman_R_vs_occupancy_p": p_l,
                "spearman_R_vs_inflow": rho_f,
                "spearman_R_vs_inflow_p": p_f,
                "top_k_overlap_R_occupancy": len(top_r & top_l),
                "top_k_overlap_R_inflow": len(top_r & top_f),
            }
        ]
    )
