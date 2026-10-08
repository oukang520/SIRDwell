"""Bootstrap uncertainty for state-level R* estimates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .scoring import ScoreThresholds


@dataclass(frozen=True)
class BootstrapConfig:
    """Bootstrap settings for R* stability."""

    replicates: int = 200
    top_k: int = 10
    random_seed: int = 20260630
    ci_low_quantile: float = 0.025
    ci_high_quantile: float = 0.975


def bootstrap_relative_dwell(
    state_counts: pd.DataFrame,
    edges: pd.DataFrame,
    thresholds: ScoreThresholds | None = None,
    bootstrap: BootstrapConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap R* by resampling state counts with a multinomial model.

    Parameters
    ----------
    state_counts:
        DataFrame with ``state`` and ``N_v`` columns.
    edges:
        DataFrame with ``source_state``, ``target_state`` and
        ``edge_probability``. ``inflow_contribution`` is recalculated per
        bootstrap replicate from resampled source occupancy.
    """

    thresholds = thresholds or ScoreThresholds()
    bootstrap = bootstrap or BootstrapConfig()
    required = {"state", "N_v"}
    missing = required.difference(state_counts.columns)
    if missing:
        raise ValueError(f"state_counts is missing columns: {sorted(missing)}")

    states = state_counts["state"].astype(str).tolist()
    counts = pd.to_numeric(state_counts["N_v"], errors="coerce").fillna(0).to_numpy(dtype=int)
    total = int(counts.sum())
    if total <= 0:
        raise ValueError("state_counts must contain at least one observation")
    probabilities = counts / total
    state_index = {state: index for index, state in enumerate(states)}

    if edges.empty:
        edge_source = np.array([], dtype=int)
        edge_target = np.array([], dtype=int)
        edge_probability = np.array([], dtype=float)
    else:
        edge_required = {"source_state", "target_state", "edge_probability"}
        edge_missing = edge_required.difference(edges.columns)
        if edge_missing:
            raise ValueError(f"edges is missing columns: {sorted(edge_missing)}")
        edge_source = np.array([state_index.get(str(state), -1) for state in edges["source_state"]], dtype=int)
        edge_target = np.array([state_index.get(str(state), -1) for state in edges["target_state"]], dtype=int)
        edge_probability = pd.to_numeric(edges["edge_probability"], errors="coerce").fillna(0).to_numpy(dtype=float)
        valid = (edge_source >= 0) & (edge_target >= 0) & np.isfinite(edge_probability) & (edge_probability > 0)
        edge_source = edge_source[valid]
        edge_target = edge_target[valid]
        edge_probability = edge_probability[valid]

    rng = np.random.default_rng(bootstrap.random_seed)
    values = np.full((bootstrap.replicates, len(states)), np.nan, dtype=float)
    top_counts = np.zeros(len(states), dtype=int)
    high_confidence_top_counts = np.zeros(len(states), dtype=int)

    for replicate in range(bootstrap.replicates):
        sampled = rng.multinomial(total, probabilities)
        sampled_l = sampled / total
        f_hat = np.zeros(len(states), dtype=float)
        if len(edge_source):
            contributions = sampled_l[edge_source] * edge_probability
            np.add.at(f_hat, edge_target, contributions)

        eligible = (sampled >= thresholds.minimum_state_count) & (f_hat >= thresholds.minimum_inflow)
        r_raw = sampled_l / (f_hat + thresholds.epsilon)
        finite = eligible & np.isfinite(r_raw)
        if not finite.any():
            continue
        normalizer = float(np.median(r_raw[finite]))
        if normalizer <= 0 or not np.isfinite(normalizer):
            continue
        r_star = r_raw / normalizer
        values[replicate] = r_star

        candidates = np.flatnonzero(finite)
        top = candidates[np.argsort(r_star[candidates])[-bootstrap.top_k :]]
        top_counts[top] += 1

        high_confidence = finite & (sampled >= thresholds.high_confidence_state_count)
        hc_candidates = np.flatnonzero(high_confidence)
        if len(hc_candidates):
            hc_top = hc_candidates[np.argsort(r_star[hc_candidates])[-bootstrap.top_k :]]
            high_confidence_top_counts[hc_top] += 1

    summary = pd.DataFrame({"state": states, "N_v": counts})
    summary["R_star_bootstrap_median"] = np.nanmedian(values, axis=0)
    summary["R_star_ci_low"] = np.nanquantile(values, bootstrap.ci_low_quantile, axis=0)
    summary["R_star_ci_high"] = np.nanquantile(values, bootstrap.ci_high_quantile, axis=0)
    summary["top_bootstrap_stability"] = top_counts / max(bootstrap.replicates, 1)
    summary["high_confidence_top_bootstrap_stability"] = high_confidence_top_counts / max(bootstrap.replicates, 1)
    summary["bootstrap_replicates"] = bootstrap.replicates

    long_rows = []
    for replicate in range(bootstrap.replicates):
        for index, state in enumerate(states):
            value = values[replicate, index]
            if np.isfinite(value):
                long_rows.append({"replicate": replicate + 1, "state": state, "R_star": float(value)})
    return summary, pd.DataFrame(long_rows)
