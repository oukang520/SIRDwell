"""Innovation-specific ablation and falsification analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..core.scoring import ScoreThresholds, compute_relative_dwell
from ..core.transitions import aggregate_inflow, same_stage_one_step_edges
from ..core.validation import require_columns


def inflow_pairing_falsification(
    scores: pd.DataFrame,
    *,
    top_k: int = 10,
    replicates: int = 400,
    seed: int = 20260630,
    epsilon: float = 1.0e-12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Break the learned L-F pairing within stage/event-count strata."""

    require_columns(scores, ["state", "stage", "event_count", "L_v", "F_hat", "R_star"], "scores")
    eligible = scores.copy()
    if "eligible_relobstq" in eligible:
        eligible = eligible[eligible["eligible_relobstq"].astype(bool)]
    eligible = eligible.replace([np.inf, -np.inf], np.nan).dropna(subset=["L_v", "F_hat", "R_star"]).reset_index(drop=True)
    top_k = min(top_k, len(eligible))
    observed_top = set(eligible.nlargest(top_k, "R_star")["state"].astype(str))
    strata = eligible["stage"].astype(str) + "|e" + eligible["event_count"].astype(str)
    groups = list(strata.groupby(strata).groups.values())
    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(1, replicates + 1):
        shuffled = eligible["F_hat"].to_numpy(dtype=float).copy()
        for index in groups:
            positions = np.asarray(index, dtype=int)
            if len(positions) > 1:
                shuffled[positions] = rng.permutation(shuffled[positions])
        raw = eligible["L_v"].to_numpy(dtype=float) / (shuffled + epsilon)
        normalizer = np.nanmedian(raw)
        shuffled_r = raw / normalizer
        selected = np.argsort(shuffled_r)[-top_k:]
        overlap = len(observed_top.intersection(eligible.iloc[selected]["state"].astype(str)))
        rows.append({"replicate": replicate, "top_k": top_k, "top_overlap_fraction": overlap / max(top_k, 1)})
    replicates_table = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "replicates": replicates,
                "top_k": top_k,
                "observed_overlap_fraction": 1.0,
                "median_shuffled_overlap": replicates_table["top_overlap_fraction"].median(),
                "q1_shuffled_overlap": replicates_table["top_overlap_fraction"].quantile(0.25),
                "q3_shuffled_overlap": replicates_table["top_overlap_fraction"].quantile(0.75),
                "median_overlap_loss": 1.0 - replicates_table["top_overlap_fraction"].median(),
            }
        ]
    )
    return replicates_table, summary


def backbone_ablation(
    occupancy: pd.DataFrame,
    events: list[str],
    learned_probabilities,
    *,
    thresholds: ScoreThresholds | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare learned MHN inflow with a uniform one-step backbone."""

    thresholds = thresholds or ScoreThresholds()

    def uniform(genotype: str) -> dict[str, float]:
        present = set() if genotype == "WT" else set(str(genotype).split("+"))
        absent = [event for event in events if event not in present]
        return {event: 1.0 / len(absent) for event in absent} if absent else {}

    variants = []
    for name, provider in [("learned_mhn", learned_probabilities), ("uniform_one_step", uniform)]:
        edges = same_stage_one_step_edges(occupancy, events, provider, rule=name)
        inflow = aggregate_inflow(
            occupancy,
            edges,
            rule=name,
            minimum_state_count=thresholds.minimum_state_count,
            minimum_inflow=thresholds.minimum_inflow,
        )
        scores, _ = compute_relative_dwell(inflow, thresholds)
        variants.append(scores[["state", "N_v", "L_v", "F_hat", "R_star"]].assign(backbone=name))
    details = pd.concat(variants, ignore_index=True)
    pivot = details.pivot(index="state", columns="backbone", values="R_star").dropna()
    rho = spearmanr(pivot["learned_mhn"], pivot["uniform_one_step"]).statistic if len(pivot) >= 3 else np.nan
    learned_top = set(pivot.nlargest(min(10, len(pivot)), "learned_mhn").index)
    uniform_top = set(pivot.nlargest(min(10, len(pivot)), "uniform_one_step").index)
    summary = pd.DataFrame(
        [
            {
                "states_compared": len(pivot),
                "spearman_learned_vs_uniform": rho,
                "top10_jaccard": len(learned_top & uniform_top) / max(len(learned_top | uniform_top), 1),
            }
        ]
    )
    return details, summary


def denominator_ablation(
    occupancy: pd.DataFrame,
    events: list[str],
    learned_probabilities,
    event_frequencies: dict[str, float],
    *,
    thresholds: ScoreThresholds | None = None,
    top_k: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare the MHN inflow denominator with objective alternatives.

    The variants retain the same observed states and one-step predecessor rule.
    Only the denominator-generating next-event probabilities are changed.
    ``occupancy_only`` removes the inflow denominator entirely.
    """

    thresholds = thresholds or ScoreThresholds()
    frequencies = {str(event): max(float(event_frequencies.get(str(event), 0.0)), 0.0) for event in events}

    def uniform(genotype: str) -> dict[str, float]:
        present = set() if genotype == "WT" else set(str(genotype).split("+"))
        absent = [event for event in events if event not in present]
        return {event: 1.0 / len(absent) for event in absent} if absent else {}

    def frequency(genotype: str) -> dict[str, float]:
        present = set() if genotype == "WT" else set(str(genotype).split("+"))
        absent = [event for event in events if event not in present]
        total = sum(frequencies[event] for event in absent)
        if not absent:
            return {}
        if total <= 0:
            return {event: 1.0 / len(absent) for event in absent}
        return {event: frequencies[event] / total for event in absent}

    variants = []
    for name, provider in [
        ("full_mhn", learned_probabilities),
        ("uniform_inflow", uniform),
        ("frequency_inflow", frequency),
    ]:
        edges = same_stage_one_step_edges(occupancy, events, provider, rule=name)
        inflow = aggregate_inflow(
            occupancy,
            edges,
            rule=name,
            minimum_state_count=thresholds.minimum_state_count,
            minimum_inflow=thresholds.minimum_inflow,
        )
        scores, _ = compute_relative_dwell(inflow, thresholds)
        variants.append(scores[["state", "N_v", "L_v", "F_hat", "R_star", "eligible_relobstq"]].assign(variant=name))

    occupancy_only = occupancy[["state", "N_v", "L_v"]].copy()
    occupancy_only["F_hat"] = np.nan
    occupancy_only["eligible_relobstq"] = occupancy_only["N_v"].ge(thresholds.minimum_state_count)
    normalizer = occupancy_only.loc[occupancy_only["eligible_relobstq"], "L_v"].median()
    occupancy_only["R_star"] = occupancy_only["L_v"] / normalizer
    occupancy_only["variant"] = "occupancy_only"
    variants.append(occupancy_only)

    details = pd.concat(variants, ignore_index=True)
    reference = details[
        details["variant"].eq("full_mhn") & details["eligible_relobstq"].astype(bool)
    ].set_index("state")["R_star"]
    reference_top = set(reference.nlargest(min(top_k, len(reference))).index)
    rows = []
    for variant, frame in details.groupby("variant", sort=False):
        values = frame[frame["eligible_relobstq"].astype(bool)].set_index("state")["R_star"]
        paired = pd.concat([reference.rename("full"), values.rename("variant")], axis=1).dropna()
        rho = spearmanr(paired["full"], paired["variant"]).statistic if len(paired) >= 3 else np.nan
        variant_top = set(values.nlargest(min(top_k, len(values))).index)
        overlap = len(reference_top & variant_top)
        union = len(reference_top | variant_top)
        rows.append(
            {
                "variant": variant,
                "states_compared": len(paired),
                "spearman_vs_full_mhn": rho,
                "top_k": min(top_k, len(reference_top)),
                "top_k_overlap": overlap,
                "top_k_retention_fraction": overlap / max(len(reference_top), 1),
                "top_k_jaccard": overlap / max(union, 1),
            }
        )
    return details, pd.DataFrame(rows)
