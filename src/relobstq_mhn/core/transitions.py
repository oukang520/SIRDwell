"""MHN-to-state inflow utilities.

The key method quantity is ``F_hat_v``: a relative expected inflow into state
``v`` under an MHN-derived one-step transition backbone and the observed
occupancy of predecessor states.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from .states import genotype_events, genotype_signature, genotype_vector


ProbabilityProvider = Callable[[str], dict[str, float]]


def softmax_addition_probabilities(
    theta: np.ndarray,
    genotype: str,
    events: Sequence[str],
) -> dict[str, float]:
    """Return event-addition probabilities from a cMHN log-rate matrix.

    For absent event ``j``, the unnormalized log-rate is
    ``theta[j, j] + sum(theta[j, present])``. The returned probabilities are
    normalized over absent events only.
    """

    theta = np.asarray(theta, dtype=float)
    if theta.shape != (len(events), len(events)):
        raise ValueError("theta must have shape (n_events, n_events)")
    present = genotype_vector(genotype, events).astype(bool)
    absent = np.flatnonzero(~present)
    if len(absent) == 0:
        return {}
    logits = np.array(
        [theta[index, index] + theta[index, present].sum() for index in absent],
        dtype=float,
    )
    scaled = np.exp(logits - logits.max())
    probabilities = scaled / scaled.sum()
    return {str(events[index]): float(prob) for index, prob in zip(absent, probabilities)}


def same_stage_one_step_edges(
    occupancy: pd.DataFrame,
    events: Sequence[str],
    probabilities: ProbabilityProvider,
    *,
    observed_sources_only: bool = True,
    rule: str = "rule_a_one_step",
) -> pd.DataFrame:
    """Build same-stage one-event predecessor edges.

    The function expects ``occupancy`` to contain ``state``, ``stage``,
    ``genotype`` and ``L_v`` columns. It does not aggregate ``F_hat``; use
    :func:`aggregate_inflow` for that step.
    """

    required = {"state", "stage", "genotype", "L_v"}
    missing = required.difference(occupancy.columns)
    if missing:
        raise ValueError(f"occupancy is missing columns: {sorted(missing)}")

    observed_states = set(occupancy["state"].astype(str))
    source_l = occupancy.set_index("state")["L_v"].astype(float).to_dict()
    rows: list[dict[str, object]] = []
    event_order = [str(event).upper() for event in events]

    for target in occupancy.itertuples(index=False):
        target_events = genotype_events(getattr(target, "genotype"))
        if not target_events:
            continue
        for added_event in target_events:
            source_events = [event for event in target_events if event != added_event]
            source_genotype = "+".join(source_events) if source_events else "WT"
            source_state = f"{getattr(target, 'stage')}::{source_genotype}"
            if observed_sources_only and source_state not in observed_states:
                continue
            if added_event not in event_order:
                continue
            edge_probability = probabilities(source_genotype).get(added_event, 0.0)
            if edge_probability <= 0:
                continue
            source_occupancy = float(source_l.get(source_state, 0.0))
            rows.append(
                {
                    "rule": rule,
                    "source_state": source_state,
                    "target_state": getattr(target, "state"),
                    "predecessor_type": "same_stage_one_event",
                    "event_added": added_event,
                    "step_distance": 1,
                    "edge_probability": float(edge_probability),
                    "source_L": source_occupancy,
                    "inflow_contribution": source_occupancy * float(edge_probability),
                }
            )
    return pd.DataFrame(rows)


def aggregate_inflow(
    occupancy: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    rule: str = "rule_a_one_step",
    minimum_state_count: int = 5,
    minimum_inflow: float = 1.0e-8,
) -> pd.DataFrame:
    """Aggregate predecessor edges into state-level ``F_hat``.

    Returns the original occupancy table with added columns:
    ``F_hat``, predecessor counts, dominant predecessor fields and eligibility
    flags used by downstream R* scoring.
    """

    required = {"state", "N_v", "L_v"}
    missing = required.difference(occupancy.columns)
    if missing:
        raise ValueError(f"occupancy is missing columns: {sorted(missing)}")

    result = occupancy.copy()
    if edges.empty:
        result["F_hat"] = 0.0
        result["n_predecessors"] = 0
        result["dominant_predecessor"] = ""
        result["dominant_edge_probability"] = 0.0
        result["dominant_contribution"] = 0.0
        result["genotype_inflow"] = 0.0
        result["stage_inflow"] = 0.0
    else:
        edge_required = {"source_state", "target_state", "edge_probability", "inflow_contribution", "predecessor_type"}
        edge_missing = edge_required.difference(edges.columns)
        if edge_missing:
            raise ValueError(f"edges is missing columns: {sorted(edge_missing)}")

        totals = edges.groupby("target_state")["inflow_contribution"].sum().rename("F_hat")
        counts = edges.groupby("target_state").size().rename("n_predecessors")
        dominant = (
            edges.sort_values("inflow_contribution", ascending=False)
            .drop_duplicates("target_state")
            .set_index("target_state")
        )
        genotype = (
            edges[edges["predecessor_type"].astype(str).str.startswith("same_stage")]
            .groupby("target_state")["inflow_contribution"]
            .sum()
            .rename("genotype_inflow")
        )
        stage = (
            edges[edges["predecessor_type"].astype(str).str.startswith("previous_stage")]
            .groupby("target_state")["inflow_contribution"]
            .sum()
            .rename("stage_inflow")
        )
        result = result.join(totals, on="state").join(counts, on="state")
        result = result.join(genotype, on="state").join(stage, on="state")
        result["dominant_predecessor"] = result["state"].map(dominant["source_state"])
        result["dominant_edge_probability"] = result["state"].map(dominant["edge_probability"])
        result["dominant_contribution"] = result["state"].map(dominant["inflow_contribution"])
        for column in [
            "F_hat",
            "n_predecessors",
            "dominant_edge_probability",
            "dominant_contribution",
            "genotype_inflow",
            "stage_inflow",
        ]:
            result[column] = result[column].fillna(0.0)
        result["n_predecessors"] = result["n_predecessors"].astype(int)
        result["dominant_predecessor"] = result["dominant_predecessor"].fillna("")

    result["rule"] = rule
    result["count_eligible"] = result["N_v"].astype(float) >= int(minimum_state_count)
    result["inflow_eligible"] = result["F_hat"].astype(float) >= float(minimum_inflow)
    result["stable_for_scoring"] = result["count_eligible"] & result["inflow_eligible"]
    result["flags"] = np.select(
        [
            ~result["count_eligible"],
            result["count_eligible"] & ~result["inflow_eligible"],
        ],
        ["rare_state", "low_or_zero_inflow"],
        default="stable",
    )
    return result.sort_values(["F_hat", "N_v"], ascending=[False, False]).reset_index(drop=True)


def probability_provider_from_theta(theta: np.ndarray, events: Sequence[str]) -> ProbabilityProvider:
    """Create a cached probability provider from a cMHN theta matrix."""

    cache: dict[str, dict[str, float]] = {}

    def provider(genotype: str) -> dict[str, float]:
        key = genotype_signature(genotype_vector(genotype, events), events)
        if key not in cache:
            cache[key] = softmax_addition_probabilities(theta, key, events)
        return cache[key]

    return provider
