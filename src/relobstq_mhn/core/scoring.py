"""Relative dwell and observation-enrichment scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScoreThresholds:
    """Thresholds used to define states eligible for R* normalization."""

    minimum_state_count: int = 5
    minimum_inflow: float = 1.0e-8
    high_confidence_state_count: int = 10
    epsilon: float = 1.0e-12


def compute_relative_dwell(
    inflow_table: pd.DataFrame,
    thresholds: ScoreThresholds | None = None,
    *,
    normalizer: float | None = None,
) -> tuple[pd.DataFrame, float]:
    """Compute raw relative dwell ``R_v`` and normalized ``R*_v``.

    ``R_v = L_v / (F_hat_v + epsilon)`` and ``R*`` is normalized by the median
    ``R_v`` among count- and inflow-eligible states. The function returns the
    scored table and the normalizer used.
    """

    thresholds = thresholds or ScoreThresholds()
    required = {"state", "N_v", "L_v", "F_hat"}
    missing = required.difference(inflow_table.columns)
    if missing:
        raise ValueError(f"inflow_table is missing columns: {sorted(missing)}")

    result = inflow_table.copy()
    result["eligible_relobstq"] = (
        (pd.to_numeric(result["N_v"], errors="coerce") >= thresholds.minimum_state_count)
        & (pd.to_numeric(result["F_hat"], errors="coerce") >= thresholds.minimum_inflow)
    )
    result["R_raw"] = pd.to_numeric(result["L_v"], errors="coerce") / (
        pd.to_numeric(result["F_hat"], errors="coerce") + thresholds.epsilon
    )

    if normalizer is None:
        values = result.loc[result["eligible_relobstq"], "R_raw"].replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            raise ValueError("No eligible states are available to normalize R*")
        normalizer = float(values.median())
    if not np.isfinite(normalizer) or normalizer <= 0:
        raise ValueError(f"R* normalizer must be positive and finite, got {normalizer!r}")

    result["R_star"] = result["R_raw"] / float(normalizer)
    result["log2_R_star"] = np.log2(result["R_star"].clip(lower=thresholds.epsilon))
    result["high_confidence_relobstq"] = result["eligible_relobstq"] & (
        pd.to_numeric(result["N_v"], errors="coerce") >= thresholds.high_confidence_state_count
    )
    return result, float(normalizer)


def compute_observation_enrichment(
    scored_table: pd.DataFrame,
    expected_progression: pd.Series | pd.DataFrame | dict[str, float],
    thresholds: ScoreThresholds | None = None,
    *,
    expected_column: str = "Lhat_progression",
) -> pd.DataFrame:
    """Compute ``O*`` from a progression-only expected occupancy distribution."""

    thresholds = thresholds or ScoreThresholds()
    result = scored_table.copy()
    if isinstance(expected_progression, pd.DataFrame):
        if "state" not in expected_progression or expected_column not in expected_progression:
            raise ValueError("expected_progression DataFrame must contain state and expected columns")
        expected = expected_progression.set_index("state")[expected_column]
    else:
        expected = pd.Series(expected_progression, dtype=float)
    result[expected_column] = result["state"].map(expected)
    result["O_star"] = pd.to_numeric(result["L_v"], errors="coerce") / (
        pd.to_numeric(result[expected_column], errors="coerce") + thresholds.epsilon
    )
    result["log2_O_star"] = np.log2(result["O_star"].clip(lower=thresholds.epsilon))
    return result


def classify_relative_states(
    scored_table: pd.DataFrame,
    *,
    r_reference: float = 1.0,
    o_reference: float = 1.0,
) -> pd.DataFrame:
    """Add qualitative interpretation flags for R* and O* quadrants."""

    result = scored_table.copy()
    result["direction_flag"] = np.select(
        [
            result["R_star"].astype(float) > r_reference,
            result["R_star"].astype(float) < r_reference,
        ],
        ["relative_bottleneck", "fast_passing"],
        default="neutral",
    )
    if "O_star" in result:
        result["interpretation_flag"] = np.select(
            [
                (result["R_star"] > r_reference) & (result["O_star"] > o_reference),
                (result["R_star"] > r_reference) & (result["O_star"] <= o_reference),
                (result["R_star"] <= r_reference) & (result["O_star"] > o_reference),
            ],
            [
                "bottleneck_with_observation_enrichment",
                "bottleneck_without_observation_enrichment",
                "observation_enriched_fast_accessible",
            ],
            default="background_or_fast",
        )
    return result
