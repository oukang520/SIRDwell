"""Small method-level orchestration helpers.

These helpers compose the pure method functions without imposing any
experiment-specific file layout. Experiment scripts can use this module once
they are ready to depend on the core package.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .bootstrap import BootstrapConfig, bootstrap_relative_dwell
from .scoring import ScoreThresholds, classify_relative_states, compute_relative_dwell
from .transitions import aggregate_inflow, probability_provider_from_theta, same_stage_one_step_edges


def score_states_from_mhn(
    occupancy: pd.DataFrame,
    theta: np.ndarray,
    events: Sequence[str],
    *,
    thresholds: ScoreThresholds | None = None,
    bootstrap: BootstrapConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Compute edges, F_hat and R* from occupancy plus cMHN theta.

    Returns ``(scores, edges, bootstrap_summary)``. Bootstrap output is ``None``
    when no ``BootstrapConfig`` is supplied.
    """

    thresholds = thresholds or ScoreThresholds()
    provider = probability_provider_from_theta(theta, events)
    edges = same_stage_one_step_edges(occupancy, events, provider)
    inflow = aggregate_inflow(
        occupancy,
        edges,
        minimum_state_count=thresholds.minimum_state_count,
        minimum_inflow=thresholds.minimum_inflow,
    )
    scores, _ = compute_relative_dwell(inflow, thresholds)
    scores = classify_relative_states(scores)
    boot_summary = None
    if bootstrap is not None:
        boot_summary, _ = bootstrap_relative_dwell(
            scores[["state", "N_v"]],
            edges,
            thresholds=thresholds,
            bootstrap=bootstrap,
        )
        scores = scores.merge(boot_summary.drop(columns=["N_v"]), on="state", how="left")
    return scores, edges, boot_summary
