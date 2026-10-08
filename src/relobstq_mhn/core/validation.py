"""Lightweight schema validation for method-level inputs."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str = "frame") -> None:
    """Raise a clear error when required DataFrame columns are missing."""

    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def assert_binary_matrix(matrix: pd.DataFrame, name: str = "matrix") -> None:
    """Validate that a DataFrame contains only 0/1 values."""

    values = matrix.to_numpy()
    if not np.isin(values, [0, 1, False, True]).all():
        raise ValueError(f"{name} must be binary")


def audit_score_table(scores: pd.DataFrame) -> pd.DataFrame:
    """Return a compact audit table for a scored state table."""

    require_columns(scores, ["state", "N_v", "L_v", "F_hat", "R_star"], "scores")
    if "eligible_relobstq" in scores:
        stable = scores[scores["eligible_relobstq"].astype(bool)]
    else:
        stable = scores[(scores["N_v"] > 0) & (scores["F_hat"] > 0)]
    return pd.DataFrame(
        [
            {
                "states": int(len(scores)),
                "observed_states": int((scores["N_v"] > 0).sum()),
                "positive_inflow_states": int((scores["F_hat"] > 0).sum()),
                "median_R_star": float(stable["R_star"].median()) if len(stable) else np.nan,
                "max_R_star": float(stable["R_star"].max()) if len(stable) else np.nan,
            }
        ]
    )
