"""Thin, experiment-independent adapter for fitting a cMHN model.

The official :mod:`mhn` dependency is imported lazily so preprocessing,
simulation, and table-only validation remain usable without the compiled
backend.  This module is the only place in the public codebase that knows the
optimizer API.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .validation import assert_binary_matrix


@dataclass(frozen=True)
class MhnFitConfig:
    """Reproducible cMHN fitting settings."""

    lambda_multipliers: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0)
    cv_folds: int = 5
    pick_1se: bool = True
    max_iterations: int = 5000
    relative_tolerance: float = 1.0e-7
    random_seed: int = 20260630


@dataclass(frozen=True)
class MhnFitResult:
    """Numerical model output and its cross-validation audit table."""

    theta: np.ndarray
    events: tuple[str, ...]
    selected_lambda: float
    cv_scores: pd.DataFrame


def fit_cmh(matrix: pd.DataFrame, config: MhnFitConfig | None = None) -> MhnFitResult:
    """Fit cMHN to a binary event matrix using L1-penalized CV.

    The selected penalty is scaled by cohort size, matching the original
    experiment implementation.  A clear runtime error is raised when the
    optional compiled MHN dependency is unavailable.
    """

    config = config or MhnFitConfig()
    if matrix.empty or matrix.shape[1] < 2:
        raise ValueError("cMHN fitting requires at least two event columns")
    assert_binary_matrix(matrix, "matrix")
    try:
        import mhn as backend
        from mhn.optimizers import Optimizer
    except ImportError as exc:  # pragma: no cover - depends on external wheel
        raise RuntimeError(
            "The official 'mhn' package is required for model fitting. "
            "Use Python 3.11 or 3.12 and install mhn==1.2.3."
        ) from exc

    values = matrix.astype(np.int32)
    sample_count = len(values)
    multipliers = np.asarray(config.lambda_multipliers, dtype=float)
    if np.any(multipliers <= 0):
        raise ValueError("lambda_multipliers must be positive")

    np.random.seed(config.random_seed)
    backend.set_seed(config.random_seed)
    optimizer = Optimizer(Optimizer.MHNType.cMHN)
    optimizer.set_device(optimizer.Device.CPU)
    optimizer.set_penalty(optimizer.Penalty.L1)
    optimizer.load_data_matrix(values)
    selected_lambda, cv = optimizer.lambda_from_cv(
        lambda_vector=multipliers / sample_count,
        nfolds=config.cv_folds,
        return_lambda_scores=True,
        pick_1se=config.pick_1se,
        show_progressbar=False,
    )
    model = optimizer.train(
        lam=float(selected_lambda),
        maxit=config.max_iterations,
        reltol=config.relative_tolerance,
        round_result=False,
    )
    theta = np.asarray(model.log_theta, dtype=float)
    if theta.shape != (matrix.shape[1], matrix.shape[1]) or not np.isfinite(theta).all():
        raise RuntimeError("cMHN returned an invalid theta matrix")

    cv_scores = cv.rename(
        columns={
            "Lambda Value": "lambda",
            "Mean Score": "mean_test_log_likelihood",
            "Standard Error": "standard_error",
        }
    ).copy()
    cv_scores["lambda_multiplier"] = cv_scores["lambda"] * sample_count
    cv_scores["selected"] = np.isclose(cv_scores["lambda"], selected_lambda)
    return MhnFitResult(
        theta=theta,
        events=tuple(matrix.columns.astype(str)),
        selected_lambda=float(selected_lambda),
        cv_scores=cv_scores,
    )
