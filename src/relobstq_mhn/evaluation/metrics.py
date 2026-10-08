"""Dependency-light metrics used by simulation and longitudinal validation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata, spearmanr


def _finite_pair(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    keep = np.isfinite(first) & np.isfinite(second)
    return first[keep], second[keep]


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC computed from ranks; returns NaN for one-class input."""

    labels, scores = _finite_pair(labels, scores)
    labels = labels.astype(int)
    positives = labels == 1
    n_positive = int(positives.sum())
    n_negative = int((~positives).sum())
    if n_positive == 0 or n_negative == 0:
        return np.nan
    ranks = rankdata(scores, method="average")
    return float((ranks[positives].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Average precision with deterministic descending score order."""

    labels, scores = _finite_pair(labels, scores)
    labels = labels.astype(int)
    positive_count = int(labels.sum())
    if positive_count == 0:
        return np.nan
    order = np.argsort(-scores, kind="mergesort")
    ordered = labels[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(precision[ordered == 1].sum() / positive_count)


def safe_rank_correlation(first: np.ndarray, second: np.ndarray, method: str = "spearman") -> tuple[float, float]:
    """Rank correlation with explicit handling of small or constant samples."""

    first, second = _finite_pair(first, second)
    if len(first) < 3 or np.unique(first).size < 2 or np.unique(second).size < 2:
        return np.nan, np.nan
    result = spearmanr(first, second) if method == "spearman" else kendalltau(first, second)
    return float(result.statistic), float(result.pvalue)


def pairwise_concordance(truth: np.ndarray, estimate: np.ndarray) -> float:
    """Fraction of non-tied pairs whose estimated and true order agrees."""

    truth, estimate = _finite_pair(truth, estimate)
    concordant = total = 0
    for left in range(len(truth)):
        for right in range(left + 1, len(truth)):
            if np.isclose(truth[left], truth[right]) or np.isclose(estimate[left], estimate[right]):
                continue
            total += 1
            concordant += int((truth[left] > truth[right]) == (estimate[left] > estimate[right]))
    return float(concordant / total) if total else np.nan


def bootstrap_interval(
    frame: pd.DataFrame,
    metric: Callable[[pd.DataFrame], float],
    *,
    replicates: int = 1000,
    seed: int = 20260630,
    quantiles: tuple[float, float] = (0.025, 0.975),
) -> tuple[float, float]:
    """Non-parametric row bootstrap confidence interval."""

    if frame.empty or replicates <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        sampled = frame.iloc[rng.integers(0, len(frame), size=len(frame))]
        value = metric(sampled)
        if np.isfinite(value):
            values.append(float(value))
    if not values:
        return np.nan, np.nan
    return tuple(float(value) for value in np.quantile(values, quantiles))


def cluster_bootstrap_interval(
    frame: pd.DataFrame,
    metric: Callable[[pd.DataFrame], float],
    *,
    group_column: str,
    replicates: int = 1000,
    seed: int = 20260630,
    quantiles: tuple[float, float] = (0.025, 0.975),
) -> tuple[float, float]:
    """Non-parametric cluster bootstrap confidence interval.

    Complete clusters are sampled with replacement, preserving dependence
    between multiple adjacent pairs contributed by the same patient.
    """

    if frame.empty or replicates <= 0 or group_column not in frame:
        return np.nan, np.nan
    groups = frame[group_column].dropna().unique()
    if len(groups) < 2:
        return np.nan, np.nan
    grouped = {group: frame[frame[group_column].eq(group)] for group in groups}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        selected = rng.choice(groups, size=len(groups), replace=True)
        sampled = pd.concat([grouped[group] for group in selected], ignore_index=True)
        value = metric(sampled)
        if np.isfinite(value):
            values.append(float(value))
    if not values:
        return np.nan, np.nan
    return tuple(float(value) for value in np.quantile(values, quantiles))
