"""Held-out longitudinal validation of cross-sectionally predicted R*."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..core.validation import require_columns
from ..evaluation.metrics import (
    average_precision,
    binary_auc,
    bootstrap_interval,
    cluster_bootstrap_interval,
    safe_rank_correlation,
)
from ..io.results import ResultWriter


@dataclass(frozen=True)
class LongitudinalConfig:
    """Evaluation settings for a prepared, leakage-controlled pair table."""

    study_column: str = "study_id"
    score_column: str = "predicted_log2_R"
    persistence_column: str = "empirical_persistent"
    dwell_proxy_column: str = "minimum_observed_dwell_interval"
    upper_quantile: float = 2.0 / 3.0
    lower_quantile: float = 1.0 / 3.0
    bootstrap_replicates: int = 1000
    random_seed: int = 20260630
    bootstrap_group_column: str = "patient_id"


def _study_metrics(frame: pd.DataFrame, config: LongitudinalConfig, seed: int) -> dict[str, float | int | str]:
    work = frame.copy()
    for column in [config.score_column, config.persistence_column, config.dwell_proxy_column]:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    evaluable = work.dropna(subset=[config.score_column, config.persistence_column]).copy()
    labels = evaluable[config.persistence_column].astype(int).to_numpy()
    scores = evaluable[config.score_column].to_numpy(dtype=float)
    prevalence = float(labels.mean()) if len(labels) else np.nan
    auc = binary_auc(labels, scores)
    ap = average_precision(labels, scores)

    low_cut = float(evaluable[config.score_column].quantile(config.lower_quantile)) if len(evaluable) else np.nan
    high_cut = float(evaluable[config.score_column].quantile(config.upper_quantile)) if len(evaluable) else np.nan
    low = evaluable[evaluable[config.score_column].le(low_cut)]
    high = evaluable[evaluable[config.score_column].ge(high_cut)]
    low_rate = float(low[config.persistence_column].mean()) if len(low) else np.nan
    high_rate = float(high[config.persistence_column].mean()) if len(high) else np.nan

    dwell = evaluable.dropna(subset=[config.dwell_proxy_column])
    rho, rho_p = safe_rank_correlation(
        dwell[config.score_column].to_numpy(dtype=float),
        np.log1p(dwell[config.dwell_proxy_column].to_numpy(dtype=float)),
    )
    bootstrap = cluster_bootstrap_interval if config.bootstrap_group_column in evaluable else bootstrap_interval
    bootstrap_kwargs = (
        {"group_column": config.bootstrap_group_column}
        if bootstrap is cluster_bootstrap_interval
        else {}
    )
    auc_ci = bootstrap(
        evaluable,
        lambda sample: binary_auc(
            sample[config.persistence_column].astype(int).to_numpy(),
            sample[config.score_column].to_numpy(dtype=float),
        ),
        replicates=config.bootstrap_replicates,
        seed=seed,
        **bootstrap_kwargs,
    )
    rho_bootstrap = cluster_bootstrap_interval if config.bootstrap_group_column in dwell else bootstrap_interval
    rho_kwargs = (
        {"group_column": config.bootstrap_group_column}
        if rho_bootstrap is cluster_bootstrap_interval
        else {}
    )
    rho_ci = rho_bootstrap(
        dwell,
        lambda sample: safe_rank_correlation(
            sample[config.score_column].to_numpy(dtype=float),
            np.log1p(sample[config.dwell_proxy_column].to_numpy(dtype=float)),
        )[0],
        replicates=config.bootstrap_replicates,
        seed=seed + 1,
        **rho_kwargs,
    )
    def persistence_contrast(sample: pd.DataFrame) -> float:
        lower = sample[config.score_column].quantile(config.lower_quantile)
        upper = sample[config.score_column].quantile(config.upper_quantile)
        low_sample = sample[sample[config.score_column].le(lower)]
        high_sample = sample[sample[config.score_column].ge(upper)]
        if low_sample.empty or high_sample.empty:
            return np.nan
        return float(
            high_sample[config.persistence_column].mean()
            - low_sample[config.persistence_column].mean()
        )

    delta_ci = bootstrap(
        evaluable,
        persistence_contrast,
        replicates=config.bootstrap_replicates,
        seed=seed + 2,
        **bootstrap_kwargs,
    )
    return {
        "evaluable_pairs": len(evaluable),
        "persistent_pairs": int(labels.sum()) if len(labels) else 0,
        "changed_pairs": int(len(labels) - labels.sum()) if len(labels) else 0,
        "persistence_rate": prevalence,
        "auc": auc,
        "auc_ci_low": auc_ci[0],
        "auc_ci_high": auc_ci[1],
        "average_precision": ap,
        "average_precision_lift": ap / prevalence if np.isfinite(ap) and prevalence > 0 else np.nan,
        "high_rstar_persistence_rate": high_rate,
        "low_rstar_persistence_rate": low_rate,
        "delta_persistence_rate_high_minus_low": high_rate - low_rate,
        "delta_persistence_ci_low": delta_ci[0],
        "delta_persistence_ci_high": delta_ci[1],
        "spearman_r_minimum_dwell_interval": rho,
        "spearman_p_minimum_dwell_interval": rho_p,
        "spearman_r_minimum_dwell_ci_low": rho_ci[0],
        "spearman_r_minimum_dwell_ci_high": rho_ci[1],
        "bootstrap_unit": config.bootstrap_group_column if bootstrap is cluster_bootstrap_interval else "row",
    }


def evaluate_longitudinal_pairs(
    pairs: pd.DataFrame,
    *,
    output_dir: str | Path | None = None,
    config: LongitudinalConfig | None = None,
    input_files: Sequence[str | Path] = (),
) -> dict[str, pd.DataFrame]:
    """Evaluate R* against persistence and minimum observed dwell proxies.

    The input must contain predictions obtained without using the evaluated
    patient's later sample.  The function records that contract but cannot infer
    leakage control from values alone; callers should include ``score_source``
    and ``split_id`` audit columns whenever available.
    """

    config = config or LongitudinalConfig()
    require_columns(
        pairs,
        [config.study_column, config.score_column, config.persistence_column, config.dwell_proxy_column],
        "longitudinal_pairs",
    )
    rows = []
    for index, (study, frame) in enumerate(pairs.groupby(config.study_column, sort=True)):
        row = _study_metrics(frame, config, config.random_seed + index * 100)
        row[config.study_column] = study
        rows.append(row)
    summary = pd.DataFrame(rows)
    columns = [config.study_column] + [column for column in summary if column != config.study_column]
    summary = summary[columns]
    outputs = {"pair_predictions": pairs.copy(), "longitudinal_metrics": summary}
    if output_dir is not None:
        writer = ResultWriter(
            output_dir,
            input_files=list(input_files),
            metadata={
                "workflow": "evaluate_longitudinal_pairs",
                "random_seed": config.random_seed,
                "bootstrap_replicates": config.bootstrap_replicates,
                "bootstrap_group_column": config.bootstrap_group_column,
            },
        )
        writer.table("pair_predictions", outputs["pair_predictions"])
        writer.table("longitudinal_metrics", summary)
        writer.json("resolved_config", asdict(config))
        writer.manifest()
    return outputs
