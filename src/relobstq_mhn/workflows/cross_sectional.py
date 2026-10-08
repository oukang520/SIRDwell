"""Canonical real-cohort workflow from prepared input to R* result tables."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.bootstrap import BootstrapConfig
from ..core.mhn import MhnFitConfig, fit_cmh
from ..core.pipeline import score_states_from_mhn
from ..core.scoring import ScoreThresholds
from ..core.states import build_state_occupancy, canonical_genotype, genotype_signature
from ..core.validation import assert_binary_matrix, audit_score_table, require_columns
from ..io.results import ResultWriter


@dataclass(frozen=True)
class CrossSectionalConfig:
    """Settings shared by all independent real-cohort analyses."""

    thresholds: ScoreThresholds = field(default_factory=ScoreThresholds)
    mhn: MhnFitConfig = field(default_factory=MhnFitConfig)
    bootstrap_replicates: int = 500
    bootstrap_top_k: int = 10
    top_state_count: int = 25
    events: tuple[str, ...] | None = None


def load_prepared_cohort(
    input_dir: str | Path,
    events: tuple[str, ...] | list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], int]:
    """Load and cross-check the public experiment-ready data contract."""

    root = Path(input_dir)
    matrix = pd.read_csv(root / "mhn_training_matrix.csv")
    row_map = pd.read_csv(root / "mhn_row_index_map.csv")
    state_table = pd.read_csv(root / "state_table.csv")
    assert_binary_matrix(matrix, "mhn_training_matrix")
    require_columns(row_map, ["analysis_id"], "mhn_row_index_map")
    require_columns(state_table, ["analysis_id", "stage_group"], "state_table")
    if len(matrix) != len(row_map):
        raise ValueError("mhn_training_matrix and mhn_row_index_map have different row counts")
    if row_map["analysis_id"].duplicated().any() or state_table["analysis_id"].duplicated().any():
        raise ValueError("analysis_id must be unique in row map and state table")

    ordered = row_map[["analysis_id"]].merge(state_table, on="analysis_id", how="left", validate="one_to_one")
    if ordered["stage_group"].isna().any():
        raise ValueError("row map contains analysis IDs absent from state_table")
    all_events = matrix.columns.astype(str).tolist()
    selected_events = list(events) if events is not None else all_events
    missing_events = sorted(set(selected_events).difference(all_events))
    if missing_events:
        raise ValueError(f"Configured events are absent from the matrix: {missing_events}")
    matrix = matrix[selected_events].copy()

    mismatch = 0
    # The stored genotype signature may have been built from a wider screening
    # panel. It is comparable only when the exact full event panel is reused.
    if selected_events == all_events and "genotype_signature" in ordered:
        reconstructed = pd.Series(
            [genotype_signature(row, selected_events) for row in matrix.to_numpy()],
            index=ordered.index,
        ).map(canonical_genotype)
        observed = ordered["genotype_signature"].fillna("WT").astype(str).map(canonical_genotype)
        mismatch = int((reconstructed != observed).sum())
        if mismatch:
            raise ValueError(f"{mismatch} genotype signatures disagree with the binary matrix")
    return matrix, ordered, selected_events, mismatch


def run_cross_sectional_cohort(
    input_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    config: CrossSectionalConfig | None = None,
    theta: np.ndarray | None = None,
) -> dict[str, pd.DataFrame | dict]:
    """Fit cMHN (or accept supplied theta), score states, and return tables."""

    config = config or CrossSectionalConfig()
    matrix, state_table, events, genotype_mismatches = load_prepared_cohort(input_dir, config.events)
    fit_metadata: dict[str, object]
    cv_scores = pd.DataFrame()
    if theta is None:
        fit = fit_cmh(matrix, config.mhn)
        theta = fit.theta
        cv_scores = fit.cv_scores
        fit_metadata = {
            "backend": "official_mhn_cMHN",
            "selected_lambda": fit.selected_lambda,
            "event_count": len(events),
            "sample_count": len(matrix),
            "theta_shape": list(theta.shape),
            "finite_theta": bool(np.isfinite(theta).all()),
            "lambda_multipliers": list(config.mhn.lambda_multipliers),
            "cv_folds": config.mhn.cv_folds,
            "pick_1se": config.mhn.pick_1se,
            "max_iterations": config.mhn.max_iterations,
            "relative_tolerance": config.mhn.relative_tolerance,
            "random_seed": config.mhn.random_seed,
            "device": "CPU",
            "penalty": "L1",
        }
    else:
        theta = np.asarray(theta, dtype=float)
        fit_metadata = {
            "backend": "supplied_theta",
            "selected_lambda": None,
            "event_count": len(events),
            "sample_count": len(matrix),
        }
    if theta.shape != (len(events), len(events)) or not np.isfinite(theta).all():
        raise ValueError("theta must be a finite square matrix in event-column order")

    occupancy = build_state_occupancy(
        state_table,
        matrix,
        events,
        stage_column="stage_group",
        analysis_id_column="analysis_id",
    )
    bootstrap = None
    if config.bootstrap_replicates > 0:
        bootstrap = BootstrapConfig(
            replicates=config.bootstrap_replicates,
            top_k=config.bootstrap_top_k,
            random_seed=config.mhn.random_seed,
        )
    scores, edges, bootstrap_summary = score_states_from_mhn(
        occupancy,
        theta,
        events,
        thresholds=config.thresholds,
        bootstrap=bootstrap,
    )
    top_states = (
        scores[scores["high_confidence_relobstq"]]
        .sort_values(["R_star", "N_v"], ascending=[False, False])
        .head(config.top_state_count)
        .reset_index(drop=True)
    )
    top_states.insert(0, "rank", np.arange(1, len(top_states) + 1))
    theta_table = pd.DataFrame(theta, index=events, columns=events).rename_axis("target_event").reset_index()
    qc = audit_score_table(scores)
    qc["matrix_rows"] = len(matrix)
    qc["events"] = len(events)
    qc["genotype_alignment_mismatches"] = genotype_mismatches

    tables: dict[str, pd.DataFrame | dict] = {
        "state_occupancy": occupancy,
        "state_edges": edges,
        "state_scores": scores,
        "top_relative_dwell_states": top_states,
        "theta": theta_table,
        "cv_scores": cv_scores,
        "bootstrap_summary": bootstrap_summary if bootstrap_summary is not None else pd.DataFrame(),
        "quality_control": qc,
        "fit_metadata": fit_metadata,
    }
    if output_dir is not None:
        input_root = Path(input_dir)
        writer = ResultWriter(
            output_dir,
            input_files=[
                input_root / "mhn_training_matrix.csv",
                input_root / "mhn_row_index_map.csv",
                input_root / "state_table.csv",
            ],
            metadata={
                "workflow": "run_cross_sectional_cohort",
                "backend": fit_metadata["backend"],
                "random_seed": config.mhn.random_seed,
                "bootstrap_replicates": config.bootstrap_replicates,
            },
        )
        for name, value in tables.items():
            if isinstance(value, pd.DataFrame):
                writer.table(name, value)
        writer.json("fit_metadata", fit_metadata)
        writer.json("resolved_config", asdict(config))
        writer.manifest()
    return tables
