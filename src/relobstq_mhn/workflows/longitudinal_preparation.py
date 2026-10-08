"""Patient-level cross-fitting for leakage-controlled longitudinal predictions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..core.mhn import MhnFitConfig, fit_cmh
from ..core.pipeline import score_states_from_mhn
from ..core.scoring import ScoreThresholds
from ..core.states import build_state_occupancy, canonical_genotype, genotype_events, genotype_signature
from ..core.validation import assert_binary_matrix, require_columns
from ..io.results import ResultWriter


@dataclass(frozen=True)
class LongitudinalPreparationConfig:
    """Configuration for patient-grouped out-of-fold R* prediction."""

    folds: int = 5
    random_seed: int = 20260827
    exclude_event_loss_pairs: bool = True
    thresholds: ScoreThresholds = field(
        default_factory=lambda: ScoreThresholds(
            minimum_state_count=2,
            minimum_inflow=1.0e-8,
            high_confidence_state_count=4,
        )
    )
    mhn: MhnFitConfig = field(
        default_factory=lambda: MhnFitConfig(
            lambda_multipliers=(1.0,),
            cv_folds=3,
            pick_1se=False,
            max_iterations=1500,
            relative_tolerance=1.0e-6,
            random_seed=20260827,
        )
    )


def _fold_assignment(patient_ids: pd.Series, folds: int, seed: int) -> dict[str, int]:
    patients = np.array(sorted(patient_ids.astype(str).unique()))
    if len(patients) < 2:
        raise ValueError("Longitudinal cross-fitting requires at least two patients")
    folds = max(2, min(int(folds), len(patients)))
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    return {patient: index % folds for index, patient in enumerate(patients)}


def prepare_longitudinal_pairs(
    sample_metadata: pd.DataFrame,
    event_matrix: pd.DataFrame,
    *,
    study_id: str,
    output_dir: str | Path | None = None,
    config: LongitudinalPreparationConfig | None = None,
    theta_by_fold: dict[int, np.ndarray] | None = None,
    input_files: Sequence[str | Path] = (),
) -> dict[str, pd.DataFrame]:
    """Create out-of-fold sample scores and adjacent longitudinal pairs.

    Required metadata columns are ``analysis_id``, ``patient_id``,
    ``collection_time`` and ``stage_group``. The matrix must contain one
    ``analysis_id`` column followed by binary events. Every patient's scores are
    learned exclusively from patients assigned to other folds.
    """

    config = config or LongitudinalPreparationConfig()
    require_columns(
        sample_metadata,
        ["analysis_id", "patient_id", "collection_time", "stage_group"],
        "sample_metadata",
    )
    require_columns(event_matrix, ["analysis_id"], "event_matrix")
    if sample_metadata["analysis_id"].duplicated().any() or event_matrix["analysis_id"].duplicated().any():
        raise ValueError("analysis_id must be unique")
    merged = sample_metadata.merge(event_matrix, on="analysis_id", how="inner", validate="one_to_one")
    events = [column for column in event_matrix.columns if column != "analysis_id"]
    assert_binary_matrix(merged[events], "event_matrix")
    merged["collection_time"] = pd.to_numeric(merged["collection_time"], errors="coerce")
    merged = merged.dropna(subset=["collection_time"]).copy()
    fold_lookup = _fold_assignment(merged["patient_id"], config.folds, config.random_seed)
    merged["fold"] = merged["patient_id"].astype(str).map(fold_lookup)
    merged["genotype"] = [genotype_signature(row, events) for row in merged[events].to_numpy()]
    merged["stage"] = merged["stage_group"].astype(str).str.lower()
    merged["state"] = merged["stage"] + "::" + merged["genotype"]

    predictions = []
    fold_audit = []
    for fold in sorted(merged["fold"].unique()):
        train = merged[merged["fold"].ne(fold)].copy()
        test = merged[merged["fold"].eq(fold)].copy()
        train_matrix = train[events].reset_index(drop=True)
        theta = None if theta_by_fold is None else theta_by_fold.get(int(fold))
        selected_lambda = np.nan
        backend = "supplied_theta"
        if theta is None:
            fit_config = MhnFitConfig(**{**asdict(config.mhn), "random_seed": config.random_seed + int(fold)})
            fit = fit_cmh(train_matrix, fit_config)
            theta = fit.theta
            selected_lambda = fit.selected_lambda
            backend = "official_mhn_cMHN"
        occupancy = build_state_occupancy(
            train,
            train_matrix,
            events,
            stage_column="stage",
            analysis_id_column="analysis_id",
        )
        scores, _, _ = score_states_from_mhn(occupancy, theta, events, thresholds=config.thresholds)
        usable_scores = scores[scores["eligible_relobstq"]].copy()
        exact = usable_scores.set_index("state")["log2_R_star"].to_dict()
        genotype_fallback = usable_scores.groupby("genotype")["log2_R_star"].median().to_dict()
        test["predicted_log2_R"] = test["state"].map(exact)
        test["score_source"] = np.where(test["predicted_log2_R"].notna(), "exact_state", "not_evaluable")
        fallback = test["genotype"].map(genotype_fallback)
        use_fallback = test["predicted_log2_R"].isna() & fallback.notna()
        test.loc[use_fallback, "predicted_log2_R"] = fallback[use_fallback]
        # This fallback intentionally pools the same genotype across stages;
        # the former label incorrectly implied stage-specific matching.
        test.loc[use_fallback, "score_source"] = "genotype_median_across_stages"
        predictions.append(test)
        fold_audit.append(
            {
                "study_id": study_id,
                "fold": int(fold),
                "training_patients": train["patient_id"].nunique(),
                "heldout_patients": test["patient_id"].nunique(),
                "training_samples": len(train),
                "heldout_samples": len(test),
                "scored_samples": int(test["predicted_log2_R"].notna().sum()),
                "fit_backend": backend,
                "selected_lambda": selected_lambda,
                "selected_lambda_multiplier": (
                    float(selected_lambda) * len(train) if np.isfinite(selected_lambda) else np.nan
                ),
                "lambda_candidates": "+".join(str(value) for value in config.mhn.lambda_multipliers),
                "lambda_at_grid_boundary": (
                    bool(
                        np.isclose(float(selected_lambda) * len(train), min(config.mhn.lambda_multipliers))
                        or np.isclose(float(selected_lambda) * len(train), max(config.mhn.lambda_multipliers))
                    )
                    if np.isfinite(selected_lambda)
                    else pd.NA
                ),
                "cv_folds": config.mhn.cv_folds,
                "pick_1se": config.mhn.pick_1se,
                "max_iterations": config.mhn.max_iterations,
                "relative_tolerance": config.mhn.relative_tolerance,
                "fit_seed": config.random_seed + int(fold),
                "event_count": len(events),
                "theta_shape": f"{len(events)}x{len(events)}",
                "finite_theta": bool(np.isfinite(theta).all()),
            }
        )
    sample_predictions = pd.concat(predictions, ignore_index=True)

    pair_rows = []
    for patient_id, group in sample_predictions.groupby("patient_id", sort=True):
        ordered = group.sort_values(["collection_time", "analysis_id"]).reset_index(drop=True)
        for pair_index in range(len(ordered) - 1):
            earlier = ordered.iloc[pair_index]
            later = ordered.iloc[pair_index + 1]
            interval = float(later["collection_time"] - earlier["collection_time"])
            if interval <= 0:
                continue
            earlier_events = set(genotype_events(earlier["genotype"]))
            later_events = set(genotype_events(later["genotype"]))
            lost_events = sorted(earlier_events.difference(later_events))
            if config.exclude_event_loss_pairs and lost_events:
                continue
            persistent = int(earlier["genotype"] == later["genotype"])
            pair_rows.append(
                {
                    "study_id": study_id,
                    "patient_id": patient_id,
                    "earlier_analysis_id": earlier["analysis_id"],
                    "later_analysis_id": later["analysis_id"],
                    "earlier_state": earlier["state"],
                    "later_state": later["state"],
                    "predicted_log2_R": earlier["predicted_log2_R"],
                    "score_source": earlier["score_source"],
                    "split_id": int(earlier["fold"]),
                    "empirical_persistent": persistent,
                    "collection_interval": interval,
                    "minimum_observed_dwell_interval": interval if persistent else 0.0,
                    "lost_event_count": len(lost_events),
                    "lost_events": "+".join(lost_events),
                }
            )
    pairs = pd.DataFrame(pair_rows)
    outputs = {
        "sample_predictions": sample_predictions,
        "pair_predictions": pairs,
        "crossfit_audit": pd.DataFrame(fold_audit),
    }
    if output_dir is not None:
        writer = ResultWriter(
            output_dir,
            input_files=list(input_files),
            metadata={
                "workflow": "prepare_longitudinal_pairs",
                "study_id": study_id,
                "fit_backend": "official_mhn_cMHN" if theta_by_fold is None else "supplied_theta",
                "random_seed": config.random_seed,
                "leakage_control": "patient_grouped_out_of_fold",
            },
        )
        for name, frame in outputs.items():
            writer.table(name, frame)
        writer.json("resolved_config", asdict(config))
        writer.manifest()
    return outputs
