"""Synthetic checks for the selected E17 calculation and table-only boundary."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from relobstq_mhn.workflows import selected_longitudinal as selected


def _config() -> dict:
    return {
        "random_seed": 20260827,
        "analysis": {
            "max_events": 2,
            "min_events_for_mhn": 2,
            "min_event_frequency": 0.005,
            "min_event_support": 2,
            "min_state_count": 1,
            "minimum_inflow": 1.0e-8,
            "high_confidence_state_count": 2,
            "top_states_per_study": 8,
            "bootstrap_replicates": 20,
            "primary_validation_studies": ["coadread_mskcc"],
            "pair_qc": {"exclude_event_loss": True},
        },
        "mhn": {"enabled": False},
    }


def _paired_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Provide predecessors, persistence, gains, and losses in both stages."""
    patterns = [
        ((0, 0), (1, 0)),
        ((1, 0), (1, 0)),
        ((1, 0), (1, 1)),
        ((0, 1), (0, 1)),
        ((1, 1), (1, 1)),
        ((1, 1), (1, 0)),
    ]
    metadata_rows = []
    matrix_rows = []
    for repeat in range(6):
        for pattern_index, pattern in enumerate(patterns):
            patient_id = f"P{repeat}_{pattern_index}"
            for timepoint, values in enumerate(pattern):
                sample_id = f"{patient_id}-{timepoint + 1}"
                metadata_rows.append(
                    {
                        "patient_id": patient_id,
                        "sample_id": sample_id,
                        "stage": "baseline" if timepoint == 0 else "progressed",
                        "sample_role": "primary" if timepoint == 0 else "metastasis",
                        "time_rank": float(timepoint),
                        "order_evaluable": True,
                    }
                )
                matrix_rows.append({"sample_id": sample_id, "APC": values[0], "TP53": values[1]})
    return pd.DataFrame(metadata_rows), pd.DataFrame(matrix_rows)


def test_frequency_cooccurrence_backbone_has_frozen_values() -> None:
    values = np.array([[0, 0], [1, 0], [1, 1], [1, 1]])
    theta = selected.build_frequency_cooccurrence_theta(values)
    np.testing.assert_allclose(theta, [[np.log(3), np.log(1.25)], [np.log(1.25), 0]])
    matrix = pd.DataFrame(values, columns=["APC", "TP53"])
    fitted, metadata = selected.fit_or_build_theta(matrix, ["APC", "TP53"], _config(), 123)
    np.testing.assert_array_equal(fitted, theta)
    assert metadata["backend"] == "frequency_cooccurrence_backbone"


def test_pair_qc_retains_gains_and_persistence_and_excludes_losses() -> None:
    metadata, matrix = _paired_fixture()
    events = ["APC", "TP53"]
    timepoints = selected.aggregate_timepoint_events(metadata, matrix, events)
    pairs = selected.build_longitudinal_pair_table("coadread_mskcc", timepoints, events)
    retained, qc = selected.apply_pair_qc(pairs, _config())
    assert qc == {
        "pair_qc_raw_pairs": 36,
        "pair_qc_retained_pairs": 30,
        "pair_qc_excluded_event_loss_pairs": 6,
    }
    assert retained["empirical_persistent"].sum() == 18
    assert retained["n_gained"].sum() == 12
    assert retained["n_lost"].eq(0).all()
    assert retained.loc[retained["empirical_persistent"].eq(0), "minimum_observed_dwell_interval"].eq(0).all()
    assert retained.loc[retained["empirical_persistent"].eq(1), "minimum_observed_dwell_interval"].eq(1).all()


def test_patient_fold_scoring_keeps_selected_full_cohort_backbone(monkeypatch: pytest.MonkeyPatch) -> None:
    metadata, matrix = _paired_fixture()
    events = ["APC", "TP53"]
    theta = selected.build_frequency_cooccurrence_theta(matrix[events].to_numpy())
    timepoints = selected.aggregate_timepoint_events(metadata, matrix, events)
    all_patients = set(metadata["patient_id"])
    held_out_sets = []
    original = selected.score_from_training_patients

    def record_scoring(metadata_arg, matrix_arg, events_arg, theta_arg, train_patients, config_arg):
        assert theta_arg is theta
        held_out_sets.append(all_patients.difference(train_patients))
        return original(metadata_arg, matrix_arg, events_arg, theta_arg, train_patients, config_arg)

    monkeypatch.setattr(selected, "score_from_training_patients", record_scoring)
    predictions, summary = selected.evaluate_rstar_dwell_persistence(
        "coadread_mskcc", metadata, matrix, timepoints, theta, events, _config()
    )
    assert len(held_out_sets) == 5
    assert set.union(*held_out_sets) == all_patients
    assert sum(len(group) for group in held_out_sets) == len(all_patients)
    assert predictions.groupby("patient_id")["fold"].nunique().eq(1).all()
    assert not predictions.loc[predictions["n_lost"].gt(0), "pair_qc_pass"].any()
    assert summary.loc[0, "total_ordered_pairs"] == 36
    assert summary.loc[0, "evaluable_pairs"] == 24
    assert summary.loc[0, "persistent_pairs"] == 18
    assert summary.loc[0, "changed_pairs"] == 6
    assert summary.loc[0, "persistence_rate"] == 0.75
    assert summary.loc[0, "exact_state_score_fraction"] == 1.0


def test_selected_runner_writes_numerical_tables_without_figures_or_reports(tmp_path: Path) -> None:
    metadata, matrix = _paired_fixture()
    study_dir = tmp_path / "data" / "coadread_mskcc"
    study_dir.mkdir(parents=True)
    clinical = metadata.rename(columns={"sample_id": "SAMPLE_ID", "patient_id": "PATIENT_ID"})
    clinical["SAMPLE_TYPE"] = np.where(clinical["stage"].eq("baseline"), "Primary", "Metastasis")
    clinical[["SAMPLE_ID", "PATIENT_ID", "SAMPLE_TYPE"]].to_csv(
        study_dir / "data_clinical_sample.txt", sep="\t", index=False
    )
    mutations = matrix.melt(id_vars="sample_id", var_name="Hugo_Symbol", value_name="present")
    mutations = mutations[mutations["present"].eq(1)].rename(columns={"sample_id": "Tumor_Sample_Barcode"})
    mutations["Variant_Classification"] = "Missense_Mutation"
    mutations[["Hugo_Symbol", "Tumor_Sample_Barcode", "Variant_Classification"]].to_csv(
        study_dir / "data_mutations.txt", sep="\t", index=False
    )
    config = _config()
    config.update(
        {
            "data_root": str(tmp_path / "data"),
            "result_root": str(tmp_path / "result"),
            "studies": {"coadread_mskcc": {"short_name": "CRC-triplets", "display_name": "Synthetic CRC"}},
        }
    )
    output = selected.run_selected_longitudinal(config)
    assert output["dwell_persistence_summary_all"].loc[0, "evaluable_pairs"] == 24
    assert output["cohort_qc"].loc[0, "backend"] == "frequency_cooccurrence_backbone"
    result_root = Path(config["result_root"])
    assert not (result_root / "figures").exists()
    assert {path.suffix for path in result_root.rglob("*") if path.is_file()} == {".tsv", ".json"}
    stored = pd.read_csv(result_root / "tables" / "dwell_persistence_summary_all.tsv", sep="\t")
    pd.testing.assert_frame_equal(stored, output["dwell_persistence_summary_all"], check_dtype=False)
    fit_metadata = json.loads((result_root / "coadread_mskcc" / "tables" / "fit_metadata.json").read_text())
    assert fit_metadata["backend"] == "frequency_cooccurrence_backbone"
