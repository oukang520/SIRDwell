from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from relobstq_mhn.core.scoring import ScoreThresholds
from relobstq_mhn.core.transitions import probability_provider_from_theta
from relobstq_mhn.workflows.controls import denominator_ablation, inflow_pairing_falsification
from relobstq_mhn.workflows.cross_sectional import CrossSectionalConfig, run_cross_sectional_cohort
from relobstq_mhn.workflows.longitudinal import LongitudinalConfig, evaluate_longitudinal_pairs
from relobstq_mhn.workflows.longitudinal_preparation import (
    LongitudinalPreparationConfig,
    prepare_longitudinal_pairs,
)
from relobstq_mhn.workflows.simulation import DwellGradientConfig, run_dwell_gradient
from relobstq_mhn.workflows.secondary import (
    inflow_computability_summary,
    information_gain_summary,
    rstar_landscape_summary,
)
from relobstq_mhn.workflows.topology_robustness import TopologyRobustnessConfig, run_topology_robustness


def _toy_prepared(root: Path) -> None:
    root.mkdir(parents=True)
    matrix = pd.DataFrame(
        {
            "A": [0] * 20 + [1] * 20 + [0] * 10 + [1] * 10,
            "B": [0] * 20 + [0] * 20 + [1] * 10 + [1] * 10,
        }
    )
    ids = [f"S{index}" for index in range(len(matrix))]
    genotype = ["WT"] * 20 + ["A"] * 20 + ["B"] * 10 + ["A+B"] * 10
    pd.DataFrame({"analysis_id": ids}).to_csv(root / "mhn_row_index_map.csv", index=False)
    pd.DataFrame(
        {
            "analysis_id": ids,
            "stage_group": ["primary"] * len(ids),
            "genotype_signature": genotype,
        }
    ).to_csv(root / "state_table.csv", index=False)
    matrix.to_csv(root / "mhn_training_matrix.csv", index=False)


def test_cross_sectional_with_supplied_theta(tmp_path: Path) -> None:
    source = tmp_path / "input"
    _toy_prepared(source)
    result = run_cross_sectional_cohort(
        source,
        theta=np.zeros((2, 2)),
        config=CrossSectionalConfig(
            thresholds=ScoreThresholds(minimum_state_count=1, minimum_inflow=1.0e-12, high_confidence_state_count=1),
            bootstrap_replicates=5,
        ),
    )
    assert not result["state_scores"].empty
    assert result["quality_control"]["genotype_alignment_mismatches"].iloc[0] == 0
    computability = inflow_computability_summary(result["state_scores"], result["state_edges"])
    landscape, summary = rstar_landscape_summary(result["state_scores"])
    details, ablation = denominator_ablation(
        result["state_occupancy"],
        ["A", "B"],
        probability_provider_from_theta(np.zeros((2, 2)), ["A", "B"]),
        {"A": 0.5, "B": 1.0 / 3.0},
        thresholds=ScoreThresholds(minimum_state_count=1, minimum_inflow=1.0e-12),
        top_k=2,
    )
    assert computability.loc[0, "finite_eligible_fraction"] == 1.0
    assert len(landscape) == summary.loc[0, "eligible_states"]
    assert set(details["variant"]) == {"full_mhn", "uniform_inflow", "frequency_inflow", "occupancy_only"}
    assert set(ablation["variant"]) == set(details["variant"])


def test_information_gain_excludes_ineligible_states() -> None:
    scores = pd.DataFrame(
        {
            "state": ["s1::A", "s1::B", "s1::C", "s1::D"],
            "L_v": [0.1, 0.2, 0.3, 0.9],
            "F_hat": [0.4, 0.3, 0.2, 0.001],
            "R_star": [0.5, 1.0, 2.0, 900.0],
            "eligible_relobstq": [True, True, True, False],
        }
    )
    summary = information_gain_summary(scores, top_k=10)
    assert summary.loc[0, "states"] == 3
    assert summary.loc[0, "top_k"] == 3


def test_continuous_gradient_smoke() -> None:
    result = run_dwell_gradient(
        config=DwellGradientConfig(
            event_count=4,
            pilot_samples=800,
            samples_per_repeat=250,
            repeats=2,
            states_per_level=1,
            maximum_events=4,
            minimum_pilot_count=1,
            thresholds=ScoreThresholds(minimum_state_count=1, minimum_inflow=1.0e-12),
        )
    )
    assert len(result["repeat_metrics"]) == 2
    assert set(result["truth_states"]["D_true"]) == {0.25, 0.5, 1.0, 2.0, 4.0}
    assert set(result["evaluation_coverage"]["D_true"]) == {0.25, 0.5, 1.0, 2.0, 4.0}
    assert result["evaluation_coverage"]["evaluation_coverage_fraction"].between(0, 1).all()


def test_topology_robustness_contract_smoke() -> None:
    result = run_topology_robustness(
        config=TopologyRobustnessConfig(
            event_count=6,
            topologies=("branching",),
            sparsities=(0.10,),
            placements=("middle",),
            states_per_level=1,
            pilot_samples=600,
            samples_per_repeat=250,
            repeats=1,
            maximum_events=5,
            minimum_pilot_count=1,
            thresholds=ScoreThresholds(minimum_state_count=1, minimum_inflow=1.0e-12),
        )
    )
    assert result["canonical_contract"].loc[0, "includes_cMHN_refit_error"] == False  # noqa: E712
    assert len(result["condition_metrics"]) == 1
    assert result["condition_metrics"].loc[0, "evaluable_fraction"] <= 1.0


def test_longitudinal_and_controls() -> None:
    pairs = pd.DataFrame(
        {
            "study_id": ["toy"] * 8,
            "predicted_log2_R": [-2, -1, -0.5, 0, 0.5, 1, 1.5, 2],
            "empirical_persistent": [0, 0, 0, 0, 1, 1, 1, 1],
            "minimum_observed_dwell_interval": [1, 1, 2, 2, 3, 4, 5, 6],
        }
    )
    metrics = evaluate_longitudinal_pairs(
        pairs, config=LongitudinalConfig(bootstrap_replicates=20)
    )["longitudinal_metrics"]
    assert metrics.loc[0, "auc"] == 1.0
    scores = pd.DataFrame(
        {
            "state": [f"primary::S{index}" for index in range(20)],
            "stage": ["primary"] * 20,
            "event_count": [1] * 10 + [2] * 10,
            "N_v": np.arange(5, 25),
            "L_v": np.linspace(0.01, 0.20, 20),
            "F_hat": np.linspace(0.20, 0.01, 20),
            "R_star": np.linspace(0.2, 4.0, 20),
            "eligible_relobstq": [True] * 20,
        }
    )
    replicates, _ = inflow_pairing_falsification(scores, top_k=3, replicates=5)
    assert len(replicates) == 5


def test_longitudinal_crossfit_with_supplied_theta() -> None:
    rows = []
    matrix_rows = []
    for patient_index in range(4):
        genotype = "A" if patient_index < 2 else "B"
        for time, state in enumerate(["WT", genotype, genotype]):
            analysis_id = f"P{patient_index}_T{time}"
            rows.append(
                {
                    "analysis_id": analysis_id,
                    "patient_id": f"P{patient_index}",
                    "collection_time": time,
                    "stage_group": "primary",
                }
            )
            matrix_rows.append(
                {
                    "analysis_id": analysis_id,
                    "A": int(state == "A"),
                    "B": int(state == "B"),
                }
            )
    result = prepare_longitudinal_pairs(
        pd.DataFrame(rows),
        pd.DataFrame(matrix_rows),
        study_id="toy",
        config=LongitudinalPreparationConfig(
            folds=2,
            thresholds=ScoreThresholds(minimum_state_count=1, minimum_inflow=1.0e-12),
        ),
        theta_by_fold={0: np.zeros((2, 2)), 1: np.zeros((2, 2))},
    )
    assert len(result["crossfit_audit"]) == 2
    assert result["pair_predictions"]["predicted_log2_R"].notna().any()
    assert set(result["pair_predictions"]["split_id"]) == {0, 1}
