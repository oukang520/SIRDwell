"""Oracle-backbone robustness across topology, sparsity, and dwell placement."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.pipeline import score_states_from_mhn
from ..core.scoring import ScoreThresholds
from ..evaluation.metrics import pairwise_concordance, safe_rank_correlation
from ..io.results import ResultWriter
from ..simulation.generator import SimulationConfig, create_sparse_theta, simulate_cohort_with_audit


@dataclass(frozen=True)
class TopologyRobustnessConfig:
    """Settings for the canonical E7 supplementary simulation contract."""

    event_count: int = 12
    topologies: tuple[str, ...] = ("linear", "branching", "mutual_exclusivity", "mixed")
    sparsities: tuple[float, ...] = (0.05, 0.10, 0.20)
    placements: tuple[str, ...] = ("early", "middle", "late")
    dwell_levels: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
    states_per_level: int = 2
    pilot_samples: int = 12000
    samples_per_repeat: int = 1000
    repeats: int = 12
    maximum_time: float = 10.0
    maximum_events: int = 7
    minimum_pilot_count: int = 8
    random_seed: int = 20260901
    thresholds: ScoreThresholds = field(
        default_factory=lambda: ScoreThresholds(minimum_state_count=5, minimum_inflow=1.0e-8)
    )


def _forced_edges(topology: str, event_count: int) -> dict[tuple[int, int], float]:
    templates: dict[str, dict[tuple[int, int], float]] = {
        "linear": {(1, 0): 1.35, (2, 1): 1.25, (3, 2): 1.15, (4, 3): 1.05},
        "branching": {(1, 0): 1.35, (2, 0): 1.35, (3, 1): 1.20, (4, 2): 1.20},
        "mutual_exclusivity": {
            (1, 0): 1.20,
            (2, 0): 1.20,
            (1, 2): -1.45,
            (2, 1): -1.45,
            (3, 1): 1.10,
            (4, 2): 1.10,
        },
        "mixed": {
            (1, 0): 1.30,
            (2, 0): 1.20,
            (3, 1): 1.15,
            (4, 2): 1.15,
            (5, 3): 1.05,
            (5, 4): 1.05,
            (1, 2): -1.10,
            (2, 1): -1.10,
        },
    }
    if topology not in templates:
        raise ValueError(f"Unsupported topology: {topology}")
    return {edge: value for edge, value in templates[topology].items() if max(edge) < event_count}


def _occupancy(snapshots: pd.DataFrame) -> pd.DataFrame:
    work = snapshots.copy()
    work["stage"] = "s1"
    work["state"] = "s1::" + work["genotype"].astype(str)
    table = (
        work.groupby(["state", "stage", "genotype", "event_count"], as_index=False)
        .size()
        .rename(columns={"size": "N_v"})
    )
    table["L_v"] = table["N_v"] / table["N_v"].sum()
    return table


def _placement_counts(placement: str) -> tuple[int, ...]:
    mapping = {"early": (1,), "middle": (2,), "late": (3, 4)}
    if placement not in mapping:
        raise ValueError(f"Unsupported dwell placement: {placement}")
    return mapping[placement]


def _select_truth(
    pilot: pd.DataFrame,
    placement: str,
    config: TopologyRobustnessConfig,
    *,
    seed: int,
) -> tuple[dict[int, float], pd.DataFrame]:
    counts = (
        pilot.groupby(["mask", "genotype", "event_count"], as_index=False)
        .size()
        .rename(columns={"size": "pilot_count"})
    )
    required = len(config.dwell_levels) * config.states_per_level
    candidates = counts[
        counts["event_count"].isin(_placement_counts(placement))
        & counts["pilot_count"].ge(config.minimum_pilot_count)
    ].nlargest(required, "pilot_count")
    if len(candidates) < required:
        raise RuntimeError(
            f"{placement}: need {required} supported states, found {len(candidates)}; "
            "increase pilot_samples or relax minimum_pilot_count"
        )
    truth = candidates.copy().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    levels = np.repeat(np.asarray(config.dwell_levels, dtype=float), config.states_per_level)
    rng.shuffle(levels)
    truth["D_true"] = levels
    truth["log2_D_true"] = np.log2(truth["D_true"])
    truth["placement"] = placement
    truth["state"] = "s1::" + truth["genotype"].astype(str)
    return dict(zip(truth["mask"].astype(int), truth["D_true"].astype(float))), truth


def run_topology_robustness(
    *,
    output_dir: str | Path | None = None,
    config: TopologyRobustnessConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Evaluate dwell-order recovery while varying an otherwise known backbone.

    This intentionally uses the generating theta for scoring. It isolates
    topology sensitivity of the R* construction and is not a cMHN refit test.
    """

    config = config or TopologyRobustnessConfig()
    events = [f"E{index + 1}" for index in range(config.event_count)]
    truth_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    condition_index = 0
    for topology in config.topologies:
        for sparsity in config.sparsities:
            condition_index += 1
            theta_seed = config.random_seed + condition_index * 1000
            theta = create_sparse_theta(
                events,
                sparsity=sparsity,
                seed=theta_seed,
                forced_edges=_forced_edges(topology, config.event_count),
            )
            _, pilot = simulate_cohort_with_audit(
                theta,
                events,
                config=SimulationConfig(
                    samples=config.pilot_samples,
                    maximum_time=config.maximum_time,
                    maximum_events=config.maximum_events,
                    random_seed=theta_seed,
                ),
            )
            for placement_index, placement in enumerate(config.placements):
                dwell, truth = _select_truth(
                    pilot,
                    placement,
                    config,
                    seed=theta_seed + placement_index + 1,
                )
                condition = f"{topology}|s={sparsity:.2f}|{placement}"
                truth.insert(0, "condition", condition)
                truth.insert(1, "topology", topology)
                truth.insert(2, "sparsity", sparsity)
                truth_rows.append(truth)
                for repeat in range(1, config.repeats + 1):
                    _, snapshots = simulate_cohort_with_audit(
                        theta,
                        events,
                        dwell,
                        config=SimulationConfig(
                            samples=config.samples_per_repeat,
                            maximum_time=config.maximum_time,
                            maximum_events=config.maximum_events,
                            random_seed=theta_seed + placement_index * 100 + repeat,
                        ),
                    )
                    scores, _, _ = score_states_from_mhn(
                        _occupancy(snapshots), theta, events, thresholds=config.thresholds
                    )
                    occupancy_normalizer = scores.loc[scores["eligible_relobstq"], "L_v"].median()
                    evaluated = scores.merge(
                        truth[["state", "D_true", "log2_D_true"]], on="state", how="inner"
                    )
                    evaluated = evaluated[evaluated["eligible_relobstq"]].copy()
                    evaluated["log2_R_star"] = np.log2(evaluated["R_star"].clip(lower=1.0e-12))
                    evaluated["log2_occupancy"] = np.log2(
                        (evaluated["L_v"] / occupancy_normalizer).clip(lower=1.0e-12)
                    )
                    rho_r, _ = safe_rank_correlation(evaluated["log2_D_true"], evaluated["log2_R_star"])
                    rho_l, _ = safe_rank_correlation(evaluated["log2_D_true"], evaluated["log2_occupancy"])
                    metric_rows.append(
                        {
                            "condition": condition,
                            "topology": topology,
                            "sparsity": sparsity,
                            "placement": placement,
                            "repeat": repeat,
                            "truth_states_evaluable": len(evaluated),
                            "truth_states_implanted": len(truth),
                            "evaluable_fraction": len(evaluated) / len(truth),
                            "spearman_R_star": rho_r,
                            "spearman_occupancy": rho_l,
                            "spearman_gain": rho_r - rho_l,
                            "pairwise_concordance_R_star": pairwise_concordance(
                                evaluated["log2_D_true"], evaluated["log2_R_star"]
                            ),
                            "pairwise_concordance_occupancy": pairwise_concordance(
                                evaluated["log2_D_true"], evaluated["log2_occupancy"]
                            ),
                        }
                    )
    metrics = pd.DataFrame(metric_rows)
    summary = (
        metrics.groupby(["topology", "sparsity", "placement"], as_index=False)
        .agg(
            repeats=("repeat", "count"),
            median_evaluable_states=("truth_states_evaluable", "median"),
            median_evaluable_fraction=("evaluable_fraction", "median"),
            minimum_evaluable_fraction=("evaluable_fraction", "min"),
            median_spearman_R_star=("spearman_R_star", "median"),
            q1_spearman_R_star=("spearman_R_star", lambda values: values.quantile(0.25)),
            q3_spearman_R_star=("spearman_R_star", lambda values: values.quantile(0.75)),
            median_spearman_occupancy=("spearman_occupancy", "median"),
            median_spearman_gain=("spearman_gain", "median"),
            median_concordance_R_star=("pairwise_concordance_R_star", "median"),
            median_concordance_occupancy=("pairwise_concordance_occupancy", "median"),
        )
    )
    contract = pd.DataFrame(
        [
            {
                "evidence_unit": "E7",
                "scope": "supplementary oracle-backbone topology robustness",
                "backbone_source": "generating theta",
                "includes_cMHN_refit_error": False,
                "varied_factors": "topology;sparsity;dwell_placement",
                "primary_metric": "spearman_R_star",
                "reference_metric": "spearman_occupancy",
            }
        ]
    )
    outputs = {
        "canonical_contract": contract,
        "truth_states": pd.concat(truth_rows, ignore_index=True),
        "condition_metrics": metrics,
        "condition_summary": summary,
    }
    if output_dir is not None:
        writer = ResultWriter(
            output_dir,
            metadata={
                "workflow": "run_topology_robustness",
                "evidence_unit": "E7",
                "scope": "supplementary_oracle_backbone",
            },
        )
        for name, frame in outputs.items():
            writer.table(name, frame)
        writer.json("resolved_config", asdict(config))
        writer.manifest()
    return outputs
