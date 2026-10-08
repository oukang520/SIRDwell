"""Controlled continuous-dwell simulation used as the primary positive control."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from ..core.pipeline import score_states_from_mhn
from ..core.scoring import ScoreThresholds
from ..evaluation.metrics import pairwise_concordance, safe_rank_correlation
from ..io.results import ResultWriter
from ..simulation.generator import SimulationConfig, create_sparse_theta, simulate_cohort_with_audit


@dataclass(frozen=True)
class DwellGradientConfig:
    """Settings for recovery of a graded relative-dwell signal."""

    event_count: int = 15
    pilot_samples: int = 30000
    samples_per_repeat: int = 2000
    repeats: int = 60
    dwell_levels: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
    states_per_level: int = 5
    maximum_time: float = 10.0
    maximum_events: int = 8
    theta_sparsity: float = 0.10
    minimum_pilot_count: int = 25
    event_count_1_states: int = 8
    event_count_2_states: int = 12
    event_count_3_4_states: int = 5
    event_count_1_minimum_count: int = 80
    event_count_2_minimum_count: int = 45
    random_seed: int = 20260901
    thresholds: ScoreThresholds = field(
        default_factory=lambda: ScoreThresholds(minimum_state_count=5, minimum_inflow=1.0e-8)
    )


def _occupancy(snapshots: pd.DataFrame) -> pd.DataFrame:
    work = snapshots.copy()
    work["stage"] = work["state"].str.split("::", n=1).str[0].str.lower()
    work["state"] = work["stage"] + "::" + work["genotype"].astype(str)
    table = (
        work.groupby(["state", "stage", "genotype", "event_count"], as_index=False)
        .size()
        .rename(columns={"size": "N_v"})
    )
    table["L_v"] = table["N_v"] / table["N_v"].sum()
    return table[["state", "stage", "genotype", "event_count", "N_v", "L_v"]]


def _select_truth(pilot: pd.DataFrame, config: DwellGradientConfig) -> tuple[dict[int, float], pd.DataFrame]:
    counts = (
        pilot.groupby(["mask", "state", "genotype", "event_count"], as_index=False)
        .size()
        .rename(columns={"size": "pilot_count"})
    )
    required = len(config.dwell_levels) * config.states_per_level
    selected = pd.concat(
        [
            counts[counts["event_count"].eq(1) & counts["pilot_count"].ge(config.event_count_1_minimum_count)]
            .nlargest(config.event_count_1_states, "pilot_count"),
            counts[counts["event_count"].eq(2) & counts["pilot_count"].ge(config.event_count_2_minimum_count)]
            .nlargest(config.event_count_2_states, "pilot_count"),
            counts[counts["event_count"].between(3, 4) & counts["pilot_count"].ge(config.minimum_pilot_count)]
            .nlargest(config.event_count_3_4_states, "pilot_count"),
        ],
        ignore_index=True,
    ).drop_duplicates("mask")
    if len(selected) < required:
        fallback = counts[
            counts["event_count"].between(1, min(4, config.maximum_events))
            & counts["pilot_count"].ge(config.minimum_pilot_count)
            & ~counts["mask"].isin(selected["mask"])
        ].nlargest(required - len(selected), "pilot_count")
        selected = pd.concat([selected, fallback], ignore_index=True)
    truth = selected.nlargest(required, "pilot_count").reset_index(drop=True)
    if len(truth) < required:
        raise RuntimeError(f"Need {required} supported truth states, found {len(truth)}")
    truth["state"] = truth["state"].str.split("::", n=1).str[0].str.lower() + "::" + truth["genotype"].astype(str)
    rng = np.random.default_rng(config.random_seed)
    assigned = []
    for _ in range(config.states_per_level):
        block = list(config.dwell_levels)
        rng.shuffle(block)
        assigned.extend(block)
    truth["D_true"] = assigned[: len(truth)]
    truth["log2_D_true"] = np.log2(truth["D_true"])
    truth["truth_level"] = truth["D_true"].map(lambda value: f"D={value:g}")
    dwell_by_mask = dict(zip(truth["mask"].astype(int), truth["D_true"].astype(float)))
    return dwell_by_mask, truth


def _adjacent_ordered_levels(frame: pd.DataFrame, value_column: str) -> tuple[int, int]:
    medians = frame.groupby("D_true", observed=True)[value_column].median().sort_index()
    if len(medians) < 2:
        return 0, 0
    differences = np.diff(medians.to_numpy(dtype=float))
    return int((differences > 0).sum()), len(differences)


def run_dwell_gradient(
    *,
    output_dir: str | Path | None = None,
    config: DwellGradientConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Recover five ordered dwell levels and return repeat-level evidence."""

    config = config or DwellGradientConfig()
    events = [f"E{index + 1}" for index in range(config.event_count)]
    scaffold_values = {
        (2, 0): 1.25,
        (2, 1): 0.85,
        (5, 2): 1.35,
        (3, 0): 1.00,
        (4, 1): 1.10,
        (6, 2): 0.90,
        (7, 3): 1.15,
        (8, 4): 1.05,
        (9, 5): 1.20,
        (10, 6): 0.80,
        (11, 7): 1.10,
        (12, 8): 0.95,
        (1, 0): -1.20,
        (0, 1): -1.20,
        (4, 3): -0.85,
        (3, 4): -0.85,
    }
    forced_edges = {
        edge: value for edge, value in scaffold_values.items() if max(edge) < config.event_count
    }
    theta = create_sparse_theta(
        events,
        sparsity=config.theta_sparsity,
        seed=config.random_seed,
        forced_edges=forced_edges,
    )
    _, pilot = simulate_cohort_with_audit(
        theta,
        events,
        config=SimulationConfig(
            samples=config.pilot_samples,
            maximum_time=config.maximum_time,
            maximum_events=config.maximum_events,
            random_seed=config.random_seed,
        ),
    )
    dwell_by_mask, truth = _select_truth(pilot, config)
    score_rows = []
    metric_rows = []
    for repeat in range(1, config.repeats + 1):
        _, snapshots = simulate_cohort_with_audit(
            theta,
            events,
            dwell_by_mask,
            config=SimulationConfig(
                samples=config.samples_per_repeat,
                maximum_time=config.maximum_time,
                maximum_events=config.maximum_events,
                random_seed=config.random_seed + repeat,
            ),
        )
        scores, _, _ = score_states_from_mhn(_occupancy(snapshots), theta, events, thresholds=config.thresholds)
        occupancy_normalizer = scores.loc[scores["eligible_relobstq"], "L_v"].median()
        scored_truth = scores.merge(
            truth[["state", "mask", "D_true", "log2_D_true", "truth_level", "pilot_count"]],
            on="state",
            how="inner",
        )
        scored_truth = scored_truth[scored_truth["eligible_relobstq"]].copy()
        scored_truth.insert(0, "repeat", repeat)
        scored_truth["log2_R_star"] = np.log2(scored_truth["R_star"].clip(lower=1.0e-12))
        occupancy_reference = scored_truth["L_v"] / occupancy_normalizer
        scored_truth["log2_occupancy_reference"] = np.log2(occupancy_reference.clip(lower=1.0e-12))
        rho_r, p_r = safe_rank_correlation(scored_truth["log2_D_true"], scored_truth["log2_R_star"])
        rho_l, p_l = safe_rank_correlation(
            scored_truth["log2_D_true"], scored_truth["log2_occupancy_reference"]
        )
        tau_r, tau_r_p = safe_rank_correlation(
            scored_truth["log2_D_true"], scored_truth["log2_R_star"], method="kendall"
        )
        tau_l, tau_l_p = safe_rank_correlation(
            scored_truth["log2_D_true"], scored_truth["log2_occupancy_reference"], method="kendall"
        )
        ordered_r, ordered_total = _adjacent_ordered_levels(scored_truth, "log2_R_star")
        ordered_l, _ = _adjacent_ordered_levels(scored_truth, "log2_occupancy_reference")
        if len(scored_truth) >= 3 and scored_truth["log2_D_true"].nunique() > 1:
            slope_r = float(np.polyfit(scored_truth["log2_D_true"], scored_truth["log2_R_star"], 1)[0])
            slope_l = float(
                np.polyfit(scored_truth["log2_D_true"], scored_truth["log2_occupancy_reference"], 1)[0]
            )
        else:
            slope_r = slope_l = np.nan
        metric_rows.append(
            {
                "repeat": repeat,
                "truth_states_evaluable": len(scored_truth),
                "spearman_R_star": rho_r,
                "spearman_R_star_p": p_r,
                "spearman_occupancy": rho_l,
                "spearman_occupancy_p": p_l,
                "spearman_gain": rho_r - rho_l,
                "kendall_R_star": tau_r,
                "kendall_R_star_p": tau_r_p,
                "kendall_occupancy": tau_l,
                "kendall_occupancy_p": tau_l_p,
                "kendall_gain": tau_r - tau_l,
                "pairwise_concordance_R_star": pairwise_concordance(
                    scored_truth["log2_D_true"], scored_truth["log2_R_star"]
                ),
                "pairwise_concordance_occupancy": pairwise_concordance(
                    scored_truth["log2_D_true"], scored_truth["log2_occupancy_reference"]
                ),
                "median_abs_error_log2_R_star": float(
                    (scored_truth["log2_R_star"] - scored_truth["log2_D_true"]).abs().median()
                ),
                "median_abs_error_log2_occupancy": float(
                    (scored_truth["log2_occupancy_reference"] - scored_truth["log2_D_true"]).abs().median()
                ),
                "absolute_error_gain": float(
                    (scored_truth["log2_occupancy_reference"] - scored_truth["log2_D_true"]).abs().median()
                    - (scored_truth["log2_R_star"] - scored_truth["log2_D_true"]).abs().median()
                ),
                "calibration_slope_R_star": slope_r,
                "calibration_slope_occupancy": slope_l,
                "calibration_slope_gain": slope_r - slope_l,
                "adjacent_ordered_levels_R_star": ordered_r,
                "adjacent_ordered_levels_occupancy": ordered_l,
                "adjacent_level_comparisons": ordered_total,
            }
        )
        score_rows.append(scored_truth)

    repeat_metrics = pd.DataFrame(metric_rows)
    state_scores = pd.concat(score_rows, ignore_index=True)
    summary_rows = []
    for metric in [column for column in repeat_metrics if column != "repeat"]:
        values = pd.to_numeric(repeat_metrics[metric], errors="coerce").dropna()
        summary_rows.append(
            {
                "metric": metric,
                "n": len(values),
                "median": float(values.median()) if len(values) else np.nan,
                "q1": float(values.quantile(0.25)) if len(values) else np.nan,
                "q3": float(values.quantile(0.75)) if len(values) else np.nan,
            }
        )
    summary = pd.DataFrame(summary_rows)
    paired_rows = []
    comparisons = [
        ("spearman", "spearman_R_star", "spearman_occupancy", "higher"),
        ("kendall", "kendall_R_star", "kendall_occupancy", "higher"),
        ("pairwise_concordance", "pairwise_concordance_R_star", "pairwise_concordance_occupancy", "higher"),
        ("calibration_slope", "calibration_slope_R_star", "calibration_slope_occupancy", "higher"),
        ("median_abs_error", "median_abs_error_log2_R_star", "median_abs_error_log2_occupancy", "lower"),
        ("adjacent_ordered_levels", "adjacent_ordered_levels_R_star", "adjacent_ordered_levels_occupancy", "higher"),
    ]
    for name, r_column, occupancy_column, direction in comparisons:
        paired = repeat_metrics[[r_column, occupancy_column]].dropna()
        if len(paired) and not np.allclose(paired[r_column], paired[occupancy_column]):
            alternative = "greater" if direction == "higher" else "less"
            pvalue = float(wilcoxon(paired[r_column], paired[occupancy_column], alternative=alternative).pvalue)
        else:
            pvalue = np.nan
        paired_rows.append(
            {
                "metric": name,
                "favorable_direction": direction,
                "paired_repeats": len(paired),
                "R_star_median": paired[r_column].median() if len(paired) else np.nan,
                "occupancy_median": paired[occupancy_column].median() if len(paired) else np.nan,
                "paired_wilcoxon_p": pvalue,
            }
        )
    comparison_summary = pd.DataFrame(paired_rows)
    level_summary = (
        state_scores.groupby(["D_true", "truth_level"], as_index=False)
        .agg(
            states=("state", "count"),
            median_log2_R_star=("log2_R_star", "median"),
            q1_log2_R_star=("log2_R_star", lambda values: values.quantile(0.25)),
            q3_log2_R_star=("log2_R_star", lambda values: values.quantile(0.75)),
            median_log2_occupancy=("log2_occupancy_reference", "median"),
        )
        .sort_values("D_true")
    )
    possible_per_level = config.repeats * config.states_per_level
    observed_coverage = (
        state_scores.groupby(["D_true", "truth_level"], as_index=False)
        .agg(
            evaluated_state_repeats=("state", "size"),
            repeats_with_evaluable_state=("repeat", "nunique"),
            distinct_truth_states_evaluated=("state", "nunique"),
        )
    )
    coverage_summary = pd.DataFrame({"D_true": config.dwell_levels})
    coverage_summary["truth_level"] = coverage_summary["D_true"].map(lambda value: f"D={value:g}")
    coverage_summary = coverage_summary.merge(
        observed_coverage, on=["D_true", "truth_level"], how="left", validate="one_to_one"
    ).sort_values("D_true")
    count_columns = [
        "evaluated_state_repeats",
        "repeats_with_evaluable_state",
        "distinct_truth_states_evaluated",
    ]
    coverage_summary[count_columns] = coverage_summary[count_columns].fillna(0).astype(int)
    coverage_summary["possible_state_repeats"] = possible_per_level
    coverage_summary["evaluation_coverage_fraction"] = (
        coverage_summary["evaluated_state_repeats"] / possible_per_level
    )
    outputs = {
        "truth_states": truth,
        "repeat_state_scores": state_scores,
        "repeat_metrics": repeat_metrics,
        "performance_summary": summary,
        "paired_comparison_summary": comparison_summary,
        "level_summary": level_summary,
        "evaluation_coverage": coverage_summary,
    }
    if output_dir is not None:
        writer = ResultWriter(output_dir)
        for name, frame in outputs.items():
            writer.table(name, frame)
        writer.json("resolved_config", asdict(config))
        writer.manifest()
    return outputs
