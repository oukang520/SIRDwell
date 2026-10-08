"""Generate result tables for falsification, information-gain, and topology analyses."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from relobstq_mhn.core.scoring import ScoreThresholds
from relobstq_mhn.core.transitions import probability_provider_from_theta
from relobstq_mhn.io import ResultWriter, load_yaml
from relobstq_mhn.workflows.controls import (
    denominator_ablation,
    inflow_pairing_falsification,
)
from relobstq_mhn.workflows.secondary import (
    inflow_computability_summary,
    information_gain_summary,
    rstar_landscape_summary,
)
from relobstq_mhn.workflows.topology import topology_route_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/secondary.yaml")
    args = parser.parse_args()
    raw = load_yaml(args.config)
    analysis = raw["analysis"]
    for cohort_index, cohort in enumerate(raw["cohorts"]):
        cross_root = Path(raw["cross_sectional_root"]) / cohort / "tables"
        prepared_root = Path(raw["prepared_data_root"]) / cohort
        source_files = {
            "scores": cross_root / "state_scores.tsv",
            "edges": cross_root / "state_edges.tsv",
            "occupancy": cross_root / "state_occupancy.tsv",
            "theta": cross_root / "theta.tsv",
            "matrix": prepared_root / "mhn_training_matrix.csv",
        }
        scores = pd.read_csv(source_files["scores"], sep="\t")
        edges = pd.read_csv(source_files["edges"], sep="\t")
        occupancy = pd.read_csv(source_files["occupancy"], sep="\t")
        theta_table = pd.read_csv(source_files["theta"], sep="\t")
        matrix = pd.read_csv(source_files["matrix"])
        events = theta_table["target_event"].astype(str).tolist()
        theta = theta_table[events].to_numpy(dtype=float)
        if theta.shape != (len(events), len(events)) or not np.isfinite(theta).all():
            raise ValueError(f"{cohort}: theta is not a finite event-by-event matrix")
        if matrix.columns.astype(str).tolist() != events:
            raise ValueError(f"{cohort}: theta event order does not match the prepared matrix")
        provider = probability_provider_from_theta(theta, events)
        frequencies = matrix.mean(axis=0).astype(float).to_dict()
        thresholds = ScoreThresholds(
            minimum_state_count=int(analysis["minimum_state_count"]),
            minimum_inflow=float(analysis["minimum_inflow"]),
        )
        writer = ResultWriter(
            Path(raw["output_root"]) / cohort,
            input_files=list(source_files.values()),
            metadata={
                "workflow": "run_secondary",
                "cohort": cohort,
                "evidence_units": ["E4", "E10", "E11", "E14", "E15B", "E16"],
                "random_seed": int(analysis["random_seed"]) + cohort_index,
            },
        )

        landscape, landscape_summary = rstar_landscape_summary(scores)
        ablation_details, ablation_summary = denominator_ablation(
            occupancy,
            events,
            provider,
            frequencies,
            thresholds=thresholds,
            top_k=int(analysis["top_k"]),
        )
        shuffled, shuffle_summary = inflow_pairing_falsification(
            scores,
            top_k=int(analysis["top_k"]),
            replicates=int(analysis["inflow_shuffle_replicates"]),
            seed=int(analysis["random_seed"]) + cohort_index,
        )
        evidence_contract = pd.DataFrame(
            [
                {"evidence_unit": "E4", "result": "inflow_computability_summary.tsv"},
                {"evidence_unit": "E10", "result": "rstar_landscape_states.tsv; rstar_landscape_summary.tsv"},
                {"evidence_unit": "E11", "result": "information_gain_summary.tsv"},
                {"evidence_unit": "E14", "result": "denominator_ablation_details.tsv; denominator_ablation_summary.tsv"},
                {"evidence_unit": "E15B", "result": "inflow_shuffle_replicates.tsv; inflow_shuffle_summary.tsv"},
                {"evidence_unit": "E16", "result": "topology_routes.tsv"},
            ]
        )
        writer.table("inflow_computability_summary", inflow_computability_summary(scores, edges))
        writer.table("rstar_landscape_states", landscape)
        writer.table("rstar_landscape_summary", landscape_summary)
        writer.table("denominator_ablation_details", ablation_details)
        writer.table("denominator_ablation_summary", ablation_summary)
        writer.table("inflow_shuffle_replicates", shuffled)
        writer.table("inflow_shuffle_summary", shuffle_summary)
        writer.table("information_gain_summary", information_gain_summary(scores, top_k=int(analysis["top_k"])))
        writer.table("topology_routes", topology_route_table(scores, target_count=int(analysis["topology_routes"])))
        writer.table("evidence_contract", evidence_contract)
        writer.manifest()
        print(f"completed secondary validations: {cohort}")


if __name__ == "__main__":
    main()
