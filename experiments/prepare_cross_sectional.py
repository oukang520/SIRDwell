"""Prepare exact p15 cMHN inputs from provider-authorized cohort tables."""

from __future__ import annotations

import argparse
from pathlib import Path

from relobstq_mhn.io import load_yaml
from relobstq_mhn.workflows import CrossSectionalPreparationConfig, prepare_cross_sectional_cohort


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cross_sectional_preparation.yaml")
    parser.add_argument("--cohort", action="append", help="Prepare only this cohort; repeatable.")
    args = parser.parse_args()
    raw = load_yaml(args.config)
    cohorts = args.cohort or list(raw["cohorts"])
    for cohort in cohorts:
        source = Path(raw["source_root"]) / cohort
        config = CrossSectionalPreparationConfig(
            events=tuple(raw["cohorts"][cohort]["events"]),
            minimum_state_count=int(raw["minimum_state_count"]),
            selection_rule=str(raw["selection_rule"]),
            dataset_version=str(raw["dataset_version"]),
        )
        prepare_cross_sectional_cohort(
            source / "analysis_metadata.csv",
            source / "mutations_long.csv",
            output_dir=Path(raw["output_root"]) / cohort,
            config=config,
        )
        print(f"completed cross-sectional preparation: {cohort}")


if __name__ == "__main__":
    main()
