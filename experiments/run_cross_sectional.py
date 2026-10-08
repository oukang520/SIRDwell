"""Fit and score the three independent real cross-sectional cohorts."""

from __future__ import annotations

import argparse
from pathlib import Path

from relobstq_mhn.core.mhn import MhnFitConfig
from relobstq_mhn.core.scoring import ScoreThresholds
from relobstq_mhn.io import load_yaml
from relobstq_mhn.workflows.cross_sectional import CrossSectionalConfig, run_cross_sectional_cohort


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cross_sectional.yaml")
    parser.add_argument("--cohort", action="append", help="Run only the named cohort; repeatable.")
    args = parser.parse_args()
    raw = load_yaml(args.config)
    score = ScoreThresholds(**raw["score"])
    mhn = MhnFitConfig(
        lambda_multipliers=tuple(raw["mhn"]["lambda_multipliers"]),
        **{key: value for key, value in raw["mhn"].items() if key != "lambda_multipliers"},
    )
    analysis = raw["analysis"]
    cohorts = args.cohort or raw["cohorts"]
    for cohort in cohorts:
        config = CrossSectionalConfig(
            thresholds=score,
            mhn=mhn,
            bootstrap_replicates=int(analysis["bootstrap_replicates"]),
            bootstrap_top_k=int(analysis["bootstrap_top_k"]),
            top_state_count=int(analysis["top_state_count"]),
            events=tuple(raw["event_panels"][cohort]),
        )
        run_cross_sectional_cohort(
            Path(raw["data_root"]) / cohort,
            output_dir=Path(raw["output_root"]) / cohort,
            config=config,
        )
        print(f"completed cross-sectional workflow: {cohort}")


if __name__ == "__main__":
    main()
