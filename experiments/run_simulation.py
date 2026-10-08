"""Run the graded relative-dwell positive-control simulation."""

from __future__ import annotations

import argparse

from relobstq_mhn.core.scoring import ScoreThresholds
from relobstq_mhn.io import load_yaml
from relobstq_mhn.workflows.simulation import DwellGradientConfig, run_dwell_gradient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/simulation.yaml")
    parser.add_argument("--repeats", type=int)
    args = parser.parse_args()
    raw = load_yaml(args.config)
    output_root = raw.pop("output_root")
    thresholds = ScoreThresholds(
        minimum_state_count=int(raw.pop("minimum_state_count")),
        minimum_inflow=float(raw.pop("minimum_inflow")),
    )
    raw["dwell_levels"] = tuple(float(value) for value in raw["dwell_levels"])
    if args.repeats is not None:
        raw["repeats"] = args.repeats
    run_dwell_gradient(output_dir=output_root, config=DwellGradientConfig(thresholds=thresholds, **raw))
    print(f"completed dwell-gradient workflow: {output_root}")


if __name__ == "__main__":
    main()
