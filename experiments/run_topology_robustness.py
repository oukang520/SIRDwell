"""Run the canonical supplementary E7 topology-robustness simulation."""

from __future__ import annotations

import argparse

from relobstq_mhn.core.scoring import ScoreThresholds
from relobstq_mhn.io import load_yaml
from relobstq_mhn.workflows.topology_robustness import TopologyRobustnessConfig, run_topology_robustness


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/topology_robustness.yaml")
    parser.add_argument("--repeats", type=int)
    args = parser.parse_args()
    raw = load_yaml(args.config)
    output_root = raw.pop("output_root")
    thresholds = ScoreThresholds(
        minimum_state_count=int(raw.pop("minimum_state_count")),
        minimum_inflow=float(raw.pop("minimum_inflow")),
    )
    for field in ("topologies", "sparsities", "placements", "dwell_levels"):
        raw[field] = tuple(raw[field])
    if args.repeats is not None:
        raw["repeats"] = args.repeats
    config = TopologyRobustnessConfig(thresholds=thresholds, **raw)
    run_topology_robustness(output_dir=output_root, config=config)
    print(f"completed topology-robustness workflow: {output_root}")


if __name__ == "__main__":
    main()
