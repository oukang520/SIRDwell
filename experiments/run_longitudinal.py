"""Run the selected E17 longitudinal calculation and write numerical tables."""

from __future__ import annotations

import argparse

from relobstq_mhn.io import load_yaml
from relobstq_mhn.workflows.selected_longitudinal import run_selected_longitudinal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/longitudinal.yaml")
    args = parser.parse_args()
    run_selected_longitudinal(load_yaml(args.config))


if __name__ == "__main__":
    main()
