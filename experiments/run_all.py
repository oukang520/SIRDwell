"""Run the configured table-producing experiments in dependency order."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    modules = (
        "prepare_cross_sectional",
        "run_cross_sectional",
        "run_secondary",
        "run_simulation",
        "run_topology_robustness",
        "run_longitudinal",
    )
    for module in modules:
        command = [sys.executable, "-m", f"experiments.{module}"]
        if args.dry_run:
            print(" ".join(command))
        else:
            subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
