"""Table representation of real-cohort evolutionary routes."""

from __future__ import annotations

import pandas as pd

from ..core.topology import build_dominant_predecessor_path, event_added, select_topology_targets


def topology_route_table(scores: pd.DataFrame, *, target_count: int = 6, max_depth: int = 8) -> pd.DataFrame:
    """Return one row per node on each selected dominant-predecessor route."""

    targets = select_topology_targets(scores, top_paths=target_count)
    indexed = scores.set_index("state", drop=False)
    rows = []
    for route_rank, target in enumerate(targets["state"].astype(str), start=1):
        path = build_dominant_predecessor_path(target, scores, max_depth=max_depth)
        for step, state in enumerate(path):
            record = indexed.loc[state] if state in indexed.index else pd.Series(dtype=object)
            rows.append(
                {
                    "route_rank": route_rank,
                    "target_state": target,
                    "step": step,
                    "state": state,
                    "event_added": event_added(path[step - 1], state) if step else "",
                    "R_star": record.get("R_star"),
                    "log2_R_star": record.get("log2_R_star"),
                    "N_v": record.get("N_v"),
                    "F_hat": record.get("F_hat"),
                }
            )
    return pd.DataFrame(rows)
