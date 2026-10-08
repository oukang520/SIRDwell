"""Topology helpers for displaying relative-dwell states on MHN paths."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .states import genotype_events


def event_added(source_state: str, target_state: str) -> str:
    """Return the event added from source to target, if uniquely defined."""

    _, source_genotype = str(source_state).split("::", 1)
    _, target_genotype = str(target_state).split("::", 1)
    source = set(genotype_events(source_genotype))
    target = set(genotype_events(target_genotype))
    added = sorted(target.difference(source))
    return added[0] if len(added) == 1 else "+".join(added)


def build_dominant_predecessor_path(
    target_state: str,
    score_by_state: dict[str, dict] | pd.DataFrame,
    *,
    predecessor_column: str = "dominant_predecessor",
    max_depth: int = 8,
) -> list[str]:
    """Trace a target state backward through dominant predecessors."""

    if isinstance(score_by_state, pd.DataFrame):
        records = score_by_state.set_index("state").to_dict(orient="index")
    else:
        records = score_by_state

    path = [str(target_state)]
    seen = {str(target_state)}
    current = str(target_state)
    for _ in range(max_depth):
        row = records.get(current)
        if row is None:
            break
        predecessor = str(row.get(predecessor_column, "") or "")
        if not predecessor or predecessor == "nan" or predecessor == current or predecessor in seen:
            break
        path.insert(0, predecessor)
        seen.add(predecessor)
        current = predecessor
        if predecessor.endswith("::WT"):
            break
    return path


def select_topology_targets(
    scores: pd.DataFrame,
    *,
    top_paths: int = 6,
    top_rstar_paths: int = 4,
    long_event_paths: int = 2,
    long_event_threshold: int = 3,
    long_event_minimum_count: int = 3,
    eligible_column: str = "eligible_relobstq",
) -> pd.DataFrame:
    """Select target states for a compact real-cohort topology display."""

    if "R_star" not in scores or "state" not in scores:
        raise ValueError("scores must contain state and R_star columns")
    work = scores.copy()
    if "event_count" not in work:
        work["event_count"] = work["state"].astype(str).map(lambda value: len(genotype_events(value.split("::", 1)[1])))
    if eligible_column in work:
        work = work[work[eligible_column].astype(bool)].copy()
    work = work.sort_values("R_star", ascending=False)

    core = work.head(min(top_rstar_paths, top_paths)).copy()
    core["selection_type"] = "top_rstar"
    selected = set(core["state"].astype(str))

    remaining = work[
        ~work["state"].astype(str).isin(selected)
        & (pd.to_numeric(work["event_count"], errors="coerce") > long_event_threshold)
        & (pd.to_numeric(work.get("N_v", long_event_minimum_count), errors="coerce") >= long_event_minimum_count)
    ].copy()
    long = remaining.head(max(0, min(long_event_paths, top_paths - len(core)))).copy()
    long["selection_type"] = "long_event_rstar"

    result = pd.concat([core, long], ignore_index=True)
    if len(result) < top_paths:
        filler = work[~work["state"].astype(str).isin(set(result["state"].astype(str)))].head(top_paths - len(result)).copy()
        filler["selection_type"] = "fallback_rstar"
        result = pd.concat([result, filler], ignore_index=True)
    result["path_id"] = np.arange(1, len(result) + 1)
    return result.reset_index(drop=True)
