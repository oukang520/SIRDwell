"""State and genotype utilities for Rel-ObsTQ-MHN.

The method represents a state as ``stage::genotype`` where genotype is either
``WT`` or a ``+``-joined set of event names. These helpers provide one canonical
implementation shared by inflow estimation, scoring and topology display.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class State:
    """Parsed stage/genotype state identifier."""

    stage: str
    genotype: str

    @property
    def state_id(self) -> str:
        return f"{self.stage}::{self.genotype}"

    @property
    def events(self) -> tuple[str, ...]:
        return tuple(genotype_events(self.genotype))

    @property
    def event_count(self) -> int:
        return len(self.events)


def genotype_events(genotype: object) -> list[str]:
    """Return sorted unique events in a genotype string.

    ``WT``, empty strings and missing values are treated as zero-event
    genotypes. Duplicate events are collapsed to keep downstream state IDs
    deterministic.
    """

    if genotype is None or (isinstance(genotype, float) and np.isnan(genotype)):
        return []
    text = str(genotype).strip()
    if not text or text.upper() == "WT" or text.lower() == "nan":
        return []
    events = [event.strip().upper() for event in text.split("+") if event.strip()]
    return sorted(set(events))


def canonical_genotype(genotype: object) -> str:
    """Canonicalize a genotype string."""

    events = genotype_events(genotype)
    return "+".join(events) if events else "WT"


def canonical_state(state: object) -> str:
    """Canonicalize ``stage::genotype`` while preserving the stage label."""

    parsed = split_state(state)
    return parsed.state_id


def split_state(state: object) -> State:
    """Parse a state string into stage and canonical genotype."""

    text = str(state).strip()
    if "::" not in text:
        raise ValueError(f"State must have 'stage::genotype' form, got: {text!r}")
    stage, genotype = text.split("::", 1)
    stage = stage.strip().lower()
    if not stage:
        raise ValueError(f"State stage is empty in: {text!r}")
    return State(stage=stage, genotype=canonical_genotype(genotype))


def event_count(genotype_or_state: object) -> int:
    """Count events in a genotype or state string."""

    text = str(genotype_or_state)
    if "::" in text:
        return split_state(text).event_count
    return len(genotype_events(text))


def genotype_signature(vector: Iterable[int | bool | float], events: Sequence[str]) -> str:
    """Convert a binary event vector to a canonical genotype string."""

    present = sorted({str(events[i]).upper() for i, value in enumerate(vector) if int(value) == 1})
    return "+".join(present) if present else "WT"


def genotype_vector(genotype: object, events: Sequence[str]) -> np.ndarray:
    """Convert a genotype string to a binary vector in the supplied event order."""

    present = set(genotype_events(genotype))
    return np.array([int(str(event).upper() in present) for event in events], dtype=np.int8)


def compact_state(state: object, max_events: int = 3, stage_style: str = "short") -> str:
    """Return a compact label for figures and reports."""

    parsed = split_state(state)
    events = list(parsed.events)
    if len(events) > max_events:
        genotype = "+".join(events[:max_events]) + "+..."
    else:
        genotype = "+".join(events) if events else "WT"
    if stage_style == "short":
        prefix = "P" if parsed.stage == "primary" else "M" if parsed.stage == "metastatic" else parsed.stage[:1].upper()
    elif stage_style == "full":
        prefix = parsed.stage
    else:
        raise ValueError("stage_style must be 'short' or 'full'")
    return f"{prefix}:{genotype}"


def build_state_occupancy(
    state_table: pd.DataFrame,
    event_matrix: pd.DataFrame,
    events: Sequence[str],
    *,
    stage_column: str = "stage",
    analysis_id_column: str | None = None,
) -> pd.DataFrame:
    """Build an occupancy table from sample states and a binary event matrix.

    Parameters
    ----------
    state_table:
        One row per analysis unit with a clean stage column.
    event_matrix:
        Binary matrix with the same row order as ``state_table``.
    events:
        Event order used by the MHN model.
    stage_column:
        Column containing stage/compartment labels.
    analysis_id_column:
        Optional identifier copied to an audit-only column when present.
    """

    if len(state_table) != len(event_matrix):
        raise ValueError("state_table and event_matrix must have the same row count")
    if list(event_matrix.columns.astype(str)) != list(events):
        raise ValueError("event_matrix columns must match the supplied event order")

    work = pd.DataFrame(index=state_table.index)
    work["stage"] = state_table[stage_column].astype(str).str.lower()
    work["genotype"] = [genotype_signature(row, events) for row in event_matrix.to_numpy()]
    work["event_count"] = event_matrix.sum(axis=1).astype(int).to_numpy()
    work["state"] = work["stage"] + "::" + work["genotype"]
    if analysis_id_column and analysis_id_column in state_table:
        work["analysis_id"] = state_table[analysis_id_column].astype(str).to_numpy()

    occupancy = (
        work.groupby(["state", "stage", "genotype", "event_count"], dropna=False)
        .size()
        .rename("N_v")
        .reset_index()
    )
    total = int(occupancy["N_v"].sum())
    occupancy["L_v"] = occupancy["N_v"] / max(total, 1)
    return occupancy.sort_values(["N_v", "state"], ascending=[False, True]).reset_index(drop=True)
