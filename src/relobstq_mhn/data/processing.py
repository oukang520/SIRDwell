"""Data-processing utilities for Rel-ObsTQ-MHN inputs.

These functions implement the method-facing preprocessing layer: harmonizing
clinical/stage fields, filtering mutation events, building binary event
matrices and deriving state tables. Dataset-specific extraction scripts can
wrap these primitives without duplicating method logic.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from ..core.states import canonical_genotype, genotype_signature
from ..core.validation import assert_binary_matrix, require_columns


MISSING_VALUES = {"", "na", "nan", "none", "null", "not available", "unknown", "[not available]"}

MAF_NONFUNCTIONAL_CLASSES = {
    "silent",
    "synonymous_variant",
    "intron",
    "intron_variant",
    "3'utr",
    "5'utr",
    "3_prime_utr_variant",
    "5_prime_utr_variant",
    "igr",
    "intergenic_region",
}


def safe_text(value: object) -> str:
    """Return a stripped string, converting missing-like values to empty text."""

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() in MISSING_VALUES else text


def standardize_stage_group(value: object) -> str:
    """Map raw stage/metastasis text to a compact stage group."""

    text = safe_text(value).lower()
    if not text:
        return "unknown"
    if text in {"early", "local_advanced", "primary", "metastatic"}:
        return text
    if "metast" in text or "distant" in text or re.search(r"\bm1\b", text):
        return "metastatic"
    clean = text.replace("stage", "").replace("ajcc", "").strip()
    clean = re.sub(r"[^ivxabc0-9]+", "", clean)
    if clean in {"i", "ia", "ib", "1", "1a", "1b"}:
        return "early"
    if clean in {
        "ii",
        "iia",
        "iib",
        "iic",
        "iii",
        "iiia",
        "iiib",
        "iiic",
        "2",
        "2a",
        "2b",
        "2c",
        "3",
        "3a",
        "3b",
        "3c",
    }:
        return "local_advanced"
    if clean in {"iv", "iva", "ivb", "ivc", "4", "4a", "4b", "4c"}:
        return "metastatic"
    if "primary" in text or "tumour" in text or "tumor" in text:
        return "primary"
    return "unknown"


def normalize_survival_event(value: object) -> str:
    """Normalize survival vital status/event fields to ``'1'``, ``'0'`` or ``''``."""

    text = safe_text(value).lower()
    if text in {"true", "1", "deceased", "dead", "yes"}:
        return "1"
    if text in {"false", "0", "alive", "living", "no"}:
        return "0"
    return ""


def is_functional_maf(alteration_type: object) -> bool:
    """Return whether a MAF consequence/type should be treated as functional."""

    text = safe_text(alteration_type).lower()
    if not text:
        return True
    return text not in MAF_NONFUNCTIONAL_CLASSES


def normalize_mutation_table(
    mutations: pd.DataFrame,
    *,
    analysis_id_col: str = "analysis_id",
    gene_col: str = "gene",
    consequence_col: str | None = None,
    functional_only: bool = True,
) -> pd.DataFrame:
    """Return a de-duplicated long mutation table with ``analysis_id``/``gene``."""

    require_columns(mutations, [analysis_id_col, gene_col], "mutations")
    work = mutations[[analysis_id_col, gene_col] + ([consequence_col] if consequence_col else [])].copy()
    work = work.rename(columns={analysis_id_col: "analysis_id", gene_col: "gene"})
    work["analysis_id"] = work["analysis_id"].astype(str)
    work["gene"] = work["gene"].astype(str).str.strip().str.upper()
    work = work[(work["analysis_id"].map(safe_text) != "") & (work["gene"].map(safe_text) != "")]
    if functional_only and consequence_col:
        work = work[work[consequence_col].map(is_functional_maf)]
    return work[["analysis_id", "gene"]].drop_duplicates().reset_index(drop=True)


def select_event_panel(
    mutations: pd.DataFrame,
    *,
    max_events: int = 25,
    min_frequency: float = 0.01,
    analysis_unit_count: int | None = None,
    exclude_genes: Iterable[str] | None = None,
) -> tuple[list[str], pd.DataFrame]:
    """Select a mutation event panel by sample support and frequency."""

    require_columns(mutations, ["analysis_id", "gene"], "mutations")
    exclude = {gene.upper() for gene in (exclude_genes or [])}
    support = (
        mutations[~mutations["gene"].isin(exclude)]
        .drop_duplicates(["analysis_id", "gene"])
        .groupby("gene")["analysis_id"]
        .nunique()
        .sort_values(ascending=False)
    )
    denominator = int(analysis_unit_count or mutations["analysis_id"].nunique() or 1)
    frequency = support / max(denominator, 1)
    table = pd.DataFrame(
        {
            "event": support.index,
            "sample_count": support.astype(int).to_numpy(),
            "frequency": frequency.reindex(support.index).to_numpy(),
        }
    )
    selected = table.loc[table["frequency"].ge(min_frequency), "event"].head(max_events).tolist()
    if len(selected) < min(max_events, 10) and len(table):
        selected = table["event"].head(min(max_events, max(10, len(selected)))).tolist()
    table["selected"] = table["event"].isin(selected)
    return selected, table


def build_event_matrix(metadata: pd.DataFrame, mutations: pd.DataFrame, events: Sequence[str]) -> pd.DataFrame:
    """Build a binary analysis-unit by event matrix with an ``analysis_id`` column."""

    require_columns(metadata, ["analysis_id"], "metadata")
    require_columns(mutations, ["analysis_id", "gene"], "mutations")
    ids = metadata["analysis_id"].drop_duplicates().astype(str)
    events = [str(event).upper() for event in events]
    matrix = pd.DataFrame({"analysis_id": ids})
    for event in events:
        matrix[event] = 0
    if events and not mutations.empty:
        work = mutations[mutations["gene"].isin(events)][["analysis_id", "gene"]].drop_duplicates()
        work["_value"] = 1
        pivot = work.pivot_table(index="analysis_id", columns="gene", values="_value", aggfunc="max", fill_value=0)
        pivot = pivot.reindex(index=ids, columns=events, fill_value=0).astype(int).reset_index()
        matrix = pivot
    assert_binary_matrix(matrix.drop(columns=["analysis_id"]), "event_matrix")
    return matrix


def build_state_table(
    metadata: pd.DataFrame,
    event_matrix: pd.DataFrame,
    events: Sequence[str],
    *,
    min_state_count: int = 5,
    stage_column: str = "stage_group",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build per-analysis-unit state assignments and state occupancy summary."""

    require_columns(metadata, ["analysis_id"], "metadata")
    require_columns(event_matrix, ["analysis_id", *events], "event_matrix")
    merged = metadata.merge(event_matrix, on="analysis_id", how="left")
    for event in events:
        merged[event] = pd.to_numeric(merged[event], errors="coerce").fillna(0).astype(int)
    merged["stage_group"] = merged.get(stage_column, "unknown")
    merged["stage_group"] = merged["stage_group"].fillna("unknown").astype(str).map(standardize_stage_group)
    merged["event_count"] = merged[list(events)].sum(axis=1).astype(int) if events else 0
    merged["genotype_signature"] = [
        genotype_signature(row, events) for row in merged[list(events)].to_numpy()
    ] if events else "WT"
    merged["genotype_signature"] = merged["genotype_signature"].map(canonical_genotype)
    merged["state_id"] = merged["stage_group"] + "::" + merged["genotype_signature"]
    counts = merged["state_id"].value_counts()
    merged["state_count"] = merged["state_id"].map(counts).astype(int)
    merged["state_count_flag"] = np.where(merged["state_count"] >= min_state_count, "valid_state", "rare_state")
    merged["usable_for_relobstq"] = merged["stage_group"].ne("unknown") & merged["state_count_flag"].eq("valid_state")

    occupancy = (
        merged.groupby(["stage_group", "genotype_signature", "state_id", "state_count_flag"], dropna=False)
        .size()
        .reset_index(name="state_count")
        .sort_values(["state_count", "state_id"], ascending=[False, True])
    )
    occupancy["occupancy_fraction"] = occupancy["state_count"] / max(len(merged), 1)
    return merged, occupancy.reset_index(drop=True)


def build_experiment_ready_tables(
    metadata: pd.DataFrame,
    mutations: pd.DataFrame,
    *,
    max_events: int = 25,
    min_event_frequency: float = 0.01,
    min_state_count: int = 5,
) -> dict[str, pd.DataFrame | list[str]]:
    """Convenience preprocessing pipeline for method-ready tables."""

    mutations_norm = normalize_mutation_table(mutations)
    events, event_frequency = select_event_panel(
        mutations_norm,
        max_events=max_events,
        min_frequency=min_event_frequency,
        analysis_unit_count=metadata["analysis_id"].nunique(),
    )
    event_matrix = build_event_matrix(metadata, mutations_norm, events)
    state_table, state_occupancy = build_state_table(
        metadata,
        event_matrix,
        events,
        min_state_count=min_state_count,
    )
    return {
        "events": events,
        "mutations_long": mutations_norm,
        "event_frequency": event_frequency,
        "event_matrix": event_matrix,
        "state_table": state_table,
        "state_occupancy": state_occupancy,
    }
