"""Canonical preparation of fixed model-panel cross-sectional inputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from ..data.processing import build_event_matrix, build_state_table, normalize_mutation_table
from ..io.results import ResultWriter


@dataclass(frozen=True)
class CrossSectionalPreparationConfig:
    """Settings for a prespecified, cohort-specific model panel."""

    events: tuple[str, ...]
    minimum_state_count: int = 5
    stage_column: str = "stage_group"
    selection_rule: str = "prespecified_model_panel"
    dataset_version: str = "AACR Project GENIE v18.0-public"


def prepare_cross_sectional_cohort(
    metadata_path: str | Path,
    mutations_path: str | Path,
    *,
    output_dir: str | Path,
    config: CrossSectionalPreparationConfig,
) -> dict[str, pd.DataFrame]:
    """Create the exact binary matrix, row map and state table used by cMHN.

    This entry point starts from already provider-authorized, cohort-filtered
    tables. It deliberately does not redistribute or infer restricted GENIE
    records. The event panel is prespecified in configuration so the screening
    panel and final cMHN panel cannot be silently conflated.
    """

    metadata_path = Path(metadata_path)
    mutations_path = Path(mutations_path)
    metadata = pd.read_csv(metadata_path)
    if metadata["analysis_id"].duplicated().any():
        raise ValueError("analysis_id must be unique in cohort metadata")
    # The historical E1 contract fixes row order before matrix construction.
    metadata = metadata.sort_values("analysis_id").reset_index(drop=True)
    mutations_raw = pd.read_csv(mutations_path)
    mutations = normalize_mutation_table(
        mutations_raw,
        consequence_col="consequence" if "consequence" in mutations_raw else None,
        functional_only=True,
    )
    events: Sequence[str] = tuple(str(event).upper() for event in config.events)
    missing = sorted(set(events).difference(set(mutations["gene"].unique())))
    if missing:
        raise ValueError(f"Prespecified events have no records in the cohort mutation table: {missing}")

    event_matrix = build_event_matrix(metadata, mutations, events)
    state_table, state_occupancy = build_state_table(
        metadata,
        event_matrix,
        events,
        min_state_count=config.minimum_state_count,
        stage_column=config.stage_column,
    )
    matrix = event_matrix[list(events)].copy()
    row_map_columns = [
        column
        for column in (
            "analysis_id",
            "patient_id",
            "sample_id",
            "stage_group",
            "genotype_signature",
            "event_count",
            "state_id",
            "state_count_flag",
            "usable_for_relobstq",
        )
        if column in state_table
    ]
    row_map = state_table[row_map_columns].copy()
    row_map.insert(0, "row_index", range(len(row_map)))
    event_panel = pd.DataFrame(
        {
            "panel_rank": range(1, len(events) + 1),
            "event": events,
            "selection_rule": config.selection_rule,
        }
    )
    qc = pd.DataFrame(
        [
            {
                "analysis_units": len(metadata),
                "unique_patients": metadata["patient_id"].nunique() if "patient_id" in metadata else pd.NA,
                "model_events": len(events),
                "matrix_binary": bool(matrix.isin([0, 1]).all().all()),
                "matrix_rows_match_metadata": len(matrix) == len(metadata),
                "unique_analysis_id": not metadata["analysis_id"].duplicated().any(),
                "dataset_version": config.dataset_version,
            }
        ]
    )
    outputs = {
        "mhn_training_matrix": matrix,
        "mhn_row_index_map": row_map,
        "state_table": state_table,
        "state_occupancy": state_occupancy,
        "event_panel": event_panel,
        "preparation_qc": qc,
    }
    writer = ResultWriter(
        output_dir,
        input_files=[metadata_path, mutations_path],
        metadata={
            "workflow": "prepare_cross_sectional_cohort",
            "dataset_version": config.dataset_version,
            "selection_rule": config.selection_rule,
        },
    )
    # These three filenames are the public cross-sectional input contract.
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    matrix.to_csv(Path(output_dir) / "mhn_training_matrix.csv", index=False)
    row_map.to_csv(Path(output_dir) / "mhn_row_index_map.csv", index=False)
    state_table.to_csv(Path(output_dir) / "state_table.csv", index=False)
    writer.track(Path(output_dir) / "mhn_training_matrix.csv")
    writer.track(Path(output_dir) / "mhn_row_index_map.csv")
    writer.track(Path(output_dir) / "state_table.csv")
    writer.table("state_occupancy", state_occupancy)
    writer.table("event_panel", event_panel)
    writer.table("preparation_qc", qc)
    writer.json("resolved_config", asdict(config))
    writer.manifest()
    return outputs
