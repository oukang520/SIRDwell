"""Split-sample and cross-cohort reproducibility summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.validation import require_columns
from ..evaluation.metrics import safe_rank_correlation


def compare_score_tables(
    first: pd.DataFrame,
    second: pd.DataFrame,
    *,
    score_column: str = "log2_R_star",
    top_k: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare independently estimated state scores on their common support."""

    require_columns(first, ["state", score_column], "first_scores")
    require_columns(second, ["state", score_column], "second_scores")
    paired = first[["state", score_column]].merge(
        second[["state", score_column]], on="state", suffixes=("_first", "_second"), validate="one_to_one"
    )
    rho, pvalue = safe_rank_correlation(paired[f"{score_column}_first"], paired[f"{score_column}_second"])
    first_top = set(first.nlargest(min(top_k, len(first)), score_column)["state"].astype(str))
    second_top = set(second.nlargest(min(top_k, len(second)), score_column)["state"].astype(str))
    summary = pd.DataFrame(
        [
            {
                "common_states": len(paired),
                "spearman_r": rho,
                "spearman_p": pvalue,
                "top_k": top_k,
                "top_k_overlap": len(first_top & second_top),
                "top_k_jaccard": len(first_top & second_top) / max(len(first_top | second_top), 1),
            }
        ]
    )
    return paired, summary
