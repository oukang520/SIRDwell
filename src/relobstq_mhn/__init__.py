"""Core Rel-ObsTQ-MHN method package.

This package contains experiment-independent method code for estimating
state-level relative dwell/accumulation from cross-sectional mutation states
and an MHN-derived transition backbone. It intentionally avoids plotting,
file-system conventions and experiment-specific thresholds so that experiment
scripts can call a stable method layer.
"""

from .core.states import (
    State,
    canonical_genotype,
    canonical_state,
    compact_state,
    event_count,
    genotype_events,
    genotype_signature,
    genotype_vector,
    split_state,
)
from .core.transitions import (
    aggregate_inflow,
    probability_provider_from_theta,
    same_stage_one_step_edges,
    softmax_addition_probabilities,
)
from .core.scoring import (
    ScoreThresholds,
    classify_relative_states,
    compute_observation_enrichment,
    compute_relative_dwell,
)
from .core.bootstrap import BootstrapConfig, bootstrap_relative_dwell
from .core.mhn import MhnFitConfig, MhnFitResult, fit_cmh
from .data.processing import (
    build_event_matrix,
    build_experiment_ready_tables,
    build_state_table,
    normalize_mutation_table,
    normalize_survival_event,
    select_event_panel,
    standardize_stage_group,
)
from .simulation.generator import (
    SimulationConfig,
    create_sparse_theta,
    event_rates_from_mask,
    implant_dwell_truth,
    simulate_cohort_with_audit,
    simulate_patient_trajectory,
    theta_edge_list,
)
from .core.topology import (
    build_dominant_predecessor_path,
    event_added,
    select_topology_targets,
)

__all__ = [
    "BootstrapConfig",
    "MhnFitConfig",
    "MhnFitResult",
    "ScoreThresholds",
    "SimulationConfig",
    "State",
    "aggregate_inflow",
    "build_event_matrix",
    "build_experiment_ready_tables",
    "bootstrap_relative_dwell",
    "build_dominant_predecessor_path",
    "build_state_table",
    "canonical_genotype",
    "canonical_state",
    "classify_relative_states",
    "compact_state",
    "compute_observation_enrichment",
    "compute_relative_dwell",
    "create_sparse_theta",
    "event_added",
    "event_count",
    "event_rates_from_mask",
    "fit_cmh",
    "genotype_events",
    "genotype_signature",
    "genotype_vector",
    "implant_dwell_truth",
    "normalize_mutation_table",
    "normalize_survival_event",
    "probability_provider_from_theta",
    "same_stage_one_step_edges",
    "select_event_panel",
    "select_topology_targets",
    "simulate_cohort_with_audit",
    "simulate_patient_trajectory",
    "softmax_addition_probabilities",
    "standardize_stage_group",
    "theta_edge_list",
    "split_state",
]
