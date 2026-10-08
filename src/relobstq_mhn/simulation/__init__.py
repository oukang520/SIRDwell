"""Simulation utilities for controlled Rel-ObsTQ-MHN validation."""

from .generator import (
    SimulationConfig,
    create_sparse_theta,
    event_rates_from_mask,
    implant_dwell_truth,
    simulate_cohort_with_audit,
    simulate_patient_trajectory,
    theta_edge_list,
)

__all__ = [
    "SimulationConfig",
    "create_sparse_theta",
    "event_rates_from_mask",
    "implant_dwell_truth",
    "simulate_cohort_with_audit",
    "simulate_patient_trajectory",
    "theta_edge_list",
]
