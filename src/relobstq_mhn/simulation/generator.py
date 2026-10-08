"""Simulation utilities for Rel-ObsTQ-MHN method validation.

The simulator creates a cMHN-like event-addition process, implants known
relative dwell multipliers and samples cross-sectional snapshots uniformly
along simulated patient trajectories. It is intentionally independent of any
experiment-specific plotting or file layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from ..core.states import genotype_signature


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for cMHN-like dwell simulations."""

    samples: int = 1000
    maximum_time: float = 8.0
    maximum_events: int = 6
    neutral_dwell: float = 1.0
    random_seed: int = 20260630


def create_sparse_theta(
    events: Sequence[str],
    *,
    sparsity: float = 0.15,
    seed: int = 20260630,
    baseline_mean: float = -1.0,
    baseline_sd: float = 0.4,
    positive_probability: float = 0.62,
    effect_range: tuple[float, float] = (0.5, 1.5),
    inhibitory_range: tuple[float, float] = (-1.5, -0.5),
    forced_edges: dict[tuple[int, int], float] | None = None,
) -> np.ndarray:
    """Create a sparse cMHN-style log-rate matrix."""

    rng = np.random.default_rng(seed)
    p = len(events)
    theta = np.zeros((p, p), dtype=float)
    theta[np.diag_indices(p)] = np.clip(rng.normal(baseline_mean, baseline_sd, p), -2.5, 1.0)
    off_diagonal = [(target, source) for target in range(p) for source in range(p) if target != source]
    selected_n = int(round(float(sparsity) * len(off_diagonal)))
    if selected_n > 0:
        selected = rng.choice(len(off_diagonal), size=selected_n, replace=False)
        for index in selected:
            target, source = off_diagonal[int(index)]
            if rng.random() < positive_probability:
                theta[target, source] = float(rng.uniform(*effect_range))
            else:
                theta[target, source] = float(rng.uniform(*inhibitory_range))
    for (target, source), value in (forced_edges or {}).items():
        theta[int(target), int(source)] = float(value)
    return theta


def mask_event_count(mask: int) -> int:
    """Count active events in an integer genotype mask."""

    return int(mask).bit_count()


def mask_to_genotype(mask: int, events: Sequence[str]) -> str:
    """Convert an integer mask to a genotype string."""

    return genotype_signature([int(bool(mask & (1 << index))) for index in range(len(events))], events)


def mask_to_state(mask: int, events: Sequence[str], *, stage: str = "S1") -> str:
    """Return a simple simulated state identifier."""

    return f"{stage}::{mask_to_genotype(mask, events)}"


def event_rates_from_mask(mask: int, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return absent event indices and their cMHN-like addition rates."""

    p = theta.shape[0]
    present = np.array([bool(mask & (1 << index)) for index in range(p)])
    absent = np.flatnonzero(~present)
    if absent.size == 0:
        return absent, np.array([], dtype=float)
    logits = np.array([theta[event, event] + theta[event, present].sum() for event in absent], dtype=float)
    return absent, np.exp(np.clip(logits, -50, 50))


def event_probabilities_from_mask(mask: int, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return absent event indices and normalized addition probabilities."""

    absent, rates = event_rates_from_mask(mask, theta)
    if rates.sum() <= 0:
        return absent, np.zeros_like(rates)
    return absent, rates / rates.sum()


def simulate_patient_trajectory(
    theta: np.ndarray,
    events: Sequence[str],
    dwell_by_mask: dict[int, float] | None = None,
    *,
    config: SimulationConfig | None = None,
    rng: np.random.Generator | None = None,
    sample_id: int = 1,
    repeat: int = 1,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Simulate one continuous-time trajectory and one cross-sectional snapshot."""

    config = config or SimulationConfig()
    rng = rng or np.random.default_rng(config.random_seed)
    dwell_by_mask = dwell_by_mask or {}
    mask = 0
    current_time = 0.0
    rows: list[dict[str, object]] = []
    step = 0
    while current_time < config.maximum_time:
        if mask_event_count(mask) >= config.maximum_events:
            break
        absent, rates = event_rates_from_mask(mask, theta)
        dwell = float(dwell_by_mask.get(mask, config.neutral_dwell))
        if absent.size == 0 or rates.sum() <= 0:
            end = config.maximum_time
            added = "STOP_NO_EVENT"
            next_mask = mask
        else:
            total_rate = float(rates.sum()) / max(dwell, 1.0e-12)
            wait = float(rng.exponential(1.0 / total_rate))
            end = min(current_time + wait, config.maximum_time)
            if end >= config.maximum_time:
                added = "CENSORED_AT_HORIZON"
                next_mask = mask
            else:
                chosen = int(rng.choice(absent, p=rates / rates.sum()))
                added = str(events[chosen])
                next_mask = mask | (1 << chosen)
        rows.append(
            {
                "repeat": repeat,
                "sample_id": sample_id,
                "step": step,
                "time_start": float(current_time),
                "time_end": float(end),
                "duration": float(max(end - current_time, 0.0)),
                "mask": int(mask),
                "state": mask_to_state(mask, events),
                "genotype": mask_to_genotype(mask, events),
                "event_count": mask_event_count(mask),
                "event_added": added,
                "target_mask_after_event": int(next_mask),
                "D_true": dwell,
            }
        )
        if end >= config.maximum_time or next_mask == mask:
            break
        mask = next_mask
        current_time = end
        step += 1

    trajectory = pd.DataFrame(rows)
    horizon = float(trajectory["time_end"].iloc[-1]) if len(trajectory) else 0.0
    observation_time = float(rng.uniform(0.0, horizon)) if horizon > 0 else 0.0
    selected = trajectory.iloc[-1]
    for _, row in trajectory.iterrows():
        if row["time_start"] <= observation_time < row["time_end"] or np.isclose(observation_time, row["time_end"]):
            selected = row
            break
    snapshot = {
        "repeat": repeat,
        "sample_id": sample_id,
        "observation_time": observation_time,
        "trajectory_step": int(selected["step"]),
        "mask": int(selected["mask"]),
        "state": str(selected["state"]),
        "genotype": str(selected["genotype"]),
        "event_count": int(selected["event_count"]),
        "D_true": float(selected["D_true"]),
    }
    for index, event in enumerate(events):
        snapshot[str(event)] = int(bool(int(selected["mask"]) & (1 << index)))
    return trajectory, snapshot


def simulate_cohort_with_audit(
    theta: np.ndarray,
    events: Sequence[str],
    dwell_by_mask: dict[int, float] | None = None,
    *,
    config: SimulationConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate a cohort and return trajectory and snapshot audit tables."""

    config = config or SimulationConfig()
    rng = np.random.default_rng(config.random_seed)
    trajectories: list[pd.DataFrame] = []
    snapshots: list[dict[str, object]] = []
    for sample_id in range(1, config.samples + 1):
        trajectory, snapshot = simulate_patient_trajectory(
            theta,
            events,
            dwell_by_mask,
            config=config,
            rng=rng,
            sample_id=sample_id,
            repeat=1,
        )
        trajectories.append(trajectory)
        snapshots.append(snapshot)
    return pd.concat(trajectories, ignore_index=True), pd.DataFrame(snapshots)


def implant_dwell_truth(
    pilot_snapshots: pd.DataFrame,
    *,
    bottleneck_states: int = 3,
    fast_states: int = 3,
    bottleneck_dwell: float = 3.0,
    fast_dwell: float = 0.3,
    neutral_dwell: float = 1.0,
    min_bottleneck_count: int = 20,
    max_bottleneck_count: int | None = None,
) -> tuple[dict[int, float], pd.DataFrame]:
    """Select truth states from a pilot run and assign dwell multipliers."""

    counts = (
        pilot_snapshots.groupby(["mask", "state", "genotype", "event_count"], dropna=False)
        .size()
        .rename("pilot_count")
        .reset_index()
    )
    bottleneck_pool = counts[counts["pilot_count"].ge(min_bottleneck_count)].copy()
    if max_bottleneck_count is not None:
        bottleneck_pool = bottleneck_pool[bottleneck_pool["pilot_count"].le(max_bottleneck_count)]
    bottleneck_pool = bottleneck_pool.sort_values(["event_count", "pilot_count"], ascending=[False, False])
    selected_bottleneck = bottleneck_pool.head(bottleneck_states)["mask"].astype(int).tolist()

    fast_pool = counts[~counts["mask"].isin(selected_bottleneck)].sort_values("pilot_count", ascending=False)
    selected_fast = fast_pool.head(fast_states)["mask"].astype(int).tolist()

    dwell = {int(mask): float(neutral_dwell) for mask in counts["mask"].astype(int)}
    dwell.update({int(mask): float(bottleneck_dwell) for mask in selected_bottleneck})
    dwell.update({int(mask): float(fast_dwell) for mask in selected_fast})

    truth = counts[counts["mask"].isin(selected_bottleneck + selected_fast)].copy()
    truth["truth_class"] = np.where(truth["mask"].isin(selected_bottleneck), "bottleneck", "fast")
    truth["D_true"] = truth["mask"].map(dwell)
    return dwell, truth.reset_index(drop=True)


def theta_edge_list(theta: np.ndarray, events: Sequence[str]) -> pd.DataFrame:
    """Return nonzero off-diagonal theta entries as an edge list."""

    rows = []
    for target in range(theta.shape[0]):
        for source in range(theta.shape[1]):
            value = float(theta[target, source])
            if target == source or np.isclose(value, 0.0):
                continue
            rows.append(
                {
                    "target_event": str(events[target]),
                    "source_event": str(events[source]),
                    "target_index": target,
                    "source_index": source,
                    "log_effect": value,
                    "effect": "promoting" if value > 0 else "inhibiting",
                }
            )
    return pd.DataFrame(rows)
