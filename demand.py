# demand.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from grid import GridData, manhattan


@dataclass(frozen=True)
class DemandData:
    """
    Destination-demand distribution over non-parking cells.

    Destination cells are indexed by Python indices:
        i = 0, ..., M-1.

    This corresponds to g_1, ..., g_M in the writeup, but with 0-indexed code.
    """

    eta_D: float
    weights: np.ndarray       # shape (M,)
    probabilities: np.ndarray # shape (M,)

    @property
    def M(self) -> int:
        return len(self.probabilities)


def build_destination_distribution(grid: GridData, config: Any) -> DemandData:
    """
    Build the destination density distribution

        q_D(i) = exp(-eta_D * d(g_i, c_0)),

    normalized over destination/non-parking cells.

    Expected config fields:
        eta_D: float
    """
    eta_D = float(config.eta_D)

    if eta_D < 0:
        raise ValueError("eta_D must be nonnegative.")

    weights = compute_destination_weights(grid, eta_D)
    probabilities = normalize_weights(weights)

    return DemandData(
        eta_D=eta_D,
        weights=weights,
        probabilities=probabilities,
    )


def compute_destination_weights(grid: GridData, eta_D: float) -> np.ndarray:
    """
    Compute unnormalized destination density weights over destination cells.

    q_D(i) = exp(-eta_D * d(g_i, c_0)).
    """
    if grid.M == 0:
        raise ValueError("Grid has no destination cells.")

    weights = np.empty(grid.M, dtype=float)

    for i, cell in enumerate(grid.destination_cells):
        weights[i] = np.exp(-eta_D * manhattan(cell, grid.center))

    return weights


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """Normalize nonnegative weights into probabilities."""
    weights = np.asarray(weights, dtype=float)

    if weights.ndim != 1:
        raise ValueError("weights must be a one-dimensional array.")
    if len(weights) == 0:
        raise ValueError("weights must be nonempty.")
    if np.any(weights < 0):
        raise ValueError("weights must be nonnegative.")

    total = weights.sum()
    if total <= 0:
        raise ValueError("weights must have positive sum.")

    return weights / total


def sample_destination_index(
    demand: DemandData,
    rng: np.random.Generator,
    *,
    exclude_index: Optional[int] = None,
) -> int:
    """
    Sample one destination index.

    If exclude_index is provided, resample from the same distribution but with
    that destination removed. This is useful if we want to avoid zero-length
    destination-to-destination initialization cases.
    """
    probs = demand.probabilities

    if exclude_index is None:
        return int(rng.choice(demand.M, p=probs))

    if not (0 <= exclude_index < demand.M):
        raise ValueError("exclude_index is out of range.")

    if demand.M <= 1:
        raise ValueError("Cannot exclude the only destination cell.")

    masked = probs.copy()
    masked[exclude_index] = 0.0
    masked = normalize_weights(masked)

    return int(rng.choice(demand.M, p=masked))


def sample_destination_indices(
    demand: DemandData,
    rng: np.random.Generator,
    n: int,
) -> np.ndarray:
    """Sample n destination indices independently."""
    if n < 0:
        raise ValueError("n must be nonnegative.")

    return rng.choice(demand.M, size=n, replace=True, p=demand.probabilities).astype(int)


def destination_probability(demand: DemandData, i: int) -> float:
    """Return probability of destination index i."""
    if not (0 <= i < demand.M):
        raise ValueError("Destination index out of range.")

    return float(demand.probabilities[i])


def destination_probability_map(
    grid: GridData,
    demand: DemandData,
    *,
    fill_value: float = np.nan,
) -> np.ndarray:
    """
    Return an L x L array with destination probabilities placed on destination cells.

    Parking cells are filled with fill_value.
    Useful for plotting the destination density heatmap.
    """
    prob_map = np.full((grid.L, grid.L), fill_value, dtype=float)

    for i, cell in enumerate(grid.destination_cells):
        prob_map[cell] = demand.probabilities[i]

    return prob_map


def destination_weight_map(
    grid: GridData,
    demand: DemandData,
    *,
    fill_value: float = np.nan,
) -> np.ndarray:
    """
    Return an L x L array with unnormalized destination weights placed on destination cells.

    Parking cells are filled with fill_value.
    """
    weight_map = np.full((grid.L, grid.L), fill_value, dtype=float)

    for i, cell in enumerate(grid.destination_cells):
        weight_map[cell] = demand.weights[i]

    return weight_map


def demand_summary(demand: DemandData) -> str:
    """Short text summary of the destination distribution."""
    probs = demand.probabilities
    return (
        f"DemandData(M={demand.M}, eta_D={demand.eta_D:.4f}, "
        f"min_prob={probs.min():.4g}, max_prob={probs.max():.4g})"
    )