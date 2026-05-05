# driver_behavior.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from grid import Cell, GridData, make_path, manhattan


@dataclass(frozen=True)
class ParkingChoiceData:
    """
    Destination-conditional parking-choice distribution.

    probabilities[i, j] = pi_i(j)
    where:
        i = destination index, corresponding to g_i in the writeup
        j = parking-cell index, corresponding to p_j in the writeup

    Code uses 0-indexed indices.
    """

    lambda_choice: float
    probabilities: np.ndarray  # shape (M, K)

    @property
    def M(self) -> int:
        return self.probabilities.shape[0]

    @property
    def K(self) -> int:
        return self.probabilities.shape[1]


@dataclass(frozen=True)
class AvailabilityDraw:
    """One availability check at a parking cell."""

    parking_index: int
    success: bool
    is_planned_cell: bool


@dataclass(frozen=True)
class AttemptResult:
    """
    Result of one parking attempt.

    planned_index:
        \tilde A_{n,k}; the planned parking-cell index.

    actual_index:
        A_{n,k}; the actual parking-cell index where the driver attempts/stops.

    success:
        S_{n,k}; whether the attempt succeeds.

    travel_path:
        Ordered cells traversed during this attempt, excluding the starting cell and
        including the actual attempted/stopping cell.

    full_planned_path:
        Ordered cells from current cell to planned parking cell, before truncating
        if the driver stops early.

    early_eligible_indices:
        Eligible early parking cells encountered before reaching the planned cell.

    availability_draws:
        Availability draws that were actually checked before the attempt outcome.
    """

    planned_index: int
    actual_index: int
    success: bool
    travel_path: list[Cell]
    full_planned_path: list[Cell]
    early_eligible_indices: list[int]
    availability_draws: list[AvailabilityDraw]
    stopped_early: bool


def build_parking_choice_distribution(grid: GridData, config: Any) -> ParkingChoiceData:
    """
    Build all destination-conditional parking-choice probabilities.

    pi_i(j) proportional to exp(-lambda_choice * d(g_i, p_j)).

    This creates the common fixed-lambda choice distribution. If driver-level
    heterogeneity is enabled, simulator.py samples trips using each driver's
    lambda_v directly through sample_planned_parking_index_with_lambda(...).
    """
    lambda_choice = float(config.lambda_choice)

    if lambda_choice < 0:
        raise ValueError("lambda_choice must be nonnegative.")
    if grid.K == 0:
        raise ValueError("Grid has no parking cells.")
    if grid.M == 0:
        raise ValueError("Grid has no destination cells.")

    probs = np.empty((grid.M, grid.K), dtype=float)

    for i in range(grid.M):
        probs[i] = parking_choice_probs(grid, i, lambda_choice)

    return ParkingChoiceData(
        lambda_choice=lambda_choice,
        probabilities=probs,
    )


def parking_choice_probs(
    grid: GridData,
    destination_index: int,
    lambda_choice: float,
) -> np.ndarray:
    """
    Compute pi_i(j) for a single destination index i.

    Returns:
        probs[j] = probability of choosing planned parking cell p_j.
    """
    if not (0 <= destination_index < grid.M):
        raise ValueError("destination_index is out of range.")
    if lambda_choice < 0:
        raise ValueError("lambda_choice must be nonnegative.")

    destination = grid.destination_cell(destination_index)

    weights = np.empty(grid.K, dtype=float)
    for j, parking_cell in enumerate(grid.parking_cells):
        weights[j] = np.exp(-lambda_choice * manhattan(destination, parking_cell))

    return normalize_weights(weights)


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """Normalize nonnegative weights into probabilities."""
    weights = np.asarray(weights, dtype=float)

    if weights.ndim != 1:
        raise ValueError("weights must be one-dimensional.")
    if len(weights) == 0:
        raise ValueError("weights must be nonempty.")
    if np.any(weights < 0):
        raise ValueError("weights must be nonnegative.")

    total = weights.sum()
    if total <= 0:
        raise ValueError("weights must have positive sum.")

    return weights / total


def sample_planned_parking_index(
    choice: ParkingChoiceData,
    destination_index: int,
    rng: np.random.Generator,
    *,
    exclude_index: Optional[int] = None,
) -> int:
    """
    Sample a planned parking-cell index from pi_i.

    If exclude_index is provided, that parking cell is removed and the remaining
    probabilities are renormalized. This is used after failure when switching away
    from the failed parking cell.
    """
    if not (0 <= destination_index < choice.M):
        raise ValueError("destination_index is out of range.")

    probs = choice.probabilities[destination_index]

    if exclude_index is None:
        return int(rng.choice(choice.K, p=probs))

    if not (0 <= exclude_index < choice.K):
        raise ValueError("exclude_index is out of range.")
    if choice.K <= 1:
        raise ValueError("Cannot exclude the only parking cell.")

    masked = probs.copy()
    masked[exclude_index] = 0.0
    masked = normalize_weights(masked)

    return int(rng.choice(choice.K, p=masked))


def sample_planned_parking_index_with_lambda(
    *,
    grid: GridData,
    destination_index: int,
    lambda_choice: float,
    rng: np.random.Generator,
    exclude_index: Optional[int] = None,
) -> int:
    """
    Sample a planned parking-cell index using a supplied lambda_choice.

    This is useful for driver-level heterogeneity, where each vehicle v may have
    its own distance sensitivity lambda_v.
    """
    if not (0 <= destination_index < grid.M):
        raise ValueError("destination_index is out of range.")
    if lambda_choice < 0:
        raise ValueError("lambda_choice must be nonnegative.")

    probs = parking_choice_probs(grid, destination_index, lambda_choice)

    if exclude_index is not None:
        if not (0 <= exclude_index < grid.K):
            raise ValueError("exclude_index is out of range.")
        if grid.K <= 1:
            raise ValueError("Cannot exclude the only parking cell.")

        probs = probs.copy()
        probs[exclude_index] = 0.0
        probs = normalize_weights(probs)

    return int(rng.choice(grid.K, p=probs))


def eligible_early_parking_indices(
    *,
    current_cell: Cell,
    planned_index: int,
    destination_index: int,
    grid: GridData,
    rng: Optional[np.random.Generator] = None,
) -> tuple[list[int], list[Cell]]:
    """
    Compute the eligible early parking cells passed before reaching the planned cell.

    A parking cell p_j is eligible if:
        1. it lies on the path from current_cell to p_planned,
        2. it is not the planned parking cell itself,
        3. it is no farther from the destination than the planned parking cell.

    Returns:
        (eligible_indices, full_path)

    eligible_indices are ordered by when the driver encounters them along the path.
    full_path is the ordered path from current_cell to the planned parking cell.
    """
    if not (0 <= planned_index < grid.K):
        raise ValueError("planned_index is out of range.")
    if not (0 <= destination_index < grid.M):
        raise ValueError("destination_index is out of range.")

    planned_cell = grid.parking_cell(planned_index)
    destination_cell = grid.destination_cell(destination_index)

    full_path = make_path(current_cell, planned_cell, rng=rng)
    planned_dist = manhattan(planned_cell, destination_cell)

    eligible: list[int] = []
    seen: set[int] = set()

    for cell in full_path:
        j = grid.parking_index_by_cell.get(cell)

        if j is None:
            continue
        if j == planned_index:
            continue
        if j in seen:
            continue

        if manhattan(cell, destination_cell) <= planned_dist:
            eligible.append(j)
            seen.add(j)

    return eligible, full_path


def simulate_attempt_with_greedy_stopping(
    *,
    current_cell: Cell,
    destination_index: int,
    planned_index: int,
    theta: np.ndarray,
    grid: GridData,
    rng: np.random.Generator,
) -> AttemptResult:
    """
    Simulate one attempt with greedy stopping.

    The driver has a planned parking cell p_planned. Along the path, the driver
    checks eligible parking cells before reaching p_planned. If any eligible early
    cell succeeds, the driver stops at the first successful one. Otherwise, the
    driver reaches the planned cell and checks it.

    This function uses true theta and is therefore part of the simulator, not the
    estimator.
    """
    theta = np.asarray(theta, dtype=float)

    if theta.shape != (grid.K,):
        raise ValueError(f"theta must have shape ({grid.K},).")
    if np.any(theta < 0) or np.any(theta > 1):
        raise ValueError("theta values must be in [0, 1].")

    eligible, full_path = eligible_early_parking_indices(
        current_cell=current_cell,
        planned_index=planned_index,
        destination_index=destination_index,
        grid=grid,
        rng=rng,
    )

    availability_draws: list[AvailabilityDraw] = []

    # Check eligible early cells in order.
    for j in eligible:
        success = bool(rng.random() < theta[j])
        availability_draws.append(
            AvailabilityDraw(
                parking_index=j,
                success=success,
                is_planned_cell=False,
            )
        )

        if success:
            actual_cell = grid.parking_cell(j)
            travel_path = truncate_path_at_cell(full_path, actual_cell)

            return AttemptResult(
                planned_index=planned_index,
                actual_index=j,
                success=True,
                travel_path=travel_path,
                full_planned_path=full_path,
                early_eligible_indices=eligible,
                availability_draws=availability_draws,
                stopped_early=True,
            )

    # If no early eligible cell succeeds, check the planned parking cell.
    planned_success = bool(rng.random() < theta[planned_index])
    availability_draws.append(
        AvailabilityDraw(
            parking_index=planned_index,
            success=planned_success,
            is_planned_cell=True,
        )
    )

    return AttemptResult(
        planned_index=planned_index,
        actual_index=planned_index,
        success=planned_success,
        travel_path=full_path,
        full_planned_path=full_path,
        early_eligible_indices=eligible,
        availability_draws=availability_draws,
        stopped_early=False,
    )


def truncate_path_at_cell(path: list[Cell], stop_cell: Cell) -> list[Cell]:
    """
    Return path up to and including stop_cell.

    If stop_cell is not in path, raises an error.
    """
    for idx, cell in enumerate(path):
        if cell == stop_cell:
            return path[: idx + 1]

    raise ValueError(f"stop_cell={stop_cell} is not in path.")


def sample_next_planned_after_failure(
    *,
    failed_index: int,
    destination_index: int,
    choice: ParkingChoiceData,
    rho_stay: float,
    rng: np.random.Generator,
) -> tuple[bool, int]:
    """
    Decide whether the driver stays/retries or switches after failure.

    Returns:
        (stay, next_planned_index)

    If stay is True:
        next_planned_index = failed_index.

    If stay is False:
        next_planned_index is sampled from pi_i with failed_index excluded.
    """
    if not (0 <= failed_index < choice.K):
        raise ValueError("failed_index is out of range.")
    if not (0 <= destination_index < choice.M):
        raise ValueError("destination_index is out of range.")
    if not (0.0 <= rho_stay <= 1.0):
        raise ValueError("rho_stay must be in [0, 1].")

    stay = bool(rng.random() < rho_stay)

    if stay:
        return True, failed_index

    next_planned = sample_planned_parking_index(
        choice,
        destination_index,
        rng,
        exclude_index=failed_index,
    )
    return False, next_planned


def sample_next_planned_after_failure_with_lambda(
    *,
    failed_index: int,
    destination_index: int,
    grid: GridData,
    lambda_choice: float,
    rho_stay: float,
    rng: np.random.Generator,
) -> tuple[bool, int]:
    """
    Decide whether the driver stays/retries or switches after failure,
    using supplied driver-specific lambda_choice and rho_stay.

    Returns:
        (stay, next_planned_index)
    """
    if not (0 <= failed_index < grid.K):
        raise ValueError("failed_index is out of range.")
    if not (0 <= destination_index < grid.M):
        raise ValueError("destination_index is out of range.")
    if lambda_choice < 0:
        raise ValueError("lambda_choice must be nonnegative.")
    if not (0.0 <= rho_stay <= 1.0):
        raise ValueError("rho_stay must be in [0, 1].")

    stay = bool(rng.random() < rho_stay)

    if stay:
        return True, failed_index

    next_planned = sample_planned_parking_index_with_lambda(
        grid=grid,
        destination_index=destination_index,
        lambda_choice=lambda_choice,
        rng=rng,
        exclude_index=failed_index,
    )
    return False, next_planned


def path_time_steps(path: list[Cell], *, time_per_grid_move: int = 1) -> int:
    """
    Number of time steps used to traverse a path.

    By convention, make_path(start, end) excludes the starting cell and includes
    the ending cell, so each cell in the path corresponds to one grid move,
    except the same-cell convention path=[start]. If path=[start], this returns 0
    because no movement occurs.
    """
    if len(path) == 1:
        return 0

    return len(path) * time_per_grid_move


def choice_summary(choice: ParkingChoiceData) -> str:
    probs = choice.probabilities
    return (
        f"ParkingChoiceData(M={choice.M}, K={choice.K}, "
        f"lambda_choice={choice.lambda_choice:.4f}, "
        f"min_prob={probs.min():.4g}, max_prob={probs.max():.4g})"
    )
