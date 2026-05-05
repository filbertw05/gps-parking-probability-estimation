# estimators.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from grid import GridData
from simulator import TripObserved, TripRecord, TripTruth


@dataclass(frozen=True)
class EstimateResult:
    """
    Result of a parking-probability estimator.

    theta_hat[j] estimates the true parking-success probability theta[j].
    successes[j] is the inferred or observed number of successes at parking cell p_j.
    exposure[j] is the inferred or observed exposure/opportunity count at p_j.
    """

    name: str
    theta_hat: np.ndarray
    successes: np.ndarray
    exposure: np.ndarray
    notes: str = ""


def estimate_oracle_exposure(
    trips: Sequence[TripRecord],
    grid: GridData,
    config: Any,
    *,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
) -> EstimateResult:
    """
    Oracle exposure estimator.

    This uses hidden simulation labels and should only be used as an upper-bound
    benchmark. Each availability draw at parking cell p_j contributes one unit of
    exposure to p_j, and each successful draw contributes one success.
    """
    alpha, beta = _get_smoothing(config, alpha, beta)

    exposure = np.zeros(grid.K, dtype=float)
    successes = np.zeros(grid.K, dtype=float)

    for trip in trips:
        truth = _get_truth(trip)

        for attempt in truth.attempts:
            for draw in attempt.availability_draws:
                j = draw.parking_index
                exposure[j] += 1.0
                if draw.success:
                    successes[j] += 1.0

    theta_hat = _smoothed_ratio(successes, exposure, alpha, beta)

    return EstimateResult(
        name="oracle_exposure",
        theta_hat=theta_hat,
        successes=successes,
        exposure=exposure,
        notes="Uses hidden availability draws; evaluation upper bound.",
    )


def estimate_trace_exposure(
    trips: Sequence[TripRecord | TripObserved],
    grid: GridData,
    config: Any,
    *,
    exposure_radius: Optional[float] = None,
    success_radius: Optional[float] = None,
    collapse_consecutive: bool = True,
    skip_first_point: bool = True,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
) -> EstimateResult:
    """
    Trace-based exposure estimator.

    Uses only observed trip-level traces. Exposure is inferred when a trace point
    passes through or near parking cell p_j. Success is inferred from the final
    trace point being at or near parking cell p_j.

    If collapse_consecutive=True, consecutive trace points near the same parking
    cell count as a single exposure episode.

    If skip_first_point=True, the first point of each trip is not counted as
    exposure, since it is the starting parking cell rather than a parking search
    opportunity.
    """
    alpha, beta = _get_smoothing(config, alpha, beta)

    if exposure_radius is None:
        exposure_radius = float(getattr(config, "exposure_radius", 0.0))
    if success_radius is None:
        success_radius = exposure_radius

    if exposure_radius < 0 or success_radius < 0:
        raise ValueError("exposure_radius and success_radius must be nonnegative.")

    exposure = np.zeros(grid.K, dtype=float)
    successes = np.zeros(grid.K, dtype=float)

    for trip in trips:
        obs = _get_observed(trip)

        exposure += _exposure_from_trace(
            obs.Gamma,
            grid,
            radius=exposure_radius,
            collapse_consecutive=collapse_consecutive,
            skip_first_point=skip_first_point,
        )

        success_j = infer_success_from_endpoint(
            obs.Gamma,
            grid,
            radius=success_radius,
        )
        if success_j is not None:
            successes[success_j] += 1.0

    theta_hat = _smoothed_ratio(successes, exposure, alpha, beta)

    return EstimateResult(
        name="trace_exposure",
        theta_hat=theta_hat,
        successes=successes,
        exposure=exposure,
        notes="Uses inferred exposure from observed traces only.",
    )


def estimate_stop_confirmed_exposure(
    trips: Sequence[TripRecord | TripObserved],
    grid: GridData,
    config: Any,
    *,
    stop_radius: Optional[float] = None,
    same_location_tolerance: float = 1e-9,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
) -> EstimateResult:
    """
    Stop-confirmed trace estimator.

    This uses only observed trip-level traces. A parking attempt at p_j is inferred
    only when two consecutive trace points are at the same parking cell p_j.

    For a repeated point z_m = z_{m-1} = p_j:
        - exposure[j] += 1
        - success[j] += 1 only if m is the final trace point
        - otherwise, it is counted as a failed exposure

    This is intentionally conservative: it avoids treating ordinary pass-throughs
    near parking cells as failed parking attempts.
    """
    alpha, beta = _get_smoothing(config, alpha, beta)

    if stop_radius is None:
        stop_radius = float(getattr(config, "exposure_radius", 0.0))

    if stop_radius < 0:
        raise ValueError("stop_radius must be nonnegative.")
    if same_location_tolerance < 0:
        raise ValueError("same_location_tolerance must be nonnegative.")

    exposure = np.zeros(grid.K, dtype=float)
    successes = np.zeros(grid.K, dtype=float)

    for trip in trips:
        obs = _get_observed(trip)

        trip_exposure, trip_successes = _stop_confirmed_counts_from_trace(
            obs.Gamma,
            grid,
            radius=stop_radius,
            same_location_tolerance=same_location_tolerance,
        )

        exposure += trip_exposure
        successes += trip_successes

    theta_hat = _smoothed_ratio(successes, exposure, alpha, beta)

    return EstimateResult(
        name="stop_confirmed",
        theta_hat=theta_hat,
        successes=successes,
        exposure=exposure,
        notes=(
            "Uses repeated observed locations at parking cells as confirmed "
            "parking attempts; success only if the repeated point ends the trace."
        ),
    )


def infer_success_from_endpoint(
    trace: Sequence[tuple[float, float, int]],
    grid: GridData,
    *,
    radius: float,
) -> Optional[int]:
    """
    Infer successful parking cell from the final trace point.

    Returns parking-cell index j if the final point is within radius of p_j,
    otherwise returns None.
    """
    if not trace:
        return None

    x, y, _t = trace[-1]
    return parking_index_near_point((x, y), grid, radius=radius)


def parking_index_near_point(
    point: tuple[float, float],
    grid: GridData,
    *,
    radius: float,
) -> Optional[int]:
    """
    Return nearest parking-cell index if it is within radius of point.

    Otherwise return None.
    """
    if radius < 0:
        raise ValueError("radius must be nonnegative.")

    j, dist = nearest_parking_cell_to_point(point, grid)

    if dist <= radius + 1e-9:
        return j

    return None


def nearest_parking_cell_to_point(
    point: tuple[float, float],
    grid: GridData,
) -> tuple[int, float]:
    """
    Return the nearest parking-cell index and its Manhattan distance to point.
    """
    if grid.K == 0:
        raise ValueError("Grid has no parking cells.")

    x, y = point

    best_j = 0
    best_dist = float("inf")

    for j, (r, c) in enumerate(grid.parking_cells):
        dist = abs(x - r) + abs(y - c)
        if dist < best_dist:
            best_dist = dist
            best_j = j

    return best_j, float(best_dist)


def parking_indices_near_point(
    point: tuple[float, float],
    grid: GridData,
    *,
    radius: float,
) -> list[int]:
    """
    Return parking-cell indices within Manhattan radius of a point.
    """
    if radius < 0:
        raise ValueError("radius must be nonnegative.")

    x, y = point
    out: list[int] = []

    for j, (r, c) in enumerate(grid.parking_cells):
        dist = abs(x - r) + abs(y - c)
        if dist <= radius + 1e-9:
            out.append(j)

    return out


def _exposure_from_trace(
    trace: Sequence[tuple[float, float, int]],
    grid: GridData,
    *,
    radius: float,
    collapse_consecutive: bool,
    skip_first_point: bool,
) -> np.ndarray:
    """
    Infer exposure counts from a single observed trace.
    """
    exposure = np.zeros(grid.K, dtype=float)

    if len(trace) == 0:
        return exposure

    points = trace[1:] if skip_first_point else trace

    if not collapse_consecutive:
        for x, y, _t in points:
            for j in parking_indices_near_point((x, y), grid, radius=radius):
                exposure[j] += 1.0
        return exposure

    previous_near: set[int] = set()

    for x, y, _t in points:
        current_near = set(parking_indices_near_point((x, y), grid, radius=radius))

        # Count a new exposure episode when the trace newly enters the
        # neighborhood of parking cell p_j.
        for j in current_near - previous_near:
            exposure[j] += 1.0

        previous_near = current_near

    return exposure


def _stop_confirmed_counts_from_trace(
    trace: Sequence[tuple[float, float, int]],
    grid: GridData,
    *,
    radius: float,
    same_location_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Infer stop-confirmed exposure and success counts from one trace.

    A stop-confirmed attempt occurs when two consecutive trace points have the
    same observed location and that location is a parking cell.

    The repeated point is a success if and only if it is the final trace point.
    """
    exposure = np.zeros(grid.K, dtype=float)
    successes = np.zeros(grid.K, dtype=float)

    if len(trace) < 2:
        return exposure, successes

    final_m = len(trace) - 1

    for m in range(1, len(trace)):
        x_prev, y_prev, _t_prev = trace[m - 1]
        x, y, _t = trace[m]

        same_location_dist = abs(x - x_prev) + abs(y - y_prev)
        if same_location_dist > same_location_tolerance:
            continue

        j = parking_index_near_point((x, y), grid, radius=radius)
        if j is None:
            continue

        exposure[j] += 1.0

        if m == final_m:
            successes[j] += 1.0

    return exposure, successes


def _smoothed_ratio(
    successes: np.ndarray,
    exposure: np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """
    Compute (alpha + successes) / (alpha + beta + exposure).

    If alpha=beta=0 and exposure[j]=0, returns NaN for that cell.
    """
    successes = np.asarray(successes, dtype=float)
    exposure = np.asarray(exposure, dtype=float)

    denom = alpha + beta + exposure
    numer = alpha + successes

    with np.errstate(divide="ignore", invalid="ignore"):
        theta_hat = numer / denom

    theta_hat = np.where(denom > 0, theta_hat, np.nan)
    return theta_hat


def _get_smoothing(
    config: Any,
    alpha: Optional[float],
    beta: Optional[float],
) -> tuple[float, float]:
    if alpha is None:
        alpha = float(getattr(config, "alpha_smoothing", 1.0))
    if beta is None:
        beta = float(getattr(config, "beta_smoothing", 1.0))

    if alpha < 0 or beta < 0:
        raise ValueError("Smoothing parameters must be nonnegative.")

    return alpha, beta


def _get_observed(trip: TripRecord | TripObserved) -> TripObserved:
    if isinstance(trip, TripObserved):
        return trip
    return trip.observed


def _get_truth(trip: TripRecord) -> TripTruth:
    return trip.truth