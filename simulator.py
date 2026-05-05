# simulator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from demand import DemandData, sample_destination_index
from driver_behavior import (
    AttemptResult,
    ParkingChoiceData,
    sample_next_planned_after_failure_with_lambda,
    sample_planned_parking_index_with_lambda,
    simulate_attempt_with_greedy_stopping,
)
from grid import Cell, GridData


TracePoint = tuple[float, float, int]


@dataclass(frozen=True)
class FailureDecision:
    """Stay/switch decision after a failed attempt."""

    attempt_number: int
    failed_index: int
    stay: bool
    next_planned_index: int


@dataclass(frozen=True)
class TripObserved:
    """
    Observed trip-level data.

    This is what the estimator is allowed to observe:
        O: starting parking-cell index
        I: desired destination index
        Gamma: observed GPS-style trace points
    """

    trip_id: int
    vehicle_id: int
    O: int
    I: int
    Gamma: list[TracePoint]
    start_time: int
    end_time: int


@dataclass(frozen=True)
class TripTruth:
    """
    Hidden simulation labels for evaluation/debugging only.

    The estimator should not use these fields.
    """

    trip_id: int
    vehicle_id: int
    O: int
    I: int
    E: int
    start_time: int
    end_time: int
    success_attempt_number: int
    attempts: list[AttemptResult]
    failure_decisions: list[FailureDecision]


@dataclass(frozen=True)
class TripRecord:
    """One simulated trip with observed data and hidden truth."""

    observed: TripObserved
    truth: TripTruth


@dataclass(frozen=True)
class SimulationResult:
    """Full simulation output."""

    grid: GridData
    demand: DemandData
    choice: ParkingChoiceData
    theta: np.ndarray
    trips: list[TripRecord]
    vehicle_traces: dict[int, list[TracePoint]]
    initial_parking_indices: np.ndarray
    lambda_by_vehicle: np.ndarray
    rho_by_vehicle: np.ndarray


def generate_theta(
    grid: GridData,
    config: Any,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Draw true parking-success probabilities.

    theta[j] is the true immediate parking-success probability at parking cell p_j.
    """
    if rng is None:
        rng = np.random.default_rng(config.seed)

    theta_low = float(config.theta_low)
    theta_high = float(config.theta_high)

    if not (0.0 <= theta_low <= theta_high <= 1.0):
        raise ValueError("Require 0 <= theta_low <= theta_high <= 1.")

    return rng.uniform(theta_low, theta_high, size=grid.K)


def sample_driver_parameters(
    *,
    n_vehicles: int,
    config: Any,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample driver-level lambda_v and rho_v.

    If config.heterogeneous_drivers is False, every vehicle receives the common
    values config.lambda_choice and config.rho_stay. If True, lambda_v and rho_v
    are sampled independently from the configured uniform ranges.
    """
    if n_vehicles <= 0:
        raise ValueError("n_vehicles must be positive.")

    if getattr(config, "heterogeneous_drivers", False):
        lambda_by_vehicle = rng.uniform(
            float(config.lambda_low),
            float(config.lambda_high),
            size=n_vehicles,
        )
        rho_by_vehicle = rng.uniform(
            float(config.rho_low),
            float(config.rho_high),
            size=n_vehicles,
        )
    else:
        lambda_by_vehicle = np.full(n_vehicles, float(config.lambda_choice), dtype=float)
        rho_by_vehicle = np.full(n_vehicles, float(config.rho_stay), dtype=float)

    return lambda_by_vehicle, rho_by_vehicle


def initialize_vehicle_parking_indices(
    *,
    n_vehicles: int,
    demand: DemandData,
    grid: GridData,
    lambda_by_vehicle: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Initialize vehicles at parking cells.

    The starting parking-lot distribution is simulated by first assigning each
    vehicle a desired non-parking destination from the density distribution, then
    having it select a parking cell according to that vehicle's parking-choice
    behavior. If drivers are heterogeneous, vehicle v uses lambda_by_vehicle[v].
    """
    if n_vehicles <= 0:
        raise ValueError("n_vehicles must be positive.")

    lambda_by_vehicle = np.asarray(lambda_by_vehicle, dtype=float)
    if lambda_by_vehicle.shape != (n_vehicles,):
        raise ValueError(f"lambda_by_vehicle must have shape ({n_vehicles},).")

    starts = np.empty(n_vehicles, dtype=int)

    for vehicle_id in range(n_vehicles):
        initial_destination = sample_destination_index(demand, rng)
        starts[vehicle_id] = sample_planned_parking_index_with_lambda(
            grid=grid,
            destination_index=initial_destination,
            lambda_choice=float(lambda_by_vehicle[vehicle_id]),
            rng=rng,
        )

    return starts


def simulate_trip(
    *,
    trip_id: int,
    vehicle_id: int,
    start_parking_index: int,
    destination_index: int,
    start_time: int,
    grid: GridData,
    theta: np.ndarray,
    config: Any,
    rng: np.random.Generator,
    lambda_choice: float,
    rho_stay: float,
) -> TripRecord:
    """
    Simulate one trip.

    The trip starts from parking cell p_O and has desired destination g_I.
    The driver repeatedly samples planned parking cells, moves toward them,
    greedily checks eligible parking cells along the way, and stops once parking
    succeeds.

    lambda_choice and rho_stay are driver-level behavior parameters for this
    trip's vehicle.
    """
    _validate_trip_inputs(
        start_parking_index=start_parking_index,
        destination_index=destination_index,
        grid=grid,
        theta=theta,
    )

    if lambda_choice < 0:
        raise ValueError("lambda_choice must be nonnegative.")
    if not (0.0 <= rho_stay <= 1.0):
        raise ValueError("rho_stay must be in [0, 1].")

    current_cell = grid.parking_cell(start_parking_index)
    current_time = int(start_time)

    full_trace: list[TracePoint] = [
        _observed_point(current_cell, current_time, config, rng)
    ]

    attempts: list[AttemptResult] = []
    failure_decisions: list[FailureDecision] = []

    planned_index = sample_planned_parking_index_with_lambda(
        grid=grid,
        destination_index=destination_index,
        lambda_choice=lambda_choice,
        rng=rng,
    )

    for attempt_number in range(1, int(config.max_attempts_per_trip) + 1):
        attempt = simulate_attempt_with_greedy_stopping(
            current_cell=current_cell,
            destination_index=destination_index,
            planned_index=planned_index,
            theta=theta,
            grid=grid,
            rng=rng,
        )
        attempts.append(attempt)

        current_time = _append_travel_path_to_trace(
            trace=full_trace,
            path=attempt.travel_path,
            current_time=current_time,
            config=config,
            rng=rng,
        )

        if attempt.success:
            ending_index = attempt.actual_index
            observed_trace = _subsample_trace(full_trace, int(config.record_every))

            observed = TripObserved(
                trip_id=trip_id,
                vehicle_id=vehicle_id,
                O=start_parking_index,
                I=destination_index,
                Gamma=observed_trace,
                start_time=start_time,
                end_time=current_time,
            )

            truth = TripTruth(
                trip_id=trip_id,
                vehicle_id=vehicle_id,
                O=start_parking_index,
                I=destination_index,
                E=ending_index,
                start_time=start_time,
                end_time=current_time,
                success_attempt_number=attempt_number,
                attempts=attempts,
                failure_decisions=failure_decisions,
            )

            return TripRecord(observed=observed, truth=truth)

        # If the attempt fails, the driver is now at the failed parking cell.
        failed_index = attempt.actual_index
        current_cell = grid.parking_cell(failed_index)

        stay, next_planned_index = sample_next_planned_after_failure_with_lambda(
            failed_index=failed_index,
            destination_index=destination_index,
            grid=grid,
            lambda_choice=lambda_choice,
            rho_stay=rho_stay,
            rng=rng,
        )

        failure_decisions.append(
            FailureDecision(
                attempt_number=attempt_number,
                failed_index=failed_index,
                stay=stay,
                next_planned_index=next_planned_index,
            )
        )

        if stay:
            current_time += int(config.time_per_wait)
            full_trace.append(_observed_point(current_cell, current_time, config, rng))

        planned_index = next_planned_index

    raise RuntimeError(
        f"Trip {trip_id} did not succeed within "
        f"{config.max_attempts_per_trip} attempts. Consider increasing "
        "max_attempts_per_trip or using larger theta values."
    )


def simulate_all_vehicles(
    *,
    grid: GridData,
    demand: DemandData,
    choice: ParkingChoiceData,
    theta: np.ndarray,
    config: Any,
    rng: Optional[np.random.Generator] = None,
) -> SimulationResult:
    """
    Simulate multiple vehicles continuously moving around the grid.

    Each vehicle starts at a parking cell. For each trip:
        1. sample a desired non-parking destination,
        2. search for parking near that destination,
        3. end at the successful parking cell,
        4. dwell there for config.dwell_time time steps,
        5. use that ending parking cell as the starting cell for the next trip.

    If config.heterogeneous_drivers=True, each vehicle has its own lambda_v and
    rho_v, sampled once at the start and reused across all trips by that vehicle.
    """
    if rng is None:
        rng = np.random.default_rng(config.seed)

    theta = np.asarray(theta, dtype=float)
    if theta.shape != (grid.K,):
        raise ValueError(f"theta must have shape ({grid.K},).")

    n_vehicles = int(config.n_vehicles)
    lambda_by_vehicle, rho_by_vehicle = sample_driver_parameters(
        n_vehicles=n_vehicles,
        config=config,
        rng=rng,
    )

    initial_parking_indices = initialize_vehicle_parking_indices(
        n_vehicles=n_vehicles,
        demand=demand,
        grid=grid,
        lambda_by_vehicle=lambda_by_vehicle,
        rng=rng,
    )

    trips: list[TripRecord] = []
    vehicle_traces: dict[int, list[TracePoint]] = {}

    trip_id = 0

    for vehicle_id in range(n_vehicles):
        current_parking_index = int(initial_parking_indices[vehicle_id])
        current_time = 0
        vehicle_trace: list[TracePoint] = []
        lambda_v = float(lambda_by_vehicle[vehicle_id])
        rho_v = float(rho_by_vehicle[vehicle_id])

        for _ in range(int(config.trips_per_vehicle)):
            destination_index = sample_destination_index(demand, rng)

            trip = simulate_trip(
                trip_id=trip_id,
                vehicle_id=vehicle_id,
                start_parking_index=current_parking_index,
                destination_index=destination_index,
                start_time=current_time,
                grid=grid,
                theta=theta,
                config=config,
                rng=rng,
                lambda_choice=lambda_v,
                rho_stay=rho_v,
            )
            trips.append(trip)

            _extend_trace_without_duplicate_time(vehicle_trace, trip.observed.Gamma)

            ending_parking_index = trip.truth.E
            ending_cell = grid.parking_cell(ending_parking_index)
            current_time = trip.truth.end_time

            # Dwell at the ending parking cell before the next trip.
            dwell_points: list[TracePoint] = []
            for dt in range(1, int(config.dwell_time) + 1):
                dwell_time = current_time + dt
                dwell_points.append(_observed_point(ending_cell, dwell_time, config, rng))

            _extend_trace_without_duplicate_time(vehicle_trace, dwell_points)

            current_time += int(config.dwell_time)
            current_parking_index = ending_parking_index
            trip_id += 1

        vehicle_traces[vehicle_id] = vehicle_trace

    return SimulationResult(
        grid=grid,
        demand=demand,
        choice=choice,
        theta=theta,
        trips=trips,
        vehicle_traces=vehicle_traces,
        initial_parking_indices=initial_parking_indices,
        lambda_by_vehicle=lambda_by_vehicle,
        rho_by_vehicle=rho_by_vehicle,
    )


def _validate_trip_inputs(
    *,
    start_parking_index: int,
    destination_index: int,
    grid: GridData,
    theta: np.ndarray,
) -> None:
    if not (0 <= start_parking_index < grid.K):
        raise ValueError("start_parking_index is out of range.")

    if not (0 <= destination_index < grid.M):
        raise ValueError("destination_index is out of range.")

    theta = np.asarray(theta, dtype=float)
    if theta.shape != (grid.K,):
        raise ValueError(f"theta must have shape ({grid.K},).")

    if np.any(theta < 0) or np.any(theta > 1):
        raise ValueError("theta values must be in [0, 1].")


def _append_travel_path_to_trace(
    *,
    trace: list[TracePoint],
    path: list[Cell],
    current_time: int,
    config: Any,
    rng: np.random.Generator,
) -> int:
    """
    Append movement along a path to the trace.

    Each adjacent grid movement takes config.time_per_grid_move time steps.
    If path=[current_cell] from the same-cell convention, no movement occurs.
    """
    if len(path) == 0:
        return current_time

    # Same-cell retry path: no movement.
    if len(path) == 1 and trace:
        last_r = int(round(trace[-1][0]))
        last_c = int(round(trace[-1][1]))
        if path[0] == (last_r, last_c):
            return current_time

    for cell in path:
        current_time += int(config.time_per_grid_move)
        trace.append(_observed_point(cell, current_time, config, rng))

    return current_time


def _observed_point(
    cell: Cell,
    time: int,
    config: Any,
    rng: np.random.Generator,
) -> TracePoint:
    """Convert a true grid cell and time into an observed GPS-style point."""
    r, c = cell

    if getattr(config, "add_gps_noise", False):
        noise_std = float(config.gps_noise_std)
        x = float(r + rng.normal(0.0, noise_std))
        y = float(c + rng.normal(0.0, noise_std))
    else:
        x = float(r)
        y = float(c)

    return (x, y, int(time))


def _subsample_trace(trace: list[TracePoint], record_every: int) -> list[TracePoint]:
    """Keep every record_every-th point, always preserving first and last."""
    if record_every <= 0:
        raise ValueError("record_every must be positive.")

    if record_every == 1 or len(trace) <= 2:
        return list(trace)

    sampled = [pt for idx, pt in enumerate(trace) if idx % record_every == 0]

    if sampled[0] != trace[0]:
        sampled.insert(0, trace[0])

    if sampled[-1] != trace[-1]:
        sampled.append(trace[-1])

    return sampled


def _extend_trace_without_duplicate_time(
    base: list[TracePoint],
    new_points: list[TracePoint],
) -> None:
    """
    Extend base with new_points while avoiding duplicate timestamps at joins.

    This is mainly for stitching trip traces and dwell points into continuous
    vehicle traces.
    """
    if not new_points:
        return

    if not base:
        base.extend(new_points)
        return

    start_idx = 0
    if base[-1][2] == new_points[0][2]:
        start_idx = 1

    base.extend(new_points[start_idx:])


def trip_records_to_observed(trips: list[TripRecord]) -> list[TripObserved]:
    """Extract observed trip-level data from trip records."""
    return [trip.observed for trip in trips]


def trip_records_to_truth(trips: list[TripRecord]) -> list[TripTruth]:
    """Extract hidden truth labels from trip records."""
    return [trip.truth for trip in trips]


def simulation_summary(result: SimulationResult) -> str:
    """Short text summary of a simulation result."""
    n_trips = len(result.trips)
    n_vehicles = len(result.vehicle_traces)
    avg_points = (
        np.mean([len(trip.observed.Gamma) for trip in result.trips])
        if result.trips
        else 0.0
    )

    lambda_min = float(np.min(result.lambda_by_vehicle)) if len(result.lambda_by_vehicle) else float("nan")
    lambda_max = float(np.max(result.lambda_by_vehicle)) if len(result.lambda_by_vehicle) else float("nan")
    rho_min = float(np.min(result.rho_by_vehicle)) if len(result.rho_by_vehicle) else float("nan")
    rho_max = float(np.max(result.rho_by_vehicle)) if len(result.rho_by_vehicle) else float("nan")

    return (
        f"SimulationResult(n_vehicles={n_vehicles}, n_trips={n_trips}, "
        f"K={result.grid.K}, M={result.grid.M}, "
        f"avg_points_per_trip={avg_points:.2f}, "
        f"lambda_range=[{lambda_min:.3f}, {lambda_max:.3f}], "
        f"rho_range=[{rho_min:.3f}, {rho_max:.3f}])"
    )
