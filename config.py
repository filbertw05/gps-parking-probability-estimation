# config.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class RingParkingSpec:
    """Optional ring-based parking probabilities for the larger city setup."""
    inner_size: int
    outer_size: int
    parking_probability: float


@dataclass
class SimConfig:
    """
    Configuration for the parking-probability simulation.

    Note: the Python implementation should use 0-indexed grid coordinates:
        (0, 0), ..., (L-1, L-1).

    The LaTeX writeup may use 1-indexed coordinates:
        (1, 1), ..., (L, L).

    Manhattan distances are unchanged up to this shift.
    """

    # ----------------------------
    # Reproducibility
    # ----------------------------
    seed: int = 0

    # ----------------------------
    # Grid / city generation
    # ----------------------------
    L: int = 10
    parking_probability: float = 0.20  # phi

    # Optional nested city structure.
    # If use_ring_parking=True, these overwrite parking_probability by region.
    use_ring_parking: bool = False
    ring_specs: list[RingParkingSpec] = field(
        default_factory=lambda: [
            RingParkingSpec(inner_size=0, outer_size=10, parking_probability=0.10),
            RingParkingSpec(inner_size=10, outer_size=20, parking_probability=0.30),
            RingParkingSpec(inner_size=20, outer_size=30, parking_probability=0.15),
        ]
    )

    # ----------------------------
    # Destination density
    # q_D(i) = exp(-eta_D * d(g_i, c_0))
    # If eta_D is None, use log(4)/(L-1), so farthest cells have
    # approximately 25% of the center's density.
    # ----------------------------
    eta_D: Optional[float] = None

    # ----------------------------
    # Parking-success probabilities
    # theta_j ~ Uniform(theta_low, theta_high)
    # ----------------------------
    theta_low: float = 0.20
    theta_high: float = 0.80

    # ----------------------------
    # Driver behavior
    # pi_i(j) proportional to exp(-lambda_choice * d(g_i, p_j))
    # rho_stay = probability of staying/retrying after failure
    # ----------------------------
    lambda_choice: float = 0.80
    rho_stay: float = 0.50
    max_attempts_per_trip: int = 50

    # Driver-level heterogeneity. If heterogeneous_drivers=False, all drivers
    # use lambda_choice and rho_stay. If True, each vehicle v gets
    # lambda_v ~ Uniform(lambda_low, lambda_high) and
    # rho_v ~ Uniform(rho_low, rho_high).
    heterogeneous_drivers: bool = False
    lambda_low: float = 0.40
    lambda_high: float = 1.20
    rho_low: float = 0.20
    rho_high: float = 0.80

    # ----------------------------
    # Continuous vehicle simulation
    # ----------------------------
    n_vehicles: int = 100
    trips_per_vehicle: int = 20
    dwell_time: int = 30  # time steps spent parked before next trip

    # Movement timing:
    # one adjacent grid move = one time step;
    # one wait/retry action = one time step.
    time_per_grid_move: int = 1
    time_per_wait: int = 1

    # ----------------------------
    # Trace observation settings
    # ----------------------------
    add_gps_noise: bool = False
    gps_noise_std: float = 0.0  # measured in grid-cell units
    record_every: int = 1       # keep every kth simulated time point

    # ----------------------------
    # Trace-based exposure estimator
    # ----------------------------
    exposure_radius: float = 0.0
    stop_min_duration: int = 2
    alpha_smoothing: float = 1.0
    beta_smoothing: float = 1.0

    # ----------------------------
    # Outputs
    # ----------------------------
    output_dir: Path = Path("outputs")
    figures_dir: Path = Path("outputs/figures")
    data_dir: Path = Path("outputs/data")

    def __post_init__(self) -> None:
        self._validate()

        if self.eta_D is None:
            # Avoid division by zero for degenerate grids.
            self.eta_D = 0.0 if self.L <= 1 else __import__("math").log(4) / (self.L - 1)

    @property
    def city_center(self) -> tuple[float, float]:
        """Center in 0-indexed Python coordinates."""
        return ((self.L - 1) / 2.0, (self.L - 1) / 2.0)

    @property
    def total_trips(self) -> int:
        return self.n_vehicles * self.trips_per_vehicle

    def ensure_output_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _validate(self) -> None:
        if self.L <= 0:
            raise ValueError("L must be positive.")

        if not (0.0 <= self.parking_probability <= 1.0):
            raise ValueError("parking_probability must be in [0, 1].")

        if not (0.0 <= self.theta_low <= self.theta_high <= 1.0):
            raise ValueError("Require 0 <= theta_low <= theta_high <= 1.")

        if self.lambda_choice < 0:
            raise ValueError("lambda_choice must be nonnegative.")

        if not (0.0 <= self.rho_stay <= 1.0):
            raise ValueError("rho_stay must be in [0, 1].")

        if self.lambda_low < 0 or self.lambda_high < self.lambda_low:
            raise ValueError("Require 0 <= lambda_low <= lambda_high.")

        if not (0.0 <= self.rho_low <= self.rho_high <= 1.0):
            raise ValueError("Require 0 <= rho_low <= rho_high <= 1.")

        if self.max_attempts_per_trip <= 0:
            raise ValueError("max_attempts_per_trip must be positive.")

        if self.n_vehicles <= 0:
            raise ValueError("n_vehicles must be positive.")

        if self.trips_per_vehicle <= 0:
            raise ValueError("trips_per_vehicle must be positive.")

        if self.dwell_time < 0:
            raise ValueError("dwell_time must be nonnegative.")

        if self.time_per_grid_move <= 0:
            raise ValueError("time_per_grid_move must be positive.")

        if self.time_per_wait <= 0:
            raise ValueError("time_per_wait must be positive.")

        if self.gps_noise_std < 0:
            raise ValueError("gps_noise_std must be nonnegative.")

        if self.record_every <= 0:
            raise ValueError("record_every must be positive.")

        if self.exposure_radius < 0:
            raise ValueError("exposure_radius must be nonnegative.")

        if self.stop_min_duration < 0:
            raise ValueError("stop_min_duration must be nonnegative.")

        if self.alpha_smoothing < 0 or self.beta_smoothing < 0:
            raise ValueError("Smoothing parameters must be nonnegative.")


def default_config() -> SimConfig:
    """Default 10x10 baseline experiment."""
    return SimConfig()


def ring_config() -> SimConfig:
    """Larger 30x30 ring-based parking experiment."""
    return SimConfig(
        L=30,
        use_ring_parking=True,
        parking_probability=0.20,  # fallback only
        n_vehicles=200,
        trips_per_vehicle=25,
    )
