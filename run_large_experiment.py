# run_large_experiment.py
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from config import SimConfig
from grid import generate_grid, grid_summary
from demand import build_destination_distribution, demand_summary
from driver_behavior import build_parking_choice_distribution, choice_summary
from simulator import (
    generate_theta,
    simulate_all_vehicles,
    simulation_summary,
)
from estimators import (
    estimate_trace_exposure,
    estimate_stop_confirmed_exposure,
    estimate_oracle_exposure,
)
from evaluation import evaluate_many, format_evaluation_table
from plotting import (
    plot_grid,
    plot_destination_density,
    plot_theta_map,
    plot_estimate_map,
    plot_error_map,
    plot_simulation_overview,
    plot_sampled_trips_to_folder,
    plot_theta_calibration,
    plot_estimator_comparison,
    plot_exposure_success_counts,
    plot_trace_length_histogram,
)


def export_data(config, grid, demand, theta, result, estimates, evals) -> None:
    """
    Export simulation inputs, observed traces, hidden truth labels, estimates,
    driver-level parameters, and evaluation metrics to config.data_dir.
    """
    config.data_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    config_dict: dict[str, Any] = asdict(config)
    config_dict["output_dir"] = str(config.output_dir)
    config_dict["figures_dir"] = str(config.figures_dir)
    config_dict["data_dir"] = str(config.data_dir)

    with open(config.data_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # ------------------------------------------------------------------
    # Driver-level behavior parameters
    # ------------------------------------------------------------------
    with open(config.data_dir / "vehicle_parameters.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["vehicle_id", "lambda_choice", "rho_stay"])
        for vehicle_id in range(len(result.lambda_by_vehicle)):
            writer.writerow(
                [
                    vehicle_id,
                    result.lambda_by_vehicle[vehicle_id],
                    result.rho_by_vehicle[vehicle_id],
                ]
            )

    # ------------------------------------------------------------------
    # Parking cells and true theta
    # ------------------------------------------------------------------
    with open(config.data_dir / "parking_cells.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["parking_index", "row", "col", "theta_true"])
        for j, (r, c) in enumerate(grid.parking_cells):
            writer.writerow([j, r, c, theta[j]])

    with open(config.data_dir / "theta_true.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["parking_index", "theta_true"])
        for j in range(grid.K):
            writer.writerow([j, theta[j]])

    # ------------------------------------------------------------------
    # Destination cells and demand probabilities
    # ------------------------------------------------------------------
    with open(config.data_dir / "destination_cells.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "destination_index",
                "row",
                "col",
                "demand_probability",
                "demand_weight",
            ]
        )
        for i, (r, c) in enumerate(grid.destination_cells):
            writer.writerow([i, r, c, demand.probabilities[i], demand.weights[i]])

    # ------------------------------------------------------------------
    # Observed trip metadata
    # ------------------------------------------------------------------
    with open(config.data_dir / "trip_observed.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "trip_id",
                "vehicle_id",
                "start_parking_O",
                "destination_I",
                "start_time",
                "end_time",
                "n_trace_points",
            ]
        )
        for trip in result.trips:
            obs = trip.observed
            writer.writerow(
                [
                    obs.trip_id,
                    obs.vehicle_id,
                    obs.O,
                    obs.I,
                    obs.start_time,
                    obs.end_time,
                    len(obs.Gamma),
                ]
            )

    # ------------------------------------------------------------------
    # Trace points in long format
    # ------------------------------------------------------------------
    with open(config.data_dir / "trace_points.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trip_id", "vehicle_id", "m", "x", "y", "t"])
        for trip in result.trips:
            obs = trip.observed
            for m, (x, y, t) in enumerate(obs.Gamma):
                writer.writerow([obs.trip_id, obs.vehicle_id, m, x, y, t])

    # ------------------------------------------------------------------
    # Hidden trip-level truth
    # ------------------------------------------------------------------
    with open(config.data_dir / "trip_truth.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "trip_id",
                "vehicle_id",
                "start_parking_O",
                "destination_I",
                "ending_parking_E",
                "success_attempt_number",
            ]
        )
        for trip in result.trips:
            truth = trip.truth
            writer.writerow(
                [
                    truth.trip_id,
                    truth.vehicle_id,
                    truth.O,
                    truth.I,
                    truth.E,
                    truth.success_attempt_number,
                ]
            )

    # ------------------------------------------------------------------
    # Hidden attempt-level truth
    # ------------------------------------------------------------------
    with open(config.data_dir / "attempts_truth.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "trip_id",
                "vehicle_id",
                "attempt_number",
                "planned_index",
                "actual_index",
                "success",
                "stopped_early",
                "early_eligible_indices",
            ]
        )
        for trip in result.trips:
            truth = trip.truth
            for k, attempt in enumerate(truth.attempts, start=1):
                writer.writerow(
                    [
                        truth.trip_id,
                        truth.vehicle_id,
                        k,
                        attempt.planned_index,
                        attempt.actual_index,
                        attempt.success,
                        attempt.stopped_early,
                        " ".join(map(str, attempt.early_eligible_indices)),
                    ]
                )

    # ------------------------------------------------------------------
    # Hidden availability draws
    # ------------------------------------------------------------------
    with open(config.data_dir / "availability_draws_truth.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "trip_id",
                "vehicle_id",
                "attempt_number",
                "draw_number",
                "parking_index",
                "success",
                "is_planned_cell",
            ]
        )
        for trip in result.trips:
            truth = trip.truth
            for k, attempt in enumerate(truth.attempts, start=1):
                for d, draw in enumerate(attempt.availability_draws, start=1):
                    writer.writerow(
                        [
                            truth.trip_id,
                            truth.vehicle_id,
                            k,
                            d,
                            draw.parking_index,
                            draw.success,
                            draw.is_planned_cell,
                        ]
                    )

    # ------------------------------------------------------------------
    # Estimates by parking cell
    # ------------------------------------------------------------------
    with open(config.data_dir / "estimates.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "estimator",
                "parking_index",
                "theta_hat",
                "successes",
                "exposure",
            ]
        )
        for est in estimates:
            for j in range(grid.K):
                writer.writerow(
                    [
                        est.name,
                        j,
                        est.theta_hat[j],
                        est.successes[j],
                        est.exposure[j],
                    ]
                )

    # ------------------------------------------------------------------
    # Evaluation metrics
    # ------------------------------------------------------------------
    with open(config.data_dir / "evaluation.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["estimator", "mae", "rmse", "corr", "n_used", "notes"])
        for ev in evals:
            writer.writerow([ev.name, ev.mae, ev.rmse, ev.corr, ev.n_used, ev.notes])


def main():
    # Larger ring-based city with driver-level lambda/rho heterogeneity.
    config = SimConfig(
        seed=0,
        L=30,
        use_ring_parking=True,
        parking_probability=0.20,  # fallback only; ring_specs determine ring probabilities
        n_vehicles=500,
        trips_per_vehicle=10,
        dwell_time=30,
        lambda_choice=0.8,
        rho_stay=0.5,
        heterogeneous_drivers=True,
        lambda_low=0.4,
        lambda_high=1.2,
        rho_low=0.2,
        rho_high=0.8,
        max_attempts_per_trip=50,
        theta_low=0.2,
        theta_high=0.8,
        exposure_radius=0.0,
        alpha_smoothing=1.0,
        beta_smoothing=1.0,
        output_dir=Path("outputs_ring"),
        figures_dir=Path("outputs_ring/figures"),
        data_dir=Path("outputs_ring/data"),
    )
    config.ensure_output_dirs()

    rng = np.random.default_rng(config.seed)

    # ------------------------------------------------------------------
    # Build environment
    # ------------------------------------------------------------------
    grid = generate_grid(config, rng)
    print(grid_summary(grid))

    demand = build_destination_distribution(grid, config)
    print(demand_summary(demand))

    # This common choice distribution is still useful for summary/debugging.
    # If heterogeneous_drivers=True, the simulator samples choices using each
    # vehicle's lambda_v instead.
    choice = build_parking_choice_distribution(grid, config)
    print(choice_summary(choice))

    theta = generate_theta(grid, config, rng)
    print("theta shape:", theta.shape)
    print("theta range:", float(theta.min()), float(theta.max()))

    # ------------------------------------------------------------------
    # Simulate
    # ------------------------------------------------------------------
    result = simulate_all_vehicles(
        grid=grid,
        demand=demand,
        choice=choice,
        theta=theta,
        config=config,
        rng=rng,
    )
    print(simulation_summary(result))

    # ------------------------------------------------------------------
    # Basic sanity checks
    # ------------------------------------------------------------------
    assert len(result.trips) == config.n_vehicles * config.trips_per_vehicle
    assert result.theta.shape == (grid.K,)
    assert result.lambda_by_vehicle.shape == (config.n_vehicles,)
    assert result.rho_by_vehicle.shape == (config.n_vehicles,)

    for trip in result.trips:
        obs = trip.observed
        truth = trip.truth

        assert 0 <= obs.O < grid.K
        assert 0 <= obs.I < grid.M
        assert len(obs.Gamma) > 0
        assert truth.E == truth.attempts[-1].actual_index
        assert truth.attempts[-1].success is True
        assert obs.Gamma[0][2] == obs.start_time
        assert obs.Gamma[-1][2] == obs.end_time

    print("Basic sanity checks passed.")

    # ------------------------------------------------------------------
    # Estimators
    # ------------------------------------------------------------------
    est_trace = estimate_trace_exposure(result.trips, grid, config)
    est_stop = estimate_stop_confirmed_exposure(result.trips, grid, config)
    est_oracle = estimate_oracle_exposure(result.trips, grid, config)

    # Keep this order everywhere: trace exposure, stop-confirmed, oracle.
    estimates = [est_trace, est_stop, est_oracle]
    evals = evaluate_many(theta, estimates)

    print(format_evaluation_table(evals))

    # ------------------------------------------------------------------
    # Export data
    # ------------------------------------------------------------------
    export_data(
        config=config,
        grid=grid,
        demand=demand,
        theta=theta,
        result=result,
        estimates=estimates,
        evals=evals,
    )
    print(f"Saved data to {config.data_dir}")

    # ------------------------------------------------------------------
    # Paper figures
    # ------------------------------------------------------------------

    # Figure 1. Simulation setup:
    # parking cells, destination density, and true theta.
    plot_simulation_overview(
        result,
        save_path=config.figures_dir / "fig1_simulation_setup.png",
    )

    # Figure 2. Main estimator performance:
    # stop-confirmed calibration, then MAE comparison.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))

    plot_theta_calibration(
        theta,
        est_stop,
        ax=axes[0],
        title="Stop-confirmed calibration",
    )

    plot_estimator_comparison(
        evals,
        metric="mae",
        ax=axes[1],
        title="Estimator comparison (MAE)",
    )

    fig.tight_layout()
    fig.savefig(
        config.figures_dir / "fig2_main_estimator_performance.png",
        bbox_inches="tight",
        dpi=200,
    )
    plt.close(fig)

    # Figure 3. Spatial stop-confirmed results:
    # true theta, stop-confirmed estimate, and absolute error.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    plot_theta_map(
        grid,
        theta,
        ax=axes[0],
        title=r"True $\theta_j$",
    )

    plot_estimate_map(
        grid,
        est_stop,
        ax=axes[1],
        title=r"Stop-confirmed $\widehat{\theta}_j$",
    )

    plot_error_map(
        grid,
        theta_true=theta,
        theta_hat=est_stop.theta_hat,
        ax=axes[2],
        title="Absolute error",
    )

    fig.tight_layout()
    fig.savefig(
        config.figures_dir / "fig3_stop_confirmed_spatial_results.png",
        bbox_inches="tight",
        dpi=200,
    )
    plt.close(fig)

    print(
        "Saved paper figures to:\n"
        f"  {config.figures_dir / 'fig1_simulation_setup.png'}\n"
        f"  {config.figures_dir / 'fig2_main_estimator_performance.png'}\n"
        f"  {config.figures_dir / 'fig3_stop_confirmed_spatial_results.png'}"
    )

    # ------------------------------------------------------------------
    # Diagnostic plots
    # ------------------------------------------------------------------

    # 1. Parking / grid map
    plot_grid(
        grid,
        title="Ring-based parking grid",
        save_path=config.figures_dir / "grid_map.png",
    )

    # 2. Destination density
    plot_destination_density(
        grid,
        demand,
        title="Destination density",
        save_path=config.figures_dir / "destination_density.png",
    )

    # 3. True theta map
    plot_theta_map(
        grid,
        theta,
        title="True parking-success probabilities",
        save_path=config.figures_dir / "theta_map.png",
    )

    # 4. Compact overview panel
    plot_simulation_overview(
        result,
        save_path=config.figures_dir / "simulation_overview.png",
    )

    # 5. Sampled trip traces
    sampled_trip_paths = plot_sampled_trips_to_folder(
        grid=grid,
        trips=result.trips,
        folder=config.figures_dir / "sampled_trip_traces",
        n_sample=20,
        seed=config.seed,
    )
    print(
        f"Saved {len(sampled_trip_paths)} sampled trip plots to "
        f"{config.figures_dir / 'sampled_trip_traces'}"
    )

    # 6. Calibration plots, ordered trace, stop, oracle.
    plot_theta_calibration(
        theta,
        est_trace,
        save_path=config.figures_dir / "calibration_trace_exposure.png",
    )

    plot_theta_calibration(
        theta,
        est_stop,
        save_path=config.figures_dir / "calibration_stop_confirmed.png",
    )

    plot_theta_calibration(
        theta,
        est_oracle,
        save_path=config.figures_dir / "calibration_oracle.png",
    )

    # 7. Spatial estimate maps, ordered trace, stop, oracle.
    plot_estimate_map(
        grid,
        est_trace,
        title="Trace-exposure estimated probabilities",
        save_path=config.figures_dir / "trace_exposure_estimate_map.png",
    )

    plot_estimate_map(
        grid,
        est_stop,
        title="Stop-confirmed estimated probabilities",
        save_path=config.figures_dir / "stop_confirmed_estimate_map.png",
    )

    plot_estimate_map(
        grid,
        est_oracle,
        title="Oracle-exposure estimated probabilities",
        save_path=config.figures_dir / "oracle_exposure_estimate_map.png",
    )

    # 8. Spatial absolute-error maps, ordered trace, stop, oracle.
    plot_error_map(
        grid,
        theta_true=theta,
        theta_hat=est_trace.theta_hat,
        title="Trace-exposure absolute estimation error",
        save_path=config.figures_dir / "trace_exposure_error_map.png",
    )

    plot_error_map(
        grid,
        theta_true=theta,
        theta_hat=est_stop.theta_hat,
        title="Stop-confirmed absolute estimation error",
        save_path=config.figures_dir / "stop_confirmed_error_map.png",
    )

    plot_error_map(
        grid,
        theta_true=theta,
        theta_hat=est_oracle.theta_hat,
        title="Oracle-exposure absolute estimation error",
        save_path=config.figures_dir / "oracle_exposure_error_map.png",
    )

    # 9. Estimator comparison. Bar order follows evals, i.e. trace, stop, oracle.
    plot_estimator_comparison(
        evals,
        metric="mae",
        save_path=config.figures_dir / "estimator_mae_comparison.png",
    )

    plot_estimator_comparison(
        evals,
        metric="rmse",
        save_path=config.figures_dir / "estimator_rmse_comparison.png",
    )

    plot_estimator_comparison(
        evals,
        metric="corr",
        save_path=config.figures_dir / "estimator_corr_comparison.png",
    )

    # 10. Exposure / success counts, ordered trace, stop, oracle.
    plot_exposure_success_counts(
        est_trace,
        title="Trace-exposure exposure and successes",
        save_path=config.figures_dir / "trace_exposure_success_counts.png",
    )

    plot_exposure_success_counts(
        est_stop,
        title="Stop-confirmed exposure and successes",
        save_path=config.figures_dir / "stop_confirmed_exposure_success_counts.png",
    )

    plot_exposure_success_counts(
        est_oracle,
        title="Oracle-exposure exposure and successes",
        save_path=config.figures_dir / "oracle_exposure_success_counts.png",
    )

    # 11. Trace length histogram
    plot_trace_length_histogram(
        result.trips,
        save_path=config.figures_dir / "trace_length_histogram.png",
    )

    print(f"Saved all figures to {config.figures_dir}")


if __name__ == "__main__":
    main()