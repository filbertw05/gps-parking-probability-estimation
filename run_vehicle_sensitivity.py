# run_vehicle_sensitivity.py
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from config import SimConfig
from grid import generate_grid
from demand import build_destination_distribution
from driver_behavior import build_parking_choice_distribution
from simulator import generate_theta, simulate_all_vehicles
from estimators import (
    estimate_trace_exposure,
    estimate_stop_confirmed_exposure,
    estimate_oracle_exposure,
)
from evaluation import evaluate_many


ESTIMATOR_ORDER = ["trace_exposure", "stop_confirmed", "oracle_exposure"]


def make_base_config(
    *,
    seed: int,
    n_vehicles: int,
    trips_per_vehicle: int,
    output_dir: Path,
) -> SimConfig:
    """
    Baseline config matching the main experiment, except that n_vehicles
    is varied by the sensitivity analysis.
    """
    return SimConfig(
        seed=seed,
        L=30,
        use_ring_parking=True,
        parking_probability=0.20,  # fallback only; ring_specs determine ring probabilities
        n_vehicles=n_vehicles,
        trips_per_vehicle=trips_per_vehicle,
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
        output_dir=output_dir,
        figures_dir=output_dir / "figures",
        data_dir=output_dir / "data",
    )


def run_one_experiment(
    *,
    config: SimConfig,
    grid,
    demand,
    choice,
    theta: np.ndarray,
    sim_seed: int,
) -> list[dict]:
    """
    Run one simulation at a fixed n_vehicles and evaluate all estimators.
    The grid, demand, and theta are passed in so that each replication can
    reuse the same environment across different n_vehicles values.
    """
    rng = np.random.default_rng(sim_seed)

    result = simulate_all_vehicles(
        grid=grid,
        demand=demand,
        choice=choice,
        theta=theta,
        config=config,
        rng=rng,
    )

    est_trace = estimate_trace_exposure(result.trips, grid, config)
    est_stop = estimate_stop_confirmed_exposure(result.trips, grid, config)
    est_oracle = estimate_oracle_exposure(result.trips, grid, config)

    estimates = [est_trace, est_stop, est_oracle]
    evals = evaluate_many(theta, estimates)

    rows = []
    for ev in evals:
        rows.append(
            {
                "estimator": ev.name,
                "mae": float(ev.mae),
                "rmse": float(ev.rmse),
                "corr": float(ev.corr),
                "n_used": int(ev.n_used),
            }
        )
    return rows


def run_sensitivity(
    *,
    n_vehicle_values: list[int],
    trips_per_vehicle: int,
    n_reps: int,
    base_seed: int,
    output_dir: Path,
) -> list[dict]:
    """
    Run sensitivity analysis over n_vehicles.

    For each replication, we generate one fixed environment
    (grid, demand distribution, true theta), then vary n_vehicles.
    This makes the comparison across n_vehicles cleaner because
    the true city and true theta map are held fixed within a replication.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    for rep in range(n_reps):
        env_seed = base_seed + 10_000 * rep
        env_config = make_base_config(
            seed=env_seed,
            n_vehicles=max(n_vehicle_values),
            trips_per_vehicle=trips_per_vehicle,
            output_dir=output_dir,
        )

        env_rng = np.random.default_rng(env_seed)

        grid = generate_grid(env_config, env_rng)
        demand = build_destination_distribution(grid, env_config)
        choice = build_parking_choice_distribution(grid, env_config)
        theta = generate_theta(grid, env_config, env_rng)

        print(
            f"\nReplication {rep + 1}/{n_reps}: "
            f"K={grid.K} parking cells, M={grid.M} destination cells"
        )

        for n_vehicles in n_vehicle_values:
            sim_seed = base_seed + 10_000 * rep + n_vehicles
            config = replace(
                env_config,
                seed=sim_seed,
                n_vehicles=n_vehicles,
                trips_per_vehicle=trips_per_vehicle,
            )

            print(
                f"  running n_vehicles={n_vehicles}, "
                f"total_trips={n_vehicles * trips_per_vehicle}..."
            )

            eval_rows = run_one_experiment(
                config=config,
                grid=grid,
                demand=demand,
                choice=choice,
                theta=theta,
                sim_seed=sim_seed,
            )

            for row in eval_rows:
                row.update(
                    {
                        "rep": rep,
                        "seed": sim_seed,
                        "n_vehicles": n_vehicles,
                        "trips_per_vehicle": trips_per_vehicle,
                        "total_trips": n_vehicles * trips_per_vehicle,
                        "K": grid.K,
                    }
                )
                all_rows.append(row)

    return all_rows


def write_raw_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "rep",
        "seed",
        "n_vehicles",
        "trips_per_vehicle",
        "total_trips",
        "K",
        "estimator",
        "mae",
        "rmse",
        "corr",
        "n_used",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        key = (row["n_vehicles"], row["total_trips"], row["estimator"])
        grouped[key].append(row)

    summary = []

    for (n_vehicles, total_trips, estimator), group_rows in sorted(grouped.items()):
        out = {
            "n_vehicles": n_vehicles,
            "total_trips": total_trips,
            "estimator": estimator,
            "n_reps": len(group_rows),
        }

        for metric in ["mae", "rmse", "corr"]:
            vals = np.array([r[metric] for r in group_rows], dtype=float)
            vals = vals[np.isfinite(vals)]

            if len(vals) == 0:
                out[f"{metric}_mean"] = np.nan
                out[f"{metric}_std"] = np.nan
            else:
                out[f"{metric}_mean"] = float(np.mean(vals))
                out[f"{metric}_std"] = float(np.std(vals, ddof=1 if len(vals) > 1 else 0))

        summary.append(out)

    return summary


def write_summary_csv(summary: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "n_vehicles",
        "total_trips",
        "estimator",
        "n_reps",
        "mae_mean",
        "mae_std",
        "rmse_mean",
        "rmse_std",
        "corr_mean",
        "corr_std",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def _summary_lookup(summary: list[dict]):
    lookup = {}
    for row in summary:
        lookup[(row["n_vehicles"], row["estimator"])] = row
    return lookup


def plot_metric_vs_nvehicles(
    summary: list[dict],
    *,
    metric: str,
    ylabel: str,
    title: str,
    save_path: Path,
) -> None:
    """
    Plot a metric against n_vehicles for all estimators.
    """
    n_values = sorted({row["n_vehicles"] for row in summary})
    lookup = _summary_lookup(summary)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for estimator in ESTIMATOR_ORDER:
        means = []
        stds = []

        for n in n_values:
            row = lookup.get((n, estimator))
            if row is None:
                means.append(np.nan)
                stds.append(np.nan)
            else:
                means.append(row[f"{metric}_mean"])
                stds.append(row[f"{metric}_std"])

        ax.errorbar(
            n_values,
            means,
            yerr=stds,
            marker="o",
            capsize=3,
            label=estimator,
        )

    ax.set_xscale("log")
    ax.set_xticks(n_values)
    ax.set_xticklabels([str(n) for n in n_values])
    ax.set_xlabel(r"number of vehicles $N_{\mathrm{veh}}$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    if metric == "corr":
        ax.set_ylim(0.0, 1.0)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def compute_improvement_rows(rows: list[dict]) -> list[dict]:
    """
    For each replication and n_vehicles, compute the percent MAE reduction
    of stop_confirmed relative to trace_exposure.
    """
    grouped = defaultdict(dict)
    for row in rows:
        key = (row["rep"], row["n_vehicles"], row["total_trips"])
        grouped[key][row["estimator"]] = row

    out = []

    for (rep, n_vehicles, total_trips), est_rows in grouped.items():
        if "trace_exposure" not in est_rows or "stop_confirmed" not in est_rows:
            continue

        trace_mae = est_rows["trace_exposure"]["mae"]
        stop_mae = est_rows["stop_confirmed"]["mae"]

        if not np.isfinite(trace_mae) or trace_mae <= 0:
            continue

        improvement_pct = 100.0 * (trace_mae - stop_mae) / trace_mae

        out.append(
            {
                "rep": rep,
                "n_vehicles": n_vehicles,
                "total_trips": total_trips,
                "trace_mae": trace_mae,
                "stop_mae": stop_mae,
                "improvement_pct": improvement_pct,
            }
        )

    return out


def plot_stop_improvement(rows: list[dict], save_path: Path) -> None:
    """
    Plot percent MAE improvement of stop_confirmed over trace_exposure.
    """
    improvement_rows = compute_improvement_rows(rows)

    grouped = defaultdict(list)
    for row in improvement_rows:
        grouped[row["n_vehicles"]].append(row["improvement_pct"])

    n_values = sorted(grouped.keys())
    means = []
    stds = []

    for n in n_values:
        vals = np.array(grouped[n], dtype=float)
        vals = vals[np.isfinite(vals)]
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals, ddof=1 if len(vals) > 1 else 0)))

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.errorbar(
        n_values,
        means,
        yerr=stds,
        marker="o",
        capsize=3,
    )

    ax.axhline(0, linestyle="--", linewidth=1)

    ax.set_xscale("log")
    ax.set_xticks(n_values)
    ax.set_xticklabels([str(n) for n in n_values])
    ax.set_xlabel(r"number of vehicles $N_{\mathrm{veh}}$")
    ax.set_ylabel("MAE reduction vs. trace exposure (%)")
    ax.set_title("Stop-confirmed improvement over trace exposure")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def plot_stop_mae_only(summary: list[dict], save_path: Path) -> None:
    """
    Plot only stop-confirmed MAE, useful for the paper if you want one clean figure.
    """
    n_values = sorted({row["n_vehicles"] for row in summary})
    lookup = _summary_lookup(summary)

    means = []
    stds = []

    for n in n_values:
        row = lookup.get((n, "stop_confirmed"))
        if row is None:
            means.append(np.nan)
            stds.append(np.nan)
        else:
            means.append(row["mae_mean"])
            stds.append(row["mae_std"])

    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    ax.errorbar(
        n_values,
        means,
        yerr=stds,
        marker="o",
        capsize=3,
    )

    ax.set_xscale("log")
    ax.set_xticks(n_values)
    ax.set_xticklabels([str(n) for n in n_values])
    ax.set_xlabel(r"number of vehicles $N_{\mathrm{veh}}$")
    ax.set_ylabel("MAE")
    ax.set_title("Stop-confirmed MAE vs. number of vehicles")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def make_plots(summary: list[dict], rows: list[dict], figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_metric_vs_nvehicles(
        summary,
        metric="mae",
        ylabel="MAE",
        title="MAE vs. number of vehicles",
        save_path=figures_dir / "sensitivity_mae_vs_nvehicles.png",
    )

    plot_metric_vs_nvehicles(
        summary,
        metric="rmse",
        ylabel="RMSE",
        title="RMSE vs. number of vehicles",
        save_path=figures_dir / "sensitivity_rmse_vs_nvehicles.png",
    )

    plot_metric_vs_nvehicles(
        summary,
        metric="corr",
        ylabel="Correlation",
        title="Correlation vs. number of vehicles",
        save_path=figures_dir / "sensitivity_corr_vs_nvehicles.png",
    )

    plot_stop_improvement(
        rows,
        save_path=figures_dir / "sensitivity_stop_improvement_vs_nvehicles.png",
    )

    plot_stop_mae_only(
        summary,
        save_path=figures_dir / "sensitivity_stop_mae_only_vs_nvehicles.png",
    )


def parse_n_values(values: Iterable[str]) -> list[int]:
    out = []
    for v in values:
        out.append(int(v))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-values",
        nargs="+",
        default=["50", "100", "200", "500", "1000", "2000", "5000"],
        help="Vehicle counts to test.",
    )
    parser.add_argument(
        "--trips-per-vehicle",
        type=int,
        default=10,
        help="Number of trips generated per vehicle.",
    )
    parser.add_argument(
        "--n-reps",
        type=int,
        default=3,
        help="Number of replications per vehicle count.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs_ring/sensitivity_nvehicles"),
        help="Output directory for sensitivity results.",
    )

    args = parser.parse_args()

    n_vehicle_values = parse_n_values(args.n_values)

    print("Running vehicle-count sensitivity analysis")
    print("n_vehicle_values:", n_vehicle_values)
    print("trips_per_vehicle:", args.trips_per_vehicle)
    print("n_reps:", args.n_reps)
    print("output_dir:", args.output_dir)

    rows = run_sensitivity(
        n_vehicle_values=n_vehicle_values,
        trips_per_vehicle=args.trips_per_vehicle,
        n_reps=args.n_reps,
        base_seed=args.seed,
        output_dir=args.output_dir,
    )

    raw_csv = args.output_dir / "vehicle_sensitivity_raw.csv"
    summary_csv = args.output_dir / "vehicle_sensitivity_summary.csv"

    write_raw_csv(rows, raw_csv)

    summary = summarize_rows(rows)
    write_summary_csv(summary, summary_csv)

    make_plots(
        summary=summary,
        rows=rows,
        figures_dir=args.output_dir / "figures",
    )

    print("\nSaved:")
    print(f"  raw results:     {raw_csv}")
    print(f"  summary results: {summary_csv}")
    print(f"  figures:         {args.output_dir / 'figures'}")


if __name__ == "__main__":
    main()