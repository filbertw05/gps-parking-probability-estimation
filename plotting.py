# plotting.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap

from demand import DemandData, destination_probability_map
from estimators import EstimateResult
from evaluation import EvaluationResult, calibration_points
from grid import Cell, GridData
from simulator import SimulationResult, TracePoint, TripObserved, TripRecord


PathLike = str | Path


# ---------------------------------------------------------------------
# Plot style constants
# ---------------------------------------------------------------------

GRID_TITLE_FONTSIZE = 14
GRID_AXIS_LABELSIZE = 13
GRID_TICK_LABELSIZE = 7
GRID_CELL_TEXT_FONTSIZE = 5

COLORBAR_LABELSIZE = 12
COLORBAR_TICK_LABELSIZE = 8

GENERAL_TITLE_FONTSIZE = 14
GENERAL_AXIS_LABELSIZE = 13
GENERAL_TICK_LABELSIZE = 8
GENERAL_LEGEND_FONTSIZE = 11


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------


def _cell_xy(cell: Cell) -> tuple[int, int]:
    """
    Convert grid cell (row, col) to plotting coordinates (x, y) = (col, row).
    """
    r, c = cell
    return c, r


def _trace_xy(trace: Sequence[TracePoint]) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert trace points (x=row, y=col, t) to plotting arrays (col, row).
    """
    rows = np.array([pt[0] for pt in trace], dtype=float)
    cols = np.array([pt[1] for pt in trace], dtype=float)
    return cols, rows


def _finish_grid_axes(ax: Axes, grid: GridData) -> None:
    ax.set_xlim(-0.5, grid.L - 0.5)
    ax.set_ylim(grid.L - 0.5, -0.5)
    ax.set_aspect("equal")

    ax.set_xticks(np.arange(grid.L))
    ax.set_yticks(np.arange(grid.L))
    ax.set_xticks(np.arange(-0.5, grid.L, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.L, 1), minor=True)
    ax.grid(which="minor", linewidth=0.5, alpha=0.4)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_xlabel("column", fontsize=GRID_AXIS_LABELSIZE)
    ax.set_ylabel("row", fontsize=GRID_AXIS_LABELSIZE)
    ax.tick_params(axis="both", which="major", labelsize=GRID_TICK_LABELSIZE)


def _style_colorbar(cbar) -> None:
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_LABELSIZE)
    cbar.ax.yaxis.label.set_size(COLORBAR_LABELSIZE)


def _style_general_axes(ax: Axes) -> None:
    ax.tick_params(axis="both", which="major", labelsize=GENERAL_TICK_LABELSIZE)
    ax.xaxis.label.set_size(GENERAL_AXIS_LABELSIZE)
    ax.yaxis.label.set_size(GENERAL_AXIS_LABELSIZE)


def _get_fig_ax(ax: Optional[Axes], figsize: tuple[float, float]) -> tuple[plt.Figure, Axes]:
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _maybe_save(fig: plt.Figure, save_path: Optional[PathLike]) -> None:
    if save_path is None:
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=200)


def _get_observed(trip: TripRecord | TripObserved) -> TripObserved:
    if isinstance(trip, TripObserved):
        return trip
    return trip.observed


# ---------------------------------------------------------------------
# Grid / map plots
# ---------------------------------------------------------------------


def plot_grid(
    grid: GridData,
    *,
    ax: Optional[Axes] = None,
    title: str = "Simulated grid",
    show_indices: bool = False,
    save_path: Optional[PathLike] = None,
) -> tuple[plt.Figure, Axes]:
    """
    Plot parking vs non-parking cells.
    """
    fig, ax = _get_fig_ax(ax, figsize=(6, 6))

    arr = grid.is_parking.astype(int)
    cmap = ListedColormap(["white", "lightgray"])
    ax.imshow(arr, origin="upper", cmap=cmap, vmin=0, vmax=1)

    ax.set_title(title, fontsize=GRID_TITLE_FONTSIZE)
    _finish_grid_axes(ax, grid)
    _maybe_save(fig, save_path)
    return fig, ax


def plot_destination_density(
    grid: GridData,
    demand: DemandData,
    *,
    ax: Optional[Axes] = None,
    title: str = "Destination density",
    save_path: Optional[PathLike] = None,
) -> tuple[plt.Figure, Axes]:
    """
    Plot destination probabilities over non-parking cells.
    """
    fig, ax = _get_fig_ax(ax, figsize=(6, 6))

    prob_map = destination_probability_map(grid, demand, fill_value=np.nan)
    im = ax.imshow(prob_map, origin="upper", cmap="Oranges")

    for cell in grid.parking_cells:
        x, y = _cell_xy(cell)
        ax.scatter(
            x,
            y,
            marker="s",
            s=35,
            facecolors="none",
            edgecolors="black",
            linewidth=0.7,
        )

    ax.set_title(title, fontsize=GRID_TITLE_FONTSIZE)
    _finish_grid_axes(ax, grid)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="probability")
    _style_colorbar(cbar)

    _maybe_save(fig, save_path)
    return fig, ax


def plot_theta_map(
    grid: GridData,
    theta: np.ndarray,
    *,
    ax: Optional[Axes] = None,
    title: str = "True parking-success probabilities",
    save_path: Optional[PathLike] = None,
) -> tuple[plt.Figure, Axes]:
    """
    Plot true theta_j on parking cells.
    """
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (grid.K,):
        raise ValueError(f"theta must have shape ({grid.K},).")

    fig, ax = _get_fig_ax(ax, figsize=(6, 6))

    theta_map = np.full((grid.L, grid.L), np.nan, dtype=float)
    for j, cell in enumerate(grid.parking_cells):
        theta_map[cell] = theta[j]

    im = ax.imshow(theta_map, origin="upper", vmin=0, vmax=1)

    ax.set_title(title, fontsize=GRID_TITLE_FONTSIZE)
    _finish_grid_axes(ax, grid)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$\theta_j$")
    _style_colorbar(cbar)

    _maybe_save(fig, save_path)
    return fig, ax


def plot_estimate_map(
    grid: GridData,
    estimate: EstimateResult | np.ndarray,
    *,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    save_path: Optional[PathLike] = None,
) -> tuple[plt.Figure, Axes]:
    """
    Plot estimated theta_hat_j on parking cells.
    """
    if isinstance(estimate, EstimateResult):
        theta_hat = np.asarray(estimate.theta_hat, dtype=float)
        if title is None:
            title = f"Estimated probabilities: {estimate.name}"
    else:
        theta_hat = np.asarray(estimate, dtype=float)
        if title is None:
            title = "Estimated parking-success probabilities"

    if theta_hat.shape != (grid.K,):
        raise ValueError(f"theta_hat must have shape ({grid.K},).")

    fig, ax = _get_fig_ax(ax, figsize=(6, 6))

    est_map = np.full((grid.L, grid.L), np.nan, dtype=float)
    for j, cell in enumerate(grid.parking_cells):
        est_map[cell] = theta_hat[j]

    im = ax.imshow(est_map, origin="upper", vmin=0, vmax=1)

    ax.set_title(title, fontsize=GRID_TITLE_FONTSIZE)
    _finish_grid_axes(ax, grid)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$\widehat{\theta}_j$")
    _style_colorbar(cbar)

    _maybe_save(fig, save_path)
    return fig, ax


def plot_error_map(
    grid: GridData,
    theta_true: np.ndarray,
    theta_hat: np.ndarray,
    *,
    ax: Optional[Axes] = None,
    title: str = "Absolute estimation error",
    save_path: Optional[PathLike] = None,
) -> tuple[plt.Figure, Axes]:
    """
    Plot |theta_hat_j - theta_j| on parking cells.
    """
    theta_true = np.asarray(theta_true, dtype=float)
    theta_hat = np.asarray(theta_hat, dtype=float)

    if theta_true.shape != (grid.K,) or theta_hat.shape != (grid.K,):
        raise ValueError(f"theta_true and theta_hat must both have shape ({grid.K},).")

    err = np.abs(theta_hat - theta_true)

    fig, ax = _get_fig_ax(ax, figsize=(6, 6))

    err_map = np.full((grid.L, grid.L), np.nan, dtype=float)
    for j, cell in enumerate(grid.parking_cells):
        err_map[cell] = err[j]

    vmax = np.nanmax(err) if np.any(np.isfinite(err)) else 1.0
    im = ax.imshow(err_map, origin="upper", vmin=0, vmax=vmax, cmap="magma")

    ax.set_title(title, fontsize=GRID_TITLE_FONTSIZE)
    _finish_grid_axes(ax, grid)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$|\widehat{\theta}_j-\theta_j|$")
    _style_colorbar(cbar)

    _maybe_save(fig, save_path)
    return fig, ax


# ---------------------------------------------------------------------
# Trace plots
# ---------------------------------------------------------------------


def plot_trip_trace(
    grid: GridData,
    trip: TripRecord | TripObserved,
    *,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    show_grid: bool = True,
    show_parking_cells: bool = True,
    save_path: Optional[PathLike] = None,
) -> tuple[plt.Figure, Axes]:
    """
    Plot one trip-level trace.

    Marks:
        start parking cell p_O
        desired destination g_I
        ending parking cell p_E if TripRecord truth is available
    """
    obs = _get_observed(trip)
    fig, ax = _get_fig_ax(ax, figsize=(6, 6))

    if show_grid:
        background = np.zeros((grid.L, grid.L))
        ax.imshow(background, origin="upper", alpha=0.05)

    if show_parking_cells:
        for cell in grid.parking_cells:
            x, y = _cell_xy(cell)
            ax.scatter(
                x,
                y,
                marker="s",
                s=35,
                facecolors="none",
                edgecolors="black",
                linewidth=0.7,
            )

    cols, rows = _trace_xy(obs.Gamma)
    ax.plot(cols, rows, marker="o", markersize=3, linewidth=1.5, label="trace")

    start_cell = grid.parking_cell(obs.O)
    sx, sy = _cell_xy(start_cell)
    ax.scatter(sx, sy, marker="o", s=90, label="start parking")

    dest_cell = grid.destination_cell(obs.I)
    dx, dy = _cell_xy(dest_cell)
    ax.scatter(dx, dy, marker="*", s=140, label="destination")

    if isinstance(trip, TripRecord):
        end_cell = grid.parking_cell(trip.truth.E)
        ex, ey = _cell_xy(end_cell)
        ax.scatter(ex, ey, marker="X", s=100, label="ending parking")

    if title is None:
        title = f"Trip {obs.trip_id}, vehicle {obs.vehicle_id}"
    ax.set_title(title, fontsize=GRID_TITLE_FONTSIZE)

    _finish_grid_axes(ax, grid)
    ax.legend(loc="best", fontsize=GENERAL_LEGEND_FONTSIZE)
    _maybe_save(fig, save_path)
    return fig, ax


def plot_vehicle_trace(
    grid: GridData,
    vehicle_trace: Sequence[TracePoint],
    *,
    ax: Optional[Axes] = None,
    title: str = "Continuous vehicle trace",
    show_parking_cells: bool = True,
    save_path: Optional[PathLike] = None,
) -> tuple[plt.Figure, Axes]:
    """
    Plot a continuous vehicle trace made of multiple trips.
    """
    fig, ax = _get_fig_ax(ax, figsize=(6, 6))

    if show_parking_cells:
        for cell in grid.parking_cells:
            x, y = _cell_xy(cell)
            ax.scatter(
                x,
                y,
                marker="s",
                s=25,
                facecolors="none",
                edgecolors="black",
                linewidth=0.5,
            )

    cols, rows = _trace_xy(vehicle_trace)
    ax.plot(cols, rows, marker="o", markersize=2, linewidth=1.0)

    ax.set_title(title, fontsize=GRID_TITLE_FONTSIZE)
    _finish_grid_axes(ax, grid)
    _maybe_save(fig, save_path)
    return fig, ax


def plot_multiple_trip_traces(
    grid: GridData,
    trips: Sequence[TripRecord | TripObserved],
    *,
    max_trips: int = 5,
    save_path: Optional[PathLike] = None,
) -> plt.Figure:
    """
    Plot several trip traces in separate small panels.
    """
    n = min(max_trips, len(trips))
    if n == 0:
        raise ValueError("No trips to plot.")

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    axes_flat = axes.ravel()

    for idx in range(n):
        plot_trip_trace(
            grid,
            trips[idx],
            ax=axes_flat[idx],
            title=f"Trip {_get_observed(trips[idx]).trip_id}",
            show_grid=True,
            show_parking_cells=True,
        )

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_sampled_trips_to_folder(
    grid: GridData,
    trips: Sequence[TripRecord | TripObserved],
    folder: PathLike,
    *,
    n_sample: int = 100,
    rng: Optional[np.random.Generator] = None,
    seed: int = 0,
    prefix: str = "trip_sample",
    show_grid: bool = True,
    show_parking_cells: bool = True,
) -> list[Path]:
    """
    Save a random sample of trip plots to a folder.

    Parameters
    ----------
    grid:
        GridData object.
    trips:
        Sequence of TripRecord or TripObserved objects.
    folder:
        Output folder.
    n_sample:
        Number of trips to plot. If n_sample exceeds len(trips), all trips are plotted.
    rng:
        Optional NumPy random generator. If None, a generator is created from seed.
    seed:
        Seed used only when rng is None.
    prefix:
        Filename prefix.
    show_grid:
        Whether to show grid background.
    show_parking_cells:
        Whether to show parking cells.

    Returns
    -------
    saved_paths:
        List of saved image paths.
    """
    if len(trips) == 0:
        raise ValueError("No trips to plot.")
    if n_sample <= 0:
        raise ValueError("n_sample must be positive.")

    if rng is None:
        rng = np.random.default_rng(seed)

    outdir = Path(folder)
    outdir.mkdir(parents=True, exist_ok=True)

    n_total = len(trips)
    n_draw = min(n_sample, n_total)
    sampled_indices = rng.choice(n_total, size=n_draw, replace=False)

    saved_paths: list[Path] = []

    for sample_rank, trip_idx in enumerate(sampled_indices):
        trip = trips[int(trip_idx)]
        obs = _get_observed(trip)

        save_path = outdir / (
            f"{prefix}_{sample_rank:04d}_"
            f"trip_{obs.trip_id:05d}_"
            f"vehicle_{obs.vehicle_id:04d}.png"
        )

        fig, _ax = plot_trip_trace(
            grid,
            trip,
            title=f"Sample {sample_rank}: trip {obs.trip_id}, vehicle {obs.vehicle_id}",
            show_grid=show_grid,
            show_parking_cells=show_parking_cells,
            save_path=save_path,
        )
        plt.close(fig)

        saved_paths.append(save_path)

    return saved_paths


def plot_all_trips_to_folder(
    grid: GridData,
    trips: Sequence[TripRecord | TripObserved],
    folder: PathLike,
    *,
    max_trips: Optional[int] = None,
    prefix: str = "trip",
    show_grid: bool = True,
    show_parking_cells: bool = True,
) -> list[Path]:
    """
    Save one image per trip into a folder.

    Use max_trips to avoid creating thousands of files by accident.
    """
    if len(trips) == 0:
        raise ValueError("No trips to plot.")

    outdir = Path(folder)
    outdir.mkdir(parents=True, exist_ok=True)

    n_plot = len(trips) if max_trips is None else min(max_trips, len(trips))
    saved_paths: list[Path] = []

    for idx in range(n_plot):
        trip = trips[idx]
        obs = _get_observed(trip)

        save_path = outdir / (
            f"{prefix}_{idx:04d}_"
            f"trip_{obs.trip_id:05d}_"
            f"vehicle_{obs.vehicle_id:04d}.png"
        )

        fig, _ax = plot_trip_trace(
            grid,
            trip,
            title=f"Trip {obs.trip_id}, vehicle {obs.vehicle_id}",
            show_grid=show_grid,
            show_parking_cells=show_parking_cells,
            save_path=save_path,
        )
        plt.close(fig)

        saved_paths.append(save_path)

    return saved_paths


# ---------------------------------------------------------------------
# Estimator / evaluation plots
# ---------------------------------------------------------------------


def plot_theta_calibration(
    theta_true: np.ndarray,
    estimate: EstimateResult | np.ndarray,
    *,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    save_path: Optional[PathLike] = None,
) -> tuple[plt.Figure, Axes]:
    """
    Plot theta_hat_j against theta_j with a 45-degree line.
    """
    theta_true = np.asarray(theta_true, dtype=float)

    if isinstance(estimate, EstimateResult):
        theta_hat = np.asarray(estimate.theta_hat, dtype=float)
        label = estimate.name
        if title is None:
            title = f"Calibration: {estimate.name}"
    else:
        theta_hat = np.asarray(estimate, dtype=float)
        label = r"$\widehat{\theta}_j$"
        if title is None:
            title = "Calibration"

    x, y = calibration_points(theta_true, theta_hat)

    fig, ax = _get_fig_ax(ax, figsize=(5, 5))
    ax.scatter(x, y, alpha=0.8, label=label)

    lo = min(np.nanmin(x), np.nanmin(y), 0.0)
    hi = max(np.nanmax(x), np.nanmax(y), 1.0)
    ax.plot([lo, hi], [lo, hi], linestyle="--", label=r"$45^\circ$ line")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"true $\theta_j$", fontsize=GENERAL_AXIS_LABELSIZE)
    ax.set_ylabel(r"estimated $\widehat{\theta}_j$", fontsize=GENERAL_AXIS_LABELSIZE)
    ax.set_title(title, fontsize=GENERAL_TITLE_FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=GENERAL_TICK_LABELSIZE)
    ax.legend(loc="best", fontsize=GENERAL_LEGEND_FONTSIZE)

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig, ax


def plot_estimator_comparison(
    results: Sequence[EvaluationResult],
    *,
    metric: str = "mae",
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    save_path: Optional[PathLike] = None,
) -> tuple[plt.Figure, Axes]:
    """
    Bar plot comparing estimators by MAE, RMSE, or correlation.
    """
    allowed = {"mae", "rmse", "corr"}
    if metric not in allowed:
        raise ValueError(f"metric must be one of {allowed}.")

    names = [r.name for r in results]
    values = [getattr(r, metric) for r in results]

    fig, ax = _get_fig_ax(ax, figsize=(7, 4))
    ax.bar(names, values)

    ylabel = {"mae": "MAE", "rmse": "RMSE", "corr": "Correlation"}[metric]
    ax.set_ylabel(ylabel, fontsize=GENERAL_AXIS_LABELSIZE)
    if title is None:
        title = f"Estimator comparison ({ylabel})"
    ax.set_title(title, fontsize=GENERAL_TITLE_FONTSIZE)

    ax.tick_params(axis="x", rotation=30, labelsize=GENERAL_TICK_LABELSIZE)
    ax.tick_params(axis="y", labelsize=GENERAL_TICK_LABELSIZE)

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig, ax


def plot_exposure_success_counts(
    estimate: EstimateResult,
    *,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    save_path: Optional[PathLike] = None,
) -> tuple[plt.Figure, Axes]:
    """
    Plot inferred successes and exposure counts by parking-cell index.
    """
    fig, ax = _get_fig_ax(ax, figsize=(8, 4))

    j = np.arange(len(estimate.successes))
    ax.plot(j, estimate.exposure, marker="o", label="exposure")
    ax.plot(j, estimate.successes, marker="o", label="successes")

    ax.set_xlabel("parking-cell index j", fontsize=GENERAL_AXIS_LABELSIZE)
    ax.set_ylabel("count", fontsize=GENERAL_AXIS_LABELSIZE)
    if title is None:
        title = f"Exposure and successes: {estimate.name}"
    ax.set_title(title, fontsize=GENERAL_TITLE_FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=GENERAL_TICK_LABELSIZE)
    ax.legend(loc="best", fontsize=GENERAL_LEGEND_FONTSIZE)

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig, ax


def plot_trace_length_histogram(
    trips: Sequence[TripRecord | TripObserved],
    *,
    ax: Optional[Axes] = None,
    title: str = "Trip trace lengths",
    save_path: Optional[PathLike] = None,
) -> tuple[plt.Figure, Axes]:
    """
    Histogram of the number of recorded points per trip.
    """
    lengths = [len(_get_observed(trip).Gamma) for trip in trips]

    fig, ax = _get_fig_ax(ax, figsize=(6, 4))
    ax.hist(lengths, bins=20)
    ax.set_xlabel("number of trace points", fontsize=GENERAL_AXIS_LABELSIZE)
    ax.set_ylabel("number of trips", fontsize=GENERAL_AXIS_LABELSIZE)
    ax.set_title(title, fontsize=GENERAL_TITLE_FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=GENERAL_TICK_LABELSIZE)

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig, ax


# ---------------------------------------------------------------------
# Convenience summary figure
# ---------------------------------------------------------------------


def plot_simulation_overview(
    result: SimulationResult,
    *,
    save_path: Optional[PathLike] = None,
) -> plt.Figure:
    """
    Make a compact overview figure:
        destination density
        true theta map
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    plot_destination_density(result.grid, result.demand, ax=axes[0], title="Destination density")
    plot_theta_map(result.grid, result.theta, ax=axes[1], title="True theta")

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig