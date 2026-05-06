# GPS-Based Inference of Lot-Level Parking Probabilities

This repository contains the simulation and analysis code for the 1.200 project **GPS-Based Inference of Lot-Level Parking Probabilities for Probability-Aware Parking Selection**.

The project studies whether GPS-style vehicle traces can be used to estimate lot-level parking success probabilities. The original probability-aware parking-selection framework assumes that each parking lot's success probability is already known; this project simulates a city, generates vehicle traces, estimates those probabilities from traces, and compares trace-based estimators against the known ground truth from the simulator.

## Repository structure

```text
.
├── 1_200_Project.pdf                  # Project writeup / paper
├── config.py                          # Simulation configuration dataclasses
├── grid.py                            # Grid generation, cells, distances, paths
├── demand.py                          # Destination-demand distribution
├── driver_behavior.py                 # Parking choice and attempt behavior
├── simulator.py                       # Vehicle/trip simulation
├── estimators.py                      # Parking-success probability estimators
├── evaluation.py                      # MAE, RMSE, correlation, evaluation tables
├── plotting.py                        # Figure-generation utilities
├── run_large_experiment.py            # Main baseline experiment
├── run_vehicle_sensitivity.py         # Sensitivity analysis over number of vehicles
└── outputs_ring/
    ├── data/                          # Main experiment CSV/JSON outputs
    ├── figures/                       # Main experiment figures
    └── sensitivity_nvehicles/         # Vehicle-count sensitivity outputs
```

## Requirements

Use Python 3.10 or newer. The code uses only the Python standard library plus:

```text
numpy
matplotlib
```

A minimal setup is:

```bash
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
# .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install numpy matplotlib
```

A conda setup is also fine:

```bash
conda create -n parking-prob python=3.11 numpy matplotlib -y
conda activate parking-prob
```

## Reproducing the main experiment

From the repository root, run:

```bash
python run_large_experiment.py
```

This regenerates the main ring-based experiment and writes outputs to:

```text
outputs_ring/data/
outputs_ring/figures/
```

The main experiment uses the following baseline configuration in `run_large_experiment.py`:

```text
seed = 0
L = 30
use_ring_parking = True
ring parking probabilities = 0.10, 0.30, 0.15 for the three square rings
n_vehicles = 500
trips_per_vehicle = 10
heterogeneous_drivers = True
lambda_v ~ Uniform(0.4, 1.2)
rho_v ~ Uniform(0.2, 0.8)
theta_j ~ Uniform(0.2, 0.8)
exposure_radius = 0.0
alpha_smoothing = beta_smoothing = 1.0
```

A successful run prints summaries of the generated grid, demand distribution, driver-choice distribution, simulation, and estimator evaluation table. The project paper's baseline comparison reports that the stop-confirmed estimator improves on the raw trace-exposure baseline, reducing MAE from about `0.2258` to `0.1015` and increasing correlation from about `0.4062` to `0.7269`. The exact metrics for a run should be checked in `outputs_ring/data/evaluation.csv`.

## Reproducing the vehicle-count sensitivity analysis

To reproduce the default sensitivity analysis over the number of vehicles, run:

```bash
python run_vehicle_sensitivity.py
```

By default, this uses:

```text
n_vehicles ∈ {50, 100, 200, 500, 1000, 2000, 5000}
trips_per_vehicle = 10
n_reps = 3
seed = 0
output_dir = outputs_ring/sensitivity_nvehicles
```

This writes:

```text
outputs_ring/sensitivity_nvehicles/vehicle_sensitivity_raw.csv
outputs_ring/sensitivity_nvehicles/vehicle_sensitivity_summary.csv
outputs_ring/sensitivity_nvehicles/figures/
```

For a quick smoke test that checks the code path without running the full sensitivity analysis, use a smaller command such as:

```bash
python run_vehicle_sensitivity.py \
  --n-values 50 100 \
  --trips-per-vehicle 2 \
  --n-reps 1 \
  --output-dir outputs_ring/sensitivity_smoke
```

## What the code does

### 1. Generate a simulated city

`grid.py` creates an `L x L` grid. Each cell is either a parking cell or a non-parking destination cell. In the main experiment, the probability that a cell is parking depends on which square ring it belongs to, allowing the simulated city to have different parking densities in the center, surrounding urban ring, and periphery.

The Python code uses 0-indexed coordinates `(0, 0), ..., (L-1, L-1)`. The paper/writeup may use 1-indexed coordinates `(1, 1), ..., (L, L)`. Manhattan distances are unchanged by this shift.

### 2. Generate destinations and true parking probabilities

`demand.py` assigns a destination-demand probability to each non-parking cell using a distance-decay model centered at the city center:

```text
q_D(i) ∝ exp(-eta_D * d(g_i, c_0)).
```

`simulator.py` draws the true immediate parking-success probability for each parking cell:

```text
theta_j ~ Uniform(theta_low, theta_high).
```

These true `theta_j` values are hidden from the trace-based estimators and are used only for evaluation.

### 3. Simulate driver behavior and traces

`driver_behavior.py` and `simulator.py` generate vehicle trips. A vehicle has a destination, chooses a planned parking cell with probability decreasing in distance from the destination, moves one grid cell per time step, and may opportunistically stop at eligible parking cells encountered along the path. If parking fails, the driver either retries at the same cell or switches to a new planned cell.

The simulator records two versions of each trip:

- **Observed data:** the starting parking cell, destination, and GPS-style trace `Gamma = [(x, y, t), ...]`.
- **Hidden truth:** the actual attempts, availability draws, successes/failures, and final parking cell. These are for evaluation and debugging only.

### 4. Estimate parking-success probabilities

`estimators.py` implements three estimators:

| Estimator | Description |
|---|---|
| `trace_exposure` | Raw trace-exposure baseline. A parking cell receives exposure when a trace passes through or near it; success is inferred from the final trace point. |
| `stop_confirmed` | Conservative trace-based estimator. A parking attempt is inferred only when consecutive trace points remain at the same parking cell. This is the main estimator in the paper. |
| `oracle_exposure` | Upper-bound benchmark using hidden simulator availability draws. This should not be interpreted as an implementable trace-only estimator. |

Each estimator returns `theta_hat[j]`, inferred successes, and inferred exposure counts for each parking cell.

### 5. Evaluate and plot

`evaluation.py` computes:

- mean absolute error (MAE),
- root mean squared error (RMSE),
- Pearson correlation,
- number of evaluated parking cells.

`plotting.py` generates spatial maps, calibration plots, estimator comparisons, exposure/success count plots, trace-length histograms, and sampled trip traces.

## Output files

### Main experiment data: `outputs_ring/data/`

| File | Contents |
|---|---|
| `config.json` | Full configuration used for the run. |
| `vehicle_parameters.csv` | Driver-level `lambda_choice` and `rho_stay` for each vehicle. |
| `parking_cells.csv` | Parking-cell index, row, column, and true `theta_j`. |
| `theta_true.csv` | True parking-success probability by parking cell. |
| `destination_cells.csv` | Destination-cell index, row, column, demand probability, and unnormalized demand weight. |
| `trip_observed.csv` | Observed trip metadata: trip ID, vehicle ID, start parking cell, destination, start/end time, and trace length. |
| `trace_points.csv` | Long-format GPS-style traces: one row per observed trace point. |
| `trip_truth.csv` | Hidden trip-level truth labels for evaluation/debugging. |
| `attempts_truth.csv` | Hidden attempt-level truth labels for evaluation/debugging. |
| `availability_draws_truth.csv` | Hidden availability draws used by the oracle estimator. |
| `estimates.csv` | Estimated `theta_hat`, successes, and exposures by estimator and parking cell. |
| `evaluation.csv` | MAE, RMSE, correlation, and number of evaluated cells for each estimator. |

### Main experiment figures: `outputs_ring/figures/`

The main script saves the paper figures:

```text
fig1_simulation_setup.png
fig2_main_estimator_performance.png
fig3_stop_confirmed_spatial_results.png
```

It also saves diagnostic figures such as:

```text
grid_map.png
destination_density.png
theta_map.png
simulation_overview.png
calibration_trace_exposure.png
calibration_stop_confirmed.png
calibration_oracle.png
trace_exposure_estimate_map.png
stop_confirmed_estimate_map.png
oracle_exposure_estimate_map.png
trace_exposure_error_map.png
stop_confirmed_error_map.png
oracle_exposure_error_map.png
estimator_mae_comparison.png
estimator_rmse_comparison.png
estimator_corr_comparison.png
trace_exposure_success_counts.png
stop_confirmed_exposure_success_counts.png
oracle_exposure_success_counts.png
trace_length_histogram.png
sampled_trip_traces/
```

### Sensitivity outputs: `outputs_ring/sensitivity_nvehicles/`

| File/folder | Contents |
|---|---|
| `vehicle_sensitivity_raw.csv` | One row per replication, vehicle count, and estimator. |
| `vehicle_sensitivity_summary.csv` | Mean and standard deviation of metrics by vehicle count and estimator. |
| `figures/` | Sensitivity plots for MAE, RMSE, correlation, and stop-confirmed improvement. |

## Reproducibility notes

- Randomness is controlled by `seed` in `SimConfig` and by the `--seed` argument in `run_vehicle_sensitivity.py`.
- The simulation uses NumPy random generators passed through the main simulation functions. With the same code, seed, and package versions, the generated data and figures should be reproducible.
- Running `run_large_experiment.py` overwrites files in `outputs_ring/data/` and `outputs_ring/figures/`.
- Running `run_vehicle_sensitivity.py` overwrites files in its selected `--output-dir`.
- To avoid overwriting existing results, change `output_dir`, `figures_dir`, and `data_dir` in `run_large_experiment.py`, or pass a different `--output-dir` to `run_vehicle_sensitivity.py`.
- Hidden truth files are included so the synthetic experiment can be checked and evaluated, but trace-only estimators should not use them.

## Troubleshooting

If imports fail, make sure you are running commands from the repository root, where all `.py` files are located:

```bash
python run_large_experiment.py
```

If `matplotlib` display issues occur on a remote machine, the scripts should still work because they save figures to disk rather than requiring an interactive window.

If the full sensitivity analysis takes too long, first run the smoke test above, then increase `--n-values`, `--trips-per-vehicle`, and `--n-reps` as needed.

## Reference

- `1_200_Project.pdf`: project writeup describing the simulation environment, estimators, and results.
- Hickert, Li, He, and Wu, **Probability-Aware Parking Selection**: original probability-aware parking-selection framework that motivates this extension.

## License

No license information is specified in the provided files.
