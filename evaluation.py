# evaluation.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from estimators import EstimateResult


@dataclass(frozen=True)
class EvaluationResult:
    """Evaluation metrics for one estimator."""

    name: str
    mae: float
    rmse: float
    corr: float
    n_used: int
    notes: str = ""


def mean_absolute_error(
    theta_true: np.ndarray,
    theta_hat: np.ndarray,
    *,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Mean absolute error between theta_hat and theta_true.
    """
    y, yhat = _clean_pair(theta_true, theta_hat, mask=mask)

    if len(y) == 0:
        return float("nan")

    return float(np.mean(np.abs(yhat - y)))


def root_mean_squared_error(
    theta_true: np.ndarray,
    theta_hat: np.ndarray,
    *,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Root mean squared error between theta_hat and theta_true.
    """
    y, yhat = _clean_pair(theta_true, theta_hat, mask=mask)

    if len(y) == 0:
        return float("nan")

    return float(np.sqrt(np.mean((yhat - y) ** 2)))


def pearson_correlation(
    theta_true: np.ndarray,
    theta_hat: np.ndarray,
    *,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Pearson correlation between theta_hat and theta_true.
    """
    y, yhat = _clean_pair(theta_true, theta_hat, mask=mask)

    if len(y) < 2:
        return float("nan")

    if np.std(y) == 0 or np.std(yhat) == 0:
        return float("nan")

    return float(np.corrcoef(y, yhat)[0, 1])


def evaluate_estimate(
    theta_true: np.ndarray,
    estimate: EstimateResult | np.ndarray,
    *,
    name: Optional[str] = None,
    mask: Optional[np.ndarray] = None,
    notes: str = "",
) -> EvaluationResult:
    """
    Evaluate one estimator against true theta.

    estimate can be either:
        - EstimateResult
        - raw theta_hat array
    """
    theta_true = np.asarray(theta_true, dtype=float)

    if isinstance(estimate, EstimateResult):
        theta_hat = estimate.theta_hat
        est_name = estimate.name
        est_notes = estimate.notes
    else:
        theta_hat = np.asarray(estimate, dtype=float)
        est_name = name or "estimate"
        est_notes = notes

    if name is not None:
        est_name = name

    y, yhat = _clean_pair(theta_true, theta_hat, mask=mask)

    return EvaluationResult(
        name=est_name,
        mae=mean_absolute_error(y, yhat),
        rmse=root_mean_squared_error(y, yhat),
        corr=pearson_correlation(y, yhat),
        n_used=len(y),
        notes=est_notes,
    )


def evaluate_many(
    theta_true: np.ndarray,
    estimates: Sequence[EstimateResult],
    *,
    mask: Optional[np.ndarray] = None,
) -> list[EvaluationResult]:
    """Evaluate several estimators."""
    return [
        evaluate_estimate(theta_true, estimate, mask=mask)
        for estimate in estimates
    ]


def format_evaluation_table(results: Sequence[EvaluationResult]) -> str:
    """
    Return a simple text table of evaluation results.
    """
    if not results:
        return "(no results)"

    header = f"{'Estimator':<24} {'MAE':>10} {'RMSE':>10} {'Corr':>10} {'n':>6}"
    line = "-" * len(header)

    rows = [header, line]
    for res in results:
        rows.append(
            f"{res.name:<24} "
            f"{_fmt(res.mae):>10} "
            f"{_fmt(res.rmse):>10} "
            f"{_fmt(res.corr):>10} "
            f"{res.n_used:>6}"
        )

    return "\n".join(rows)


def results_as_dicts(results: Sequence[EvaluationResult]) -> list[dict[str, Any]]:
    """
    Convert evaluation results to dictionaries.
    Useful if you want to make a pandas DataFrame later.
    """
    return [
        {
            "name": res.name,
            "mae": res.mae,
            "rmse": res.rmse,
            "corr": res.corr,
            "n_used": res.n_used,
            "notes": res.notes,
        }
        for res in results
    ]


def calibration_points(
    theta_true: np.ndarray,
    theta_hat: np.ndarray,
    *,
    mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return cleaned (theta_true, theta_hat) arrays for calibration plots.

    Plot theta_hat against theta_true and add a 45-degree line.
    """
    return _clean_pair(theta_true, theta_hat, mask=mask)


def exposure_mask(
    exposure: np.ndarray,
    *,
    min_exposure: float = 1.0,
) -> np.ndarray:
    """
    Return mask for parking cells with at least min_exposure.
    Useful if you want to evaluate only cells with enough observations.
    """
    exposure = np.asarray(exposure, dtype=float)
    return np.isfinite(exposure) & (exposure >= min_exposure)


def _clean_pair(
    theta_true: np.ndarray,
    theta_hat: np.ndarray,
    *,
    mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align and remove invalid entries.
    """
    theta_true = np.asarray(theta_true, dtype=float)
    theta_hat = np.asarray(theta_hat, dtype=float)

    if theta_true.shape != theta_hat.shape:
        raise ValueError(
            f"theta_true and theta_hat must have same shape; got "
            f"{theta_true.shape} and {theta_hat.shape}."
        )

    valid = np.isfinite(theta_true) & np.isfinite(theta_hat)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != theta_true.shape:
            raise ValueError(
                f"mask must have shape {theta_true.shape}; got {mask.shape}."
            )
        valid &= mask

    return theta_true[valid], theta_hat[valid]


def _fmt(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{x:.4f}"