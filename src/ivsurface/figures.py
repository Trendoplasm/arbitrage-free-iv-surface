"""The study's four figures."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import matplotlib

# A non-interactive backend keeps the study runnable headless, in CI and over SSH.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ivsurface.models import Row

FIGURE_DPI = 180


def _finish(
    figure: Figure,
    axis: Axes,
    output_path: Path,
    *,
    title: str,
    ylabel: str,
    xlabel: str | None = None,
    grid_axis: Literal["both", "x", "y"] = "both",
    legend: bool = True,
) -> None:
    """Apply the shared styling and write the file."""
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    if xlabel is not None:
        axis.set_xlabel(xlabel)
    if legend:
        axis.legend()
    axis.grid(axis=grid_axis, alpha=0.25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(figure)


def plot_term_structure(curve: Sequence[Row], output_path: Path) -> None:
    """Plot the observed volatility curve and the total-variance curve beside it.

    The pair makes the study's central distinction visible. The left panel can slope downward in a
    panic -- a "curve inversion" -- while the right panel, which is what no-arbitrage constrains,
    keeps rising regardless.
    """
    horizons = [row["horizon_days"] for row in curve]
    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 5.5))

    left.plot(horizons, [100 * row["mean_iv"] for row in curve], marker="o", label="Mean")
    left.fill_between(
        horizons,
        [100 * row["p10_iv"] for row in curve],
        [100 * row["p90_iv"] for row in curve],
        alpha=0.2,
        label="10th to 90th percentile",
    )
    left.set_xscale("log")
    left.set_xticks(horizons, [str(int(h)) for h in horizons])
    left.set_title("Observed implied volatility")
    left.set_xlabel("Horizon (calendar days, log scale)")
    left.set_ylabel("Implied volatility (%)")
    left.legend()
    left.grid(alpha=0.25)

    right.plot(
        horizons,
        [row["mean_total_variance"] for row in curve],
        marker="s",
        color="tab:orange",
        label="Mean total variance",
    )
    right.set_xscale("log")
    right.set_xticks(horizons, [str(int(h)) for h in horizons])
    right.set_title("Total variance, which no-arbitrage constrains")
    right.set_xlabel("Horizon (calendar days, log scale)")
    right.set_ylabel("Total variance")
    right.legend()
    right.grid(alpha=0.25)

    figure.suptitle("The Observed S&P 500 Variance Term Structure")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(figure)


def plot_calibration_tradeoff(comparison: Sequence[Row], output_path: Path) -> None:
    """Plot what enforcing no-arbitrage costs in fit and buys in validity."""
    labels = [row["regime"] for row in comparison]
    positions = np.arange(len(labels))
    width = 0.36

    figure, axis = plt.subplots(figsize=(9, 6))
    fit_bars = axis.bar(
        positions - width / 2,
        [row["skewness_rmse"] for row in comparison],
        width,
        label="Skewness fit error (lower is better)",
        color="tab:red",
    )
    axis.set_ylabel("Root-mean-square skewness error")
    axis.set_xticks(positions, labels)
    axis.grid(axis="y", alpha=0.25)

    twin = axis.twinx()
    valid_bars = twin.bar(
        positions + width / 2,
        [100 * row["arbitrage_free_rate"] for row in comparison],
        width,
        label="Arbitrage-free days (higher is better)",
        color="tab:green",
    )
    twin.set_ylabel("Days free of static arbitrage (%)")
    twin.set_ylim(0, 105)

    axis.set_title("What Enforcing No-Arbitrage Costs, and What It Buys")
    axis.legend(handles=[fit_bars, valid_bars], loc="upper center")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(figure)


def plot_fitted_surface(grid: Sequence[Row], output_path: Path) -> None:
    """Plot the calibrated surface as volatility smiles, one per maturity."""
    figure, axis = plt.subplots(figsize=(9, 6))
    thetas = sorted({row["atm_total_variance"] for row in grid})
    for theta in thetas:
        points = sorted(
            (row for row in grid if row["atm_total_variance"] == theta),
            key=lambda row: row["log_moneyness"],
        )
        axis.plot(
            [row["log_moneyness"] for row in points],
            [np.sqrt(max(row["total_variance"], 0.0)) for row in points],
            marker="",
            linewidth=1.6,
            label=f"ATM total variance {theta:.4f}",
        )
    axis.axvline(0, linestyle="--", linewidth=0.8, color="black")
    _finish(
        figure,
        axis,
        output_path,
        title="The Calibrated Arbitrage-Free Surface",
        xlabel="Log-moneyness  k = ln(K / F)",
        ylabel="Square root of total variance",
    )


def plot_correlation_history(parameters: Sequence[Row], output_path: Path) -> None:
    """Plot the daily correlation parameter under each calibration regime.

    How far the parameter has to travel is itself a result: a family that needs a very different
    parameter every day is describing the market less well than its fit statistics suggest.
    """
    figure, axis = plt.subplots(figsize=(12, 6))
    for regime in sorted({row["regime"] for row in parameters}):
        points = sorted(
            (row for row in parameters if row["regime"] == regime and row["rho"] is not None),
            key=lambda row: row["date"],
        )
        if not points:
            continue
        axis.plot(
            [row["date"] for row in points],
            [row["rho"] for row in points],
            linewidth=1.0,
            label=regime,
        )
    axis.axhline(0, linewidth=0.8, color="black")
    _finish(
        figure,
        axis,
        output_path,
        title="Daily Calibrated Correlation Parameter",
        xlabel="Date",
        ylabel="rho",
    )
