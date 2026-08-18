"""End-to-end orchestration: load, measure, calibrate, diagnose, export."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from ivsurface.aggregate import (
    bootstrap_interval,
    calendar_summary,
    extreme_days,
    held_out_summary,
    inversion_summary,
    parameter_stability,
    stress_episodes,
    yearly_summary,
)
from ivsurface.calibrate import (
    calibrate_day,
    fit_daily_rho,
    fit_ssvi_to_skewness,
    model_skewness,
)
from ivsurface.config import (
    CORE_TERM_STRUCTURE,
    SKEW_INDEX,
    VOL_OF_VOL_INDEX,
    StudyConfig,
)
from ivsurface.diagnostics import diagnose_day, diagnostic_grid
from ivsurface.figures import (
    plot_calibration_tradeoff,
    plot_correlation_history,
    plot_fitted_surface,
    plot_term_structure,
)
from ivsurface.loaders import load_series
from ivsurface.models import Row, Table
from ivsurface.svi import SSVI
from ivsurface.termstructure import (
    average_curve,
    build_panel,
    build_term_structure,
    maturity_years,
    skew_to_skewness,
)
from ivsurface.writers import write_csv, write_json

logger = logging.getLogger(__name__)

#: Restated in every export so no downstream reader can mistake the study's scope.
SCOPE_NOTE = (
    "The maturity dimension of this surface is observed: Cboe publishes at-the-money expected "
    "volatility at six horizons, and the calendar no-arbitrage test is run directly on that "
    "published curve. The strike dimension is not observed. No free source gives implied "
    "volatility strike by strike, so the surface's skew is calibrated to reproduce the one "
    "strike-related quantity that is published -- the risk-neutral skewness encoded in Cboe's SKEW "
    "index. One number per day is a far thinner constraint than a quoted option chain, and the "
    "study's fit statistics should be read in that light. What does not depend on the data at all "
    "are the no-arbitrage conditions themselves, which are theorems about the parameterisation and "
    "are verified numerically here."
)

#: Log-moneyness grid used when integrating the density for skewness during calibration.
SKEWNESS_GRID = np.linspace(-1.5, 1.5, 601)

#: Calibration regimes compared by the study.
UNCONSTRAINED = "Unconstrained"
CONSTRAINED = "No-arbitrage enforced"


@dataclass(frozen=True)
class StudyResults:
    """Everything one run produces."""

    config: StudyConfig
    panel: Table = field(repr=False)
    observed_curve: Table = field(repr=False)
    calendar: Row = field(repr=False)
    inversions: Row = field(repr=False)
    held_out: Table = field(repr=False)
    held_out_stats: Row = field(repr=False)
    global_fits: Table = field(repr=False)
    comparison: Table = field(repr=False)
    daily_parameters: Table = field(repr=False)
    diagnostics: Table = field(repr=False)
    stability: Table = field(repr=False)
    episodes: Table = field(repr=False)
    yearly: Table = field(repr=False)
    surface_grid: Table = field(repr=False)
    worst_days: Table = field(repr=False)
    bootstrap: dict[str, tuple[float, float, float] | None] = field(repr=False)
    surfaces: dict[str, SSVI] = field(repr=False, default_factory=dict)

    @property
    def constrained_surface(self) -> SSVI:
        """Return the surface fitted under the no-arbitrage constraint."""
        return self.surfaces[CONSTRAINED]


def _calibration_dates(dates: list[date], stride: int) -> list[date]:
    """Return an evenly spaced subsample of dates for the global fit.

    The global fit integrates a density for every candidate parameter set on every date, so it runs
    on a subsample. Sampling evenly rather than taking a block keeps every market regime
    represented.
    """
    return dates[::stride]


def run_study(data_dir: Path, config: StudyConfig) -> StudyResults:
    """Load the inputs and run the complete study.

    Args:
        data_dir: Directory holding the downloaded Cboe histories.
        config: Study period and settings.

    Returns:
        Every measurement, calibration, and diagnostic the study produces.
    """
    series = load_series(data_dir)
    dates = sorted(day for day in series["VIX"] if config.start() <= day <= config.end())
    panel = build_panel(series, dates, CORE_TERM_STRUCTURE)
    panel_dates = [row["date"] for row in panel]

    curves = {
        row["date"]: build_term_structure(series, row["date"], CORE_TERM_STRUCTURE) for row in panel
    }

    # --- Maturity dimension: fit without one maturity, then predict it -------------------
    held_out: Table = []
    for trading_date in panel_dates:
        curve = curves[trading_date]
        if curve is None:
            continue
        fitted = calibrate_day(curve, config.calibration_maturities(), config.held_out_maturity)
        if fitted is None:
            continue
        variance_curve, predicted, relative = fitted
        held_out.append(
            {
                "date": trading_date,
                "v_short": variance_curve.v_short,
                "v_long": variance_curve.v_long,
                "kappa": variance_curve.kappa,
                "held_out_maturity": config.held_out_maturity,
                "observed_total_variance": curve.theta(config.held_out_maturity),
                "predicted_total_variance": predicted,
                "held_out_relative_error": relative,
                "curve_calendar_free": variance_curve.is_calendar_free(curve.maturities),
            }
        )

    # --- Strike dimension: global fits, with and without the constraint ------------------
    calibration_dates = _calibration_dates(panel_dates, max(1, len(panel_dates) // 200))
    thetas: list[float] = []
    skewnesses: list[float] = []
    for trading_date in calibration_dates:
        curve = curves[trading_date]
        level = series[SKEW_INDEX].get(trading_date)
        if curve is None or level is None or "VIX" not in curve.names:
            continue
        thetas.append(curve.theta("VIX"))
        skewnesses.append(skew_to_skewness(level))

    all_thetas = sorted(
        {
            value
            for curve in curves.values()
            if curve
            for value in curve.total_variances
            if value > 0
        }
    )
    constraint_points = [all_thetas[0], all_thetas[len(all_thetas) // 2], all_thetas[-1]]

    surfaces: dict[str, SSVI] = {}
    global_fits: Table = []
    for label, enforce in ((UNCONSTRAINED, False), (CONSTRAINED, True)):
        surface, error = fit_ssvi_to_skewness(
            thetas,
            skewnesses,
            SKEWNESS_GRID,
            enforce_no_arbitrage=enforce,
            condition_thetas=constraint_points,
        )
        surfaces[label] = surface
        global_fits.append(
            {
                "regime": label,
                "no_arbitrage_enforced": enforce,
                "calibration_dates": len(thetas),
                "rho": surface.rho,
                "eta": surface.eta,
                "gamma": surface.gamma,
                "global_skewness_rmse": error,
            }
        )

    # --- Daily refit and diagnostics under each regime -----------------------------------
    grid = diagnostic_grid()
    daily_parameters: Table = []
    diagnostics: Table = []
    for label, enforce in ((UNCONSTRAINED, False), (CONSTRAINED, True)):
        surface = surfaces[label]
        for trading_date in panel_dates:
            curve = curves[trading_date]
            level = series[SKEW_INDEX].get(trading_date)
            if curve is None or level is None or "VIX" not in curve.names:
                continue
            theta = curve.theta("VIX")
            target = skew_to_skewness(level)
            rho = fit_daily_rho(
                surface,
                theta,
                target,
                SKEWNESS_GRID,
                enforce_no_arbitrage=enforce,
                condition_thetas=curve.total_variances if enforce else None,
            )
            if rho is None:
                daily_parameters.append(
                    {
                        "date": trading_date,
                        "regime": label,
                        "rho": None,
                        "target_skewness": target,
                        "fitted_skewness": None,
                        "skewness_error": None,
                        "feasible": False,
                    }
                )
                continue
            daily = SSVI(rho=rho, eta=surface.eta, gamma=surface.gamma)
            fitted_skewness = model_skewness(daily, theta, SKEWNESS_GRID)
            daily_parameters.append(
                {
                    "date": trading_date,
                    "regime": label,
                    "rho": rho,
                    "target_skewness": target,
                    "fitted_skewness": fitted_skewness,
                    "skewness_error": fitted_skewness - target,
                    "feasible": True,
                }
            )
            diagnostics.append(
                {
                    "date": trading_date,
                    "regime": label,
                    **diagnose_day(daily, curve.total_variances, curve.names, grid),
                }
            )

    comparison = _compare_regimes(daily_parameters, diagnostics, global_fits)

    rng = np.random.default_rng(config.random_seed)
    bootstrap = {
        "held_out_absolute_error": bootstrap_interval(
            [abs(row["held_out_relative_error"]) for row in held_out],
            rng,
            config.bootstrap_iterations,
        ),
        "risk_neutral_skewness": bootstrap_interval(
            [
                row["risk_neutral_skewness"]
                for row in panel
                if row.get("risk_neutral_skewness") is not None
            ],
            rng,
            config.bootstrap_iterations,
        ),
    }

    return StudyResults(
        config=config,
        panel=panel,
        observed_curve=average_curve(panel, CORE_TERM_STRUCTURE),
        calendar=calendar_summary(panel),
        inversions=inversion_summary(panel, CORE_TERM_STRUCTURE),
        held_out=held_out,
        held_out_stats=held_out_summary(held_out),
        global_fits=global_fits,
        comparison=comparison,
        daily_parameters=daily_parameters,
        diagnostics=diagnostics,
        stability=parameter_stability(daily_parameters, config.stability_window_days),
        episodes=stress_episodes(panel, config.event_window_days),
        yearly=yearly_summary(panel),
        surface_grid=_surface_grid(surfaces[CONSTRAINED], all_thetas),
        worst_days=extreme_days(panel, "risk_neutral_skewness", largest=False),
        bootstrap=bootstrap,
        surfaces=surfaces,
    )


def _compare_regimes(daily_parameters: Table, diagnostics: Table, global_fits: Table) -> Table:
    """Build the study's central table: what enforcing no-arbitrage costs and buys."""
    comparison: Table = []
    for fit in global_fits:
        label = fit["regime"]
        parameters = [row for row in daily_parameters if row["regime"] == label]
        checks = [row for row in diagnostics if row["regime"] == label]
        errors = np.array(
            [row["skewness_error"] for row in parameters if row["skewness_error"] is not None],
            dtype=float,
        )
        comparison.append(
            {
                "regime": label,
                "no_arbitrage_enforced": fit["no_arbitrage_enforced"],
                "rho": fit["rho"],
                "eta": fit["eta"],
                "gamma": fit["gamma"],
                "days": len(parameters),
                "feasible_days": sum(row["feasible"] for row in parameters),
                "skewness_rmse": float(np.sqrt(np.mean(errors**2))) if errors.size else None,
                "median_absolute_skewness_error": (
                    float(np.median(np.abs(errors))) if errors.size else None
                ),
                "arbitrage_free_days": sum(row["arbitrage_free"] for row in checks),
                "arbitrage_free_rate": (
                    sum(row["arbitrage_free"] for row in checks) / len(checks) if checks else None
                ),
                "sufficient_conditions_rate": (
                    sum(row["sufficient_conditions_hold"] for row in checks) / len(checks)
                    if checks
                    else None
                ),
                "worst_butterfly_g": min((row["min_butterfly_g"] for row in checks), default=None),
                "worst_density_mass_error": max(
                    (row["max_density_mass_error"] for row in checks), default=None
                ),
                "valid_density_rate": (
                    sum(row["density_valid"] for row in checks) / len(checks) if checks else None
                ),
            }
        )
    return comparison


def _surface_grid(surface: SSVI, thetas: list[float]) -> Table:
    """Export the fitted surface on a grid, so it can be inspected without rerunning anything."""
    sample = [thetas[0], thetas[len(thetas) // 4], thetas[len(thetas) // 2], thetas[-1]]
    moneyness = np.linspace(-0.6, 0.6, 25)
    rows: Table = []
    for theta in sample:
        slice_ = surface.to_raw_svi(theta)
        for k in moneyness:
            total = float(slice_.total_variance(k))
            rows.append(
                {
                    "atm_total_variance": theta,
                    "log_moneyness": float(k),
                    "total_variance": total,
                    "implied_volatility_30d": float(
                        np.sqrt(max(total, 0.0) / maturity_years("VIX"))
                    ),
                }
            )
    return rows


def summary_payload(results: StudyResults) -> dict[str, Any]:
    """Assemble the machine-readable run summary."""
    return {
        "configuration": asdict(results.config),
        "panel_days": len(results.panel),
        "observed_curve": results.observed_curve,
        "calendar_arbitrage": results.calendar,
        "volatility_curve_inversions": results.inversions,
        "held_out_calibration": results.held_out_stats,
        "global_fits": results.global_fits,
        "calibration_comparison": results.comparison,
        "parameter_stability": results.stability,
        "stress_episodes": results.episodes,
        "bootstrap": results.bootstrap,
        "scope_note": SCOPE_NOTE,
    }


def write_outputs(results: StudyResults, output_dir: Path, *, with_plots: bool = True) -> None:
    """Write every table, figure, and summary of a completed run."""
    tables_dir = output_dir / "tables"
    tables: dict[str, Table] = {
        "term_structure_panel": results.panel,
        "observed_curve": results.observed_curve,
        "calendar_arbitrage": [results.calendar],
        "volatility_curve_inversions": [results.inversions],
        "held_out_calibration": results.held_out,
        "held_out_summary": [results.held_out_stats],
        "global_fits": results.global_fits,
        "calibration_comparison": results.comparison,
        "daily_parameters": results.daily_parameters,
        "arbitrage_diagnostics": results.diagnostics,
        "parameter_stability": results.stability,
        "stress_episodes": results.episodes,
        "yearly_summary": results.yearly,
        "fitted_surface_grid": results.surface_grid,
        "most_negative_skewness_days": results.worst_days,
    }
    empty = [name for name, rows in tables.items() if not rows]
    for name, rows in tables.items():
        if rows:
            write_csv(tables_dir / f"{name}.csv", rows)
    if empty:
        logger.warning("No rows to write for: %s", ", ".join(sorted(empty)))

    if with_plots:
        plots_dir = output_dir / "plots"
        plot_term_structure(results.observed_curve, plots_dir / "observed_term_structure.png")
        plot_calibration_tradeoff(results.comparison, plots_dir / "calibration_tradeoff.png")
        plot_fitted_surface(results.surface_grid, plots_dir / "fitted_surface.png")
        plot_correlation_history(results.daily_parameters, plots_dir / "correlation_history.png")

    write_json(output_dir / "summary.json", summary_payload(results))
    logger.info("Wrote %d tables to %s", len(tables) - len(empty), output_dir)


def headline(results: StudyResults) -> str:
    """Render the one-line result used to confirm a successful reproduction."""
    calendar = results.calendar
    held_out = results.held_out_stats
    constrained = next(row for row in results.comparison if row["regime"] == CONSTRAINED)
    unconstrained = next(row for row in results.comparison if row["regime"] == UNCONSTRAINED)
    return (
        f"Completed {calendar['days_tested']} days: {calendar['days_with_violation']} calendar "
        f"violations in the observed curve, held-out maturity predicted to "
        f"{held_out['median_absolute_error']:.2%} median error, and enforcing no-arbitrage moved "
        f"the surface from {unconstrained['arbitrage_free_rate']:.1%} to "
        f"{constrained['arbitrage_free_rate']:.1%} valid at a skewness RMSE of "
        f"{unconstrained['skewness_rmse']:.3f} against {constrained['skewness_rmse']:.3f}."
    )


#: Series names re-exported for convenience.
INPUT_SERIES = (*CORE_TERM_STRUCTURE, SKEW_INDEX, VOL_OF_VOL_INDEX)
