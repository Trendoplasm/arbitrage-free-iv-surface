"""Static-arbitrage diagnostics on a calibrated surface.

A surface that fits the data can still be impossible. Two things must hold, and both are checked
here numerically rather than assumed from the parameters:

* **Butterfly.** The implied risk-neutral density must be non-negative everywhere. Where it is not,
  a butterfly spread has a negative price -- you would be paid to hold a position that cannot lose.
  The test is the sign of Gatheral's ``g(k)``.
* **Calendar.** Total variance must not fall as maturity grows, or a calendar spread is free money.

A third check is not an arbitrage condition but a sanity condition: the implied density must
integrate to one. A surface whose density sums to 0.98 is not describing a probability
distribution, however well it fits.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from ivsurface.config import (
    DENSITY_CORE_LIMIT,
    DENSITY_CORE_POINTS,
    DENSITY_K_LIMIT,
    DENSITY_TOLERANCE,
    DENSITY_WING_POINTS,
    DIAGNOSTIC_K_MAX,
    DIAGNOSTIC_K_MIN,
    DIAGNOSTIC_K_POINTS,
)
from ivsurface.models import Row
from ivsurface.svi import (
    SSVI,
    ArbitrageCheck,
    RawSVI,
    butterfly_g,
    implied_density,
    ssvi_butterfly_conditions,
    ssvi_calendar_free,
)

logger = logging.getLogger(__name__)


def diagnostic_grid(
    lower: float = DIAGNOSTIC_K_MIN,
    upper: float = DIAGNOSTIC_K_MAX,
    points: int = DIAGNOSTIC_K_POINTS,
) -> np.ndarray:
    """Return the log-moneyness grid the diagnostics are evaluated on."""
    return np.linspace(lower, upper, points)


def density_grid(atm_variance: float | None = None) -> np.ndarray:
    """Return an integration grid scaled to the width of the distribution being integrated.

    Two competing demands settle this. The range must be **wide**, because an SVI slice has linear
    wings and therefore fatter tails than a lognormal -- truncating at ten standard deviations of
    the short-dated slice loses about a tenth of a percent of the mass. The spacing must be
    **fine**,
    because a nine-day slice has a standard deviation near 0.03 in log-moneyness, and a grid that
    steps over it reports a mass far from one for reasons that have nothing to do with the surface.

    A uniform grid cannot satisfy both without becoming enormous, so the grid is built in two
    parts: a dense core covering the region that holds the mass, and sparse wings carrying the
    tails out to where they are negligible. Trapezoidal integration handles the uneven spacing
    correctly, and the result is more accurate than a uniform grid twice its size.

    Args:
        atm_variance: Retained so callers can reason about resolution; the grid is fixed, because
            a grid that changed with maturity would make the daily density check incomparable
            across maturities.

    Returns:
        An ascending, unevenly spaced grid of log-moneyness.
    """
    del atm_variance
    core = np.linspace(-DENSITY_CORE_LIMIT, DENSITY_CORE_LIMIT, DENSITY_CORE_POINTS)
    left = np.linspace(-DENSITY_K_LIMIT, -DENSITY_CORE_LIMIT, DENSITY_WING_POINTS, endpoint=False)
    right = np.linspace(DENSITY_CORE_LIMIT, DENSITY_K_LIMIT, DENSITY_WING_POINTS + 1)[1:]
    return np.concatenate([left, core, right])


def density_mass(
    slice_: RawSVI, atm_variance: float | None = None, grid: np.ndarray | None = None
) -> float:
    """Return the total mass of the density a slice implies.

    Args:
        slice_: The slice.
        atm_variance: At-the-money total variance, used to scale the integration grid.
        grid: Explicit integration grid, overriding the scaled default.

    Returns:
        The integral of the density, which should be one.
    """
    points = density_grid(atm_variance) if grid is None else grid
    return float(np.trapezoid(np.maximum(implied_density(slice_, points), 0.0), points))


def check_slice(slice_: RawSVI, grid: np.ndarray | None = None) -> tuple[bool, float, int]:
    """Scan one slice for butterfly arbitrage.

    Args:
        slice_: The slice.
        grid: Log-moneyness grid; the diagnostic default is used when omitted.

    Returns:
        Whether the slice passed, the smallest ``g(k)`` found, and the number of failing points.
    """
    points = diagnostic_grid() if grid is None else grid
    g = butterfly_g(slice_, points)
    negatives = int(np.sum(g < 0))
    return negatives == 0, float(np.min(g)), negatives


def check_surface(
    surface: SSVI, thetas: Sequence[float], grid: np.ndarray | None = None
) -> ArbitrageCheck:
    """Scan a whole surface for both forms of static arbitrage.

    Args:
        surface: The calibrated surface.
        thetas: At-the-money total variance at each maturity, ordered by increasing maturity.
        grid: Log-moneyness grid for the butterfly scan.

    Returns:
        The combined outcome.
    """
    points = diagnostic_grid() if grid is None else grid
    worst_g = float("inf")
    butterfly_failures = 0
    for theta in thetas:
        if theta <= 0:
            continue
        passed, minimum, failures = check_slice(surface.to_raw_svi(float(theta)), points)
        worst_g = min(worst_g, minimum)
        butterfly_failures += failures
        del passed

    values = np.asarray(thetas, dtype=float)
    steps = np.diff(values)
    calendar_failures = int(np.sum(steps < 0))
    return ArbitrageCheck(
        butterfly_free=butterfly_failures == 0,
        calendar_free=ssvi_calendar_free(values),
        min_butterfly_g=worst_g if np.isfinite(worst_g) else float("nan"),
        min_calendar_slope=float(steps.min()) if steps.size else 0.0,
        butterfly_violations=butterfly_failures,
        calendar_violations=calendar_failures,
    )


def diagnose_day(
    surface: SSVI,
    thetas: Sequence[float],
    names: Sequence[str],
    grid: np.ndarray | None = None,
) -> Row:
    """Produce the full diagnostic record for one date.

    Args:
        surface: Calibrated surface for the date.
        thetas: Observed at-the-money total variance per maturity, ascending.
        names: Maturity names matching ``thetas``.
        grid: Log-moneyness grid for the butterfly scan.

    Returns:
        A row carrying the arbitrage outcome, the parameter-level sufficient conditions, and the
        density mass at each maturity.
    """
    check = check_surface(surface, thetas, grid)
    row: Row = {
        "rho": surface.rho,
        "eta": surface.eta,
        "gamma": surface.gamma,
        "butterfly_free": check.butterfly_free,
        "calendar_free": check.calendar_free,
        "arbitrage_free": check.arbitrage_free,
        "min_butterfly_g": check.min_butterfly_g,
        "butterfly_violations": check.butterfly_violations,
        "calendar_violations": check.calendar_violations,
        "min_calendar_slope": check.min_calendar_slope,
    }

    conditions_hold = True
    worst_first = 0.0
    worst_second = 0.0
    worst_mass_error = 0.0
    for name, theta in zip(names, thetas, strict=True):
        if theta <= 0:
            continue
        holds, first, second = ssvi_butterfly_conditions(surface, float(theta))
        conditions_hold = conditions_hold and holds
        worst_first = max(worst_first, first)
        worst_second = max(worst_second, second)
        mass = density_mass(surface.to_raw_svi(float(theta)), float(theta))
        worst_mass_error = max(worst_mass_error, abs(mass - 1.0))
        row[f"density_mass_{name}"] = mass

    row["sufficient_conditions_hold"] = conditions_hold
    row["max_condition_1"] = worst_first
    row["max_condition_2"] = worst_second
    row["max_density_mass_error"] = worst_mass_error
    row["density_valid"] = worst_mass_error <= DENSITY_TOLERANCE
    return row
