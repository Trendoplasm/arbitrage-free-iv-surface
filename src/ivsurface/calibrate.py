"""Fitting a surface to observed market data.

The calibration splits along the two dimensions of a volatility surface, because different data is
available for each.

**Along maturity**, the backbone ``theta(T)`` is *observed* -- Cboe publishes at-the-money expected
volatility at several horizons. Nothing has to be fitted to know it. What can be tested is whether
a smooth variance-curve model, fitted to some of those maturities, predicts a maturity it was never
shown. That is a genuine held-out test, and it is the one this module runs.

**Along strike**, no free data gives volatility strike by strike. What is available is Cboe's SKEW
index, which encodes the risk-neutral *skewness* of the 30-day distribution. So the surface's
strike dimension is calibrated to reproduce that observed skewness -- computed from the surface's
own implied density, not from an approximation -- and the study is explicit that one number per day
is a far thinner constraint than a quoted option chain.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from ivsurface.models import TermStructure
from ivsurface.stats_utils import Samples
from ivsurface.svi import (
    SSVI,
    RawSVI,
    implied_density,
    max_absolute_correlation,
    ssvi_butterfly_conditions,
)

logger = logging.getLogger(__name__)

#: Bounds on the SSVI parameters during fitting: rho strictly inside (-1, 1), eta positive, and
#: gamma in (0, 1) where the power-law shape function behaves as intended.
SSVI_BOUNDS = ((-0.999, 1e-4, 1e-3), (0.999, 20.0, 0.999))

#: Starting point for the SSVI fit: an equity-like downward skew of moderate strength.
SSVI_INITIAL = (-0.7, 1.0, 0.4)

#: Bounds on the variance-curve parameters: all three are variances or rates and must be positive.
CURVE_BOUNDS = ((1e-8, 1e-8, 1e-3), (5.0, 5.0, 50.0))


@dataclass(frozen=True)
class VarianceCurve:
    """A smooth term structure of variance, in the Heston mean-reverting form.

    Average variance to horizon ``T`` is

        vbar(T) = v_long + (v_short - v_long) * (1 - exp(-kappa*T)) / (kappa*T)

    and total variance is ``theta(T) = T * vbar(T)``. The shape is standard because it captures
    what variance curves actually do: start near today's level, decay toward a long-run level, at a
    speed the third parameter sets.

    Attributes:
        v_short: Instantaneous variance, the level the curve starts from.
        v_long: Long-run variance the curve decays toward.
        kappa: Speed of that decay.
    """

    v_short: float
    v_long: float
    kappa: float

    def average_variance(self, maturity: Samples | float) -> np.ndarray:
        """Return average variance to each horizon."""
        years = np.asarray(maturity, dtype=float)
        decay = self.kappa * years
        # The limit of (1 - exp(-x))/x as x -> 0 is 1; computing it directly would divide by zero
        # at the front of the curve, which is exactly where the shortest maturity sits.
        weight = np.where(
            decay > 1e-8, (1.0 - np.exp(-decay)) / np.where(decay > 0, decay, 1.0), 1.0
        )
        return np.asarray(self.v_long + (self.v_short - self.v_long) * weight, dtype=float)

    def total_variance(self, maturity: Samples | float) -> np.ndarray:
        """Return total variance to each horizon."""
        years = np.asarray(maturity, dtype=float)
        return np.asarray(years * self.average_variance(years), dtype=float)

    def is_calendar_free(self, maturities: Samples) -> bool:
        """Whether the fitted curve is non-decreasing in total variance across the maturities."""
        totals = self.total_variance(np.asarray(maturities, dtype=float))
        return bool(np.all(np.diff(totals) >= 0))


def fit_variance_curve(maturities: Samples, total_variances: Samples) -> VarianceCurve:
    """Fit the variance curve to observed total variance.

    Args:
        maturities: Horizons in years.
        total_variances: Observed total variance at each horizon.

    Returns:
        The fitted curve.

    Raises:
        ValueError: If fewer than three points are supplied, which cannot pin three parameters.
    """
    years = np.asarray(maturities, dtype=float)
    observed = np.asarray(total_variances, dtype=float)
    if years.size < 3:
        raise ValueError("Fitting three parameters needs at least three maturities")

    # Start from the data itself: the short end sets v_short, the long end sets v_long.
    initial = (
        float(observed[0] / max(years[0], 1e-8)),
        float(observed[-1] / max(years[-1], 1e-8)),
        2.0,
    )
    clipped = tuple(
        float(np.clip(value, low, high))
        for value, low, high in zip(initial, CURVE_BOUNDS[0], CURVE_BOUNDS[1], strict=True)
    )

    def residual(parameters: np.ndarray) -> np.ndarray:
        curve = VarianceCurve(*parameters)
        errors: np.ndarray = curve.total_variance(years) - observed
        return errors

    # Converge to the floating-point floor rather than to SciPy's default 1e-8. At the default,
    # the search stops as soon as a step looks unproductive, and *where* it stops depends on the
    # host's linear-algebra library: the same day fitted on macOS and on Linux terminated at
    # visibly different parameters. Driving the tolerances down removes that dependence, and
    # tightens the spread of the fit under a perturbed starting point by roughly 600x.
    solution = least_squares(
        residual, clipped, bounds=CURVE_BOUNDS, method="trf", ftol=1e-15, xtol=1e-15, gtol=1e-15
    )
    return VarianceCurve(*solution.x)


def slice_moments(slice_: RawSVI, grid: np.ndarray) -> tuple[float, float, float]:
    """Return the mean, variance, and skewness of the density a slice implies.

    Integrated on a fixed grid with the trapezoid rule and renormalised, so the moments are those of
    a proper distribution even where the grid truncates a thin tail.

    Args:
        slice_: The slice.
        grid: Log-moneyness points to integrate over, ascending and evenly spaced.

    Returns:
        Mean, variance, and skewness of log-moneyness.
    """
    density = np.maximum(implied_density(slice_, grid), 0.0)
    mass = np.trapezoid(density, grid)
    if mass <= 0:
        return float("nan"), float("nan"), float("nan")
    density = density / mass
    mean = float(np.trapezoid(grid * density, grid))
    centered = grid - mean
    variance = float(np.trapezoid(centered**2 * density, grid))
    if variance <= 0:
        return mean, variance, float("nan")
    third = float(np.trapezoid(centered**3 * density, grid))
    return mean, variance, third / variance**1.5


def model_skewness(surface: SSVI, theta: float, grid: np.ndarray) -> float:
    """Return the skewness the surface implies at one at-the-money variance."""
    return slice_moments(surface.to_raw_svi(theta), grid)[2]


#: Weight on the no-arbitrage penalty in a constrained fit. Large enough that a violation always
#: costs more than any improvement in fit it could buy.
ARBITRAGE_PENALTY = 100.0


def condition_excess(surface: SSVI, thetas: Samples) -> float:
    """Return how far a surface breaches the sufficient butterfly conditions.

    Args:
        surface: The surface to test.
        thetas: At-the-money total variances the surface has to cover.

    Returns:
        The largest amount by which either condition exceeds its limit of four, across the given
        maturities. Zero means every slice is comfortably inside the arbitrage-free region.
    """
    worst = 0.0
    for theta in thetas:
        if theta <= 0:
            continue
        _, first, second = ssvi_butterfly_conditions(surface, float(theta))
        worst = max(worst, first - 4.0, second - 4.0)
    return max(worst, 0.0)


#: Smallest scale factor tried when projecting a surface into the arbitrage-free region. Below this
#: the surface has no skew left to speak of, and the right conclusion is that the family cannot
#: represent the target at all.
MIN_ETA_SCALE = 1e-4


def project_to_arbitrage_free(surface: SSVI, thetas: Samples) -> SSVI | None:
    """Shrink a surface's skew scale until it satisfies the butterfly conditions everywhere.

    A penalty term discourages violations; it does not prevent them. Since both conditions are
    increasing in ``eta`` -- the shape function is proportional to it -- scaling ``eta`` down moves
    a surface monotonically toward the feasible region, so a bisection finds the largest scale that
    is admissible. That turns "no-arbitrage enforced" from an aspiration of the objective function
    into a property of the returned surface.

    Args:
        surface: The fitted surface.
        thetas: At-the-money variances the conditions must hold across.

    Returns:
        A surface satisfying the conditions at every maturity, or None if even a vanishing skew
        cannot -- which would indicate a problem with the maturities rather than with the fit.
    """
    if condition_excess(surface, thetas) <= 0:
        return surface

    low, high = MIN_ETA_SCALE, 1.0
    if condition_excess(SSVI(surface.rho, surface.eta * low, surface.gamma), thetas) > 0:
        return None

    for _ in range(60):
        middle = 0.5 * (low + high)
        candidate = SSVI(surface.rho, surface.eta * middle, surface.gamma)
        if condition_excess(candidate, thetas) > 0:
            high = middle
        else:
            low = middle
    projected = SSVI(surface.rho, surface.eta * low, surface.gamma)
    logger.info(
        "Projected eta from %.4f to %.4f to satisfy the butterfly conditions",
        surface.eta,
        projected.eta,
    )
    return projected


def fit_ssvi_to_skewness(
    thetas: Samples,
    skewnesses: Samples,
    grid: np.ndarray,
    *,
    enforce_no_arbitrage: bool = False,
    condition_thetas: Samples | None = None,
) -> tuple[SSVI, float]:
    """Fit global SSVI parameters so the surface reproduces observed skewness.

    The surface's maturity backbone is already observed, so this fit concerns only the strike
    dimension. Fitting one global set of parameters across many days -- rather than three free
    parameters per day -- is deliberate: SSVI is a *global* model, and a surface refitted freely
    every day would tell you nothing about whether the family describes the market.

    Args:
        thetas: Observed at-the-money total variance at the skew horizon, one per date.
        skewnesses: Observed risk-neutral skewness, one per date.
        grid: Log-moneyness grid for the density integration.
        enforce_no_arbitrage: When true, penalise parameter sets that breach the sufficient
            butterfly conditions, so the fit is confined to the arbitrage-free region. When false,
            the fit chases skewness alone -- which is the comparison the study is built around.
        condition_thetas: At-the-money variances the conditions must hold across. Defaults to the
            calibration sample, but a surface used at shorter maturities should be constrained
            there too, since short slices are where the conditions bite hardest.

    Returns:
        The fitted surface and the root-mean-square skewness error.

    Raises:
        ValueError: If no usable observations are supplied.
    """
    atm = np.asarray(thetas, dtype=float)
    target = np.asarray(skewnesses, dtype=float)
    usable = np.isfinite(atm) & np.isfinite(target) & (atm > 0)
    if not usable.any():
        raise ValueError("No usable observations to calibrate against")
    atm, target = atm[usable], target[usable]

    constraint_points = np.asarray(
        atm if condition_thetas is None else condition_thetas, dtype=float
    )

    def residual(parameters: np.ndarray) -> np.ndarray:
        surface = SSVI(
            rho=float(parameters[0]), eta=float(parameters[1]), gamma=float(parameters[2])
        )
        modelled = np.array([model_skewness(surface, float(value), grid) for value in atm])
        # A parameter set that produces an unusable density is pushed away rather than crashing
        # the optimiser.
        errors = np.where(np.isfinite(modelled), modelled - target, 1e3)
        if not enforce_no_arbitrage:
            return errors
        excess = condition_excess(surface, constraint_points)
        return np.append(errors, ARBITRAGE_PENALTY * excess)

    # xtol alone still leaves ftol and gtol at 1e-8, which is where this fit was terminating.
    solution = least_squares(
        residual,
        SSVI_INITIAL,
        bounds=SSVI_BOUNDS,
        method="trf",
        ftol=1e-15,
        xtol=1e-15,
        gtol=1e-15,
    )
    surface = SSVI(rho=float(solution.x[0]), eta=float(solution.x[1]), gamma=float(solution.x[2]))
    if enforce_no_arbitrage:
        projected = project_to_arbitrage_free(surface, constraint_points)
        if projected is None:
            raise ValueError(
                "No surface in this family satisfies the butterfly conditions at the requested "
                "maturities"
            )
        surface = projected
    modelled = np.array([model_skewness(surface, float(value), grid) for value in atm])
    error = float(np.sqrt(np.nanmean((modelled - target) ** 2)))
    logger.info(
        "Calibrated SSVI on %d dates: rho=%.4f eta=%.4f gamma=%.4f, skewness RMSE %.4f",
        atm.size,
        surface.rho,
        surface.eta,
        surface.gamma,
        error,
    )
    return surface, error


def held_out_error(
    curve: VarianceCurve, held_out_maturity: float, observed_total_variance: float
) -> tuple[float, float]:
    """Return the predicted and observed total variance at a maturity the fit never saw.

    Args:
        curve: Curve fitted without the held-out maturity.
        held_out_maturity: Horizon in years.
        observed_total_variance: What the market actually showed there.

    Returns:
        The prediction and its relative error.
    """
    predicted = float(curve.total_variance(held_out_maturity))
    relative = (
        (predicted - observed_total_variance) / observed_total_variance
        if observed_total_variance > 0
        else float("nan")
    )
    return predicted, relative


def calibrate_day(
    curve_input: TermStructure, calibration_names: Sequence[str], held_out_name: str
) -> tuple[VarianceCurve, float, float] | None:
    """Fit the variance curve for one date and score it on the held-out maturity.

    Args:
        curve_input: The observed curve for the date.
        calibration_names: Maturities the fit is allowed to see.
        held_out_name: Maturity kept back.

    Returns:
        The fitted curve, the predicted total variance at the held-out maturity, and the relative
        error there. None when the date lacks the maturities required.
    """
    available = set(curve_input.names)
    if held_out_name not in available or not set(calibration_names).issubset(available):
        return None

    pairs = [
        (curve_input.maturities[curve_input.names.index(name)], curve_input.theta(name))
        for name in calibration_names
    ]
    pairs.sort()
    curve = fit_variance_curve([p[0] for p in pairs], [p[1] for p in pairs])
    held_out_maturity = curve_input.maturities[curve_input.names.index(held_out_name)]
    predicted, relative = held_out_error(curve, held_out_maturity, curve_input.theta(held_out_name))
    return curve, predicted, relative


#: Bounds on the daily correlation refit.
RHO_BOUNDS = (-0.999, 0.999)


def fit_daily_rho(
    surface: SSVI,
    theta: float,
    target_skewness: float,
    grid: np.ndarray,
    *,
    enforce_no_arbitrage: bool = False,
    condition_thetas: Samples | None = None,
) -> float | None:
    """Refit only the correlation parameter for one date, holding the shape function fixed.

    The comparison this enables is the point. A global surface asks whether one parameter set
    describes the market across years; a daily refit asks how much the market moves *within* the
    family. Freeing one parameter against one target is exactly identified, so any remaining error
    is a statement about the family rather than about the fit.

    Args:
        surface: Surface supplying the fixed shape function.
        theta: Observed at-the-money total variance for the date.
        target_skewness: Observed risk-neutral skewness for the date.
        grid: Log-moneyness grid for the density integration.
        enforce_no_arbitrage: When true, the search is bounded by the largest correlation the
            butterfly conditions permit, so the result cannot be an arbitrageable surface.
        condition_thetas: Maturities the bound must respect. The binding one is the shortest, so
            passing the full curve matters.

    Returns:
        The fitted correlation, or None if the date has no usable inputs, or if no correlation can
        satisfy the conditions at the requested maturities.
    """
    if not np.isfinite(theta) or theta <= 0 or not np.isfinite(target_skewness):
        return None

    lower, upper = RHO_BOUNDS
    if enforce_no_arbitrage:
        points = [theta] if condition_thetas is None else list(condition_thetas)
        limits = [
            max_absolute_correlation(surface.eta, surface.gamma, float(value))
            for value in points
            if value > 0
        ]
        if not limits or any(limit is None for limit in limits):
            return None
        bound = min(limit for limit in limits if limit is not None)
        lower, upper = -bound, bound

    def residual(parameters: np.ndarray) -> np.ndarray:
        candidate = SSVI(rho=float(parameters[0]), eta=surface.eta, gamma=surface.gamma)
        modelled = model_skewness(candidate, theta, grid)
        return np.array([modelled - target_skewness if np.isfinite(modelled) else 1e3])

    start = float(np.clip(surface.rho, lower + 1e-6, upper - 1e-6))
    # Tightened for the same reason as the variance-curve fit above: the daily correlation feeds
    # the butterfly and density diagnostics, so an early termination here shows up as
    # cross-platform drift in the arbitrage numbers this study exists to report.
    solution = least_squares(
        residual,
        [start],
        bounds=([lower], [upper]),
        method="trf",
        ftol=1e-15,
        xtol=1e-15,
        gtol=1e-15,
    )
    return float(solution.x[0])
