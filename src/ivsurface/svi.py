"""SVI and SSVI volatility surfaces, and the conditions that keep them arbitrage-free.

Coordinates
-----------
Everything is expressed in log-moneyness ``k = ln(K / F)`` and **total implied variance**
``w(k, T) = IV(k, T)^2 * T``. Total variance is the natural coordinate because the no-arbitrage
conditions are statements about it: calendar arbitrage is a statement that ``w`` cannot fall as
maturity grows, and butterfly arbitrage is a statement about the curvature of ``w`` in ``k``.

The two parameterisations
-------------------------
**Raw SVI** fits one maturity slice at a time:

    w(k) = a + b * [ rho * (k - m) + sqrt((k - m)^2 + sigma^2) ]

It is a hyperbola in ``k``: linear wings whose slopes are ``b(1 ± rho)``, joined by a smooth
minimum whose depth and position are set by ``sigma`` and ``m``. Five parameters, one slice.

**SSVI** fits the whole surface at once, tying every slice to a single at-the-money variance curve
``theta(T)`` and a shape function ``phi(theta)``:

    w(k, theta) = theta/2 * { 1 + rho*phi(theta)*k + sqrt([phi(theta)*k + rho]^2 + 1 - rho^2) }

That is the form this study calibrates, for a practical reason: ``theta(T)`` is *observable*. Cboe
publishes at-the-money expected volatility at six maturities, so the surface's backbone is market
data rather than a fitted guess.

Why the arbitrage conditions matter
-----------------------------------
A surface that fits quoted prices well can still be nonsense: it can imply a negative probability
density, or that a longer-dated option is worth less than a shorter-dated one. Gatheral and
Jacquier's conditions on SSVI rule both out at the level of the parameters, before any pricing
happens. They are checked here rather than assumed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

logger = logging.getLogger(__name__)

#: A scalar or an array of them.
Numeric: TypeAlias = float | np.ndarray

#: Guard against division by zero in denominators that are positive by construction.
EPSILON = 1e-12


@dataclass(frozen=True)
class RawSVI:
    """Raw SVI parameters for a single maturity slice.

    Attributes:
        a: Vertical level; shifts the whole slice.
        b: Wing slope scale; must not be negative or the wings turn inward.
        rho: Asymmetry between the wings, in (-1, 1). Negative tilts the left wing up, which is
            what an equity index surface looks like.
        m: Horizontal position of the minimum.
        sigma: Curvature at the minimum; must be positive or the slice develops a kink.
    """

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def total_variance(self, k: Numeric) -> np.ndarray:
        """Return total implied variance at log-moneyness ``k``."""
        moneyness = np.asarray(k, dtype=float) - self.m
        return np.asarray(
            self.a + self.b * (self.rho * moneyness + np.sqrt(moneyness**2 + self.sigma**2)),
            dtype=float,
        )

    def implied_volatility(self, k: Numeric, maturity: float) -> np.ndarray:
        """Return implied volatility at ``k`` for a slice of the given maturity."""
        if maturity <= 0:
            raise ValueError("Maturity must be positive to convert total variance to volatility")
        volatility: np.ndarray = np.sqrt(np.maximum(self.total_variance(k), 0.0) / maturity)
        return volatility

    def is_valid(self) -> bool:
        """Whether the parameters describe a usable slice.

        These are the domain constraints of the parameterisation itself, not the no-arbitrage
        conditions: ``b`` negative flips the wings, ``|rho|`` at or beyond one degenerates the
        hyperbola into a line, and a non-positive ``sigma`` puts a kink at the minimum.
        """
        return self.b >= 0 and abs(self.rho) < 1 and self.sigma > 0

    def minimum_variance(self) -> float:
        """Return the lowest total variance the slice attains.

        Negative total variance is meaningless, so this is the quantity that has to stay
        non-negative for the slice to be usable at all.
        """
        return float(self.a + self.b * self.sigma * np.sqrt(max(1.0 - self.rho**2, 0.0)))


def power_law_phi(theta: Numeric, eta: float, gamma: float) -> np.ndarray:
    """Return SSVI's shape function under the power-law choice.

    ``phi(theta) = eta / (theta^gamma * (1 + theta)^(1 - gamma))``

    This is the standard choice because it interpolates between the two behaviours a real surface
    shows: at short maturities (small ``theta``) the skew is steep, and it flattens as maturity
    grows. ``gamma`` controls how fast that flattening happens.

    Args:
        theta: At-the-money total variance.
        eta: Overall skew scale; must be positive.
        gamma: Decay exponent, conventionally in (0, 1).

    Returns:
        The shape function evaluated at ``theta``.

    Raises:
        ValueError: If ``eta`` is not positive or any ``theta`` is not positive.
    """
    values = np.asarray(theta, dtype=float)
    if eta <= 0:
        raise ValueError("eta must be positive")
    if np.any(values <= 0):
        raise ValueError("theta must be positive")
    shape: np.ndarray = eta / (values**gamma * (1.0 + values) ** (1.0 - gamma))
    return shape


@dataclass(frozen=True)
class SSVI:
    """A global SSVI surface.

    Attributes:
        rho: Correlation parameter in (-1, 1), controlling skew. Negative for equity indexes.
        eta: Skew scale of the shape function.
        gamma: Decay exponent of the shape function.
    """

    rho: float
    eta: float
    gamma: float

    def phi(self, theta: Numeric) -> np.ndarray:
        """Return the shape function at the given at-the-money total variance."""
        return power_law_phi(theta, self.eta, self.gamma)

    def total_variance(self, k: Numeric, theta: Numeric) -> np.ndarray:
        """Return total implied variance at log-moneyness ``k`` and at-the-money variance ``theta``.

        Args:
            k: Log-moneyness.
            theta: At-the-money total variance for the maturity in question.

        Returns:
            Total implied variance.
        """
        moneyness = np.asarray(k, dtype=float)
        atm = np.asarray(theta, dtype=float)
        shape = self.phi(atm)
        inner = shape * moneyness + self.rho
        return np.asarray(
            0.5
            * atm
            * (1.0 + self.rho * shape * moneyness + np.sqrt(inner**2 + 1.0 - self.rho**2)),
            dtype=float,
        )

    def implied_volatility(self, k: Numeric, theta: float, maturity: float) -> np.ndarray:
        """Return implied volatility at ``k`` for a slice with at-the-money variance ``theta``."""
        if maturity <= 0:
            raise ValueError("Maturity must be positive")
        volatility: np.ndarray = np.sqrt(np.maximum(self.total_variance(k, theta), 0.0) / maturity)
        return volatility

    def to_raw_svi(self, theta: float) -> RawSVI:
        """Return the equivalent raw-SVI slice for one maturity.

        SSVI is a constrained family inside raw SVI, so every SSVI slice has an exact raw-SVI
        representation. Producing it makes the two parameterisations directly comparable and lets
        slice-level diagnostics run unchanged on a global fit.
        """
        shape = float(self.phi(theta))
        return RawSVI(
            a=0.5 * theta * (1.0 - self.rho**2),
            b=0.5 * theta * shape,
            rho=self.rho,
            m=-self.rho / shape,
            sigma=np.sqrt(1.0 - self.rho**2) / shape,
        )


def butterfly_g(slice_: RawSVI, k: Numeric) -> np.ndarray:
    """Return Gatheral's ``g(k)``, whose sign decides butterfly arbitrage.

    A slice is free of butterfly arbitrage exactly when ``g(k) >= 0`` everywhere. Where ``g`` goes
    negative the implied risk-neutral density is negative there -- a probability that cannot exist,
    which a butterfly spread would monetise.

    Args:
        slice_: The slice to test.
        k: Log-moneyness values to evaluate at.

    Returns:
        ``g(k)`` at each point.
    """
    moneyness = np.asarray(k, dtype=float)
    w = slice_.total_variance(moneyness)
    shifted = moneyness - slice_.m
    root = np.sqrt(shifted**2 + slice_.sigma**2)
    # First and second derivatives of the raw-SVI hyperbola, in closed form.
    first = slice_.b * (slice_.rho + shifted / root)
    second = slice_.b * slice_.sigma**2 / root**3

    safe_w = np.maximum(w, EPSILON)
    term = 1.0 - moneyness * first / (2.0 * safe_w)
    return np.asarray(term**2 - 0.25 * first**2 * (1.0 / safe_w + 0.25) + 0.5 * second, dtype=float)


def implied_density(slice_: RawSVI, k: Numeric) -> np.ndarray:
    """Return the risk-neutral density of log-moneyness implied by a slice.

    Derived from the slice rather than by differentiating prices numerically, so it is exact for
    the parameterisation. A negative value anywhere means the slice is not a valid set of option
    prices.

    Args:
        slice_: The slice.
        k: Log-moneyness values.

    Returns:
        Density values at each point.
    """
    moneyness = np.asarray(k, dtype=float)
    w = np.maximum(slice_.total_variance(moneyness), EPSILON)
    g = butterfly_g(slice_, moneyness)
    # Gatheral's density: g(k) scales the lognormal density implied by the local total variance.
    # With a flat slice g is identically one and this collapses to the Black-Scholes density,
    # which is the check the tests use.
    d_minus = -moneyness / np.sqrt(w) - 0.5 * np.sqrt(w)
    density: np.ndarray = g * np.exp(-0.5 * d_minus**2) / np.sqrt(2.0 * np.pi * w)
    return density


@dataclass(frozen=True)
class ArbitrageCheck:
    """Outcome of testing a surface for static arbitrage.

    Attributes:
        butterfly_free: Whether the density stayed non-negative everywhere tested.
        calendar_free: Whether total variance was non-decreasing in maturity everywhere tested.
        min_butterfly_g: Smallest value of ``g(k)`` found; negative means a violation.
        min_calendar_slope: Smallest increase in total variance between adjacent maturities;
            negative means a violation.
        butterfly_violations: Number of grid points where ``g(k)`` was negative.
        calendar_violations: Number of maturity steps where total variance fell.
    """

    butterfly_free: bool
    calendar_free: bool
    min_butterfly_g: float
    min_calendar_slope: float
    butterfly_violations: int
    calendar_violations: int

    @property
    def arbitrage_free(self) -> bool:
        """Whether the surface passed both tests."""
        return self.butterfly_free and self.calendar_free


def ssvi_butterfly_conditions(surface: SSVI, theta: float) -> tuple[bool, float, float]:
    """Evaluate Gatheral and Jacquier's sufficient conditions for a butterfly-free SSVI slice.

    The two conditions are

        theta * phi(theta) * (1 + |rho|) < 4
        theta * phi(theta)^2 * (1 + |rho|) <= 4

    They are *sufficient*, not necessary: a slice that fails them may still be arbitrage-free, which
    is why the numerical ``g(k)`` scan is run as well. Their value is that they are checkable
    directly from the parameters, before any surface is evaluated.

    Args:
        surface: The SSVI surface.
        theta: At-the-money total variance of the slice.

    Returns:
        Whether both conditions hold, and the two condition values.
    """
    shape = float(surface.phi(theta))
    weight = 1.0 + abs(surface.rho)
    first = theta * shape * weight
    second = theta * shape**2 * weight
    return bool(first < 4.0 and second <= 4.0), first, second


def ssvi_calendar_free(thetas: np.ndarray) -> bool:
    """Whether an at-the-money variance curve is free of calendar arbitrage.

    For SSVI with a fixed shape function, the surface is calendar-arbitrage-free when the
    at-the-money total variance curve is non-decreasing in maturity. That makes the test a property
    of the observed market data rather than of the fit.

    Args:
        thetas: At-the-money total variance, ordered by increasing maturity.

    Returns:
        True when the curve never falls.
    """
    values = np.asarray(thetas, dtype=float)
    if values.size < 2:
        return True
    return bool(np.all(np.diff(values) >= 0))


def max_absolute_correlation(eta: float, gamma: float, theta: float) -> float | None:
    """Return the largest ``|rho|`` a slice can carry and stay butterfly-free.

    Rearranging Gatheral and Jacquier's two sufficient conditions for ``|rho|`` turns them from a
    test into a *bound*. That is what makes constrained calibration possible: instead of fitting
    freely and checking afterwards, the fit can be confined to the region where no arbitrage is
    possible by construction.

        theta * phi * (1 + |rho|) < 4      ->   |rho| < 4 / (theta * phi) - 1
        theta * phi^2 * (1 + |rho|) <= 4   ->   |rho| <= 4 / (theta * phi^2) - 1

    Args:
        eta: Skew scale of the shape function.
        gamma: Decay exponent of the shape function.
        theta: At-the-money total variance of the slice.

    Returns:
        The binding bound on ``|rho|``, capped just inside one, or None when the shape function is
        already too steep for any correlation to rescue -- in which case ``eta`` or ``gamma`` has to
        change, not ``rho``.
    """
    if theta <= 0:
        return None
    shape = float(power_law_phi(theta, eta, gamma))
    product = theta * shape
    if product <= 0:
        return None
    bounds = [4.0 / product - 1.0, 4.0 / (theta * shape**2) - 1.0]
    limit = min(bounds)
    if limit <= 0:
        return None
    return float(min(limit, 0.999))
