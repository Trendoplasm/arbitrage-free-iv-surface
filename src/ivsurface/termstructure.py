"""The observed at-the-money variance curve, and the calendar condition it must satisfy.

Cboe publishes expected S&P 500 volatility at six horizons. Converting each to total variance,
``theta(T) = sigma(T)^2 * T``, produces an *observed* backbone for the surface -- the one input an
SSVI calibration would otherwise have to guess.

It also makes one no-arbitrage condition directly testable on published market data. Calendar
arbitrage is precisely the statement that total variance cannot fall as maturity grows: if it did,
a longer-dated option would be cheaper than a shorter-dated one covering the same outcomes, and the
calendar spread between them would be free money. Whether the published curve ever violates that is
an empirical question, and this module is what asks it.

A caveat the study states rather than buries: these indexes are model-free variance-swap-style
measures, not traded option prices. A violation in the published curve is not by itself a tradeable
arbitrage -- it is a sign that the two horizons were measured under conditions different enough to
break the ordering, which usually means a fast-moving market.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import date

import numpy as np

from ivsurface.config import (
    CALENDAR_DAYS_PER_YEAR,
    SKEW_INDEX,
    SKEW_INTERCEPT,
    SKEW_SLOPE,
    TERM_STRUCTURE,
)
from ivsurface.models import LevelByDate, Row, Table, TermStructure

logger = logging.getLogger(__name__)


def maturity_years(name: str) -> float:
    """Return the horizon of a Cboe volatility index in years.

    Raises:
        KeyError: If the name is not one of the study's maturities.
    """
    return TERM_STRUCTURE[name] / CALENDAR_DAYS_PER_YEAR


def build_term_structure(
    series: Mapping[str, LevelByDate], trading_date: date, names: Sequence[str]
) -> TermStructure | None:
    """Assemble the observed variance curve for one date.

    Args:
        series: Loaded Cboe histories keyed by index name.
        trading_date: Date to assemble.
        names: Maturities to include, in any order; the result is sorted by maturity.

    Returns:
        The curve, or None if fewer than two of the requested maturities published that day. One
        point is not a term structure and cannot be tested for anything.
    """
    points: list[tuple[float, str, float]] = []
    for name in names:
        level = series[name].get(trading_date)
        if level is not None and level > 0:
            points.append((maturity_years(name), name, level))
    if len(points) < 2:
        return None

    points.sort()
    maturities = tuple(point[0] for point in points)
    labels = tuple(point[1] for point in points)
    volatilities = tuple(point[2] for point in points)
    total_variances = tuple(
        volatility**2 * maturity
        for volatility, maturity in zip(volatilities, maturities, strict=True)
    )
    return TermStructure(
        date=trading_date,
        maturities=maturities,
        names=labels,
        volatilities=volatilities,
        total_variances=total_variances,
    )


def calendar_violations(curve: TermStructure) -> list[tuple[str, str, float]]:
    """Return every adjacent maturity pair where total variance falls.

    Args:
        curve: The observed curve.

    Returns:
        Tuples of ``(shorter maturity, longer maturity, drop in total variance)``, one per
        violation. An empty list means the curve is free of calendar arbitrage.
    """
    violations: list[tuple[str, str, float]] = []
    for index in range(len(curve.total_variances) - 1):
        change = curve.total_variances[index + 1] - curve.total_variances[index]
        if change < 0:
            violations.append((curve.names[index], curve.names[index + 1], change))
    return violations


def forward_variance(curve: TermStructure) -> list[tuple[str, str, float]]:
    """Return the annualised forward variance implied between adjacent maturities.

    Forward variance is the variance the market prices for the period *between* two maturities. It
    is the quantity that goes negative when the calendar condition fails, which makes it the more
    interpretable way to report a violation: a negative forward variance is an impossible price for
    a real period of time.

    Args:
        curve: The observed curve.

    Returns:
        Tuples of ``(shorter maturity, longer maturity, annualised forward variance)``.
    """
    forwards: list[tuple[str, str, float]] = []
    for index in range(len(curve.maturities) - 1):
        span = curve.maturities[index + 1] - curve.maturities[index]
        if span <= 0:
            continue
        change = curve.total_variances[index + 1] - curve.total_variances[index]
        forwards.append((curve.names[index], curve.names[index + 1], change / span))
    return forwards


def skew_to_skewness(skew_level: float) -> float:
    """Convert Cboe's SKEW index into the risk-neutral skewness it encodes.

    Cboe defines ``SKEW = 100 - 10 * S`` where ``S`` is the skewness of the 30-day log return under
    the risk-neutral measure. Inverting gives an observed skewness, which is the quantity SSVI's
    correlation parameter controls -- so this is what ties the surface's strike dimension to market
    data rather than to an assumption.

    Args:
        skew_level: The published index level, in its own units.

    Returns:
        Risk-neutral skewness. Negative for equity indexes: the left tail is fatter.
    """
    return (SKEW_INTERCEPT - skew_level) / SKEW_SLOPE


def build_panel(
    series: Mapping[str, LevelByDate],
    dates: Sequence[date],
    names: Sequence[str],
) -> Table:
    """Build the daily record of the observed curve and its calendar diagnostics.

    Args:
        series: Loaded Cboe histories.
        dates: Trading dates to cover.
        names: Maturities to include.

    Returns:
        One row per date on which a curve could be assembled.
    """
    rows: Table = []
    for trading_date in dates:
        curve = build_term_structure(series, trading_date, names)
        if curve is None:
            continue
        violations = calendar_violations(curve)
        forwards = forward_variance(curve)
        skew_level = series[SKEW_INDEX].get(trading_date)

        row: Row = {
            "date": trading_date,
            "maturities_available": len(curve.names),
            "calendar_violations": len(violations),
            "calendar_free": not violations,
            "worst_variance_drop": min((change for _, _, change in violations), default=0.0),
            "min_forward_variance": min((value for _, _, value in forwards), default=float("nan")),
            "skew_index": skew_level,
            "risk_neutral_skewness": (
                skew_to_skewness(skew_level) if skew_level is not None else None
            ),
        }
        for name, volatility, total in zip(
            curve.names, curve.volatilities, curve.total_variances, strict=True
        ):
            row[f"iv_{name}"] = volatility
            row[f"theta_{name}"] = total
        rows.append(row)

    violation_count = sum(not row["calendar_free"] for row in rows)
    logger.info(
        "Built term-structure panel: %d days, %d with a calendar violation (%.2f%%)",
        len(rows),
        violation_count,
        100.0 * violation_count / len(rows) if rows else 0.0,
    )
    return rows


def slope_summary(curve: TermStructure) -> float:
    """Return the average annualised slope of the variance curve.

    A positive slope is the normal state: uncertainty compounds with horizon. A negative slope means
    the market expects the near term to be more turbulent than the long term, which is what a curve
    looks like in a panic.
    """
    if len(curve.maturities) < 2:
        return float("nan")
    span = curve.maturities[-1] - curve.maturities[0]
    change = curve.total_variances[-1] - curve.total_variances[0]
    return float(change / span) if span > 0 else float("nan")


def average_curve(rows: Sequence[Row], names: Sequence[str]) -> Table:
    """Summarise the observed curve across the sample, one row per maturity."""
    summary: Table = []
    for name in names:
        volatilities = [row[f"iv_{name}"] for row in rows if f"iv_{name}" in row]
        totals = [row[f"theta_{name}"] for row in rows if f"theta_{name}" in row]
        if not volatilities:
            continue
        summary.append(
            {
                "index": name,
                "horizon_days": TERM_STRUCTURE[name],
                "maturity_years": maturity_years(name),
                "n": len(volatilities),
                "mean_iv": float(np.mean(volatilities)),
                "median_iv": float(np.median(volatilities)),
                "p10_iv": float(np.quantile(volatilities, 0.10)),
                "p90_iv": float(np.quantile(volatilities, 0.90)),
                "mean_total_variance": float(np.mean(totals)),
            }
        )
    return summary
