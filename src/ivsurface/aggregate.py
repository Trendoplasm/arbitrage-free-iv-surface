"""Summaries of the calibration and its diagnostics."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import date, timedelta

import numpy as np

from ivsurface.config import (
    BOOTSTRAP_LOWER_QUANTILE,
    BOOTSTRAP_UPPER_QUANTILE,
    EXTREME_DAY_COUNT,
    STRESS_EPISODES,
)
from ivsurface.models import Row, Table
from ivsurface.stats_utils import mean_or_none, median_or_none, std_dev_or_none

logger = logging.getLogger(__name__)


def calendar_summary(panel: Sequence[Row]) -> Row:
    """Summarise the calendar-arbitrage test on the observed curve.

    Args:
        panel: Daily term-structure rows.

    Returns:
        How many days were tested and how many breached the condition.

    Raises:
        ValueError: If the panel is empty.
    """
    if not panel:
        raise ValueError("No term-structure days to summarise")
    violations = [row for row in panel if not row["calendar_free"]]
    forwards = [
        row["min_forward_variance"]
        for row in panel
        if np.isfinite(row.get("min_forward_variance", np.nan))
    ]
    return {
        "days_tested": len(panel),
        "first_date": min(row["date"] for row in panel),
        "last_date": max(row["date"] for row in panel),
        "days_with_violation": len(violations),
        "violation_rate": len(violations) / len(panel),
        "worst_variance_drop": min((row["worst_variance_drop"] for row in violations), default=0.0),
        "min_forward_variance": min(forwards) if forwards else None,
        "mean_min_forward_variance": mean_or_none(forwards),
        "days_forward_variance_negative": sum(1 for value in forwards if value < 0),
    }


def inversion_summary(panel: Sequence[Row], names: Sequence[str]) -> Row:
    """Count days on which the volatility curve inverted but total variance still rose.

    This is the distinction the study exists to make precise. A "VIX curve inversion" -- short-dated
    implied volatility above long-dated -- is a familiar stress signal, and it is routinely confused
    with a calendar arbitrage. It is not one. Total variance carries a factor of maturity, so a
    curve can invert in volatility while total variance keeps rising, and no arbitrage exists.

    Args:
        panel: Daily term-structure rows.
        names: Maturity names, ascending.

    Returns:
        Counts of each state.
    """
    short, long = names[0], names[-1]
    inverted = 0
    inverted_but_free = 0
    for row in panel:
        if f"iv_{short}" not in row or f"iv_{long}" not in row:
            continue
        if row[f"iv_{short}"] > row[f"iv_{long}"]:
            inverted += 1
            if row["calendar_free"]:
                inverted_but_free += 1
    return {
        "days_tested": len(panel),
        "short_maturity": short,
        "long_maturity": long,
        "days_volatility_curve_inverted": inverted,
        "inversion_rate": inverted / len(panel) if panel else 0.0,
        "inverted_yet_calendar_free": inverted_but_free,
        "inverted_and_arbitrageable": inverted - inverted_but_free,
    }


def held_out_summary(rows: Sequence[Row]) -> Row:
    """Summarise the held-out maturity test.

    Args:
        rows: Daily calibration rows carrying a relative error.

    Returns:
        Central tendency and tail of the prediction error.

    Raises:
        ValueError: If no usable rows are supplied.
    """
    errors = np.array(
        [
            row["held_out_relative_error"]
            for row in rows
            if row["held_out_relative_error"] is not None
        ],
        dtype=float,
    )
    if errors.size == 0:
        raise ValueError("No held-out predictions to summarise")
    absolute = np.abs(errors)
    return {
        "n": int(errors.size),
        "mean_relative_error": float(np.mean(errors)),
        "median_absolute_error": float(np.median(absolute)),
        "p90_absolute_error": float(np.quantile(absolute, 0.90)),
        "p99_absolute_error": float(np.quantile(absolute, 0.99)),
        "worst_absolute_error": float(np.max(absolute)),
        "share_within_1pct": float(np.mean(absolute <= 0.01)),
        "share_within_5pct": float(np.mean(absolute <= 0.05)),
    }


def calibration_comparison(regimes: Sequence[Row]) -> Table:
    """Lay the constrained and unconstrained fits side by side.

    This is the study's central table: what enforcing no-arbitrage costs, and what it buys.
    """
    return list(regimes)


def parameter_stability(rows: Sequence[Row], window: int) -> Table:
    """Summarise how much the daily correlation parameter moves.

    A surface family that needs a wildly different parameter every day is describing the market
    less well than its fit statistics suggest, so the movement is reported alongside the fit.

    Args:
        rows: Daily parameter rows, ascending by date.
        window: Window over which rolling movement is measured.

    Returns:
        One row per calibration regime.
    """
    summary: Table = []
    for regime in sorted({row["regime"] for row in rows}):
        values = np.array(
            [row["rho"] for row in rows if row["regime"] == regime and row["rho"] is not None],
            dtype=float,
        )
        if values.size < 2:
            continue
        changes = np.abs(np.diff(values))
        rolling = (
            [float(np.std(values[i : i + window], ddof=1)) for i in range(values.size - window)]
            if values.size > window
            else []
        )
        summary.append(
            {
                "regime": regime,
                "n": int(values.size),
                "mean_rho": float(np.mean(values)),
                "median_rho": float(np.median(values)),
                "sd_rho": std_dev_or_none(values),
                "min_rho": float(np.min(values)),
                "max_rho": float(np.max(values)),
                "mean_absolute_daily_change": float(np.mean(changes)),
                "max_absolute_daily_change": float(np.max(changes)),
                "mean_rolling_sd": mean_or_none(rolling),
                "share_at_bound": float(np.mean(np.abs(values) > 0.99)),
            }
        )
    return summary


def stress_episodes(panel: Sequence[Row], window: int) -> Table:
    """Tabulate the observed curve around each named market shock.

    Args:
        panel: Daily term-structure rows, ascending by date.
        window: Calendar days either side of the episode to include.

    Returns:
        One row per episode for which data exists.
    """
    by_date = {row["date"]: row for row in panel}
    rows: Table = []
    for label, iso in STRESS_EPISODES.items():
        event = date.fromisoformat(iso)
        nearby = [row for row in panel if abs((row["date"] - event).days) <= window]
        if not nearby:
            continue
        on_the_day = by_date.get(event)
        rows.append(
            {
                "episode": label,
                "date": event,
                "days_in_window": len(nearby),
                "index_iv_on_the_day": None if on_the_day is None else on_the_day.get("iv_VIX"),
                "peak_index_iv": max(
                    (row["iv_VIX"] for row in nearby if "iv_VIX" in row), default=None
                ),
                "peak_skew_index": max(
                    (row["skew_index"] for row in nearby if row.get("skew_index") is not None),
                    default=None,
                ),
                "most_negative_skewness": min(
                    (
                        row["risk_neutral_skewness"]
                        for row in nearby
                        if row.get("risk_neutral_skewness") is not None
                    ),
                    default=None,
                ),
                "calendar_violations_in_window": sum(not row["calendar_free"] for row in nearby),
                "volatility_curve_inverted_days": sum(
                    1
                    for row in nearby
                    if "iv_VIX9D" in row and "iv_VIX1Y" in row and row["iv_VIX9D"] > row["iv_VIX1Y"]
                ),
            }
        )
    return rows


def extreme_days(panel: Sequence[Row], key: str, *, largest: bool) -> Table:
    """Return the days with the most extreme value of one field."""
    usable = [row for row in panel if row.get(key) is not None and np.isfinite(row[key])]
    ordered = sorted(usable, key=lambda row: row[key], reverse=largest)
    return ordered[:EXTREME_DAY_COUNT]


def bootstrap_interval(
    values: Sequence[float], rng: np.random.Generator, iterations: int
) -> tuple[float, float, float] | None:
    """Bootstrap a mean and its 95% interval, or None for an empty sample."""
    sample = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if sample.size == 0:
        return None
    draws = rng.integers(0, sample.size, size=(iterations, sample.size))
    means = sample[draws].mean(axis=1)
    return (
        float(np.quantile(means, BOOTSTRAP_LOWER_QUANTILE)),
        float(np.quantile(means, BOOTSTRAP_UPPER_QUANTILE)),
        float(np.mean(means)),
    )


def yearly_summary(panel: Sequence[Row]) -> Table:
    """Summarise the observed curve and skew by calendar year."""
    years = sorted({row["date"].year for row in panel})
    summary: Table = []
    for year in years:
        subset = [row for row in panel if row["date"].year == year]
        summary.append(
            {
                "year": year,
                "n": len(subset),
                "mean_index_iv": mean_or_none([row["iv_VIX"] for row in subset if "iv_VIX" in row]),
                "mean_skew_index": mean_or_none(
                    [row["skew_index"] for row in subset if row.get("skew_index") is not None]
                ),
                "mean_risk_neutral_skewness": mean_or_none(
                    [
                        row["risk_neutral_skewness"]
                        for row in subset
                        if row.get("risk_neutral_skewness") is not None
                    ]
                ),
                "median_risk_neutral_skewness": median_or_none(
                    [
                        row["risk_neutral_skewness"]
                        for row in subset
                        if row.get("risk_neutral_skewness") is not None
                    ]
                ),
                "calendar_violations": sum(not row["calendar_free"] for row in subset),
            }
        )
    return summary


#: Re-exported for callers that build windows around episodes.
EPISODE_WINDOW = timedelta
