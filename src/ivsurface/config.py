"""Study parameters and the fixed data contract."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Final

# --- External data contract -------------------------------------------------------------

#: Observed at-the-money term structure: Cboe index name -> nominal horizon in calendar days.
#: Six published maturities from one day to one year, which together are the backbone an SSVI
#: surface needs. That this curve is observed rather than fitted is what anchors the study.
TERM_STRUCTURE: Final[dict[str, float]] = {
    "VIX1D": 1.0,
    "VIX9D": 9.0,
    "VIX": 30.0,
    "VIX3M": 93.0,
    "VIX6M": 186.0,
    "VIX1Y": 365.0,
}

#: Maturities available across the whole study period. VIX1D only begins in May 2022, so it is
#: reported but excluded from the calibration backbone to keep the sample consistent.
CORE_TERM_STRUCTURE: Final[tuple[str, ...]] = ("VIX9D", "VIX", "VIX3M", "VIX6M", "VIX1Y")

#: Cboe's skew index, which encodes the risk-neutral skewness of the 30-day distribution.
SKEW_INDEX: Final[str] = "SKEW"

#: Volatility of VIX, carried as a stress indicator rather than a calibration input.
VOL_OF_VOL_INDEX: Final[str] = "VVIX"

#: The 30-day point, which SKEW refers to and which the skew calibration is anchored at.
SKEW_HORIZON_DAYS: Final[float] = 30.0

#: Every Cboe series the study downloads.
REQUIRED_CBOE_FILES: Final[dict[str, str]] = {
    name: f"{name}_History.csv" for name in (*TERM_STRUCTURE, SKEW_INDEX, VOL_OF_VOL_INDEX)
}

# --- Conventions ------------------------------------------------------------------------

CALENDAR_DAYS_PER_YEAR: Final[float] = 365.0

#: Cboe quotes volatility in percentage points; the models work in decimals.
POINTS_PER_UNIT: Final[float] = 100.0

#: Cboe defines SKEW = 100 - 10 * S, where S is the risk-neutral skewness of the 30-day
#: log return. Inverting that recovers an observed skewness.
SKEW_INTERCEPT: Final[float] = 100.0
SKEW_SLOPE: Final[float] = 10.0

#: Divisor applied when loading each series. The volatility indexes are quoted in percentage
#: points; SKEW is an index in its own units and must not be rescaled.
SERIES_SCALE: Final[dict[str, float]] = {SKEW_INDEX: 1.0}

#: Log-moneyness grid on which arbitrage diagnostics are evaluated. Wide enough to reach the wings
#: where butterfly violations appear first, fine enough not to step over a narrow one.
DIAGNOSTIC_K_MIN: Final[float] = -1.5
DIAGNOSTIC_K_MAX: Final[float] = 1.5
DIAGNOSTIC_K_POINTS: Final[int] = 601

#: Widest range over which the implied density is integrated when checking it sums to one.
DENSITY_K_LIMIT: Final[float] = 12.0


#: The density grid is built in two parts, because a uniform grid cannot be both wide enough for
#: SVI's fat wings and fine enough for a nine-day slice without becoming enormous. A dense core
#: covers where the mass is; sparse wings carry the tails.
DENSITY_CORE_LIMIT: Final[float] = 2.0
DENSITY_CORE_POINTS: Final[int] = 4001
DENSITY_WING_POINTS: Final[int] = 400

#: Tolerance on the density integral. Numerical quadrature over a truncated range will not return
#: exactly one, and this is the band inside which the surface is treated as a valid distribution.
DENSITY_TOLERANCE: Final[float] = 1e-4

#: Quantile bounds of a two-sided 95% bootstrap interval.
BOOTSTRAP_LOWER_QUANTILE: Final[float] = 0.025
BOOTSTRAP_UPPER_QUANTILE: Final[float] = 0.975

#: Number of days listed in the extreme-condition tables.
EXTREME_DAY_COUNT: Final[int] = 15

#: Market episodes examined individually. Each is a date on which the volatility surface repriced
#: sharply, and they are named so the reader can check the study against their own memory of them.
STRESS_EPISODES: Final[dict[str, str]] = {
    "Volmageddon": "2018-02-05",
    "Covid crash": "2020-03-16",
    "Rate shock": "2022-06-13",
    "Yen carry unwind": "2024-08-05",
    "Tariff shock": "2025-04-04",
}


@dataclass(frozen=True)
class StudyConfig:
    """Windows and settings for one run of the study.

    Attributes:
        start_date: First date of the study period. Defaults to the first date on which the core
            term structure is complete.
        end_date: Last date of the study period. Frozen deliberately: Cboe extends these series
            every trading day, so an open-ended sample would answer differently on every download.
        held_out_maturity: Maturity excluded from calibration and used to test the fit out of
            sample. Holding one out is the difference between fitting a curve and testing one.
        event_window_days: Trading days either side of a stress episode to tabulate.
        stability_window_days: Window over which parameter stability is measured.
        bootstrap_iterations: Resamples used for confidence intervals.
        random_seed: Seed for the bootstrap generator.
    """

    start_date: str = "2011-01-07"
    end_date: str = "2026-06-30"
    held_out_maturity: str = "VIX6M"
    event_window_days: int = 10
    stability_window_days: int = 21
    bootstrap_iterations: int = 10_000
    random_seed: int = 20_260_820

    def start(self) -> date:
        """Return :attr:`start_date` as a date."""
        return datetime.strptime(self.start_date, "%Y-%m-%d").date()

    def end(self) -> date:
        """Return :attr:`end_date` as a date."""
        return datetime.strptime(self.end_date, "%Y-%m-%d").date()

    def calibration_maturities(self) -> tuple[str, ...]:
        """Return the maturities used to calibrate, excluding the held-out one."""
        return tuple(name for name in CORE_TERM_STRUCTURE if name != self.held_out_maturity)
