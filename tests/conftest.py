"""Shared fixtures.

Most of this suite needs no fixtures at all: the propositions being tested are mathematical
identities about the SVI family, and a test can simply construct the parameters it wants. What is
shared here are a few representative surfaces -- one that behaves like an equity index, one that is
deliberately arbitrageable -- and the synthetic Cboe files the loader and command-line tests read.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from ivsurface.config import REQUIRED_CBOE_FILES, StudyConfig
from ivsurface.models import LevelByDate
from ivsurface.svi import SSVI, RawSVI

#: A surface with an equity-index shape: downward skew, moderate steepness, arbitrage-free.
HEALTHY = SSVI(rho=-0.6, eta=1.0, gamma=0.45)

#: A surface steep enough to imply a negative density at every maturity the study uses. This is
#: what genuine butterfly arbitrage looks like, and it is what the numerical detector must catch.
ARBITRAGEABLE = SSVI(rho=-0.90, eta=8.0, gamma=0.50)

#: A surface that *fails the sufficient conditions yet has a perfectly valid density*. The
#: conditions are sufficient, not necessary, so this case must exist -- and keeping it in the
#: fixtures stops the suite from quietly assuming the parameter test and the numerical test are
#: the same test.
CONDITIONS_FAIL_BUT_VALID = SSVI(rho=-0.95, eta=8.0, gamma=0.05)

#: At-the-money total variances spanning the study's maturities, from nine days to a year.
THETAS: tuple[float, ...] = (0.0009, 0.0031, 0.0112, 0.0251, 0.0534)

#: Levels the synthetic Cboe files are written at, chosen so the term structure rises.
SYNTHETIC_LEVELS: dict[str, float] = {
    "VIX1D": 14.0,
    "VIX9D": 16.0,
    "VIX": 18.0,
    "VIX3M": 20.0,
    "VIX6M": 21.5,
    "VIX1Y": 23.0,
    "SKEW": 125.0,
    "VVIX": 90.0,
}


def flat_slice(total_variance: float) -> RawSVI:
    """Return a slice with no smile at all.

    A flat slice is the reference case for every density test: its implied distribution is exactly
    lognormal, so results can be checked against a closed form rather than against each other.
    """
    return RawSVI(a=total_variance, b=0.0, rho=0.0, m=0.0, sigma=1.0)


def trading_dates(start: date, count: int) -> list[date]:
    """Generate ascending weekday dates."""
    dates: list[date] = []
    current = start
    while len(dates) < count:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    return dates


def write_cboe_csv(
    path: Path, dates: Sequence[date], levels: Sequence[float], *, value_column: str = "CLOSE"
) -> None:
    """Write a Cboe-format history file in either published shape."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        if value_column == "CLOSE":
            writer.writerow(["DATE", "OPEN", "HIGH", "LOW", "CLOSE"])
            for day, level in zip(dates, levels, strict=True):
                writer.writerow([day.strftime("%m/%d/%Y"), level, level, level, level])
        else:
            writer.writerow(["DATE", value_column])
            for day, level in zip(dates, levels, strict=True):
                writer.writerow([day.strftime("%m/%d/%Y"), level])


def write_synthetic_inputs(directory: Path, dates: Sequence[date]) -> None:
    """Write a complete set of Cboe files at levels that rise with maturity.

    A gentle wave is added to the skew index so the daily calibration has something to track;
    a constant series would make every day identical and hide any date-handling error.
    """
    for name in REQUIRED_CBOE_FILES:
        base = SYNTHETIC_LEVELS[name]
        if name == "SKEW":
            levels = [base + 8.0 * np.sin(index / 40.0) for index in range(len(dates))]
            write_cboe_csv(directory / f"{name}_History.csv", dates, levels, value_column="SKEW")
        elif name == "VVIX":
            write_cboe_csv(
                directory / f"{name}_History.csv", dates, [base] * len(dates), value_column="VVIX"
            )
        else:
            levels = [base * (1.0 + 0.05 * np.sin(index / 30.0)) for index in range(len(dates))]
            write_cboe_csv(directory / f"{name}_History.csv", dates, levels)


def series_from(dates: Sequence[date], level: float) -> LevelByDate:
    """Return a constant series."""
    return dict.fromkeys(dates, level)


@pytest.fixture
def dates() -> list[date]:
    """Return 400 weekday trading dates from 2011."""
    return trading_dates(date(2011, 1, 7), 400)


#: Days in the command-line fixture. Deliberately short: those tests check wiring, messages and
#: exit codes, and the pipeline calibrates every day under two regimes, so a long calendar buys no
#: extra coverage at several times the runtime.
CLI_DAYS = 90


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write a synthetic input tree once and share it across the module."""
    root = tmp_path_factory.mktemp("cli")
    write_synthetic_inputs(root / "raw", trading_dates(date(2011, 1, 7), CLI_DAYS))
    return root


@pytest.fixture
def config() -> StudyConfig:
    """Return a configuration sized for the synthetic calendar."""
    return StudyConfig(start_date="2011-01-07", end_date="2026-06-30", bootstrap_iterations=100)


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parent.parent
