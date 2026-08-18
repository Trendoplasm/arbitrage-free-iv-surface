"""Arbitrage-free implied-volatility surface construction and calibration.

The maturity dimension is calibrated to Cboe's published term structure and the strike dimension to
its published skew index, so the surface is anchored to observed data. The no-arbitrage conditions
it is tested against are theorems about the parameterisation, and are verified numerically rather
than assumed. See ``README.md`` for what the study does and does not claim.
"""

from __future__ import annotations

from ivsurface.config import StudyConfig
from ivsurface.pipeline import StudyResults, headline, run_study, write_outputs
from ivsurface.svi import SSVI, ArbitrageCheck, RawSVI

__version__ = "1.0.0"

__all__ = [
    "SSVI",
    "ArbitrageCheck",
    "RawSVI",
    "StudyConfig",
    "StudyResults",
    "__version__",
    "headline",
    "run_study",
    "write_outputs",
]
