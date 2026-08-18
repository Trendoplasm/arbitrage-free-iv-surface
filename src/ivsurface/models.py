"""Typed records passed between the stages of the study."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, TypeAlias

#: One output record; key order is the exported column order.
Row: TypeAlias = dict[str, Any]

#: An output table.
Table: TypeAlias = list[Row]

#: A daily level series keyed by trading date.
LevelByDate: TypeAlias = dict[date, float]


@dataclass(frozen=True)
class TermStructure:
    """The observed at-the-money variance curve on one date.

    Attributes:
        date: Trading date.
        maturities: Horizons in years, ascending.
        names: Cboe index name behind each maturity.
        volatilities: Observed at-the-money implied volatility at each maturity, as decimals.
        total_variances: ``volatility^2 * maturity`` at each maturity -- the coordinate every
            no-arbitrage condition is stated in.
    """

    date: date
    maturities: tuple[float, ...]
    names: tuple[str, ...]
    volatilities: tuple[float, ...]
    total_variances: tuple[float, ...]

    def theta(self, name: str) -> float:
        """Return the at-the-money total variance for one named maturity.

        Raises:
            KeyError: If the maturity is not present on this date.
        """
        return self.total_variances[self.names.index(name)]
