"""Reader for the study's single input: Cboe index history.

Every reader fails loudly. A silently dropped observation or a coerced date would change a
published statistic without changing anything visible.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path

from ivsurface.config import POINTS_PER_UNIT, REQUIRED_CBOE_FILES, SERIES_SCALE
from ivsurface.models import LevelByDate

logger = logging.getLogger(__name__)

CBOE_DATE_COLUMN = "DATE"
CBOE_CLOSE_COLUMN = "CLOSE"
CBOE_DATE_FORMAT = "%m/%d/%Y"


def resolve_value_column(fieldnames: Sequence[str]) -> str:
    """Return the column holding the daily level.

    Cboe publishes these files in two shapes. The volatility and correlation indexes carry
    ``DATE,OPEN,HIGH,LOW,CLOSE``; the ivsurface index carries ``DATE`` plus a single column named
    after the index itself. Rather than special-casing series names, take ``CLOSE`` when it exists
    and otherwise the sole remaining column.

    Args:
        fieldnames: Header of the file.

    Returns:
        Name of the value column.

    Raises:
        ValueError: If there is no ``DATE`` column, or no unambiguous value column.
    """
    if CBOE_DATE_COLUMN not in fieldnames:
        raise ValueError(f"No {CBOE_DATE_COLUMN} column; got {list(fieldnames)}")
    if CBOE_CLOSE_COLUMN in fieldnames:
        return CBOE_CLOSE_COLUMN
    others = [name for name in fieldnames if name != CBOE_DATE_COLUMN]
    if len(others) != 1:
        raise ValueError(f"Cannot identify the value column among {list(fieldnames)}")
    return others[0]


def load_cboe_history(path: Path, *, scale: float = POINTS_PER_UNIT) -> LevelByDate:
    """Load a Cboe index history as a decimal level keyed by trading date.

    Cboe quotes its volatility indexes in percentage points, so they are divided by 100 here,
    once, and no downstream formula has to remember the convention. SKEW is different: it is an
    index near 100 that encodes a skewness through ``SKEW = 100 - 10 * S``, not a percentage, so
    it is loaded unscaled. Making the scale an argument keeps that distinction explicit rather than
    hidden in a special case.

    Args:
        path: Path to a ``*_History.csv`` file as published by Cboe.
        scale: Divisor applied to every level. 100 for a percentage-quoted index, 1 for an index
            that is already in its natural units.

    Returns:
        Decimal level keyed by trading date.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If the header is unusable, a row will not parse, a level is not positive, or
            the file holds no data rows.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing Cboe input: {path}")

    levels: LevelByDate = {}
    skipped: list[date] = []
    # utf-8-sig: Cboe's files carry a byte-order mark that would corrupt the "DATE" header.
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header row")
        try:
            value_column = resolve_value_column(reader.fieldnames)
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from exc
        for raw in reader:
            try:
                trading_date = datetime.strptime(raw[CBOE_DATE_COLUMN], CBOE_DATE_FORMAT).date()
                level = float(raw[value_column])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid row in {path}: {raw}") from exc
            if level <= 0:
                # Cboe encodes an occasional missing print as zero -- the ivsurface index has one,
                # on 8 February 2018, sitting between values of 14.05 and 20.92. A volatility or
                # correlation level of zero is not a real observation, so it is treated as missing
                # and logged. Dropping it silently is what this loader exists to avoid.
                skipped.append(trading_date)
                continue
            levels[trading_date] = level / scale

    if not levels:
        raise ValueError(f"No data rows found in {path}")
    if skipped:
        logger.warning(
            "%s: skipped %d nonpositive level(s), treated as missing: %s",
            path.name,
            len(skipped),
            ", ".join(day.isoformat() for day in skipped[:5]),
        )
    return levels


def load_series(data_dir: Path) -> dict[str, LevelByDate]:
    """Load every Cboe series the study requires.

    Args:
        data_dir: Directory holding the downloaded ``*_History.csv`` files.

    Returns:
        Decimal levels keyed by series name.
    """
    return {
        name: load_cboe_history(data_dir / filename, scale=SERIES_SCALE.get(name, POINTS_PER_UNIT))
        for name, filename in REQUIRED_CBOE_FILES.items()
    }
