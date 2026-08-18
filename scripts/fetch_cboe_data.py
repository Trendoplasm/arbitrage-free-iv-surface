#!/usr/bin/env python3
"""Download the Cboe index histories this study calibrates against.

Cboe's data is published under Cboe's terms of use and is not redistributed with this repository,
so a fresh clone fetches it from the source:

    python scripts/fetch_cboe_data.py

The six volatility indexes are the study's backbone. Together they are an *observed* at-the-money
term structure of the S&P 500 -- expected volatility over one day, nine days, one month, three
months, six months and a year -- which is exactly the input an SSVI surface needs. SKEW supplies the
strike dimension: it encodes the risk-neutral skewness of the 30-day distribution, which is the
quantity SSVI's correlation parameter controls. VVIX is carried as a stress indicator.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices"

#: Series required by the study, with the role each one plays.
SERIES: dict[str, str] = {
    "VIX1D": "1-day expected volatility (shortest observed maturity)",
    "VIX9D": "9-day expected volatility",
    "VIX": "30-day expected volatility",
    "VIX3M": "93-day expected volatility",
    "VIX6M": "186-day expected volatility",
    "VIX1Y": "365-day expected volatility (longest observed maturity)",
    "SKEW": "Risk-neutral skewness of the 30-day S&P 500 distribution",
    "VVIX": "Volatility of VIX, carried as a stress indicator",
}

DEFAULT_DEST = Path("data/raw")
TIMEOUT_SECONDS = 60


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Cboe index histories.")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="Download directory.")
    parser.add_argument("--force", action="store_true", help="Re-download existing files.")
    return parser.parse_args(argv)


def fetch(series: str, dest: Path, *, force: bool) -> bool:
    """Download one series, returning True if a file was written."""
    filename = f"{series}_History.csv"
    target = dest / filename
    if target.exists() and not force:
        print(f"  {filename}: already present, skipping")
        return False

    url = f"{BASE_URL}/{filename}"
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:
            payload = response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Could not download {url}: {exc}") from exc

    if b"DATE" not in payload[:200].upper():
        raise RuntimeError(f"{url} did not return a Cboe history file")

    dest.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    print(f"  {filename}: {len(payload) / 1024:.0f} KiB written")
    return True


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"Fetching {len(SERIES)} Cboe index histories into {args.dest}")
    try:
        written = sum(fetch(series, args.dest, force=args.force) for series in SERIES)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Done: {written} written, {len(SERIES) - written} already present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
