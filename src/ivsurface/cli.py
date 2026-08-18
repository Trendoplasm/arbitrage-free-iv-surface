"""Command-line interface for running the study."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from ivsurface import __version__
from ivsurface.config import CORE_TERM_STRUCTURE, StudyConfig
from ivsurface.pipeline import headline, run_study, write_outputs

logger = logging.getLogger(__name__)

DESCRIPTION = "Measure the implied-correlation premium and backtest the ivsurface trade."

EPILOG = """\
example:
  ivsurface --data-dir data/raw --output-dir outputs

Inputs are downloaded first:
  python scripts/fetch_cboe_data.py
"""


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="ivsurface",
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory holding the downloaded histories. (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for tables, plots, and the JSON summary. (default: %(default)s)",
    )
    parser.add_argument(
        "--held-out-maturity",
        default=StudyConfig.held_out_maturity,
        help=(
            "Maturity excluded from the term-structure fit and used to test it out of sample. "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--start-date",
        default=StudyConfig.start_date,
        help="First date of the study period. (default: %(default)s)",
    )
    parser.add_argument(
        "--end-date",
        default=StudyConfig.end_date,
        help=(
            "Last date of the study period. Frozen by default so results reproduce from a "
            "download taken at any later date. (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=StudyConfig.bootstrap_iterations,
        help="Bootstrap resamples for confidence intervals. (default: %(default)s)",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG plot generation.")
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("--quiet", action="store_true", help="Report errors only.")
    verbosity.add_argument("--verbose", action="store_true", help="Report per-stage progress.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the study from command-line arguments.

    Returns:
        0 on success, 2 on a bad argument, 1 on a data or computation failure. Failures are
        reported as one readable line rather than a traceback, because their realistic causes -- a
        missing download, a malformed reference file -- are the user's to fix.
    """
    args = build_parser().parse_args(argv)

    level = logging.WARNING if args.quiet else logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr
    )

    if args.bootstrap_iterations <= 0:
        print("error: --bootstrap-iterations must be positive", file=sys.stderr)
        return 2
    if args.held_out_maturity not in CORE_TERM_STRUCTURE:
        print(
            f"error: --held-out-maturity must be one of {', '.join(CORE_TERM_STRUCTURE)}",
            file=sys.stderr,
        )
        return 2

    config = StudyConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        held_out_maturity=args.held_out_maturity,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    try:
        results = run_study(args.data_dir, config)
        write_outputs(results, args.output_dir, with_plots=not args.no_plots)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(headline(results))
        print(f"Outputs written to {args.output_dir.resolve()}")
    return 0
