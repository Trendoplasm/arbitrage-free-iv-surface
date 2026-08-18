"""End-to-end reproduction of the published results.

This is the test that guards the numbers. It runs the complete study against the downloaded Cboe
histories and compares every exported table with the committed results.

It skips itself when the inputs are absent, which is the case on a fresh clone and in continuous
integration, because Cboe's index history is not redistributed here. Populate it with
``python scripts/fetch_cboe_data.py``.

It is slow -- the study calibrates every trading day under two regimes -- which is why it carries
its own marker and is excluded from the default fast run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ivsurface.config import StudyConfig
from ivsurface.models import Row
from ivsurface.pipeline import (
    CONSTRAINED,
    UNCONSTRAINED,
    StudyResults,
    headline,
    run_study,
    write_outputs,
)
from ivsurface.verify import compare_output_dirs

EXPECTED_HEADLINE = (
    "Completed 3891 days: 0 calendar violations in the observed curve, held-out maturity "
    "predicted to 1.82% median error, and enforcing no-arbitrage moved the surface from 78.9% to "
    "100.0% valid at a skewness RMSE of 0.067 against 0.633."
)

EXPECTED_PANEL_DAYS = 3891
EXPECTED_TABLE_COUNT = 15

pytestmark = pytest.mark.golden


@pytest.fixture(scope="module")
def data_dir() -> Path:
    root = Path(__file__).resolve().parent.parent
    directory = root / "data" / "raw"
    if not directory.is_dir():
        pytest.skip("downloaded inputs absent; run scripts/fetch_cboe_data.py to enable")
    return directory


@pytest.fixture(scope="module")
def results(data_dir: Path) -> StudyResults:
    return run_study(data_dir, StudyConfig())


def test_headline_result_is_unchanged(results: StudyResults) -> None:
    assert headline(results) == EXPECTED_HEADLINE


def test_sample_size(results: StudyResults) -> None:
    assert len(results.panel) == EXPECTED_PANEL_DAYS


def test_the_observed_curve_never_breached_the_calendar_condition(
    results: StudyResults,
) -> None:
    # The study's first finding, and a clean one: across fifteen years of published data, including
    # every major volatility shock in that period, total variance never once fell with maturity.
    assert results.calendar["days_with_violation"] == 0
    assert results.calendar["min_forward_variance"] > 0


def test_an_inverted_volatility_curve_is_not_an_arbitrage(results: StudyResults) -> None:
    # The distinction the study exists to make precise. The volatility curve inverted on about a
    # tenth of all days, and not one of those days was arbitrageable, because total variance
    # carries a factor of maturity that a volatility comparison discards.
    inversions = results.inversions
    assert inversions["days_volatility_curve_inverted"] > 300
    assert inversions["inverted_and_arbitrageable"] == 0
    assert inversions["inverted_yet_calendar_free"] == inversions["days_volatility_curve_inverted"]


def test_the_term_structure_interpolates_a_maturity_it_never_saw(
    results: StudyResults,
) -> None:
    stats = results.held_out_stats
    assert stats["median_absolute_error"] < 0.02
    assert stats["share_within_5pct"] > 0.90


def test_enforcing_no_arbitrage_makes_every_day_valid(results: StudyResults) -> None:
    constrained = next(row for row in results.comparison if row["regime"] == CONSTRAINED)
    assert constrained["arbitrage_free_rate"] == pytest.approx(1.0)
    assert constrained["sufficient_conditions_rate"] == pytest.approx(1.0)
    assert constrained["valid_density_rate"] == pytest.approx(1.0)
    assert constrained["worst_butterfly_g"] > 0


def test_fitting_skew_alone_does_not(results: StudyResults) -> None:
    # Chasing the observed skew without a constraint produces surfaces implying negative
    # probabilities on a fifth of days. That is the failure the study is named for.
    unconstrained = next(row for row in results.comparison if row["regime"] == UNCONSTRAINED)
    assert unconstrained["arbitrage_free_rate"] < 0.85
    assert unconstrained["sufficient_conditions_rate"] == pytest.approx(0.0)
    assert unconstrained["worst_butterfly_g"] < 0


def test_the_constraint_costs_fit_quality(results: StudyResults) -> None:
    by_regime = {row["regime"]: row for row in results.comparison}
    assert by_regime[CONSTRAINED]["skewness_rmse"] > by_regime[UNCONSTRAINED]["skewness_rmse"]


def test_the_skew_the_constraint_cannot_reach_is_the_extreme_skew(
    results: StudyResults,
) -> None:
    # Where the constrained surface misses, it misses on the days the market priced the most
    # negative skew -- so the limitation is a property of those days, not a scattering of noise.
    rows = [
        row
        for row in results.daily_parameters
        if row["regime"] == CONSTRAINED and row["skewness_error"] is not None
    ]
    matched = [row for row in rows if abs(row["skewness_error"]) < 1e-6]
    missed = [row for row in rows if abs(row["skewness_error"]) >= 1e-6]
    assert matched and missed

    def average_target(group: list[Row]) -> float:
        targets = [float(row["target_skewness"]) for row in group]
        return sum(targets) / len(targets)

    assert average_target(missed) < average_target(matched)


def test_no_stress_episode_produced_a_violation(results: StudyResults) -> None:
    assert results.episodes
    for episode in results.episodes:
        assert episode["calendar_violations_in_window"] == 0


def test_the_frozen_end_date_bounds_the_sample(results: StudyResults) -> None:
    cutoff = results.config.end()
    assert max(row["date"] for row in results.panel) <= cutoff


def test_all_tables_match_the_committed_results(results: StudyResults, tmp_path: Path) -> None:
    committed = Path(__file__).resolve().parent.parent / "outputs"
    if not (committed / "tables").is_dir():
        pytest.skip("no committed outputs to compare against")

    write_outputs(results, tmp_path, with_plots=False)
    comparison = compare_output_dirs(committed, tmp_path)
    assert comparison.matches, "\n".join(comparison.discrepancies[:20])
    assert comparison.compared_files == EXPECTED_TABLE_COUNT
