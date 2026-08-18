"""Summaries of the observed curve and the calibration."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import pytest

from ivsurface.aggregate import (
    bootstrap_interval,
    calendar_summary,
    extreme_days,
    held_out_summary,
    inversion_summary,
    parameter_stability,
    stress_episodes,
    yearly_summary,
)
from ivsurface.config import STRESS_EPISODES


def panel_row(
    day: date,
    *,
    calendar_free: bool = True,
    short_iv: float = 0.15,
    long_iv: float = 0.20,
    skewness: float = -2.0,
) -> dict[str, Any]:
    return {
        "date": day,
        "calendar_free": calendar_free,
        "worst_variance_drop": 0.0 if calendar_free else -0.01,
        "min_forward_variance": 0.02 if calendar_free else -0.005,
        "iv_VIX9D": short_iv,
        "iv_VIX": 0.18,
        "iv_VIX1Y": long_iv,
        "skew_index": 100.0 - 10.0 * skewness,
        "risk_neutral_skewness": skewness,
    }


def panel(days: int = 60) -> list[dict[str, Any]]:
    start = date(2015, 1, 5)
    return [panel_row(start + timedelta(days=index)) for index in range(days)]


class TestCalendarSummary:
    def test_counts_a_clean_panel_as_free(self) -> None:
        summary = calendar_summary(panel())
        assert summary["days_with_violation"] == 0
        assert summary["violation_rate"] == pytest.approx(0.0)
        assert summary["days_forward_variance_negative"] == 0

    def test_counts_violations(self) -> None:
        rows = panel()
        rows[3]["calendar_free"] = False
        rows[3]["worst_variance_drop"] = -0.02
        rows[3]["min_forward_variance"] = -0.01
        summary = calendar_summary(rows)
        assert summary["days_with_violation"] == 1
        assert summary["worst_variance_drop"] == pytest.approx(-0.02)
        assert summary["days_forward_variance_negative"] == 1

    def test_an_empty_panel_is_refused(self) -> None:
        with pytest.raises(ValueError, match="No term-structure days"):
            calendar_summary([])


class TestInversionSummary:
    def test_counts_inversions_separately_from_arbitrage(self) -> None:
        # The distinction the study exists to make: an inverted volatility curve is not an
        # arbitrage, and counting the two together would be the error.
        rows = panel(20)
        for row in rows[:5]:
            row["iv_VIX9D"] = 0.60
            row["iv_VIX1Y"] = 0.30
        summary = inversion_summary(rows, ("VIX9D", "VIX", "VIX1Y"))
        assert summary["days_volatility_curve_inverted"] == 5
        assert summary["inverted_yet_calendar_free"] == 5
        assert summary["inverted_and_arbitrageable"] == 0

    def test_an_inverted_and_violating_day_is_counted_as_such(self) -> None:
        rows = panel(20)
        rows[0]["iv_VIX9D"] = 0.60
        rows[0]["iv_VIX1Y"] = 0.30
        rows[0]["calendar_free"] = False
        summary = inversion_summary(rows, ("VIX9D", "VIX", "VIX1Y"))
        assert summary["inverted_and_arbitrageable"] == 1


class TestHeldOutSummary:
    def test_summarises_the_error_distribution(self) -> None:
        rows: list[dict[str, Any]] = [
            {"held_out_relative_error": value} for value in (0.01, -0.02, 0.005, 0.10)
        ]
        summary = held_out_summary(rows)
        assert summary["n"] == 4
        assert summary["worst_absolute_error"] == pytest.approx(0.10)
        assert summary["share_within_5pct"] == pytest.approx(0.75)

    def test_ignores_missing_predictions(self) -> None:
        rows: list[dict[str, Any]] = [
            {"held_out_relative_error": 0.01},
            {"held_out_relative_error": None},
        ]
        assert held_out_summary(rows)["n"] == 1

    def test_no_predictions_is_refused(self) -> None:
        with pytest.raises(ValueError, match="No held-out predictions"):
            held_out_summary([{"held_out_relative_error": None}])


class TestParameterStability:
    def test_reports_movement_per_regime(self) -> None:
        rows: list[dict[str, Any]] = [
            {"regime": "A", "rho": -0.5 + 0.01 * index, "date": date(2015, 1, 1)}
            for index in range(50)
        ] + [{"regime": "B", "rho": -0.5, "date": date(2015, 1, 1)} for _ in range(50)]
        summary = {row["regime"]: row for row in parameter_stability(rows, 21)}
        assert summary["A"]["mean_absolute_daily_change"] == pytest.approx(0.01)
        assert summary["B"]["mean_absolute_daily_change"] == pytest.approx(0.0)
        assert summary["B"]["sd_rho"] == pytest.approx(0.0)

    def test_counts_days_pinned_at_the_bound(self) -> None:
        rows = [{"regime": "A", "rho": -0.999, "date": date(2015, 1, 1)} for _ in range(10)]
        assert parameter_stability(rows, 5)[0]["share_at_bound"] == pytest.approx(1.0)

    def test_ignores_days_with_no_parameter(self) -> None:
        rows = [{"regime": "A", "rho": None, "date": date(2015, 1, 1)} for _ in range(5)]
        assert parameter_stability(rows, 3) == []


class TestStressEpisodes:
    def test_covers_the_episodes_present_in_the_data(self) -> None:
        rows: list[dict[str, Any]] = []
        for iso in STRESS_EPISODES.values():
            event = date.fromisoformat(iso)
            rows.extend(panel_row(event + timedelta(days=offset)) for offset in range(-5, 6))
        episodes = stress_episodes(rows, 10)
        assert {row["episode"] for row in episodes} == set(STRESS_EPISODES)
        for row in episodes:
            assert row["days_in_window"] > 0

    def test_omits_an_episode_outside_the_sample(self) -> None:
        rows = [panel_row(date(1999, 1, 4) + timedelta(days=index)) for index in range(10)]
        assert stress_episodes(rows, 10) == []


class TestExtremeDays:
    def test_finds_the_most_negative_skewness(self) -> None:
        rows = panel(20)
        rows[7]["risk_neutral_skewness"] = -9.0
        worst = extreme_days(rows, "risk_neutral_skewness", largest=False)
        assert worst[0]["risk_neutral_skewness"] == pytest.approx(-9.0)

    def test_finds_the_largest(self) -> None:
        rows = panel(20)
        rows[3]["risk_neutral_skewness"] = 1.0
        assert extreme_days(rows, "risk_neutral_skewness", largest=True)[0][
            "risk_neutral_skewness"
        ] == pytest.approx(1.0)


class TestBootstrapInterval:
    def test_brackets_the_mean(self) -> None:
        values = list(np.random.default_rng(0).normal(-2.0, 0.5, 200))
        result = bootstrap_interval(values, np.random.default_rng(1), 400)
        assert result is not None
        low, high, mean = result
        assert low < mean < high

    def test_is_reproducible(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]
        assert bootstrap_interval(values, np.random.default_rng(5), 200) == bootstrap_interval(
            values, np.random.default_rng(5), 200
        )

    def test_an_empty_sample_has_no_interval(self) -> None:
        assert bootstrap_interval([], np.random.default_rng(1), 100) is None

    def test_ignores_non_finite_values(self) -> None:
        assert (
            bootstrap_interval([float("nan"), 1.0, 1.0], np.random.default_rng(1), 100) is not None
        )


def test_yearly_summary_groups_by_calendar_year() -> None:
    rows = [panel_row(date(2015, 6, 1)), panel_row(date(2016, 6, 1)), panel_row(date(2016, 7, 1))]
    summary = yearly_summary(rows)
    assert [row["year"] for row in summary] == [2015, 2016]
    assert [row["n"] for row in summary] == [1, 2]
