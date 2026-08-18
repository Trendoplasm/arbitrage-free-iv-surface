"""The observed variance curve and the calendar condition."""

from __future__ import annotations

from datetime import date

import pytest

from ivsurface.config import CALENDAR_DAYS_PER_YEAR, CORE_TERM_STRUCTURE, TERM_STRUCTURE
from ivsurface.termstructure import (
    average_curve,
    build_panel,
    build_term_structure,
    calendar_violations,
    forward_variance,
    maturity_years,
    skew_to_skewness,
    slope_summary,
)

from .conftest import series_from, trading_dates

DAY = date(2015, 6, 15)


def series(levels: dict[str, float], day: date = DAY) -> dict[str, dict[date, float]]:
    """Build one-day series at the given levels."""
    return {name: {day: level} for name, level in levels.items()}


RISING = {"VIX9D": 0.15, "VIX": 0.16, "VIX3M": 0.18, "VIX6M": 0.19, "VIX1Y": 0.20}


class TestMaturityYears:
    @pytest.mark.parametrize("name", CORE_TERM_STRUCTURE)
    def test_matches_the_published_horizon(self, name: str) -> None:
        assert maturity_years(name) == pytest.approx(TERM_STRUCTURE[name] / CALENDAR_DAYS_PER_YEAR)

    def test_a_year_index_is_one_year(self) -> None:
        assert maturity_years("VIX1Y") == pytest.approx(1.0)

    def test_an_unknown_index_is_refused(self) -> None:
        with pytest.raises(KeyError):
            maturity_years("NOTANINDEX")


class TestBuildTermStructure:
    def test_orders_by_maturity_and_converts_to_total_variance(self) -> None:
        curve = build_term_structure(series(RISING), DAY, list(RISING))
        assert curve is not None
        assert curve.names == ("VIX9D", "VIX", "VIX3M", "VIX6M", "VIX1Y")
        assert list(curve.maturities) == sorted(curve.maturities)
        for name, volatility, total in zip(
            curve.names, curve.volatilities, curve.total_variances, strict=True
        ):
            assert total == pytest.approx(volatility**2 * maturity_years(name))

    def test_reads_a_named_maturity_back(self) -> None:
        curve = build_term_structure(series(RISING), DAY, list(RISING))
        assert curve is not None
        assert curve.theta("VIX") == pytest.approx(0.16**2 * maturity_years("VIX"))

    def test_uses_only_the_maturities_that_published(self) -> None:
        partial = series({"VIX9D": 0.15, "VIX": 0.16, "VIX3M": 0.18})
        partial["VIX6M"] = {}
        partial["VIX1Y"] = {}
        curve = build_term_structure(partial, DAY, list(RISING))
        assert curve is not None
        assert curve.names == ("VIX9D", "VIX", "VIX3M")

    def test_a_single_maturity_is_not_a_term_structure(self) -> None:
        one: dict[str, dict[date, float]] = {name: {} for name in RISING}
        one["VIX"] = {DAY: 0.16}
        assert build_term_structure(one, DAY, list(RISING)) is None

    def test_skips_a_nonpositive_level(self) -> None:
        broken = series(RISING)
        broken["VIX3M"] = {DAY: 0.0}
        curve = build_term_structure(broken, DAY, list(RISING))
        assert curve is not None
        assert "VIX3M" not in curve.names


class TestCalendarCondition:
    def test_a_rising_curve_has_no_violation(self) -> None:
        curve = build_term_structure(series(RISING), DAY, list(RISING))
        assert curve is not None
        assert calendar_violations(curve) == []

    def test_a_falling_total_variance_is_a_violation(self) -> None:
        # Total variance has to fall, not merely volatility: the year point is dropped far enough
        # that even the maturity factor cannot keep the curve rising.
        broken = dict(RISING)
        broken["VIX1Y"] = 0.05
        curve = build_term_structure(series(broken), DAY, list(broken))
        assert curve is not None
        violations = calendar_violations(curve)
        assert len(violations) == 1
        assert violations[0][0] == "VIX6M"
        assert violations[0][2] < 0

    def test_an_inverted_volatility_curve_can_still_be_arbitrage_free(self) -> None:
        # The distinction the whole study turns on. Short-dated volatility far above long-dated is
        # a stress signal, not an arbitrage: the maturity factor keeps total variance rising.
        inverted = {"VIX9D": 0.80, "VIX": 0.60, "VIX3M": 0.45, "VIX6M": 0.40, "VIX1Y": 0.35}
        curve = build_term_structure(series(inverted), DAY, list(inverted))
        assert curve is not None
        assert curve.volatilities[0] > curve.volatilities[-1]
        assert calendar_violations(curve) == []

    def test_forward_variance_is_positive_when_the_curve_rises(self) -> None:
        curve = build_term_structure(series(RISING), DAY, list(RISING))
        assert curve is not None
        assert all(value > 0 for _, _, value in forward_variance(curve))

    def test_forward_variance_goes_negative_at_a_violation(self) -> None:
        broken = dict(RISING)
        broken["VIX1Y"] = 0.05
        curve = build_term_structure(series(broken), DAY, list(broken))
        assert curve is not None
        assert min(value for _, _, value in forward_variance(curve)) < 0

    def test_slope_is_positive_for_a_rising_curve(self) -> None:
        curve = build_term_structure(series(RISING), DAY, list(RISING))
        assert curve is not None
        assert slope_summary(curve) > 0


class TestSkewConversion:
    def test_inverts_cboes_definition(self) -> None:
        # SKEW = 100 - 10 * S, so a level of 100 means no skew at all.
        assert skew_to_skewness(100.0) == pytest.approx(0.0)
        assert skew_to_skewness(130.0) == pytest.approx(-3.0)
        assert skew_to_skewness(90.0) == pytest.approx(1.0)

    def test_a_higher_index_means_a_more_negative_skew(self) -> None:
        assert skew_to_skewness(140.0) < skew_to_skewness(120.0)


class TestBuildPanel:
    def test_produces_one_row_per_usable_day(self) -> None:
        dates = trading_dates(date(2015, 1, 5), 30)
        data = {name: series_from(dates, level) for name, level in RISING.items()}
        data["SKEW"] = series_from(dates, 125.0)
        panel = build_panel(data, dates, list(RISING))
        assert len(panel) == len(dates)
        assert all(row["calendar_free"] for row in panel)
        assert all(row["risk_neutral_skewness"] == pytest.approx(-2.5) for row in panel)

    def test_records_each_maturity_as_its_own_columns(self) -> None:
        dates = trading_dates(date(2015, 1, 5), 5)
        data = {name: series_from(dates, level) for name, level in RISING.items()}
        data["SKEW"] = series_from(dates, 120.0)
        row = build_panel(data, dates, list(RISING))[0]
        for name in RISING:
            assert f"iv_{name}" in row
            assert f"theta_{name}" in row

    def test_a_day_without_a_curve_is_skipped(self) -> None:
        dates = trading_dates(date(2015, 1, 5), 5)
        data: dict[str, dict[date, float]] = {name: {} for name in RISING}
        data["VIX"] = series_from(dates, 0.16)
        data["SKEW"] = series_from(dates, 120.0)
        assert build_panel(data, dates, list(RISING)) == []


class TestAverageCurve:
    def test_summarises_each_maturity(self) -> None:
        dates = trading_dates(date(2015, 1, 5), 40)
        data = {name: series_from(dates, level) for name, level in RISING.items()}
        data["SKEW"] = series_from(dates, 120.0)
        panel = build_panel(data, dates, list(RISING))
        summary = average_curve(panel, list(RISING))
        assert [row["index"] for row in summary] == list(RISING)
        for row, (name, level) in zip(summary, RISING.items(), strict=True):
            assert row["mean_iv"] == pytest.approx(level)
            assert row["horizon_days"] == TERM_STRUCTURE[name]

    def test_total_variance_rises_across_the_summary(self) -> None:
        dates = trading_dates(date(2015, 1, 5), 40)
        data = {name: series_from(dates, level) for name, level in RISING.items()}
        data["SKEW"] = series_from(dates, 120.0)
        summary = average_curve(build_panel(data, dates, list(RISING)), list(RISING))
        totals = [row["mean_total_variance"] for row in summary]
        assert totals == sorted(totals)
