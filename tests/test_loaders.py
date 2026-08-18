"""Input parsing, including the two shapes Cboe publishes and the scale each series needs."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pytest

from ivsurface.config import REQUIRED_CBOE_FILES, SKEW_INDEX
from ivsurface.loaders import load_cboe_history, load_series, resolve_value_column

from .conftest import trading_dates, write_cboe_csv, write_synthetic_inputs


class TestResolveValueColumn:
    def test_prefers_close_when_present(self) -> None:
        assert resolve_value_column(["DATE", "OPEN", "HIGH", "LOW", "CLOSE"]) == "CLOSE"

    def test_falls_back_to_the_single_remaining_column(self) -> None:
        # SKEW and VVIX publish a single value column named after the index itself.
        assert resolve_value_column(["DATE", "SKEW"]) == "SKEW"

    def test_a_missing_date_column_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="No DATE column"):
            resolve_value_column(["WHEN", "CLOSE"])

    def test_an_ambiguous_header_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Cannot identify the value column"):
            resolve_value_column(["DATE", "ONE", "TWO"])


class TestLoadCboeHistory:
    def test_reads_the_five_column_shape(self, tmp_path: Path) -> None:
        dates = trading_dates(date(2015, 1, 5), 3)
        write_cboe_csv(tmp_path / "VIX_History.csv", dates, [18.0, 19.0, 20.0])
        levels = load_cboe_history(tmp_path / "VIX_History.csv")
        assert list(levels.values()) == pytest.approx([0.18, 0.19, 0.20])

    def test_reads_the_two_column_shape(self, tmp_path: Path) -> None:
        dates = trading_dates(date(2015, 1, 5), 2)
        write_cboe_csv(tmp_path / "SKEW_History.csv", dates, [120.0, 130.0], value_column="SKEW")
        levels = load_cboe_history(tmp_path / "SKEW_History.csv", scale=1.0)
        assert list(levels.values()) == pytest.approx([120.0, 130.0])

    def test_the_scale_is_explicit_rather_than_a_special_case(self, tmp_path: Path) -> None:
        # SKEW is an index near 100, not a percentage. Dividing it by 100 would silently corrupt
        # every skewness the study derives from it, so the divisor is a parameter.
        dates = trading_dates(date(2015, 1, 5), 1)
        write_cboe_csv(tmp_path / "SKEW_History.csv", dates, [130.0], value_column="SKEW")
        path = tmp_path / "SKEW_History.csv"
        assert next(iter(load_cboe_history(path, scale=1.0).values())) == pytest.approx(130.0)
        assert next(iter(load_cboe_history(path).values())) == pytest.approx(1.30)

    def test_a_nonpositive_level_is_treated_as_missing_and_logged(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        dates = trading_dates(date(2015, 1, 5), 3)
        write_cboe_csv(tmp_path / "VIX_History.csv", dates, [18.0, 0.0, 20.0])
        with caplog.at_level(logging.WARNING):
            levels = load_cboe_history(tmp_path / "VIX_History.csv")
        assert len(levels) == 2
        assert "nonpositive" in caplog.text

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Missing Cboe input"):
            load_cboe_history(tmp_path / "absent.csv")

    def test_unparseable_date(self, tmp_path: Path) -> None:
        path = tmp_path / "x.csv"
        path.write_text("DATE,CLOSE\n2015-01-05,18\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid row"):
            load_cboe_history(path)

    def test_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "x.csv"
        path.write_text("DATE,CLOSE\n", encoding="utf-8")
        with pytest.raises(ValueError, match="No data rows"):
            load_cboe_history(path)


class TestLoadSeries:
    def test_loads_every_required_series(self, tmp_path: Path) -> None:
        dates = trading_dates(date(2015, 1, 5), 10)
        write_synthetic_inputs(tmp_path, dates)
        loaded = load_series(tmp_path)
        assert set(loaded) == set(REQUIRED_CBOE_FILES)

    def test_applies_the_right_scale_to_each_series(self, tmp_path: Path) -> None:
        dates = trading_dates(date(2015, 1, 5), 10)
        write_synthetic_inputs(tmp_path, dates)
        loaded = load_series(tmp_path)
        # Volatility indexes come back as decimals; SKEW keeps its own units.
        assert max(loaded["VIX"].values()) < 1.0
        assert min(loaded[SKEW_INDEX].values()) > 50.0

    def test_a_missing_series_is_reported(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_series(tmp_path)
