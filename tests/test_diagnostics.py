"""Static-arbitrage diagnostics on a calibrated surface."""

from __future__ import annotations

import numpy as np
import pytest

from ivsurface.config import DENSITY_TOLERANCE
from ivsurface.diagnostics import (
    check_slice,
    check_surface,
    density_grid,
    density_mass,
    diagnose_day,
    diagnostic_grid,
)

from .conftest import ARBITRAGEABLE, CONDITIONS_FAIL_BUT_VALID, HEALTHY, THETAS, flat_slice

NAMES = ("VIX9D", "VIX", "VIX3M", "VIX6M", "VIX1Y")


class TestGrids:
    def test_the_diagnostic_grid_straddles_the_money(self) -> None:
        grid = diagnostic_grid()
        assert grid[0] < 0 < grid[-1]
        assert np.all(np.diff(grid) > 0)

    def test_the_density_grid_is_wide_with_a_dense_core(self) -> None:
        # Wide enough for SVI's fat wings, fine enough for a nine-day slice. A uniform grid meeting
        # both would be several times larger.
        grid = density_grid()
        spacings = np.diff(grid)
        middle = spacings[len(spacings) // 2]
        assert grid[0] <= -12.0
        assert grid[-1] >= 12.0
        assert middle < 0.001
        assert spacings.max() > 10 * middle


class TestDensityMass:
    @pytest.mark.parametrize("theta", THETAS)
    def test_a_healthy_surface_integrates_to_one(self, theta: float) -> None:
        assert density_mass(HEALTHY.to_raw_svi(theta), theta) == pytest.approx(
            1.0, abs=DENSITY_TOLERANCE
        )

    def test_a_flat_slice_integrates_to_one(self) -> None:
        assert density_mass(flat_slice(0.04), 0.04) == pytest.approx(1.0, abs=1e-9)

    def test_an_arbitrageable_surface_does_not(self) -> None:
        # Clipping the negative region away leaves less than a whole distribution behind, which is
        # exactly how a negative density shows up in the mass.
        mass = density_mass(ARBITRAGEABLE.to_raw_svi(0.0534), 0.0534)
        assert abs(mass - 1.0) > DENSITY_TOLERANCE


class TestCheckSlice:
    @pytest.mark.parametrize("theta", THETAS)
    def test_passes_a_healthy_slice(self, theta: float) -> None:
        passed, minimum, failures = check_slice(HEALTHY.to_raw_svi(theta))
        assert passed
        assert minimum > 0
        assert failures == 0

    def test_fails_an_arbitrageable_slice_and_counts_the_points(self) -> None:
        passed, minimum, failures = check_slice(ARBITRAGEABLE.to_raw_svi(0.0534))
        assert not passed
        assert minimum < 0
        assert failures > 0

    def test_a_flat_slice_has_g_of_one(self) -> None:
        _, minimum, _ = check_slice(flat_slice(0.04))
        assert minimum == pytest.approx(1.0)


class TestCheckSurface:
    def test_a_healthy_surface_on_a_rising_curve_passes_both_tests(self) -> None:
        check = check_surface(HEALTHY, THETAS)
        assert check.arbitrage_free
        assert check.butterfly_free
        assert check.calendar_free
        assert check.min_butterfly_g > 0
        assert check.min_calendar_slope > 0

    def test_a_dip_in_the_curve_fails_only_the_calendar_test(self) -> None:
        dipped = (0.0009, 0.0031, 0.0112, 0.0080, 0.0534)
        check = check_surface(HEALTHY, dipped)
        assert check.butterfly_free
        assert not check.calendar_free
        assert not check.arbitrage_free
        assert check.calendar_violations == 1
        assert check.min_calendar_slope < 0

    def test_a_steep_surface_fails_only_the_butterfly_test(self) -> None:
        check = check_surface(ARBITRAGEABLE, THETAS)
        assert not check.butterfly_free
        assert check.calendar_free
        assert not check.arbitrage_free
        assert check.butterfly_violations > 0

    def test_failing_the_sufficient_conditions_is_not_the_same_as_arbitrage(self) -> None:
        # The Gatheral-Jacquier conditions are sufficient, not necessary. A surface can breach them
        # and still imply a perfectly valid density, which is precisely why the numerical scan is
        # run alongside the parameter test rather than instead of it.
        from ivsurface.svi import ssvi_butterfly_conditions

        for theta in THETAS:
            holds, _, _ = ssvi_butterfly_conditions(CONDITIONS_FAIL_BUT_VALID, theta)
            if not holds:
                break
        else:
            pytest.fail("fixture no longer breaches the sufficient conditions")

        check = check_surface(CONDITIONS_FAIL_BUT_VALID, THETAS)
        assert check.butterfly_free
        assert check.min_butterfly_g > 0

    def test_ignores_nonpositive_variances_in_the_butterfly_scan(self) -> None:
        check = check_surface(HEALTHY, (0.0009, 0.0031))
        assert check.butterfly_free


class TestDiagnoseDay:
    def test_records_parameters_outcomes_and_density_mass(self) -> None:
        row = diagnose_day(HEALTHY, THETAS, NAMES)
        assert row["rho"] == HEALTHY.rho
        assert row["arbitrage_free"] is True
        assert row["sufficient_conditions_hold"] is True
        assert row["density_valid"] is True
        for name in NAMES:
            assert row[f"density_mass_{name}"] == pytest.approx(1.0, abs=DENSITY_TOLERANCE)

    def test_reports_the_binding_condition_values(self) -> None:
        row = diagnose_day(HEALTHY, THETAS, NAMES)
        # Both sufficient conditions are bounded by four; reporting how close they came makes the
        # margin visible rather than only the verdict.
        assert row["max_condition_1"] < 4.0
        assert row["max_condition_2"] <= 4.0

    def test_flags_an_arbitrageable_surface(self) -> None:
        row = diagnose_day(ARBITRAGEABLE, THETAS, NAMES)
        assert row["arbitrage_free"] is False
        assert row["sufficient_conditions_hold"] is False
        assert row["min_butterfly_g"] < 0
        assert row["density_valid"] is False

    def test_flattens_to_scalars_for_export(self) -> None:
        row = diagnose_day(HEALTHY, THETAS, NAMES)
        assert all(not isinstance(value, (list, dict, tuple)) for value in row.values())
