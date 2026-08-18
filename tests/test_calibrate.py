"""Fitting the surface: the variance curve, the held-out test, and the constrained fit."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from ivsurface.calibrate import (
    VarianceCurve,
    calibrate_day,
    condition_excess,
    fit_daily_rho,
    fit_ssvi_to_skewness,
    fit_variance_curve,
    held_out_error,
    model_skewness,
    slice_moments,
)
from ivsurface.svi import SSVI, ssvi_butterfly_conditions
from ivsurface.termstructure import build_term_structure

from .conftest import ARBITRAGEABLE, HEALTHY, THETAS, flat_slice

GRID = np.linspace(-1.5, 1.5, 601)
DAY = date(2015, 6, 15)


class TestVarianceCurve:
    def test_a_flat_curve_has_constant_average_variance(self) -> None:
        curve = VarianceCurve(v_short=0.04, v_long=0.04, kappa=2.0)
        assert np.allclose(curve.average_variance(np.array([0.1, 1.0, 5.0])), 0.04)

    def test_total_variance_is_average_variance_times_time(self) -> None:
        curve = VarianceCurve(v_short=0.03, v_long=0.05, kappa=1.5)
        years = np.array([0.05, 0.25, 1.0])
        assert np.allclose(curve.total_variance(years), years * curve.average_variance(years))

    def test_the_short_end_approaches_the_instantaneous_variance(self) -> None:
        # The limit of (1 - exp(-x))/x as x goes to zero is one, which is what makes the front of
        # the curve well behaved rather than a division by zero.
        curve = VarianceCurve(v_short=0.09, v_long=0.02, kappa=3.0)
        assert float(curve.average_variance(1e-9)) == pytest.approx(0.09, abs=1e-6)

    def test_the_long_end_approaches_the_long_run_variance(self) -> None:
        curve = VarianceCurve(v_short=0.09, v_long=0.02, kappa=3.0)
        assert float(curve.average_variance(500.0)) == pytest.approx(0.02, abs=1e-3)

    def test_a_rising_curve_is_calendar_free(self) -> None:
        curve = VarianceCurve(v_short=0.02, v_long=0.05, kappa=1.0)
        assert curve.is_calendar_free([0.05, 0.25, 0.5, 1.0])

    def test_total_variance_never_falls_even_when_average_variance_does(self) -> None:
        # Average variance decays from 0.09 toward 0.02, yet total variance still rises, because
        # it is multiplied by maturity. This is the same distinction the calendar test turns on.
        curve = VarianceCurve(v_short=0.09, v_long=0.02, kappa=4.0)
        years = np.array([0.02, 0.08, 0.25, 0.5, 1.0])
        assert np.all(np.diff(curve.average_variance(years)) < 0)
        assert curve.is_calendar_free(years)


class TestFitVarianceCurve:
    def test_recovers_a_curve_it_generated(self) -> None:
        truth = VarianceCurve(v_short=0.05, v_long=0.03, kappa=2.5)
        years = np.array([9 / 365, 30 / 365, 93 / 365, 186 / 365, 1.0])
        fitted = fit_variance_curve(years, truth.total_variance(years))
        assert np.allclose(fitted.total_variance(years), truth.total_variance(years), atol=1e-10)

    def test_fits_a_flat_curve_exactly(self) -> None:
        years = np.array([0.1, 0.3, 0.6, 1.0])
        fitted = fit_variance_curve(years, 0.04 * years)
        assert np.allclose(fitted.total_variance(years), 0.04 * years, atol=1e-10)

    def test_too_few_maturities_are_refused(self) -> None:
        with pytest.raises(ValueError, match="at least three maturities"):
            fit_variance_curve([0.1, 0.5], [0.004, 0.02])


class TestHeldOutTest:
    def test_a_perfect_fit_predicts_the_held_out_maturity(self) -> None:
        truth = VarianceCurve(v_short=0.05, v_long=0.03, kappa=2.5)
        predicted, relative = held_out_error(truth, 0.5, float(truth.total_variance(0.5)))
        assert relative == pytest.approx(0.0, abs=1e-12)
        assert predicted == pytest.approx(float(truth.total_variance(0.5)))

    def test_reports_the_relative_error_signed(self) -> None:
        curve = VarianceCurve(v_short=0.05, v_long=0.03, kappa=2.5)
        observed = float(curve.total_variance(0.5)) * 1.10
        _, relative = held_out_error(curve, 0.5, observed)
        assert relative < 0

    def test_calibrate_day_holds_a_maturity_back(self) -> None:
        levels = {"VIX9D": 0.15, "VIX": 0.16, "VIX3M": 0.18, "VIX6M": 0.19, "VIX1Y": 0.20}
        curve = build_term_structure({n: {DAY: v} for n, v in levels.items()}, DAY, list(levels))
        assert curve is not None
        result = calibrate_day(curve, ("VIX9D", "VIX", "VIX3M", "VIX1Y"), "VIX6M")
        assert result is not None
        _, predicted, relative = result
        assert predicted > 0
        # A smooth curve through four points should land close to the fifth.
        assert abs(relative) < 0.05

    def test_returns_none_when_a_required_maturity_is_missing(self) -> None:
        levels = {"VIX9D": 0.15, "VIX": 0.16, "VIX3M": 0.18}
        curve = build_term_structure({n: {DAY: v} for n, v in levels.items()}, DAY, list(levels))
        assert curve is not None
        assert calibrate_day(curve, ("VIX9D", "VIX", "VIX3M", "VIX1Y"), "VIX6M") is None


class TestSliceMoments:
    def test_a_flat_slice_has_lognormal_moments(self) -> None:
        # Log-moneyness is normal with mean -w/2 and variance w, and no skew.
        total = 0.04
        grid = np.linspace(-2.5, 2.5, 20001)
        mean, variance, skewness = slice_moments(flat_slice(total), grid)
        assert mean == pytest.approx(-total / 2, abs=1e-6)
        assert variance == pytest.approx(total, abs=1e-6)
        assert skewness == pytest.approx(0.0, abs=1e-6)

    def test_negative_correlation_produces_negative_skew(self) -> None:
        assert model_skewness(HEALTHY, 0.04, GRID) < 0

    def test_positive_correlation_produces_positive_skew(self) -> None:
        assert model_skewness(SSVI(rho=0.6, eta=1.0, gamma=0.45), 0.04, GRID) > 0

    def test_stronger_correlation_produces_stronger_skew(self) -> None:
        mild = model_skewness(SSVI(rho=-0.3, eta=1.0, gamma=0.45), 0.04, GRID)
        strong = model_skewness(SSVI(rho=-0.8, eta=1.0, gamma=0.45), 0.04, GRID)
        assert strong < mild


class TestConditionExcess:
    def test_is_zero_for_a_healthy_surface(self) -> None:
        assert condition_excess(HEALTHY, THETAS) == pytest.approx(0.0)

    def test_is_positive_for_an_arbitrageable_surface(self) -> None:
        assert condition_excess(ARBITRAGEABLE, THETAS) > 0

    def test_ignores_nonpositive_variances(self) -> None:
        assert condition_excess(HEALTHY, [0.0, -1.0]) == pytest.approx(0.0)


class TestGlobalFit:
    def test_recovers_parameters_it_generated(self) -> None:
        # Generate skewness from a known surface, then fit it back.
        truth = SSVI(rho=-0.5, eta=1.2, gamma=0.4)
        thetas = list(THETAS)
        targets = [model_skewness(truth, theta, GRID) for theta in thetas]
        fitted, error = fit_ssvi_to_skewness(thetas, targets, GRID)
        assert error < 0.05
        modelled = [model_skewness(fitted, theta, GRID) for theta in thetas]
        assert np.allclose(modelled, targets, atol=0.1)

    def test_the_constrained_fit_stays_inside_the_arbitrage_free_region(self) -> None:
        # The study's central mechanism: constrain the fit and every slice satisfies the
        # sufficient conditions, whatever the skew target asks for.
        impossible = [-8.0] * len(THETAS)
        fitted, _ = fit_ssvi_to_skewness(
            list(THETAS), impossible, GRID, enforce_no_arbitrage=True, condition_thetas=THETAS
        )
        for theta in THETAS:
            holds, _, _ = ssvi_butterfly_conditions(fitted, theta)
            assert holds

    def test_the_unconstrained_fit_may_leave_it(self) -> None:
        impossible = [-8.0] * len(THETAS)
        fitted, _ = fit_ssvi_to_skewness(list(THETAS), impossible, GRID)
        assert condition_excess(fitted, THETAS) > 0

    def test_no_usable_observations_is_refused(self) -> None:
        with pytest.raises(ValueError, match="No usable observations"):
            fit_ssvi_to_skewness([0.0], [float("nan")], GRID)


class TestDailyRho:
    def test_hits_a_reachable_target(self) -> None:
        target = model_skewness(SSVI(rho=-0.45, eta=HEALTHY.eta, gamma=HEALTHY.gamma), 0.04, GRID)
        fitted = fit_daily_rho(HEALTHY, 0.04, target, GRID)
        assert fitted is not None
        assert fitted == pytest.approx(-0.45, abs=0.02)

    def test_the_constrained_refit_respects_the_bound(self) -> None:
        # Ask for far more skew than the conditions permit; the answer must stop at the bound
        # rather than sail past it.
        steep = SSVI(rho=-0.5, eta=2.0, gamma=0.45)
        fitted = fit_daily_rho(
            steep, 0.0534, -8.0, GRID, enforce_no_arbitrage=True, condition_thetas=THETAS
        )
        assert fitted is not None
        holds, _, _ = ssvi_butterfly_conditions(
            SSVI(rho=fitted, eta=steep.eta, gamma=steep.gamma), 0.0534
        )
        assert holds

    def test_returns_none_when_no_correlation_can_satisfy_the_conditions(self) -> None:
        assert (
            fit_daily_rho(
                ARBITRAGEABLE,
                0.0534,
                -3.0,
                GRID,
                enforce_no_arbitrage=True,
                condition_thetas=THETAS,
            )
            is None
        )

    @pytest.mark.parametrize(("theta", "target"), [(0.0, -3.0), (0.04, float("nan"))])
    def test_unusable_inputs_return_none(self, theta: float, target: float) -> None:
        assert fit_daily_rho(HEALTHY, theta, target, GRID) is None
