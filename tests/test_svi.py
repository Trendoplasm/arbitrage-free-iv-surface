"""The SVI and SSVI parameterisations, and the no-arbitrage theorems about them.

These are the strongest tests in the suite, because the propositions are mathematical rather than
empirical. A flat slice must reproduce the lognormal density exactly. An SSVI slice must equal its
raw-SVI representation exactly. The density implied by the smile must equal the one recovered from
option prices by Breeden and Litzenberger. None of that depends on any market data being right.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from ivsurface.blackscholes import straddle_value
from ivsurface.svi import (
    SSVI,
    RawSVI,
    butterfly_g,
    implied_density,
    max_absolute_correlation,
    power_law_phi,
    ssvi_butterfly_conditions,
    ssvi_calendar_free,
)

from .conftest import ARBITRAGEABLE, HEALTHY, THETAS, flat_slice


class TestRawSVI:
    def test_evaluates_its_own_formula(self) -> None:
        slice_ = RawSVI(a=0.02, b=0.1, rho=-0.4, m=0.05, sigma=0.2)
        k = 0.3
        shifted = k - slice_.m
        expected = slice_.a + slice_.b * (
            slice_.rho * shifted + math.sqrt(shifted**2 + slice_.sigma**2)
        )
        assert float(slice_.total_variance(k)) == pytest.approx(expected)

    def test_wings_approach_the_asymptotic_slopes(self) -> None:
        # Far from the minimum the hyperbola becomes two straight lines with slopes b(1 + rho) to
        # the right and b(rho - 1) to the left. Those slopes are what Lee's moment formula bounds.
        slice_ = RawSVI(a=0.02, b=0.1, rho=-0.4, m=0.0, sigma=0.2)
        far = 5000.0
        right = float(slice_.total_variance(far + 1) - slice_.total_variance(far))
        left = float(slice_.total_variance(-far) - slice_.total_variance(-far - 1))
        assert right == pytest.approx(slice_.b * (1 + slice_.rho), abs=1e-6)
        assert left == pytest.approx(slice_.b * (slice_.rho - 1), abs=1e-6)

    def test_minimum_is_where_the_formula_says(self) -> None:
        slice_ = RawSVI(a=0.02, b=0.1, rho=-0.4, m=0.05, sigma=0.2)
        grid = np.linspace(-3, 3, 60001)
        assert float(slice_.total_variance(grid).min()) == pytest.approx(
            slice_.minimum_variance(), abs=1e-7
        )

    def test_a_flat_slice_is_flat(self) -> None:
        assert np.allclose(flat_slice(0.04).total_variance(np.linspace(-2, 2, 11)), 0.04)

    def test_converts_total_variance_to_volatility(self) -> None:
        slice_ = flat_slice(0.04)
        assert float(slice_.implied_volatility(0.0, 1.0)) == pytest.approx(0.2)
        assert float(slice_.implied_volatility(0.0, 0.25)) == pytest.approx(0.4)

    def test_zero_maturity_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Maturity must be positive"):
            flat_slice(0.04).implied_volatility(0.0, 0.0)

    @pytest.mark.parametrize(
        "slice_",
        [
            RawSVI(a=0.02, b=-0.1, rho=-0.4, m=0.0, sigma=0.2),
            RawSVI(a=0.02, b=0.1, rho=-1.0, m=0.0, sigma=0.2),
            RawSVI(a=0.02, b=0.1, rho=-0.4, m=0.0, sigma=0.0),
        ],
    )
    def test_rejects_parameters_outside_the_family(self, slice_: RawSVI) -> None:
        assert not slice_.is_valid()

    def test_accepts_a_well_formed_slice(self) -> None:
        assert RawSVI(a=0.02, b=0.1, rho=-0.4, m=0.0, sigma=0.2).is_valid()


class TestPowerLawPhi:
    def test_decreases_as_variance_grows(self) -> None:
        # The skew flattens with maturity, which is what the shape function encodes.
        values = power_law_phi(np.array([0.001, 0.01, 0.05, 0.2]), eta=1.0, gamma=0.4)
        assert np.all(np.diff(values) < 0)

    def test_scales_linearly_in_eta(self) -> None:
        single = power_law_phi(0.01, eta=1.0, gamma=0.4)
        double = power_law_phi(0.01, eta=2.0, gamma=0.4)
        assert float(double) == pytest.approx(2.0 * float(single))

    @pytest.mark.parametrize(("eta", "theta"), [(0.0, 0.01), (-1.0, 0.01)])
    def test_nonpositive_eta_is_rejected(self, eta: float, theta: float) -> None:
        with pytest.raises(ValueError, match="eta must be positive"):
            power_law_phi(theta, eta, 0.4)

    def test_nonpositive_variance_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="theta must be positive"):
            power_law_phi(0.0, 1.0, 0.4)


class TestSSVI:
    @pytest.mark.parametrize("theta", THETAS)
    def test_matches_its_raw_svi_representation_exactly(self, theta: float) -> None:
        # SSVI is a constrained subfamily of raw SVI, so the two must agree to machine precision.
        # If they ever drift apart, every slice-level diagnostic run on a global fit is invalid.
        grid = np.linspace(-1.0, 1.0, 51)
        assert np.allclose(
            HEALTHY.total_variance(grid, theta),
            HEALTHY.to_raw_svi(theta).total_variance(grid),
            atol=1e-15,
        )

    @pytest.mark.parametrize("theta", THETAS)
    def test_is_at_the_money_by_construction(self, theta: float) -> None:
        # At k = 0 the formula collapses to theta exactly, which is what makes the observed
        # at-the-money term structure usable as the surface's backbone without any fitting.
        assert float(HEALTHY.total_variance(0.0, theta)) == pytest.approx(theta)

    def test_negative_correlation_tilts_the_smile_left(self) -> None:
        # An equity surface is worth more below the money than above it.
        left = float(HEALTHY.total_variance(-0.2, 0.04))
        right = float(HEALTHY.total_variance(0.2, 0.04))
        assert left > right

    def test_zero_correlation_is_symmetric(self) -> None:
        symmetric = SSVI(rho=0.0, eta=1.0, gamma=0.45)
        assert float(symmetric.total_variance(-0.3, 0.04)) == pytest.approx(
            float(symmetric.total_variance(0.3, 0.04))
        )

    def test_converts_to_volatility(self) -> None:
        theta, maturity = 0.04, 1.0
        assert float(HEALTHY.implied_volatility(0.0, theta, maturity)) == pytest.approx(0.2)

    def test_zero_maturity_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Maturity must be positive"):
            HEALTHY.implied_volatility(0.0, 0.04, 0.0)


class TestButterflyFunction:
    def test_is_identically_one_for_a_flat_slice(self) -> None:
        # With no smile the density is exactly lognormal, and g(k) = 1 everywhere is the algebraic
        # statement of that.
        assert np.allclose(butterfly_g(flat_slice(0.04), np.linspace(-2, 2, 101)), 1.0)

    @pytest.mark.parametrize("theta", THETAS)
    def test_stays_positive_on_a_healthy_surface(self, theta: float) -> None:
        assert float(butterfly_g(HEALTHY.to_raw_svi(theta), np.linspace(-2, 2, 2001)).min()) > 0

    def test_goes_negative_on_an_arbitrageable_surface(self) -> None:
        g = butterfly_g(ARBITRAGEABLE.to_raw_svi(0.09), np.linspace(-2, 2, 2001))
        assert float(g.min()) < 0


class TestImpliedDensity:
    def test_reproduces_the_lognormal_density_for_a_flat_slice(self) -> None:
        total = 0.04
        grid = np.linspace(-2, 2, 401)
        expected = norm.pdf(grid, loc=-total / 2, scale=math.sqrt(total))
        assert np.allclose(implied_density(flat_slice(total), grid), expected, atol=1e-14)

    @pytest.mark.parametrize("theta", THETAS)
    def test_integrates_to_one(self, theta: float) -> None:
        # A density that does not sum to one is not a probability distribution, however well the
        # surface it came from fits.
        grid = np.concatenate(
            [
                np.linspace(-12, -2, 400, endpoint=False),
                np.linspace(-2, 2, 4001),
                np.linspace(2, 12, 400),
            ]
        )
        mass = float(np.trapezoid(implied_density(HEALTHY.to_raw_svi(theta), grid), grid))
        assert mass == pytest.approx(1.0, abs=1e-5)

    @pytest.mark.parametrize("theta", THETAS)
    def test_matches_the_density_recovered_from_option_prices(self, theta: float) -> None:
        # Breeden and Litzenberger: the risk-neutral density is the second derivative of the call
        # price in strike. Recovering it that way uses an entirely separate code path -- the
        # Black-Scholes pricer -- so agreement is real corroboration, not a restatement.
        slice_ = HEALTHY.to_raw_svi(theta)
        maturity = 1.0
        forward = 1.0

        def call_price(strike: float) -> float:
            k = math.log(strike / forward)
            volatility = float(slice_.implied_volatility(k, maturity))
            straddle = float(straddle_value(forward, strike, 0.0, 0.0, volatility, maturity))
            # With zero rates a straddle is C + P and put-call parity gives C - P = F - K.
            return 0.5 * (straddle + forward - strike)

        # Evaluate within a couple of standard deviations of the money. Further out the density is
        # genuinely negligible -- eight deviations from a nine-day slice is around 1e-11 -- and the
        # second difference of call prices underflows, which would test double precision rather
        # than the formula.
        deviation = math.sqrt(theta)
        for multiple in (-2.0, -1.0, 0.0, 1.0, 2.0):
            k = multiple * deviation
            strike = forward * math.exp(k)
            step = 1e-4 * strike
            second = (
                call_price(strike + step) - 2 * call_price(strike) + call_price(strike - step)
            ) / step**2
            # Converting the density in strike to a density in log-moneyness multiplies by K.
            assert second * strike == pytest.approx(float(implied_density(slice_, k)), rel=2e-3)

    def test_goes_negative_where_the_surface_is_arbitrageable(self) -> None:
        grid = np.linspace(-1.5, 1.5, 2001)
        assert float(implied_density(ARBITRAGEABLE.to_raw_svi(0.09), grid).min()) < 0


class TestSufficientConditions:
    @pytest.mark.parametrize("theta", THETAS)
    def test_hold_for_a_healthy_surface(self, theta: float) -> None:
        holds, first, second = ssvi_butterfly_conditions(HEALTHY, theta)
        assert holds
        assert first < 4.0
        assert second <= 4.0

    def test_fail_for_an_arbitrageable_surface(self) -> None:
        holds, first, second = ssvi_butterfly_conditions(ARBITRAGEABLE, 0.09)
        assert not holds
        assert max(first, second) > 4.0

    @pytest.mark.parametrize("theta", THETAS)
    def test_when_they_hold_the_density_is_non_negative(self, theta: float) -> None:
        # The conditions are sufficient, so satisfying them must imply a valid density. This is the
        # test that connects the parameter-level check to the numerical one.
        holds, _, _ = ssvi_butterfly_conditions(HEALTHY, theta)
        assert holds
        assert float(butterfly_g(HEALTHY.to_raw_svi(theta), np.linspace(-3, 3, 2001)).min()) >= 0


class TestCorrelationBound:
    @pytest.mark.parametrize("theta", THETAS)
    def test_a_surface_at_the_bound_satisfies_the_conditions(self, theta: float) -> None:
        bound = max_absolute_correlation(HEALTHY.eta, HEALTHY.gamma, theta)
        assert bound is not None
        at_bound = SSVI(rho=-bound, eta=HEALTHY.eta, gamma=HEALTHY.gamma)
        holds, _, _ = ssvi_butterfly_conditions(at_bound, theta)
        assert holds

    def test_a_surface_beyond_the_bound_breaches_them(self) -> None:
        theta = 0.0534
        bound = max_absolute_correlation(2.0, 0.45, theta)
        assert bound is not None and bound < 0.999
        beyond = SSVI(rho=-(bound + 0.05), eta=2.0, gamma=0.45)
        holds, _, _ = ssvi_butterfly_conditions(beyond, theta)
        assert not holds

    def test_reports_no_bound_when_the_shape_function_is_hopeless(self) -> None:
        # Past a point no correlation can rescue the slice; eta or gamma has to change instead.
        assert max_absolute_correlation(8.0, 0.05, 0.0534) is None

    def test_nonpositive_variance_has_no_bound(self) -> None:
        assert max_absolute_correlation(1.0, 0.45, 0.0) is None


class TestCalendarCondition:
    def test_accepts_a_rising_curve(self) -> None:
        assert ssvi_calendar_free(np.array(THETAS))

    def test_rejects_a_curve_that_dips(self) -> None:
        assert not ssvi_calendar_free(np.array([0.001, 0.05, 0.04, 0.09]))

    def test_accepts_a_flat_curve(self) -> None:
        # Equality is allowed: no arbitrage requires non-decreasing, not strictly increasing.
        assert ssvi_calendar_free(np.array([0.02, 0.02, 0.02]))

    def test_a_single_maturity_is_vacuously_free(self) -> None:
        assert ssvi_calendar_free(np.array([0.02]))
