"""Tests for the implied volatility solver."""

import numpy as np
import pytest

from bs_lab.greeks import vega
from bs_lab.implied_vol import implied_vol, solve_implied_vol
from bs_lab.pricing import bs_price

R, Q = 0.03, 0.01


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_round_trip_recovers_volatility(option_type):
    """Price with a known vol, invert, and get that vol back."""
    strikes, maturities, vols = np.meshgrid(
        np.linspace(50.0, 160.0, 12),
        np.array([0.05, 0.25, 1.0, 3.0]),
        np.array([0.05, 0.2, 0.5, 1.0, 2.0]),
    )
    prices = bs_price(100.0, strikes, maturities, R, vols, option_type, Q)
    solved = implied_vol(prices, 100.0, strikes, maturities, R, option_type, Q)

    # Where vega is negligible the price holds no information about vol, since
    # many vols round to the same price. The check is only meaningful elsewhere.
    informative = vega(100.0, strikes, maturities, R, vols, Q) > 1e-4
    assert informative.sum() > 0.5 * informative.size
    np.testing.assert_allclose(solved[informative], vols[informative], atol=1e-5)


def test_solved_vol_reprices_the_option_everywhere():
    """Even where vol itself is ill-defined, the price must be reproduced."""
    strikes = np.linspace(50.0, 160.0, 111)
    prices = bs_price(100.0, strikes, 0.25, R, 0.4, "call", Q)
    solved = implied_vol(prices, 100.0, strikes, 0.25, R, "call", Q)
    np.testing.assert_allclose(
        bs_price(100.0, strikes, 0.25, R, solved, "call", Q), prices, atol=1e-7
    )


def test_newton_handles_ordinary_options_without_fallback():
    prices = bs_price(100.0, np.array([90.0, 100.0, 110.0]), 1.0, R, 0.25)
    result = solve_implied_vol(prices, 100.0, np.array([90.0, 100.0, 110.0]), 1.0, R)
    assert not result.used_bisection.any()


def test_bisection_rescues_flat_vega_starting_point():
    """A low start on a far-OTM option sits where vega is zero and Newton cannot move."""
    S, K, T, true_vol = 100.0, 200.0, 0.25, 1.5
    price = bs_price(S, K, T, R, true_vol)
    assert vega(S, K, T, R, 0.01) < 1e-10
    result = solve_implied_vol(price, S, K, T, R, sigma0=0.01)
    assert result.used_bisection
    assert result.vol == pytest.approx(true_vol, abs=1e-6)


def test_bisection_rescues_newton_step_that_escapes():
    """A deep-OTM start with tiny vega overshoots far outside the valid range."""
    S, K, T, true_vol = 100.0, 130.0, 0.25, 0.2
    price = bs_price(S, K, T, R, true_vol)
    result = solve_implied_vol(price, S, K, T, R, sigma0=0.07)
    assert result.used_bisection
    assert result.vol == pytest.approx(true_vol, abs=1e-6)


def test_prices_outside_no_arbitrage_bounds_give_nan():
    S, K, T = 100.0, 90.0, 1.0
    below_intrinsic = 0.5 * (S - K * np.exp(-R * T))
    above_stock = S + 1.0
    assert np.isnan(implied_vol(below_intrinsic, S, K, T, R))
    assert np.isnan(implied_vol(above_stock, S, K, T, R))
    assert np.isnan(implied_vol(0.0, 100.0, 120.0, 1.0, R))


def test_vol_beyond_the_search_range_gives_nan_not_the_edge():
    """Below the upper bound, yet needing more than 1000% vol: refuse to answer."""
    S, K, T = 100.0, 100.0, 1.0
    price = bs_price(S, K, T, R, 12.0)
    assert price < S
    assert np.isnan(implied_vol(price, S, K, T, R))


def test_expired_option_gives_nan():
    assert np.isnan(implied_vol(5.0, 100.0, 100.0, 0.0, R))


def test_mixed_array_keeps_valid_entries():
    good = bs_price(100.0, 100.0, 1.0, R, 0.3)
    solved = implied_vol(np.array([good, -1.0, good]), 100.0, 100.0, 1.0, R)
    assert solved[0] == pytest.approx(0.3, abs=1e-6)
    assert np.isnan(solved[1])
    assert solved[2] == pytest.approx(0.3, abs=1e-6)


def test_put_and_call_give_the_same_vol():
    """Parity means one vol per strike, whichever side of the market we use."""
    call = bs_price(100.0, 105.0, 0.5, R, 0.27, "call", Q)
    put = bs_price(100.0, 105.0, 0.5, R, 0.27, "put", Q)
    assert implied_vol(call, 100.0, 105.0, 0.5, R, "call", Q) == pytest.approx(
        implied_vol(put, 100.0, 105.0, 0.5, R, "put", Q), abs=1e-7
    )
