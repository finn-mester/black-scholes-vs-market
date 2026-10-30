"""Tests for the closed-form Black-Scholes prices."""

import numpy as np
import pytest

from bs_lab.pricing import bs_price, call_price, put_price

SPOTS = np.array([60.0, 80.0, 100.0, 120.0, 150.0])


@pytest.mark.parametrize(
    "S, K, T, r, sigma, expected_call, expected_put",
    [
        # Textbook example: Hull, Options, Futures and Other Derivatives.
        (42.0, 40.0, 0.5, 0.10, 0.20, 4.7594, 0.8086),
        # Standard at-the-money benchmark.
        (100.0, 100.0, 1.0, 0.05, 0.20, 10.4506, 5.5735),
    ],
)
def test_matches_published_values(S, K, T, r, sigma, expected_call, expected_put):
    assert call_price(S, K, T, r, sigma) == pytest.approx(expected_call, abs=5e-5)
    assert put_price(S, K, T, r, sigma) == pytest.approx(expected_put, abs=5e-5)


@pytest.mark.parametrize("q", [0.0, 0.02])
@pytest.mark.parametrize("T", [0.01, 0.5, 3.0])
@pytest.mark.parametrize("sigma", [0.05, 0.2, 0.8])
def test_put_call_parity(T, sigma, q):
    """C - P = S e^(-qT) - K e^(-rT), a no-arbitrage identity with no model in it."""
    K, r = 100.0, 0.03
    lhs = call_price(SPOTS, K, T, r, sigma, q) - put_price(SPOTS, K, T, r, sigma, q)
    rhs = SPOTS * np.exp(-q * T) - K * np.exp(-r * T)
    np.testing.assert_allclose(lhs, rhs, atol=1e-10)


def test_expiry_gives_intrinsic_value():
    K = 100.0
    np.testing.assert_allclose(
        call_price(SPOTS, K, 0.0, 0.05, 0.2), np.maximum(SPOTS - K, 0.0)
    )
    np.testing.assert_allclose(
        put_price(SPOTS, K, 0.0, 0.05, 0.2), np.maximum(K - SPOTS, 0.0)
    )


def test_price_tends_to_intrinsic_as_maturity_vanishes():
    """Away from the strike the time value disappears almost instantly."""
    K = 100.0
    away = SPOTS != K
    near_expiry = call_price(SPOTS, K, 1e-10, 0.05, 0.2)
    np.testing.assert_allclose(
        near_expiry[away], np.maximum(SPOTS - K, 0.0)[away], atol=1e-10
    )


def test_at_the_money_time_value_shrinks_like_sqrt_maturity():
    """Near expiry, an ATM option is worth about 0.4 * S * sigma * sqrt(T)."""
    S, sigma, T = 100.0, 0.2, 1e-10
    expected = S * sigma * np.sqrt(T) / np.sqrt(2 * np.pi)
    assert call_price(S, S, T, 0.0, sigma) == pytest.approx(expected, rel=1e-6)


def test_zero_volatility_gives_discounted_forward_payoff():
    """With no uncertainty the option is worth its certain payoff, discounted."""
    K, T, r = 100.0, 1.0, 0.05
    call = call_price(SPOTS, K, T, r, 0.0)
    put = put_price(SPOTS, K, T, r, 0.0)
    np.testing.assert_allclose(call, np.maximum(SPOTS - K * np.exp(-r * T), 0.0))
    np.testing.assert_allclose(put, np.maximum(K * np.exp(-r * T) - SPOTS, 0.0))


def test_small_volatility_is_continuous_with_zero_volatility():
    K, T, r = 100.0, 1.0, 0.05
    np.testing.assert_allclose(
        call_price(SPOTS, K, T, r, 1e-6), call_price(SPOTS, K, T, r, 0.0), atol=1e-8
    )


def test_at_the_forward_with_no_uncertainty_is_worth_zero():
    assert call_price(100.0, 100.0, 0.0, 0.05, 0.2) == 0.0
    assert put_price(100.0, 100.0, 1.0, 0.0, 0.0) == 0.0


def test_call_respects_no_arbitrage_bounds():
    K, T, r, sigma, q = 100.0, 1.0, 0.05, 0.3, 0.01
    call = call_price(SPOTS, K, T, r, sigma, q)
    lower = np.maximum(SPOTS * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    upper = SPOTS * np.exp(-q * T)
    assert np.all(call >= lower) and np.all(call <= upper)


def test_price_increases_with_volatility():
    sigmas = np.linspace(0.05, 1.0, 20)
    assert np.all(np.diff(call_price(100.0, 110.0, 1.0, 0.03, sigmas)) > 0)
    assert np.all(np.diff(put_price(100.0, 90.0, 1.0, 0.03, sigmas)) > 0)


def test_vectorised_inputs_broadcast():
    strikes = np.array([90.0, 100.0, 110.0])
    maturities = np.array([[0.25], [1.0]])
    prices = call_price(100.0, strikes, maturities, 0.03, 0.2)
    assert prices.shape == (2, 3)
    assert prices[1, 2] == pytest.approx(call_price(100.0, 110.0, 1.0, 0.03, 0.2))


def test_bs_price_dispatches_on_option_type():
    assert bs_price(100, 100, 1, 0.05, 0.2, "call") == call_price(100, 100, 1, 0.05, 0.2)
    assert bs_price(100, 100, 1, 0.05, 0.2, "put") == put_price(100, 100, 1, 0.05, 0.2)
    with pytest.raises(ValueError):
        bs_price(100, 100, 1, 0.05, 0.2, "straddle")


@pytest.mark.parametrize(
    "S, K, T, sigma",
    [(0.0, 100.0, 1.0, 0.2), (100.0, -1.0, 1.0, 0.2), (100.0, 100.0, -1.0, 0.2), (100.0, 100.0, 1.0, -0.2)],
)
def test_invalid_inputs_are_rejected(S, K, T, sigma):
    with pytest.raises(ValueError):
        call_price(S, K, T, 0.05, sigma)
