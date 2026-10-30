"""Tests for the closed-form Greeks.

The main check bumps each input in the pricing formula and compares the
numerical slope with the closed-form Greek. The two share no derivative code,
so an error in either one shows up as a disagreement.
"""

import numpy as np
import pytest

from bs_lab.greeks import delta, gamma, rho, theta, vega
from bs_lab.pricing import bs_price

CASES = [
    # S, K, T, r, sigma, q
    (100.0, 100.0, 1.0, 0.05, 0.20, 0.00),
    (100.0, 120.0, 0.25, 0.02, 0.35, 0.01),
    (130.0, 100.0, 2.0, 0.03, 0.15, 0.02),
    (80.0, 100.0, 0.5, 0.00, 0.60, 0.00),
]
OPTION_TYPES = ["call", "put"]


@pytest.mark.parametrize("S, K, T, r, sigma, q", CASES)
@pytest.mark.parametrize("option_type", OPTION_TYPES)
def test_delta_matches_finite_difference(S, K, T, r, sigma, q, option_type):
    h = 1e-4
    up = bs_price(S + h, K, T, r, sigma, option_type, q)
    down = bs_price(S - h, K, T, r, sigma, option_type, q)
    assert delta(S, K, T, r, sigma, option_type, q) == pytest.approx(
        (up - down) / (2 * h), abs=1e-7
    )


@pytest.mark.parametrize("S, K, T, r, sigma, q", CASES)
@pytest.mark.parametrize("option_type", OPTION_TYPES)
def test_gamma_matches_finite_difference(S, K, T, r, sigma, q, option_type):
    h = 1e-2
    up = bs_price(S + h, K, T, r, sigma, option_type, q)
    mid = bs_price(S, K, T, r, sigma, option_type, q)
    down = bs_price(S - h, K, T, r, sigma, option_type, q)
    assert gamma(S, K, T, r, sigma, q) == pytest.approx(
        (up - 2 * mid + down) / h**2, abs=1e-6
    )


@pytest.mark.parametrize("S, K, T, r, sigma, q", CASES)
@pytest.mark.parametrize("option_type", OPTION_TYPES)
def test_vega_matches_finite_difference(S, K, T, r, sigma, q, option_type):
    h = 1e-5
    up = bs_price(S, K, T, r, sigma + h, option_type, q)
    down = bs_price(S, K, T, r, sigma - h, option_type, q)
    assert vega(S, K, T, r, sigma, q) == pytest.approx((up - down) / (2 * h), abs=1e-5)


@pytest.mark.parametrize("S, K, T, r, sigma, q", CASES)
@pytest.mark.parametrize("option_type", OPTION_TYPES)
def test_theta_matches_finite_difference(S, K, T, r, sigma, q, option_type):
    # Minus sign: theta is the effect of time passing, so maturity shrinks.
    h = 1e-5
    up = bs_price(S, K, T + h, r, sigma, option_type, q)
    down = bs_price(S, K, T - h, r, sigma, option_type, q)
    assert theta(S, K, T, r, sigma, option_type, q) == pytest.approx(
        -(up - down) / (2 * h), abs=1e-5
    )


@pytest.mark.parametrize("S, K, T, r, sigma, q", CASES)
@pytest.mark.parametrize("option_type", OPTION_TYPES)
def test_rho_matches_finite_difference(S, K, T, r, sigma, q, option_type):
    h = 1e-5
    up = bs_price(S, K, T, r + h, sigma, option_type, q)
    down = bs_price(S, K, T, r - h, sigma, option_type, q)
    assert rho(S, K, T, r, sigma, option_type, q) == pytest.approx(
        (up - down) / (2 * h), abs=1e-5
    )


@pytest.mark.parametrize("S, K, T, r, sigma, q", CASES)
def test_black_scholes_pde(S, K, T, r, sigma, q):
    """Theta, delta and gamma must balance: a hedged option earns the risk-free rate."""
    for option_type in OPTION_TYPES:
        price = bs_price(S, K, T, r, sigma, option_type, q)
        pde = (
            theta(S, K, T, r, sigma, option_type, q)
            + (r - q) * S * delta(S, K, T, r, sigma, option_type, q)
            + 0.5 * sigma**2 * S**2 * gamma(S, K, T, r, sigma, q)
            - r * price
        )
        assert pde == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("S, K, T, r, sigma, q", CASES)
def test_parity_links_call_and_put_delta(S, K, T, r, sigma, q):
    gap = delta(S, K, T, r, sigma, "call", q) - delta(S, K, T, r, sigma, "put", q)
    assert gap == pytest.approx(np.exp(-q * T))


def test_gamma_and_vega_peak_near_the_money():
    spots = np.linspace(50.0, 150.0, 1001)
    assert 95.0 < spots[np.argmax(gamma(spots, 100.0, 0.5, 0.02, 0.2))] < 105.0
    assert 95.0 < spots[np.argmax(vega(spots, 100.0, 0.5, 0.02, 0.2))] < 105.0


def test_greeks_at_expiry_are_finite_and_match_the_payoff():
    spots = np.array([90.0, 110.0])
    np.testing.assert_allclose(delta(spots, 100.0, 0.0, 0.05, 0.2), [0.0, 1.0])
    np.testing.assert_allclose(delta(spots, 100.0, 0.0, 0.05, 0.2, "put"), [-1.0, 0.0])
    np.testing.assert_allclose(gamma(spots, 100.0, 0.0, 0.05, 0.2), 0.0)
    np.testing.assert_allclose(vega(spots, 100.0, 0.0, 0.05, 0.2), 0.0)
    np.testing.assert_allclose(rho(spots, 100.0, 0.0, 0.05, 0.2), 0.0)
    assert np.all(np.isfinite(theta(spots, 100.0, 0.0, 0.05, 0.2)))


def test_greeks_with_zero_volatility_are_finite():
    spots = np.array([90.0, 110.0])
    np.testing.assert_allclose(gamma(spots, 100.0, 1.0, 0.05, 0.0), 0.0)
    np.testing.assert_allclose(vega(spots, 100.0, 1.0, 0.05, 0.0), 0.0)
    assert np.all(np.isfinite(theta(spots, 100.0, 1.0, 0.05, 0.0)))


def test_greeks_are_vectorised():
    spots = np.linspace(80.0, 120.0, 5)
    maturities = np.array([[0.25], [1.0]])
    assert delta(spots, 100.0, maturities, 0.02, 0.2).shape == (2, 5)
    assert theta(spots, 100.0, maturities, 0.02, 0.2).shape == (2, 5)


def test_unknown_option_type_is_rejected():
    with pytest.raises(ValueError):
        delta(100.0, 100.0, 1.0, 0.05, 0.2, "straddle")
