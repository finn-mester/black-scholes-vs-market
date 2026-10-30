"""Tests for the Monte Carlo pricer. All seeds are fixed, so results are repeatable."""

import numpy as np
import pytest

from bs_lab.montecarlo import mc_price, simulate_terminal_prices
from bs_lab.pricing import bs_price

S, K, T, R, SIGMA = 100.0, 105.0, 1.0, 0.03, 0.25


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("q", [0.0, 0.02])
def test_converges_to_closed_form(option_type, q):
    result = mc_price(S, K, T, R, SIGMA, 2_000_000, option_type, q, seed=1)
    exact = bs_price(S, K, T, R, SIGMA, option_type, q)
    assert abs(result.price - exact) < 4 * result.std_error
    assert result.std_error < 0.02


def test_error_shrinks_like_one_over_sqrt_n():
    """100 times more paths should give about 10 times less noise."""
    small = mc_price(S, K, T, R, SIGMA, 10_000, seed=2)
    large = mc_price(S, K, T, R, SIGMA, 1_000_000, seed=2)
    assert small.std_error / large.std_error == pytest.approx(10.0, rel=0.05)


def test_empirical_rmse_matches_reported_std_error():
    """Check the error bar itself: repeated runs must scatter by that amount."""
    exact = bs_price(S, K, T, R, SIGMA)
    runs = [mc_price(S, K, T, R, SIGMA, 20_000, seed=seed) for seed in range(300)]
    errors = np.array([run.price - exact for run in runs])
    rmse = np.sqrt(np.mean(errors**2))
    assert rmse == pytest.approx(np.mean([run.std_error for run in runs]), rel=0.15)


def test_confidence_interval_covers_about_95_percent():
    exact = bs_price(S, K, T, R, SIGMA)
    intervals = [mc_price(S, K, T, R, SIGMA, 5_000, seed=seed).ci_95 for seed in range(1_000)]
    coverage = np.mean([low <= exact <= high for low, high in intervals])
    assert 0.93 <= coverage <= 0.97


def test_expected_terminal_price_is_the_forward():
    """The -sigma^2 / 2 drift term must make the stock a fair forward."""
    z = np.random.default_rng(3).standard_normal(4_000_000)
    terminal = simulate_terminal_prices(S, T, R, SIGMA, z, q=0.01)
    assert terminal.mean() == pytest.approx(S * np.exp((R - 0.01) * T), rel=1e-3)


def test_antithetic_sampling_reduces_noise_and_stays_unbiased():
    n_paths = 1_000_000
    plain = mc_price(S, K, T, R, SIGMA, n_paths, seed=4)
    mirrored = mc_price(S, K, T, R, SIGMA, n_paths, seed=4, antithetic=True)
    assert mirrored.std_error < plain.std_error
    assert abs(mirrored.price - bs_price(S, K, T, R, SIGMA)) < 4 * mirrored.std_error


def test_same_seed_gives_same_estimate():
    first = mc_price(S, K, T, R, SIGMA, 1_000, seed=5)
    assert first == mc_price(S, K, T, R, SIGMA, 1_000, seed=5)


def test_invalid_arguments_are_rejected():
    with pytest.raises(ValueError):
        mc_price(S, K, T, R, SIGMA, 1_000, "straddle")
    with pytest.raises(ValueError):
        mc_price(S, K, T, R, SIGMA, 1)
