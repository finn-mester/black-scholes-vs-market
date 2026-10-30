"""Tests for the CRR binomial tree."""

import numpy as np
import pytest

from bs_lab.binomial import crr_price
from bs_lab.pricing import bs_price

S, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.2


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_european_tree_converges_to_black_scholes(option_type):
    tree = crr_price(S, K, T, R, SIGMA, 2000, option_type, american=False, q=0.01)
    assert tree == pytest.approx(bs_price(S, K, T, R, SIGMA, option_type, 0.01), abs=2e-3)


def test_error_falls_as_steps_increase():
    exact = bs_price(S, 95.0, T, R, SIGMA, "put")
    errors = [abs(crr_price(S, 95.0, T, R, SIGMA, n, "put", american=False) - exact) for n in (50, 500)]
    assert errors[1] < errors[0]


def test_published_american_put_value():
    """Hull's standard American put example, whose converged value is about 4.28."""
    assert crr_price(50.0, 50.0, 5 / 12, 0.10, 0.40, 5000) == pytest.approx(4.28, abs=0.01)


def test_american_call_without_dividends_equals_european():
    """Exercising early throws away time value, so it is never optimal."""
    american = crr_price(S, K, T, R, SIGMA, 1000, "call", american=True)
    european = crr_price(S, K, T, R, SIGMA, 1000, "call", american=False)
    assert american == pytest.approx(european, abs=1e-9)


def test_american_put_is_worth_more_than_european():
    american = crr_price(S, K, T, R, SIGMA, 1000, "put", american=True)
    european = crr_price(S, K, T, R, SIGMA, 1000, "put", american=False)
    assert american > european + 0.1


def test_dividends_can_make_early_exercise_of_a_call_worthwhile():
    american = crr_price(S, K, T, R, SIGMA, 1000, "call", american=True, q=0.06)
    european = crr_price(S, K, T, R, SIGMA, 1000, "call", american=False, q=0.06)
    assert american > european + 0.01


def test_deep_in_the_money_american_put_is_worth_intrinsic():
    assert crr_price(50.0, 100.0, T, R, SIGMA, 500, "put") == pytest.approx(50.0)


def test_zero_rate_removes_the_early_exercise_value_of_a_put():
    american = crr_price(S, K, T, 0.0, SIGMA, 1000, "put", american=True)
    european = crr_price(S, K, T, 0.0, SIGMA, 1000, "put", american=False)
    assert american == pytest.approx(european, abs=1e-9)


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        crr_price(S, K, T, R, SIGMA, 100, "straddle")
    with pytest.raises(ValueError):
        crr_price(S, K, T, R, SIGMA, 0)
    with pytest.raises(ValueError, match="Too few steps"):
        crr_price(S, K, T, 1.0, 0.01, 1)
