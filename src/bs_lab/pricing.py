"""Closed-form Black-Scholes prices for European calls and puts.

Financial intuition
-------------------
An option can be replicated by continuously trading the stock and a bond.
Because that hedge removes the risk, the option's price cannot depend on the
stock's expected return, only on its volatility. The price is therefore the
discounted expected payoff in a "risk-neutral" world where the stock drifts at
the risk-free rate (minus the dividend yield).

Every function accepts floats or numpy arrays and broadcasts them together.
"""

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm

OptionType = Literal["call", "put"]


def _as_arrays(*values: ArrayLike) -> tuple[np.ndarray, ...]:
    """Convert inputs to float arrays that share a common shape."""
    return tuple(np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in values)))


def validate_inputs(
    S: np.ndarray, K: np.ndarray, T: np.ndarray, sigma: np.ndarray
) -> None:
    """Reject inputs for which the model has no meaning."""
    if np.any(S <= 0) or np.any(K <= 0):
        raise ValueError("S and K must be strictly positive")
    if np.any(T < 0) or np.any(sigma < 0):
        raise ValueError("T and sigma must be non-negative")


def d1_d2(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the two standardised moneyness terms of Black-Scholes.

    Both measure how far the forward sits above the strike, in units of the
    total volatility sigma * sqrt(T). N(d2) is the risk-neutral probability of
    finishing in the money. N(d1) is that same probability under a measure
    weighted by the stock price, which is why it also gives the hedge ratio.

    When sigma * sqrt(T) = 0 the outcome is certain: d1 and d2 go to +inf or
    -inf depending on the side of the strike, so the price formula naturally
    collapses to the discounted intrinsic value. Exactly at the forward the
    ratio is 0/0 and we return 0, which still gives the right price of zero.
    """
    S, K, T, r, sigma, q = _as_arrays(S, K, T, r, sigma, q)
    validate_inputs(S, K, T, sigma)
    total_vol = sigma * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / total_vol
    d1 = np.nan_to_num(d1, nan=0.0, posinf=np.inf, neginf=-np.inf)
    return d1, d1 - total_vol


def call_price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
) -> np.ndarray:
    """Price a European call.

    Read the formula as: receive the stock if exercised (first term) and pay
    the discounted strike if exercised (second term).
    C = S e^(-qT) N(d1) - K e^(-rT) N(d2)
    """
    S, K, T, r, sigma, q = _as_arrays(S, K, T, r, sigma, q)
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    return (S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))[()]


def put_price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
) -> np.ndarray:
    """Price a European put.

    Mirror image of the call: receive the discounted strike and deliver the
    stock if exercised. Written directly, not derived from parity, so that the
    parity check in the tests is a genuine check.
    P = K e^(-rT) N(-d2) - S e^(-qT) N(-d1)
    """
    S, K, T, r, sigma, q = _as_arrays(S, K, T, r, sigma, q)
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    return (K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1))[()]


def bs_price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    option_type: OptionType = "call",
    q: ArrayLike = 0.0,
) -> np.ndarray:
    """Price a European option of the given type."""
    if option_type == "call":
        return call_price(S, K, T, r, sigma, q)
    if option_type == "put":
        return put_price(S, K, T, r, sigma, q)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
