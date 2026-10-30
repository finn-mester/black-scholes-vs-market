"""Closed-form Black-Scholes Greeks.

Greeks are the sensitivities a trading desk actually manages: each one answers
"how much does this position move if one input moves?". They are the partial
derivatives of the price.

Units, chosen to match the derivatives exactly:
- vega and rho are per 1.00 change (1.00 = 100 points of vol or rate), so
  divide by 100 for the value of one point;
- theta is per year, so divide by 365 for the value of one calendar day.

At expiry or with zero volatility the option has a certain payoff, so gamma,
vega and the time-decay term of theta are set to zero. Exactly at the money
these Greeks diverge as T -> 0; zero is a convention at that single point.
"""

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm

from bs_lab.pricing import OptionType, _as_arrays, d1_d2


def _check_option_type(option_type: OptionType) -> None:
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def delta(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    option_type: OptionType = "call",
    q: ArrayLike = 0.0,
) -> np.ndarray:
    """Change in price per 1.00 move in the spot.

    This is the number of shares that hedges one option. A call delta runs
    from 0 (far out of the money) to 1 (deep in the money): the option behaves
    more and more like the stock itself.
    """
    _check_option_type(option_type)
    S, K, T, r, sigma, q = _as_arrays(S, K, T, r, sigma, q)
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    carry = np.exp(-q * T)
    if option_type == "call":
        return (carry * norm.cdf(d1))[()]
    return (carry * (norm.cdf(d1) - 1.0))[()]


def gamma(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
) -> np.ndarray:
    """Change in delta per 1.00 move in the spot, identical for calls and puts.

    Gamma says how fast the hedge goes stale. It peaks at the money and grows
    sharply near expiry, because the option is then flipping between being
    worth something and being worth nothing.
    """
    S, K, T, r, sigma, q = _as_arrays(S, K, T, r, sigma, q)
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    total_vol = sigma * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.exp(-q * T) * norm.pdf(d1) / (S * total_vol)
    return np.where(total_vol > 0, value, 0.0)[()]


def vega(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
) -> np.ndarray:
    """Change in price per 1.00 change in volatility, identical for calls and puts.

    Vega is largest for at-the-money, long-dated options: that is where extra
    uncertainty has the most room to change the outcome.
    """
    S, K, T, r, sigma, q = _as_arrays(S, K, T, r, sigma, q)
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    return (S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T))[()]


def theta(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    option_type: OptionType = "call",
    q: ArrayLike = 0.0,
) -> np.ndarray:
    """Change in price per year as time passes, with spot held fixed.

    The first term is the pure time decay, the rent paid for holding gamma.
    The other terms come from the carry on the stock and the cash in the hedge.
    """
    _check_option_type(option_type)
    S, K, T, r, sigma, q = _as_arrays(S, K, T, r, sigma, q)
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    with np.errstate(divide="ignore", invalid="ignore"):
        decay = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2.0 * np.sqrt(T))
    decay = np.where(sigma * np.sqrt(T) > 0, decay, 0.0)
    stock_flow = S * np.exp(-q * T) * q
    cash_flow = K * np.exp(-r * T) * r
    if option_type == "call":
        return (decay + stock_flow * norm.cdf(d1) - cash_flow * norm.cdf(d2))[()]
    return (decay - stock_flow * norm.cdf(-d1) + cash_flow * norm.cdf(-d2))[()]


def rho(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    option_type: OptionType = "call",
    q: ArrayLike = 0.0,
) -> np.ndarray:
    """Change in price per 1.00 change in the interest rate.

    Higher rates make the strike cheaper in present value, which helps a call
    and hurts a put. The effect scales with maturity and is usually the
    smallest of the Greeks for short-dated equity options.
    """
    _check_option_type(option_type)
    S, K, T, r, sigma, q = _as_arrays(S, K, T, r, sigma, q)
    _, d2 = d1_d2(S, K, T, r, sigma, q)
    discounted_strike = K * T * np.exp(-r * T)
    if option_type == "call":
        return (discounted_strike * norm.cdf(d2))[()]
    return (-discounted_strike * norm.cdf(-d2))[()]
