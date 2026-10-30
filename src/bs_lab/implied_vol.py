"""Implied volatility: the sigma that makes Black-Scholes match a price.

Financial intuition
-------------------
Every input to the formula is observable except volatility, so traders run the
formula backwards and quote options in vol. Implied vol is then the market's
price of future uncertainty, expressed in a common unit that lets options of
different strikes and maturities be compared.

The price strictly increases with sigma (vega > 0), so there is exactly one
solution whenever the price respects the no-arbitrage bounds. Outside those
bounds no volatility can explain the price and we return NaN.

Method
------
Newton-Raphson uses vega, the slope of price against vol, and usually converges
in a few steps. It fails when vega is nearly flat (deep in or out of the money,
or very short-dated), because it then divides by almost zero and jumps away.
Those cases fall back to bisection, which is slower but cannot fail once the
solution is bracketed.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike

from bs_lab.greeks import vega
from bs_lab.pricing import OptionType, _as_arrays, bs_price

VOL_MIN: Final = 1e-6
VOL_MAX: Final = 10.0
MIN_VEGA: Final = 1e-10


@dataclass(frozen=True)
class ImpliedVolResult:
    """Solved vols, with a mask showing which entries needed bisection."""

    vol: np.ndarray
    used_bisection: np.ndarray


def _no_arbitrage_bounds(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    option_type: OptionType,
) -> tuple[np.ndarray, np.ndarray]:
    """Lowest and highest prices that any positive volatility can produce."""
    forward_stock = S * np.exp(-q * T)
    discounted_strike = K * np.exp(-r * T)
    if option_type == "call":
        return np.maximum(forward_stock - discounted_strike, 0.0), forward_stock
    return np.maximum(discounted_strike - forward_stock, 0.0), discounted_strike


def _bisect(
    price: np.ndarray,
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    option_type: OptionType,
    tol: float,
    max_iter: int = 200,
) -> np.ndarray:
    """Halve the bracket [VOL_MIN, VOL_MAX] until it is narrower than tol.

    A price the bracket cannot reach gives NaN, not a vol stuck at the edge.
    """
    bracketed = (bs_price(S, K, T, r, VOL_MIN, option_type, q) <= price) & (
        price <= bs_price(S, K, T, r, VOL_MAX, option_type, q)
    )
    low = np.full_like(price, VOL_MIN)
    high = np.full_like(price, VOL_MAX)
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        too_low = bs_price(S, K, T, r, mid, option_type, q) < price
        low = np.where(too_low, mid, low)
        high = np.where(too_low, high, mid)
        if np.all(high - low < tol):
            break
    return np.where(bracketed, 0.5 * (low + high), np.nan)


def solve_implied_vol(
    price: ArrayLike,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    option_type: OptionType = "call",
    q: ArrayLike = 0.0,
    sigma0: float = 0.3,
    tol: float = 1e-10,
    max_iter: int = 50,
) -> ImpliedVolResult:
    """Solve for implied vol by Newton-Raphson, falling back on bisection."""
    price, S, K, T, r, q = _as_arrays(price, S, K, T, r, q)
    lower, upper = _no_arbitrage_bounds(S, K, T, r, q, option_type)
    solvable = (T > 0) & (price > lower) & (price < upper)

    vol = np.where(solvable, sigma0, np.nan)
    active = solvable.copy()
    needs_bisection = np.zeros_like(solvable)

    for _ in range(max_iter):
        if not active.any():
            break
        error = bs_price(S, K, T, r, np.where(active, vol, sigma0), option_type, q) - price
        slope = vega(S, K, T, r, np.where(active, vol, sigma0), q)

        converged = active & (np.abs(error) < tol)
        active &= ~converged
        stalled = active & (slope < MIN_VEGA)

        with np.errstate(divide="ignore", invalid="ignore"):
            step = vol - error / slope
        escaped = active & ~stalled & ~((step > VOL_MIN) & (step < VOL_MAX))
        needs_bisection |= stalled | escaped
        active &= ~(stalled | escaped)
        vol = np.where(active, step, vol)

    needs_bisection |= active  # Newton ran out of iterations.
    if needs_bisection.any():
        vol[needs_bisection] = _bisect(
            price[needs_bisection],
            S[needs_bisection],
            K[needs_bisection],
            T[needs_bisection],
            r[needs_bisection],
            q[needs_bisection],
            option_type,
            tol,
        )
    return ImpliedVolResult(vol[()], needs_bisection[()])


def implied_vol(
    price: ArrayLike,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    option_type: OptionType = "call",
    q: ArrayLike = 0.0,
) -> np.ndarray:
    """Return the implied vol, or NaN where the price breaks no-arbitrage bounds."""
    return solve_implied_vol(price, S, K, T, r, option_type, q).vol
