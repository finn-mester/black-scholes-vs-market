"""Monte Carlo pricing of European options under geometric Brownian motion.

Financial intuition
-------------------
The closed form says the price is a discounted risk-neutral expectation. Monte
Carlo computes that same expectation by brute force: simulate many possible
final stock prices, average the payoffs, discount. It needs no formula, so it
gives an independent check here and still works for payoffs with no closed form.

Under risk-neutral geometric Brownian motion the terminal price is
S_T = S exp((r - q - sigma^2 / 2) T + sigma sqrt(T) Z), with Z standard normal.
The -sigma^2 / 2 correction keeps the expected stock price equal to the
forward: without it, the convexity of the exponential would inflate the mean.
A European payoff depends only on S_T, so one exact step is enough and there
is no time-discretisation error.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from bs_lab.pricing import OptionType

Z_95 = 1.96


@dataclass(frozen=True)
class MonteCarloResult:
    """Estimated price with its standard error (the noise of the estimate)."""

    price: float
    std_error: float
    n_paths: int

    @property
    def ci_95(self) -> tuple[float, float]:
        """95% confidence interval for the true price."""
        half_width = Z_95 * self.std_error
        return self.price - half_width, self.price + half_width


def simulate_terminal_prices(
    S: float,
    T: float,
    r: float,
    sigma: float,
    z: np.ndarray,
    q: float = 0.0,
) -> np.ndarray:
    """Map standard normal draws to risk-neutral terminal stock prices."""
    drift = (r - q - 0.5 * sigma**2) * T
    return S * np.exp(drift + sigma * np.sqrt(T) * z)


def _discounted_payoffs(
    S_T: ArrayLike, K: float, T: float, r: float, option_type: OptionType
) -> np.ndarray:
    if option_type == "call":
        payoff = np.maximum(S_T - K, 0.0)
    elif option_type == "put":
        payoff = np.maximum(K - S_T, 0.0)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    return np.exp(-r * T) * payoff


def mc_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int,
    option_type: OptionType = "call",
    q: float = 0.0,
    seed: int | None = None,
    antithetic: bool = False,
) -> MonteCarloResult:
    """Estimate a European option price by simulation.

    With `antithetic=True`, each draw Z is paired with -Z. If one path ends
    high, its mirror ends low, so sampling noise partly cancels. Each mirrored
    pair is averaged first, because the two values are not independent and the
    standard error must be computed on the pairs.
    """
    if n_paths < 2:
        raise ValueError("n_paths must be at least 2")
    rng = np.random.default_rng(seed)
    if antithetic:
        z = rng.standard_normal(n_paths // 2)
        up = _discounted_payoffs(simulate_terminal_prices(S, T, r, sigma, z, q), K, T, r, option_type)
        down = _discounted_payoffs(simulate_terminal_prices(S, T, r, sigma, -z, q), K, T, r, option_type)
        samples = 0.5 * (up + down)
    else:
        z = rng.standard_normal(n_paths)
        samples = _discounted_payoffs(simulate_terminal_prices(S, T, r, sigma, z, q), K, T, r, option_type)
    std_error = samples.std(ddof=1) / np.sqrt(samples.size)
    return MonteCarloResult(float(samples.mean()), float(std_error), n_paths)
