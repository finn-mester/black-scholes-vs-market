"""Cox-Ross-Rubinstein binomial tree for European and American options.

Financial intuition
-------------------
The tree cuts time into steps where the stock moves up or down by a fixed
factor, chosen to match the volatility. Working backwards from expiry, each
node's value is the discounted risk-neutral average of its two successors.
With many steps this converges to Black-Scholes.

Its advantage is that an American option can be handled at every node: the
holder takes the better of holding on or exercising now. There is no simple
closed form for that decision, which is why the tree is useful here.
"""

import numpy as np

from bs_lab.pricing import OptionType


def crr_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_steps: int = 1000,
    option_type: OptionType = "put",
    american: bool = True,
    q: float = 0.0,
) -> float:
    """Price an option on a CRR tree of `n_steps` time steps."""
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")
    sign = 1.0 if option_type == "call" else -1.0

    dt = T / n_steps
    up = np.exp(sigma * np.sqrt(dt))
    down = 1.0 / up
    # Risk-neutral up probability: the one that makes the stock earn r - q on average.
    p_up = (np.exp((r - q) * dt) - down) / (up - down)
    if not 0.0 < p_up < 1.0:
        raise ValueError("Too few steps for these inputs: the tree implies an invalid probability")
    discount = np.exp(-r * dt)

    # At step i, the node with j up-moves has stock price S * up^(2j - i).
    j = np.arange(n_steps + 1)
    stock = S * up ** (2 * j - n_steps)
    value = np.maximum(sign * (stock - K), 0.0)

    for step in range(n_steps - 1, -1, -1):
        value = discount * (p_up * value[1:] + (1.0 - p_up) * value[:-1])
        if american:
            j = np.arange(step + 1)
            stock = S * up ** (2 * j - step)
            value = np.maximum(value, sign * (stock - K))
    return float(value[0])
