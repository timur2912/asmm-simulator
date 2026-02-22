"""Mid-price process generators: Rademacher and Gaussian."""

from __future__ import annotations

import numpy as np


def generate_increments(
    n_steps: int,
    sigma: float,
    dt: float,
    model: str = "rademacher",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate *n_steps* price increments (vectorized).

    Parameters
    ----------
    n_steps : number of increments
    sigma : volatility
    dt : time step
    model : ``"rademacher"`` (±σ√dt) or ``"gaussian"`` (N(0, σ²dt))
    rng : numpy Generator (uses default if *None*)
    """
    if rng is None:
        rng = np.random.default_rng()
    sqrt_dt = np.sqrt(dt)
    if model == "rademacher":
        signs = 2 * rng.integers(0, 2, size=n_steps) - 1
        return signs.astype(np.float64) * sigma * sqrt_dt
    else:
        return rng.normal(0.0, sigma * sqrt_dt, size=n_steps)


# Backward-compat alias used by simulator
pregenerate_increments = generate_increments


def generate_price_path(
    S0: float,
    n_steps: int,
    sigma: float,
    dt: float,
    model: str = "rademacher",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a full mid-price path of length *n_steps + 1*.

    ``path[0] == S0``; subsequent values are cumulative sums of increments.
    """
    inc = generate_increments(n_steps, sigma, dt, model, rng)
    path = np.empty(n_steps + 1)
    path[0] = S0
    np.cumsum(inc, out=path[1:])
    path[1:] += S0
    return path
