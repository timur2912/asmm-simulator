from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelParams:
    s0: float = 100.0
    T: float = 1.0
    sigma: float = 2.0
    dt: float = 0.005
    q0: int = 0
    A: float = 140.0
    k: float = 1.5


def reservation_price(*, s: float, q: int, t: float, params: ModelParams, gamma: float) -> float:
    """Inventory-dependent reservation (indifference) price.

    r(s,q,t) = s - q * gamma * sigma^2 * (T - t)
    """
    return s - q * gamma * (params.sigma**2) * (params.T - t)


def spread_constant(*, gamma: float, k: float) -> float:
    """Table-consistent constant spread: (2/gamma) * ln(1 + gamma/k)."""
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    return (2.0 / gamma) * math.log(1.0 + gamma / k)


def spread_time_varying(*, t: float, params: ModelParams, gamma: float) -> float:
    """Theory-consistent time-varying spread.

    Sp_t = gamma*sigma^2*(T-t) + (2/gamma)*ln(1 + gamma/k)
    """
    return gamma * (params.sigma**2) * (params.T - t) + spread_constant(gamma=gamma, k=params.k)


def intensity(*, delta: float, A: float, k: float) -> float:
    """Poisson intensity as function of quote distance delta >= 0."""
    if delta < 0:
        raise ValueError("delta must be >= 0")
    return A * math.exp(-k * delta)
