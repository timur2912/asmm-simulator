"""Execution intensity functions: λ(δ) = A·exp(-k·δ)."""

from __future__ import annotations

from typing import Union

import numpy as np


def execution_intensity(
    delta: Union[float, np.ndarray],
    A: float,
    k: float,
) -> Union[float, np.ndarray]:
    """Compute execution intensity λ(δ) = A·exp(-k·δ).

    Works for scalars and arrays.  For large positive δ the intensity
    decays toward 0; for δ < 0 (quote crosses mid) it exceeds *A*.
    """
    return A * np.exp(-k * delta)


# Convenience alias
intensity = execution_intensity


def execution_probability(
    delta: float,
    A: float,
    k: float,
    dt: float,
    clip: bool = True,
    *,
    prob_cap: float = 1.0,
    clip_delta: bool = False,
) -> float:
    """Bernoulli execution probability p = λ(δ)·dt.

    Parameters
    ----------
    delta : distance from mid-price
    A, k : intensity parameters
    dt : time step
    clip : if *True*, cap probability at *prob_cap* (default 1.0)
    prob_cap : maximum allowed probability
    clip_delta : if *True*, enforce δ ≥ 0 before computing λ
    """
    d = max(0.0, delta) if clip_delta else delta
    lam = float(execution_intensity(d, A, k))
    p = lam * dt
    if clip:
        p = min(p, prob_cap)
    return p
