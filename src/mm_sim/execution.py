"""Execution models: Bernoulli and Poisson fill simulation."""

from __future__ import annotations

import numpy as np


def simulate_fills_bernoulli(
    p_ask: float,
    p_bid: float,
    rng: np.random.Generator,
) -> tuple[bool, bool]:
    """Simulate independent Bernoulli fills for ask and bid.

    Parameters
    ----------
    p_ask : execution probability for the ask side
    p_bid : execution probability for the bid side
    rng : numpy Generator

    Returns ``(ask_filled, bid_filled)``.
    """
    ask = bool(rng.random() < p_ask)
    bid = bool(rng.random() < p_bid)
    return ask, bid


def simulate_fills_poisson(
    lam_dt_ask: float,
    lam_dt_bid: float,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Simulate Poisson fill counts for ask and bid.

    Parameters
    ----------
    lam_dt_ask : λ·dt for the ask side
    lam_dt_bid : λ·dt for the bid side
    rng : numpy Generator

    Returns ``(n_ask_fills, n_bid_fills)``.
    """
    a = int(rng.poisson(max(0.0, lam_dt_ask)))
    b = int(rng.poisson(max(0.0, lam_dt_bid)))
    return a, b
