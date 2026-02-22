"""Seed management, RNG streams, and common random numbers (CRN)."""

from __future__ import annotations

import numpy as np

# Type alias for RNG streams dict
RNGStreams = dict[str, np.random.Generator]


def make_streams(seed: int) -> RNGStreams:
    """Create independent RNG streams from a single seed via SeedSequence.

    Returns a dict with keys ``"price"`` and ``"execution"``.
    """
    ss = np.random.SeedSequence(seed)
    children = ss.spawn(2)
    return {
        "price": np.random.default_rng(children[0]),
        "execution": np.random.default_rng(children[1]),
    }


def make_crn_streams(seed: int, n_strategies: int) -> list[RNGStreams]:
    """Create streams that share the same random draws across strategies (CRN)."""
    return [make_streams(seed) for _ in range(n_strategies)]


def make_independent_streams(seed: int, n_strategies: int) -> list[RNGStreams]:
    """Create fully independent streams for each strategy."""
    ss = np.random.SeedSequence(seed)
    children = ss.spawn(n_strategies)
    return [make_streams(int(cs.entropy)) for cs in children]  # type: ignore[arg-type]
