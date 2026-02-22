"""Symmetric (naive) benchmark strategy: quotes centered around mid-price."""

from __future__ import annotations

from mm_sim.config import SimulationConfig
from mm_sim.state import SimState
from mm_sim.strategies.base import Strategy


class SymmetricStrategy(Strategy):
    """Symmetric strategy: same spread as inventory, centered on S_t."""

    @property
    def name(self) -> str:
        return "symmetric"

    def quotes(self, state: SimState, config: SimulationConfig) -> tuple[float, float]:
        sp = self.spread(config, state.t)
        half = sp / 2.0
        p_bid = state.S - half
        p_ask = state.S + half
        return p_bid, p_ask
