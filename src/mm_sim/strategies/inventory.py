"""Inventory-based strategy: quotes centered around reservation price."""

from __future__ import annotations

from mm_sim.config import SimulationConfig
from mm_sim.state import SimState
from mm_sim.strategies.base import Strategy


def _compute_reservation_price(
    S: float, q: int, gamma: float, sigma: float, T: float, t: float,
) -> float:
    """r(S, q, t) = S - q·γ·σ²·(T - t)"""
    return S - q * gamma * sigma**2 * (T - t)


class InventoryStrategy(Strategy):
    """Inventory strategy: quotes centered around reservation price r_t."""

    @property
    def name(self) -> str:
        return "inventory"

    def reservation_price(self, state: SimState, config: SimulationConfig) -> float:
        """Compute the reservation (indifference) price."""
        return _compute_reservation_price(
            state.S, state.q, config.gamma, config.sigma, config.T, state.t,
        )

    def quotes(self, state: SimState, config: SimulationConfig) -> tuple[float, float]:
        r = self.reservation_price(state, config)
        sp = self.spread(config, state.t)
        half = sp / 2.0
        p_bid = r - half
        p_ask = r + half
        return p_bid, p_ask
