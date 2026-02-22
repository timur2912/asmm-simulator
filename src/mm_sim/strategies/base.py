"""Base strategy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from mm_sim.config import SimulationConfig
from mm_sim.state import SimState


class Strategy(ABC):
    """Abstract base for quoting strategies."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def quotes(self, state: SimState, config: SimulationConfig) -> tuple[float, float]:
        """Return ``(p_bid, p_ask)`` given current state and config."""
        ...

    def spread(self, config: SimulationConfig, t: float = 0.0) -> float:
        """Compute the spread used by this strategy."""
        if config.spread_model.value == "constant":
            return config.constant_spread()
        return config.time_varying_spread(t)
