"""A9: Edge case probability tests."""

from __future__ import annotations

import numpy as np
import pytest

from mm_sim.config import SimulationConfig
from mm_sim.intensities import execution_intensity, execution_probability
from mm_sim.state import SimState
from mm_sim.strategies import InventoryStrategy


class TestHighIntensityEdgeCases:
    def test_delta_zero_probability(self):
        """At δ=0, λ*dt = 140*0.005 = 0.7, which is valid but high."""
        p = execution_probability(0.0, 140.0, 1.5, 0.005, clip=True)
        assert abs(p - 0.7) < 1e-10

    def test_negative_delta_clipped(self):
        """Negative δ gives λ > A, so λ*dt > 1; must be clipped."""
        p = execution_probability(-1.0, 140.0, 1.5, 0.005, clip=True)
        assert p <= 1.0

    def test_negative_delta_unclipped(self):
        """Without clipping, probability can exceed 1."""
        p = execution_probability(-1.0, 140.0, 1.5, 0.005, clip=False)
        assert p > 1.0

    def test_very_large_delta_probability_near_zero(self):
        p = execution_probability(100.0, 140.0, 1.5, 0.005, clip=True)
        assert p < 1e-50

    def test_inventory_strategy_can_produce_negative_delta(self):
        """With large inventory, reservation price shifts enough to cross mid."""
        cfg = SimulationConfig(gamma=0.5, sigma=2.0, T=1.0, k=1.5)
        strat = InventoryStrategy()
        # Large positive inventory → r << S → ask = r + Sp/2 could be < S
        state = SimState(S=100.0, q=10, X=0.0, t=0.0)
        bid, ask = strat.quotes(state, cfg)
        delta_a = ask - state.S
        # With q=10, γ=0.5, σ²=4, T-t=1: r = 100 - 10*0.5*4*1 = 80
        # Sp ≈ 1.15, ask = 80 + 0.575 = 80.575, δ_a = -19.425
        assert delta_a < 0

    def test_simulation_handles_high_prob_gracefully(self):
        """Run a short simulation with parameters that produce high probabilities."""
        from mm_sim.simulator import simulate_path

        cfg = SimulationConfig(
            gamma=0.5, sigma=2.0, T=0.1, dt=0.005, n_paths=1,
            seed=42, A=140.0, k=1.5,
        )
        strat = InventoryStrategy()
        result = simulate_path(strat, cfg, np.random.default_rng(42))
        assert np.isfinite(result.pnl)
        assert np.isfinite(result.q_T)
