"""D3: Convergence as γ→0 tests (H2)."""

from __future__ import annotations

import numpy as np
import pytest

from mm_sim.config import SimulationConfig
from mm_sim.simulator import simulate_mc
from mm_sim.strategies import InventoryStrategy, SymmetricStrategy


class TestConvergenceGammaToZero:
    """As γ→0, inventory and symmetric strategies should converge."""

    def test_very_small_gamma_strategies_converge(self):
        gamma = 1e-6
        cfg = SimulationConfig(
            S0=100.0, T=1.0, dt=0.005, sigma=2.0,
            A=140.0, k=1.5, gamma=gamma, n_paths=200,
            seed=42, use_crn=True,
        )
        inv = simulate_mc(InventoryStrategy(), cfg)
        sym = simulate_mc(SymmetricStrategy(), cfg)

        # With γ≈0, reservation price ≈ mid-price, so strategies are nearly identical
        assert abs(inv.summary.mean_profit - sym.summary.mean_profit) < 1.0
        assert abs(inv.summary.std_profit - sym.summary.std_profit) < 1.0
        assert abs(inv.summary.mean_q - sym.summary.mean_q) < 0.5
        assert abs(inv.summary.std_q - sym.summary.std_q) < 0.5

    def test_pnl_arrays_nearly_identical_at_tiny_gamma(self):
        gamma = 1e-8
        cfg = SimulationConfig(
            S0=100.0, T=1.0, dt=0.005, sigma=2.0,
            A=140.0, k=1.5, gamma=gamma, n_paths=100,
            seed=42, use_crn=True,
        )
        inv = simulate_mc(InventoryStrategy(), cfg)
        sym = simulate_mc(SymmetricStrategy(), cfg)

        # Arrays should be nearly identical with CRN
        np.testing.assert_allclose(
            inv.pnl_array, sym.pnl_array, rtol=1e-3, atol=0.1
        )
