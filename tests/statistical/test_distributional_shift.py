"""D2: Distributional shift tests."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from mm_sim.config import SimulationConfig
from mm_sim.simulator import simulate_mc
from mm_sim.strategies import InventoryStrategy, SymmetricStrategy


class TestDistributionalShift:
    """Inventory strategy produces more concentrated q_T distribution."""

    def test_inventory_smaller_iqr(self):
        cfg = SimulationConfig(
            S0=100.0, T=1.0, dt=0.005, sigma=2.0,
            A=140.0, k=1.5, gamma=0.1, n_paths=500,
            seed=42, use_crn=True,
        )
        inv = simulate_mc(InventoryStrategy(), cfg)
        sym = simulate_mc(SymmetricStrategy(), cfg)

        iqr_inv = np.percentile(inv.q_array, 75) - np.percentile(inv.q_array, 25)
        iqr_sym = np.percentile(sym.q_array, 75) - np.percentile(sym.q_array, 25)
        assert iqr_inv < iqr_sym

    def test_inventory_higher_concentration_near_zero(self):
        cfg = SimulationConfig(
            S0=100.0, T=1.0, dt=0.005, sigma=2.0,
            A=140.0, k=1.5, gamma=0.1, n_paths=500,
            seed=42, use_crn=True,
        )
        inv = simulate_mc(InventoryStrategy(), cfg)
        sym = simulate_mc(SymmetricStrategy(), cfg)

        # Fraction of paths with |q_T| ≤ 3
        frac_inv = np.mean(np.abs(inv.q_array) <= 3)
        frac_sym = np.mean(np.abs(sym.q_array) <= 3)
        assert frac_inv > frac_sym

    def test_ks_test_pnl_distributions_differ(self):
        cfg = SimulationConfig(
            S0=100.0, T=1.0, dt=0.005, sigma=2.0,
            A=140.0, k=1.5, gamma=0.5, n_paths=500,
            seed=42, use_crn=True,
        )
        inv = simulate_mc(InventoryStrategy(), cfg)
        sym = simulate_mc(SymmetricStrategy(), cfg)

        stat, pval = stats.ks_2samp(inv.pnl_array, sym.pnl_array)
        assert pval < 0.05  # Distributions should differ significantly
