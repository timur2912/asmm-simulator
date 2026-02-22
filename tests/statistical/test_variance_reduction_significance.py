"""D1: Variance reduction significance tests (H1)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from mm_sim.config import SimulationConfig
from mm_sim.simulator import simulate_mc
from mm_sim.strategies import InventoryStrategy, SymmetricStrategy


class TestVarianceReductionSignificance:
    """Test that variance reduction is statistically significant."""

    @pytest.mark.parametrize("gamma", [0.1, 0.5])
    def test_levene_profit_variance(self, gamma):
        """Levene's test for equality of variances in P&L."""
        cfg = SimulationConfig(
            S0=100.0, T=1.0, dt=0.005, sigma=2.0,
            A=140.0, k=1.5, gamma=gamma, n_paths=500,
            seed=42, use_crn=True,
        )
        inv = simulate_mc(InventoryStrategy(), cfg)
        sym = simulate_mc(SymmetricStrategy(), cfg)

        stat, pval = stats.levene(inv.pnl_array, sym.pnl_array)
        # Expect significant difference (p < 0.05)
        assert pval < 0.05, (
            f"γ={gamma}: Levene p={pval:.4f}, not significant"
        )
        # And inventory should have lower variance
        assert np.var(inv.pnl_array) < np.var(sym.pnl_array)

    @pytest.mark.parametrize("gamma", [0.1, 0.5])
    def test_bootstrap_std_difference(self, gamma):
        """Bootstrap test: std(inventory) < std(symmetric) with high confidence."""
        cfg = SimulationConfig(
            S0=100.0, T=1.0, dt=0.005, sigma=2.0,
            A=140.0, k=1.5, gamma=gamma, n_paths=500,
            seed=42, use_crn=True,
        )
        inv = simulate_mc(InventoryStrategy(), cfg)
        sym = simulate_mc(SymmetricStrategy(), cfg)

        rng = np.random.default_rng(123)
        n_boot = 1000
        diffs = []
        n = len(inv.pnl_array)
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            std_inv = np.std(inv.pnl_array[idx], ddof=1)
            std_sym = np.std(sym.pnl_array[idx], ddof=1)
            diffs.append(std_inv - std_sym)

        diffs = np.array(diffs)
        # 95% of bootstrap samples should show inventory < symmetric
        frac_negative = np.mean(diffs < 0)
        assert frac_negative > 0.95, (
            f"γ={gamma}: Only {frac_negative:.2%} of bootstrap samples show "
            f"std(inv) < std(sym)"
        )
