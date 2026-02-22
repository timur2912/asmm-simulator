"""B2: Monte Carlo reproducibility tests."""

from __future__ import annotations

import numpy as np
import pytest

from mm_sim.config import SimulationConfig
from mm_sim.simulator import simulate_mc
from mm_sim.strategies import InventoryStrategy, SymmetricStrategy


@pytest.fixture
def mc_config():
    return SimulationConfig(
        S0=100.0, T=1.0, dt=0.005, sigma=2.0,
        A=140.0, k=1.5, gamma=0.1, n_paths=50, seed=42,
    )


class TestMCReproducibility:
    def test_same_seed_same_results_inventory(self, mc_config):
        strat = InventoryStrategy()
        r1 = simulate_mc(strat, mc_config)
        r2 = simulate_mc(strat, mc_config)
        np.testing.assert_array_equal(r1.pnl_array, r2.pnl_array)
        np.testing.assert_array_equal(r1.q_array, r2.q_array)

    def test_same_seed_same_results_symmetric(self, mc_config):
        strat = SymmetricStrategy()
        r1 = simulate_mc(strat, mc_config)
        r2 = simulate_mc(strat, mc_config)
        np.testing.assert_array_equal(r1.pnl_array, r2.pnl_array)
        np.testing.assert_array_equal(r1.q_array, r2.q_array)

    def test_different_seed_different_results(self, mc_config):
        strat = InventoryStrategy()
        cfg1 = mc_config.model_copy(update={"seed": 42})
        cfg2 = mc_config.model_copy(update={"seed": 99})
        r1 = simulate_mc(strat, cfg1)
        r2 = simulate_mc(strat, cfg2)
        assert not np.array_equal(r1.pnl_array, r2.pnl_array)

    def test_summary_stats_match(self, mc_config):
        strat = InventoryStrategy()
        r1 = simulate_mc(strat, mc_config)
        r2 = simulate_mc(strat, mc_config)
        assert abs(r1.summary.mean_profit - r2.summary.mean_profit) < 1e-10
        assert abs(r1.summary.std_profit - r2.summary.std_profit) < 1e-10
