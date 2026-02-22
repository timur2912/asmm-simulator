"""B4: Strategy comparison directional tests (H1)."""

from __future__ import annotations

import pytest

from mm_sim.config import SimulationConfig
from mm_sim.simulator import simulate_mc
from mm_sim.strategies import InventoryStrategy, SymmetricStrategy


@pytest.fixture
def mc_config_300():
    """300 paths for reasonable CI speed."""
    return SimulationConfig(
        S0=100.0, T=1.0, dt=0.005, sigma=2.0,
        A=140.0, k=1.5, gamma=0.1, n_paths=300,
        seed=42, use_crn=True,
    )


class TestStrategyComparisonDirectional:
    """H1: Inventory strategy reduces risk vs symmetric."""

    @pytest.mark.parametrize("gamma", [0.01, 0.1, 0.5])
    def test_std_profit_lower_inventory(self, gamma):
        cfg = SimulationConfig(
            S0=100.0, T=1.0, dt=0.005, sigma=2.0,
            A=140.0, k=1.5, gamma=gamma, n_paths=300,
            seed=42, use_crn=True,
        )
        inv = simulate_mc(InventoryStrategy(), cfg)
        sym = simulate_mc(SymmetricStrategy(), cfg)
        assert inv.summary.std_profit < sym.summary.std_profit, (
            f"γ={gamma}: inv std_profit={inv.summary.std_profit:.2f} "
            f">= sym std_profit={sym.summary.std_profit:.2f}"
        )

    @pytest.mark.parametrize("gamma", [0.01, 0.1, 0.5])
    def test_std_q_lower_inventory(self, gamma):
        cfg = SimulationConfig(
            S0=100.0, T=1.0, dt=0.005, sigma=2.0,
            A=140.0, k=1.5, gamma=gamma, n_paths=300,
            seed=42, use_crn=True,
        )
        inv = simulate_mc(InventoryStrategy(), cfg)
        sym = simulate_mc(SymmetricStrategy(), cfg)
        assert inv.summary.std_q < sym.summary.std_q, (
            f"γ={gamma}: inv std_q={inv.summary.std_q:.2f} "
            f">= sym std_q={sym.summary.std_q:.2f}"
        )
