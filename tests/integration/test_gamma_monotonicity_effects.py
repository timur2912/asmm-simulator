"""B5: Gamma monotonicity effects (H2)."""

from __future__ import annotations

import pytest

from mm_sim.config import SimulationConfig
from mm_sim.simulator import simulate_mc
from mm_sim.strategies import InventoryStrategy, SymmetricStrategy


class TestGammaMonotonicity:
    """H2: Increasing γ reduces inventory dispersion under inventory strategy."""

    def _run_inventory(self, gamma: float, n_paths: int = 300) -> float:
        cfg = SimulationConfig(
            S0=100.0, T=1.0, dt=0.005, sigma=2.0,
            A=140.0, k=1.5, gamma=gamma, n_paths=n_paths,
            seed=42, use_crn=True,
        )
        return simulate_mc(InventoryStrategy(), cfg).summary.std_q

    def test_std_q_decreases_with_gamma(self):
        std_001 = self._run_inventory(0.01)
        std_01 = self._run_inventory(0.1)
        std_05 = self._run_inventory(0.5)
        assert std_05 < std_01 < std_001, (
            f"Expected std_q(0.5)={std_05:.2f} < std_q(0.1)={std_01:.2f} "
            f"< std_q(0.01)={std_001:.2f}"
        )

    def test_convergence_gamma_near_zero(self):
        """As γ→0, inventory and symmetric strategies should converge."""
        gamma_small = 1e-6
        cfg = SimulationConfig(
            S0=100.0, T=1.0, dt=0.005, sigma=2.0,
            A=140.0, k=1.5, gamma=gamma_small, n_paths=200,
            seed=42, use_crn=True,
        )
        inv = simulate_mc(InventoryStrategy(), cfg)
        sym = simulate_mc(SymmetricStrategy(), cfg)

        # With CRN and γ≈0, results should be very close
        assert abs(inv.summary.mean_profit - sym.summary.mean_profit) < 2.0
        assert abs(inv.summary.std_profit - sym.summary.std_profit) < 2.0
        assert abs(inv.summary.std_q - sym.summary.std_q) < 1.0
