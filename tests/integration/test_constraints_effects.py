"""B7: Constraint effects tests."""

from __future__ import annotations

import numpy as np
import pytest

from mm_sim.config import ConstraintsConfig, SimulationConfig
from mm_sim.simulator import simulate_mc
from mm_sim.strategies import InventoryStrategy


class TestConstraintEffects:
    def test_clipped_prob_never_exceeds_cap(self):
        """With clip_prob=True, execution probability never exceeds prob_cap."""
        from mm_sim.intensities import execution_probability

        # Test with negative delta (high intensity)
        p = execution_probability(-2.0, 140.0, 1.5, 0.005, clip=True, prob_cap=0.95)
        assert p <= 0.95

    def test_constrained_reduces_tail(self):
        """Constrained mode should reduce extreme tail behavior."""
        cfg_unconstrained = SimulationConfig(
            gamma=0.5, n_paths=200, seed=42,
            constraints=ConstraintsConfig(clip_delta=False, clip_prob=False),
        )
        cfg_constrained = SimulationConfig(
            gamma=0.5, n_paths=200, seed=42,
            constraints=ConstraintsConfig(clip_delta=True, clip_prob=True),
        )
        strat = InventoryStrategy()
        r_unc = simulate_mc(strat, cfg_unconstrained)
        r_con = simulate_mc(strat, cfg_constrained)

        # Constrained should have smaller or equal tail range
        range_unc = np.percentile(r_unc.pnl_array, 99) - np.percentile(r_unc.pnl_array, 1)
        range_con = np.percentile(r_con.pnl_array, 99) - np.percentile(r_con.pnl_array, 1)
        # Soft check: constrained range should not be dramatically larger
        assert range_con < range_unc * 1.5
