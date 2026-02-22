"""B6: Spread variant tests — verify formula matches paper values."""

from __future__ import annotations

import math

import pytest

from mm_sim.config import SimulationConfig


class TestSpreadVariantsMatchPaper:
    """Verify constant spread formula yields reported values."""

    @pytest.mark.parametrize(
        "gamma,expected_spread",
        [
            (0.01, 1.33),
            (0.1, 1.29),
            (0.5, 1.15),
        ],
    )
    def test_constant_spread_matches_paper(self, gamma, expected_spread):
        cfg = SimulationConfig(gamma=gamma, k=1.5)
        sp = cfg.constant_spread()
        assert abs(sp - expected_spread) < 0.01, (
            f"γ={gamma}: computed spread={sp:.4f}, expected≈{expected_spread}"
        )

    def test_time_varying_spread_larger_at_t0(self):
        cfg = SimulationConfig(gamma=0.1, k=1.5, sigma=2.0, T=1.0)
        sp_const = cfg.constant_spread()
        sp_tv = cfg.time_varying_spread(0.0)
        assert sp_tv > sp_const

    def test_time_varying_equals_constant_at_T(self):
        cfg = SimulationConfig(gamma=0.1, k=1.5, sigma=2.0, T=1.0)
        sp_const = cfg.constant_spread()
        sp_tv = cfg.time_varying_spread(1.0)
        assert abs(sp_tv - sp_const) < 1e-10
