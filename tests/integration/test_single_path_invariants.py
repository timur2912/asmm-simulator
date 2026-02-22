"""B1: Single path invariant tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mm_sim.config import SimulationConfig
from mm_sim.simulator import simulate_path
from mm_sim.strategies import InventoryStrategy, SymmetricStrategy


@pytest.fixture
def short_config():
    return SimulationConfig(
        S0=100.0, T=1.0, dt=0.005, sigma=2.0,
        A=140.0, k=1.5, gamma=0.1, n_paths=1, seed=42,
    )


class TestSinglePathInvariants:
    def test_no_nan_or_inf_inventory(self, short_config):
        result = simulate_path(
            InventoryStrategy(), short_config, np.random.default_rng(42)
        )
        assert math.isfinite(result.pnl)
        assert math.isfinite(result.q_T)

    def test_no_nan_or_inf_symmetric(self, short_config):
        result = simulate_path(
            SymmetricStrategy(), short_config, np.random.default_rng(42)
        )
        assert math.isfinite(result.pnl)
        assert math.isfinite(result.q_T)

    def test_correct_number_of_steps(self, short_config):
        result = simulate_path(
            InventoryStrategy(), short_config, np.random.default_rng(42),
            record_trajectory=True,
        )
        assert result.state is not None
        assert len(result.state.S_history) == short_config.n_steps + 1

    def test_inventory_is_integer(self, short_config):
        result = simulate_path(
            InventoryStrategy(), short_config, np.random.default_rng(42),
            record_trajectory=True,
        )
        assert result.state is not None
        for q in result.state.q_history:
            assert q == int(q)

    def test_price_evolves_correctly_rademacher(self, short_config):
        result = simulate_path(
            InventoryStrategy(), short_config, np.random.default_rng(42),
            record_trajectory=True,
        )
        assert result.state is not None
        diffs = np.diff(result.state.S_history)
        expected_abs = short_config.sigma * math.sqrt(short_config.dt)
        assert np.allclose(np.abs(diffs), expected_abs, atol=1e-10)

    def test_terminal_pnl_matches_state(self, short_config):
        result = simulate_path(
            InventoryStrategy(), short_config, np.random.default_rng(42),
            record_trajectory=True,
        )
        assert result.state is not None
        expected = result.state.X + result.state.q * result.state.S
        assert abs(result.pnl - expected) < 1e-10

    def test_inventory_equals_buys_minus_sells(self, short_config):
        result = simulate_path(
            InventoryStrategy(), short_config, np.random.default_rng(42),
            record_trajectory=True,
        )
        assert result.state is not None
        assert result.state.q == result.state.n_buys - result.state.n_sells

    def test_starts_at_initial_values(self, short_config):
        result = simulate_path(
            InventoryStrategy(), short_config, np.random.default_rng(42),
            record_trajectory=True,
        )
        assert result.state is not None
        assert result.state.S_history[0] == short_config.S0
        assert result.state.q_history[0] == short_config.q0
