"""E1: Runtime budget tests."""

from __future__ import annotations

import time

import pytest

from mm_sim.config import SimulationConfig
from mm_sim.simulator import simulate_mc
from mm_sim.strategies import InventoryStrategy


class TestRuntimeBudget:
    def test_300_paths_under_30_seconds(self):
        """300 paths should complete well within 30 seconds."""
        cfg = SimulationConfig(
            S0=100.0, T=1.0, dt=0.005, sigma=2.0,
            A=140.0, k=1.5, gamma=0.1, n_paths=300, seed=42,
        )
        start = time.perf_counter()
        simulate_mc(InventoryStrategy(), cfg)
        elapsed = time.perf_counter() - start
        assert elapsed < 30.0, f"300 paths took {elapsed:.1f}s (budget: 30s)"

    def test_1000_paths_under_120_seconds(self):
        """Full 1000-path run should complete within 120 seconds."""
        cfg = SimulationConfig(
            S0=100.0, T=1.0, dt=0.005, sigma=2.0,
            A=140.0, k=1.5, gamma=0.1, n_paths=1000, seed=42,
        )
        start = time.perf_counter()
        simulate_mc(InventoryStrategy(), cfg)
        elapsed = time.perf_counter() - start
        assert elapsed < 120.0, f"1000 paths took {elapsed:.1f}s (budget: 120s)"
