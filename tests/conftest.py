"""Shared fixtures for the test suite."""

from __future__ import annotations

import numpy as np
import pytest

from mm_sim.config import SimulationConfig
from mm_sim.rng import make_streams
from mm_sim.state import SimState
from mm_sim.strategies import InventoryStrategy, SymmetricStrategy


@pytest.fixture
def paper_config() -> SimulationConfig:
    """Paper Section 3.3 default config (γ=0.1)."""
    return SimulationConfig(
        S0=100.0,
        T=1.0,
        dt=0.005,
        sigma=2.0,
        A=140.0,
        k=1.5,
        q0=0,
        X0=0.0,
        gamma=0.1,
        n_paths=1000,
        seed=42,
        use_crn=True,
    )


@pytest.fixture
def small_config() -> SimulationConfig:
    """Small config for fast tests."""
    return SimulationConfig(
        S0=100.0,
        T=1.0,
        dt=0.005,
        sigma=2.0,
        A=140.0,
        k=1.5,
        q0=0,
        X0=0.0,
        gamma=0.1,
        n_paths=100,
        seed=42,
        use_crn=True,
    )


@pytest.fixture
def rng_streams():
    return make_streams(42)


@pytest.fixture
def initial_state() -> SimState:
    return SimState(S=100.0, q=0, X=0.0, t=0.0)


@pytest.fixture
def inventory_strategy() -> InventoryStrategy:
    return InventoryStrategy()


@pytest.fixture
def symmetric_strategy() -> SymmetricStrategy:
    return SymmetricStrategy()


@pytest.fixture
def rng_gen():
    return np.random.default_rng(42)
