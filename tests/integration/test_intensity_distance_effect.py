"""B3: Intensity-distance effect tests (H3)."""

from __future__ import annotations

import numpy as np
import pytest

from mm_sim.execution import simulate_fills_bernoulli
from mm_sim.intensities import execution_probability


class TestIntensityDistanceEffect:
    """H3: Execution rate decays with distance from mid-price."""

    def test_fill_rate_decreases_with_distance(self):
        """Empirically verify that increasing δ reduces fill rate."""
        rng = np.random.default_rng(42)
        A, k, dt = 140.0, 1.5, 0.005
        deltas = [0.2, 0.5, 1.0, 1.5, 2.0]
        n_trials = 20_000

        fill_rates = []
        for delta in deltas:
            p = execution_probability(delta, A, k, dt, clip=True)
            fills = sum(
                simulate_fills_bernoulli(p, p, rng)[0] for _ in range(n_trials)
            )
            fill_rates.append(fills / n_trials)

        # Fill rates should be monotonically decreasing
        for i in range(len(fill_rates) - 1):
            assert fill_rates[i] > fill_rates[i + 1], (
                f"Fill rate at δ={deltas[i]} ({fill_rates[i]:.4f}) "
                f"not > fill rate at δ={deltas[i+1]} ({fill_rates[i+1]:.4f})"
            )

    def test_probability_monotonically_decreasing(self):
        """Analytical check: probability decreases with δ."""
        A, k, dt = 140.0, 1.5, 0.005
        deltas = np.linspace(0.0, 5.0, 100)
        probs = [execution_probability(d, A, k, dt, clip=True) for d in deltas]
        for i in range(len(probs) - 1):
            assert probs[i] >= probs[i + 1]
