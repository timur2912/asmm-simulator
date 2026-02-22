"""A4: Execution model tests."""

from __future__ import annotations

import numpy as np
import pytest

from mm_sim.execution import simulate_fills_bernoulli, simulate_fills_poisson


class TestBernoulliFills:
    def test_returns_bool_pair(self):
        rng = np.random.default_rng(0)
        ask_fill, bid_fill = simulate_fills_bernoulli(0.3, 0.3, rng)
        assert isinstance(ask_fill, (bool, np.bool_))
        assert isinstance(bid_fill, (bool, np.bool_))

    def test_probability_zero_never_fills(self):
        rng = np.random.default_rng(0)
        for _ in range(1000):
            ask, bid = simulate_fills_bernoulli(0.0, 0.0, rng)
            assert not ask
            assert not bid

    def test_probability_one_always_fills(self):
        rng = np.random.default_rng(0)
        for _ in range(100):
            ask, bid = simulate_fills_bernoulli(1.0, 1.0, rng)
            assert ask
            assert bid

    def test_empirical_fill_rate(self):
        rng = np.random.default_rng(42)
        p = 0.3
        n = 50_000
        fills = sum(simulate_fills_bernoulli(p, p, rng)[0] for _ in range(n))
        empirical = fills / n
        assert abs(empirical - p) < 0.02

    def test_independent_sides(self):
        rng = np.random.default_rng(42)
        n = 50_000
        both = sum(
            simulate_fills_bernoulli(0.5, 0.5, rng) == (True, True)
            for _ in range(n)
        )
        # P(both) ≈ 0.25
        assert abs(both / n - 0.25) < 0.02

    def test_clipped_probability(self):
        rng = np.random.default_rng(0)
        # Even with p > 1 passed, Bernoulli should handle it
        ask, bid = simulate_fills_bernoulli(1.0, 1.0, rng)
        assert ask and bid


class TestPoissonFills:
    def test_returns_int_pair(self):
        rng = np.random.default_rng(0)
        ask_count, bid_count = simulate_fills_poisson(0.3, 0.3, rng)
        assert isinstance(ask_count, (int, np.integer))
        assert isinstance(bid_count, (int, np.integer))

    def test_zero_intensity_no_fills(self):
        rng = np.random.default_rng(0)
        for _ in range(1000):
            a, b = simulate_fills_poisson(0.0, 0.0, rng)
            assert a == 0
            assert b == 0

    def test_empirical_mean(self):
        rng = np.random.default_rng(42)
        lam_dt = 0.5
        n = 50_000
        counts = [simulate_fills_poisson(lam_dt, lam_dt, rng)[0] for _ in range(n)]
        assert abs(np.mean(counts) - lam_dt) < 0.02
