"""A2: Price process tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mm_sim.price_process import generate_increments, generate_price_path


class TestRademacherIncrements:
    def test_values_are_plus_minus_sigma_sqrt_dt(self):
        sigma, dt = 2.0, 0.005
        rng = np.random.default_rng(0)
        inc = generate_increments(200, sigma, dt, "rademacher", rng)
        expected_abs = sigma * math.sqrt(dt)
        assert np.allclose(np.abs(inc), expected_abs)

    def test_mean_approximately_zero(self):
        sigma, dt = 2.0, 0.005
        rng = np.random.default_rng(0)
        inc = generate_increments(100_000, sigma, dt, "rademacher", rng)
        assert abs(np.mean(inc)) < 0.01

    def test_variance_approximately_sigma2_dt(self):
        sigma, dt = 2.0, 0.005
        rng = np.random.default_rng(0)
        inc = generate_increments(100_000, sigma, dt, "rademacher", rng)
        expected_var = sigma**2 * dt
        assert abs(np.var(inc) - expected_var) < 0.001

    def test_deterministic_with_seed(self):
        sigma, dt = 2.0, 0.005
        inc1 = generate_increments(200, sigma, dt, "rademacher", np.random.default_rng(42))
        inc2 = generate_increments(200, sigma, dt, "rademacher", np.random.default_rng(42))
        np.testing.assert_array_equal(inc1, inc2)


class TestGaussianIncrements:
    def test_mean_approximately_zero(self):
        sigma, dt = 2.0, 0.005
        rng = np.random.default_rng(0)
        inc = generate_increments(100_000, sigma, dt, "gaussian", rng)
        assert abs(np.mean(inc)) < 0.02

    def test_variance_approximately_sigma2_dt(self):
        sigma, dt = 2.0, 0.005
        rng = np.random.default_rng(0)
        inc = generate_increments(100_000, sigma, dt, "gaussian", rng)
        expected_var = sigma**2 * dt
        assert abs(np.var(inc) - expected_var) < 0.001


class TestPricePath:
    def test_path_length(self):
        rng = np.random.default_rng(0)
        path = generate_price_path(100.0, 200, 2.0, 0.005, "rademacher", rng)
        assert len(path) == 201  # includes S0

    def test_starts_at_S0(self):
        rng = np.random.default_rng(0)
        path = generate_price_path(100.0, 200, 2.0, 0.005, "rademacher", rng)
        assert path[0] == 100.0

    def test_rademacher_path_increments(self):
        rng = np.random.default_rng(0)
        path = generate_price_path(100.0, 200, 2.0, 0.005, "rademacher", rng)
        diffs = np.diff(path)
        expected_abs = 2.0 * math.sqrt(0.005)
        assert np.allclose(np.abs(diffs), expected_abs)
