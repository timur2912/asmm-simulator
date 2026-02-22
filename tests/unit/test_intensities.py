"""A3: Intensity function tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mm_sim.intensities import execution_intensity, execution_probability


class TestExecutionIntensity:
    def test_at_zero_distance(self):
        assert execution_intensity(0.0, 140.0, 1.5) == 140.0

    def test_decreases_with_distance(self):
        lam1 = execution_intensity(0.5, 140.0, 1.5)
        lam2 = execution_intensity(1.0, 140.0, 1.5)
        lam3 = execution_intensity(2.0, 140.0, 1.5)
        assert lam1 > lam2 > lam3

    def test_large_distance_approaches_zero(self):
        lam = execution_intensity(100.0, 140.0, 1.5)
        assert lam < 1e-50

    def test_no_nan_or_inf(self):
        for delta in [0.0, 0.1, 1.0, 10.0, 100.0]:
            lam = execution_intensity(delta, 140.0, 1.5)
            assert math.isfinite(lam)

    def test_negative_distance_gives_higher_intensity(self):
        lam_neg = execution_intensity(-0.5, 140.0, 1.5)
        lam_zero = execution_intensity(0.0, 140.0, 1.5)
        assert lam_neg > lam_zero

    def test_formula_correctness(self):
        delta, A, k = 0.5, 140.0, 1.5
        expected = A * math.exp(-k * delta)
        assert abs(execution_intensity(delta, A, k) - expected) < 1e-10

    def test_vectorized(self):
        deltas = np.array([0.0, 0.5, 1.0, 2.0])
        lams = execution_intensity(deltas, 140.0, 1.5)
        assert len(lams) == 4
        assert np.all(np.diff(lams) < 0)


class TestExecutionProbability:
    def test_basic_probability(self):
        p = execution_probability(0.5, 140.0, 1.5, 0.005, clip=True)
        assert 0.0 <= p <= 1.0

    def test_clipping_at_high_intensity(self):
        # delta < 0 can give λ > A, so λ*dt > 1
        p = execution_probability(-2.0, 140.0, 1.5, 0.005, clip=True)
        assert p <= 1.0

    def test_no_clip_can_exceed_one(self):
        p = execution_probability(-2.0, 140.0, 1.5, 0.005, clip=False)
        assert p > 1.0  # λ*dt > 1 for large negative delta

    def test_custom_prob_cap(self):
        p = execution_probability(-2.0, 140.0, 1.5, 0.005, clip=True, prob_cap=0.9)
        assert p <= 0.9
