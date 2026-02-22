"""A8: RNG stream and CRN tests."""

from __future__ import annotations

import numpy as np
import pytest

from mm_sim.rng import make_streams


class TestRNGStreams:
    def test_deterministic_with_same_seed(self):
        s1 = make_streams(42)
        s2 = make_streams(42)
        v1 = s1["price"].random(10)
        v2 = s2["price"].random(10)
        np.testing.assert_array_equal(v1, v2)

    def test_different_seeds_differ(self):
        s1 = make_streams(42)
        s2 = make_streams(99)
        v1 = s1["price"].random(10)
        v2 = s2["price"].random(10)
        assert not np.array_equal(v1, v2)

    def test_streams_are_independent(self):
        s = make_streams(42)
        v_price = s["price"].random(1000)
        v_exec = s["execution"].random(1000)
        # Correlation should be near zero
        corr = np.corrcoef(v_price, v_exec)[0, 1]
        assert abs(corr) < 0.1

    def test_has_expected_keys(self):
        s = make_streams(42)
        assert "price" in s
        assert "execution" in s
