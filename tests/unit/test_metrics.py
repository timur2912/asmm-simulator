"""A7: Metrics tests."""

from __future__ import annotations

import numpy as np
import pytest

from mm_sim.metrics import compute_summary, terminal_pnl


class TestTerminalPnL:
    def test_basic(self):
        assert terminal_pnl(50.0, 3, 100.0) == 350.0

    def test_zero_inventory(self):
        assert terminal_pnl(100.0, 0, 100.0) == 100.0

    def test_negative_inventory(self):
        assert terminal_pnl(250.0, -2, 100.0) == 50.0


class TestComputeSummary:
    def test_mean_and_std(self):
        pnl = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        q = np.array([1, -1, 2, -2, 0])
        summary = compute_summary(pnl, q, spread=1.0)
        assert abs(summary.mean_profit - np.mean(pnl)) < 1e-10
        assert abs(summary.std_profit - np.std(pnl, ddof=1)) < 1e-10
        assert abs(summary.mean_q - np.mean(q)) < 1e-10
        assert abs(summary.std_q - np.std(q, ddof=1)) < 1e-10

    def test_n_paths(self):
        pnl = np.array([10.0, 20.0, 30.0])
        q = np.array([1, -1, 0])
        summary = compute_summary(pnl, q, spread=1.0)
        assert summary.n_paths == 3

    def test_spread_stored(self):
        pnl = np.array([10.0, 20.0])
        q = np.array([1, -1])
        summary = compute_summary(pnl, q, spread=1.29)
        assert abs(summary.spread - 1.29) < 1e-10

    def test_single_path(self):
        pnl = np.array([42.0])
        q = np.array([3])
        summary = compute_summary(pnl, q, spread=1.0)
        assert summary.mean_profit == 42.0
        assert summary.std_profit == 0.0  # single value
