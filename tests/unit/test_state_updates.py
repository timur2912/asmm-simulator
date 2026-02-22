"""A5: State update tests."""

from __future__ import annotations

import pytest

from mm_sim.state import SimState


class TestStateUpdates:
    def test_ask_fill_decreases_inventory(self):
        state = SimState(S=100.0, q=5, X=0.0, t=0.0)
        ask_price = 101.0
        state.apply_ask_fill(ask_price)
        assert state.q == 4
        assert state.X == 101.0

    def test_bid_fill_increases_inventory(self):
        state = SimState(S=100.0, q=5, X=0.0, t=0.0)
        bid_price = 99.0
        state.apply_bid_fill(bid_price)
        assert state.q == 6
        assert state.X == -99.0

    def test_ask_fill_cash_sign(self):
        state = SimState(S=100.0, q=0, X=0.0, t=0.0)
        state.apply_ask_fill(105.0)
        assert state.X == 105.0  # Selling: receive cash
        assert state.q == -1

    def test_bid_fill_cash_sign(self):
        state = SimState(S=100.0, q=0, X=0.0, t=0.0)
        state.apply_bid_fill(95.0)
        assert state.X == -95.0  # Buying: pay cash
        assert state.q == 1

    def test_inventory_equals_buys_minus_sells(self):
        state = SimState(S=100.0, q=0, X=0.0, t=0.0)
        buys, sells = 0, 0
        # Simulate some fills
        state.apply_bid_fill(99.0)
        buys += 1
        state.apply_bid_fill(98.0)
        buys += 1
        state.apply_ask_fill(101.0)
        sells += 1
        assert state.q == buys - sells

    def test_multiple_fills_accumulate(self):
        state = SimState(S=100.0, q=0, X=0.0, t=0.0)
        state.apply_bid_fill(99.0)
        state.apply_bid_fill(98.0)
        state.apply_ask_fill(101.0)
        assert state.q == 1
        assert state.X == -99.0 - 98.0 + 101.0

    def test_terminal_pnl(self):
        state = SimState(S=100.0, q=3, X=50.0, t=1.0)
        pnl = state.terminal_pnl()
        assert pnl == 50.0 + 3 * 100.0

    def test_terminal_pnl_negative_inventory(self):
        state = SimState(S=100.0, q=-2, X=250.0, t=1.0)
        pnl = state.terminal_pnl()
        assert pnl == 250.0 + (-2) * 100.0

    def test_update_price(self):
        state = SimState(S=100.0, q=0, X=0.0, t=0.0)
        state.update_price(101.5)
        assert state.S == 101.5

    def test_advance_time(self):
        state = SimState(S=100.0, q=0, X=0.0, t=0.0)
        state.advance_time(0.005)
        assert abs(state.t - 0.005) < 1e-15
