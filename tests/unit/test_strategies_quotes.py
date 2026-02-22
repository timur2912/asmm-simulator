"""A6: Strategy quoting tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mm_sim.config import SimulationConfig
from mm_sim.state import SimState
from mm_sim.strategies import InventoryStrategy, SymmetricStrategy


class TestSymmetricStrategy:
    def test_quotes_symmetric_around_mid(self):
        cfg = SimulationConfig(gamma=0.1, k=1.5)
        strat = SymmetricStrategy()
        state = SimState(S=100.0, q=5, X=0.0, t=0.5)
        bid, ask = strat.quotes(state, cfg)
        sp = cfg.constant_spread()
        assert abs(ask - (100.0 + sp / 2)) < 1e-10
        assert abs(bid - (100.0 - sp / 2)) < 1e-10

    def test_delta_equals_half_spread(self):
        cfg = SimulationConfig(gamma=0.1, k=1.5)
        strat = SymmetricStrategy()
        state = SimState(S=100.0, q=0, X=0.0, t=0.0)
        bid, ask = strat.quotes(state, cfg)
        sp = cfg.constant_spread()
        delta_a = ask - state.S
        delta_b = state.S - bid
        assert abs(delta_a - sp / 2) < 1e-10
        assert abs(delta_b - sp / 2) < 1e-10

    def test_independent_of_inventory(self):
        cfg = SimulationConfig(gamma=0.1, k=1.5)
        strat = SymmetricStrategy()
        bid1, ask1 = strat.quotes(SimState(S=100.0, q=0, X=0.0, t=0.5), cfg)
        bid2, ask2 = strat.quotes(SimState(S=100.0, q=10, X=0.0, t=0.5), cfg)
        assert abs(bid1 - bid2) < 1e-10
        assert abs(ask1 - ask2) < 1e-10

    def test_time_varying_spread(self):
        cfg = SimulationConfig(gamma=0.1, k=1.5, spread_model="time_varying")
        strat = SymmetricStrategy()
        bid0, ask0 = strat.quotes(SimState(S=100.0, q=0, X=0.0, t=0.0), cfg)
        bid1, ask1 = strat.quotes(SimState(S=100.0, q=0, X=0.0, t=0.5), cfg)
        # Spread at t=0 > spread at t=0.5
        assert (ask0 - bid0) > (ask1 - bid1)


class TestInventoryStrategy:
    def test_reservation_price_at_zero_inventory(self):
        cfg = SimulationConfig(gamma=0.1, k=1.5, sigma=2.0, T=1.0)
        strat = InventoryStrategy()
        state = SimState(S=100.0, q=0, X=0.0, t=0.0)
        r = strat.reservation_price(state, cfg)
        assert abs(r - 100.0) < 1e-10

    def test_reservation_price_positive_inventory(self):
        cfg = SimulationConfig(gamma=0.1, k=1.5, sigma=2.0, T=1.0)
        strat = InventoryStrategy()
        state = SimState(S=100.0, q=5, X=0.0, t=0.0)
        r = strat.reservation_price(state, cfg)
        # r = S - q*γ*σ²*(T-t) = 100 - 5*0.1*4*1 = 100 - 2 = 98
        assert abs(r - 98.0) < 1e-10

    def test_reservation_price_negative_inventory(self):
        cfg = SimulationConfig(gamma=0.1, k=1.5, sigma=2.0, T=1.0)
        strat = InventoryStrategy()
        state = SimState(S=100.0, q=-3, X=0.0, t=0.0)
        r = strat.reservation_price(state, cfg)
        # r = 100 - (-3)*0.1*4*1 = 100 + 1.2 = 101.2
        assert abs(r - 101.2) < 1e-10

    def test_quotes_centered_around_reservation(self):
        cfg = SimulationConfig(gamma=0.1, k=1.5, sigma=2.0, T=1.0)
        strat = InventoryStrategy()
        state = SimState(S=100.0, q=5, X=0.0, t=0.0)
        bid, ask = strat.quotes(state, cfg)
        r = strat.reservation_price(state, cfg)
        sp = cfg.constant_spread()
        assert abs(ask - (r + sp / 2)) < 1e-10
        assert abs(bid - (r - sp / 2)) < 1e-10

    def test_equals_symmetric_when_q_zero(self):
        cfg = SimulationConfig(gamma=0.1, k=1.5, sigma=2.0, T=1.0)
        inv = InventoryStrategy()
        sym = SymmetricStrategy()
        state = SimState(S=100.0, q=0, X=0.0, t=0.0)
        bid_inv, ask_inv = inv.quotes(state, cfg)
        bid_sym, ask_sym = sym.quotes(state, cfg)
        # When q=0, r=S, so quotes should be identical
        assert abs(bid_inv - bid_sym) < 1e-10
        assert abs(ask_inv - ask_sym) < 1e-10

    def test_positive_inventory_shifts_quotes_down(self):
        cfg = SimulationConfig(gamma=0.1, k=1.5, sigma=2.0, T=1.0)
        strat = InventoryStrategy()
        state0 = SimState(S=100.0, q=0, X=0.0, t=0.0)
        state5 = SimState(S=100.0, q=5, X=0.0, t=0.0)
        bid0, ask0 = strat.quotes(state0, cfg)
        bid5, ask5 = strat.quotes(state5, cfg)
        # Positive inventory → lower reservation → lower quotes
        assert ask5 < ask0
        assert bid5 < bid0

    def test_negative_inventory_shifts_quotes_up(self):
        cfg = SimulationConfig(gamma=0.1, k=1.5, sigma=2.0, T=1.0)
        strat = InventoryStrategy()
        state0 = SimState(S=100.0, q=0, X=0.0, t=0.0)
        state_neg = SimState(S=100.0, q=-5, X=0.0, t=0.0)
        bid0, ask0 = strat.quotes(state0, cfg)
        bid_neg, ask_neg = strat.quotes(state_neg, cfg)
        assert ask_neg > ask0
        assert bid_neg > bid0

    def test_reservation_at_terminal_time(self):
        cfg = SimulationConfig(gamma=0.1, k=1.5, sigma=2.0, T=1.0)
        strat = InventoryStrategy()
        state = SimState(S=100.0, q=5, X=0.0, t=1.0)
        r = strat.reservation_price(state, cfg)
        # At T, r = S (T-t=0)
        assert abs(r - 100.0) < 1e-10
