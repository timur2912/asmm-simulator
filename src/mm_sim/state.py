"""Simulation state: S, q, X, t and update logic."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SimState:
    """Mutable simulation state for a single path."""

    S: float  # mid-price
    q: int  # inventory (shares)
    X: float  # cash
    t: float  # current time
    n_buys: int = 0  # cumulative buys
    n_sells: int = 0  # cumulative sells

    # Trajectory recording
    S_history: list[float] = field(default_factory=list)
    q_history: list[int] = field(default_factory=list)
    X_history: list[float] = field(default_factory=list)
    t_history: list[float] = field(default_factory=list)
    bid_history: list[float] = field(default_factory=list)
    ask_history: list[float] = field(default_factory=list)
    r_history: list[float] = field(default_factory=list)

    def record(
        self,
        p_bid: float = np.nan,
        p_ask: float = np.nan,
        r: float = np.nan,
    ) -> None:
        self.S_history.append(self.S)
        self.q_history.append(self.q)
        self.X_history.append(self.X)
        self.t_history.append(self.t)
        self.bid_history.append(p_bid)
        self.ask_history.append(p_ask)
        self.r_history.append(r)

    # ------------------------------------------------------------------
    # Fill handling – each execution is 1 share
    # ------------------------------------------------------------------

    def apply_ask_fill(self, ask_price: float) -> None:
        """Ask filled: agent sells 1 share at *ask_price*."""
        self.q -= 1
        self.X += ask_price
        self.n_sells += 1

    def apply_bid_fill(self, bid_price: float) -> None:
        """Bid filled: agent buys 1 share at *bid_price*."""
        self.q += 1
        self.X -= bid_price
        self.n_buys += 1

    # ------------------------------------------------------------------
    # Price / time advancement
    # ------------------------------------------------------------------

    def update_price(self, new_S: float) -> None:
        self.S = new_S

    def advance_time(self, dt: float) -> None:
        self.t += dt

    # ------------------------------------------------------------------
    # Terminal P&L
    # ------------------------------------------------------------------

    def terminal_pnl(self) -> float:
        """Π_T = X_T + q_T · S_T"""
        return self.X + self.q * self.S

    def verify_inventory_invariant(self) -> bool:
        return self.q == self.n_buys - self.n_sells
