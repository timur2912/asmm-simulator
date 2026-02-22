"""Single-path and multi-path simulation loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

import numpy as np

from mm_sim.config import SimulationConfig
from mm_sim.execution import simulate_fills_bernoulli, simulate_fills_poisson
from mm_sim.intensities import execution_probability
from mm_sim.metrics import SummaryStats, compute_summary
from mm_sim.price_process import generate_increments
from mm_sim.rng import RNGStreams, make_streams
from mm_sim.state import SimState
from mm_sim.strategies.base import Strategy
from mm_sim.strategies.inventory import InventoryStrategy


@dataclass
class PathResult:
    """Result of a single simulation path."""

    pnl: float
    q_T: int
    terminal_S: float
    terminal_X: float
    n_buys: int
    n_sells: int
    state: SimState | None = None  # optionally keep full trajectory


@dataclass
class MCResult:
    """Result of a Monte Carlo experiment for one strategy."""

    strategy_name: str
    pnl_array: np.ndarray
    q_array: np.ndarray
    spread_used: float
    config: SimulationConfig
    paths: list[PathResult] = field(default_factory=list)

    @property
    def summary(self) -> SummaryStats:
        return compute_summary(
            self.pnl_array,
            self.q_array,
            spread=self.spread_used,
            strategy_name=self.strategy_name,
        )


def simulate_path(
    strategy: Strategy,
    config: SimulationConfig,
    rng: Union[np.random.Generator, RNGStreams, None] = None,
    record_trajectory: bool = False,
) -> PathResult:
    """Run a single simulation path.

    *rng* can be a ``numpy.random.Generator`` (used for both price and
    execution draws), a dict of streams (``{"price": ..., "execution": ...}``),
    or *None* (creates a default RNG).
    """
    # Normalise rng argument
    if rng is None:
        streams = make_streams(config.seed)
    elif isinstance(rng, np.random.Generator):
        streams = {"price": rng, "execution": rng}
    else:
        streams = rng

    state = SimState(S=config.S0, q=config.q0, X=config.X0, t=0.0)

    increments = generate_increments(
        config.n_steps,
        config.sigma,
        config.dt,
        config.price_model.value,
        streams["price"],
    )

    clip_delta = config.constraints.clip_delta
    clip_prob = config.constraints.clip_prob
    prob_cap = config.constraints.prob_cap
    use_poisson = config.execution_model.value == "poisson"
    exec_rng = streams["execution"]

    for i in range(config.n_steps):
        # 1. Compute quotes → (p_bid, p_ask)
        p_bid, p_ask = strategy.quotes(state, config)
        delta_ask = p_ask - state.S
        delta_bid = state.S - p_bid

        if record_trajectory:
            r = (
                strategy.reservation_price(state, config)
                if isinstance(strategy, InventoryStrategy)
                else state.S
            )
            state.record(p_bid=p_bid, p_ask=p_ask, r=r)

        # 2. Compute execution probabilities / intensities
        d_a = max(0.0, delta_ask) if clip_delta else delta_ask
        d_b = max(0.0, delta_bid) if clip_delta else delta_bid

        if use_poisson:
            from mm_sim.intensities import execution_intensity

            lam_a = float(execution_intensity(d_a, config.A, config.k)) * config.dt
            lam_b = float(execution_intensity(d_b, config.A, config.k)) * config.dt
            n_ask, n_bid = simulate_fills_poisson(lam_a, lam_b, exec_rng)
        else:
            p_a = execution_probability(
                d_a, config.A, config.k, config.dt,
                clip=clip_prob, prob_cap=prob_cap,
            )
            p_b = execution_probability(
                d_b, config.A, config.k, config.dt,
                clip=clip_prob, prob_cap=prob_cap,
            )
            ask_filled, bid_filled = simulate_fills_bernoulli(p_a, p_b, exec_rng)
            n_ask = int(ask_filled)
            n_bid = int(bid_filled)

        # 3. Update state
        for _ in range(n_ask):
            state.apply_ask_fill(p_ask)
        for _ in range(n_bid):
            state.apply_bid_fill(p_bid)

        # 4. Advance price and time
        state.update_price(state.S + increments[i])
        state.advance_time(config.dt)

    # Record final state
    if record_trajectory:
        state.record()

    assert state.verify_inventory_invariant(), "Inventory invariant violated!"

    return PathResult(
        pnl=state.terminal_pnl(),
        q_T=state.q,
        terminal_S=state.S,
        terminal_X=state.X,
        n_buys=state.n_buys,
        n_sells=state.n_sells,
        state=state if record_trajectory else None,
    )


def simulate_mc(
    strategy: Strategy,
    config: SimulationConfig,
) -> MCResult:
    """Run a full Monte Carlo experiment for one strategy.

    Each path gets its own RNG derived from ``config.seed + path_index``.
    """
    spread = strategy.spread(config)
    pnl_arr = np.empty(config.n_paths)
    q_arr = np.empty(config.n_paths, dtype=int)
    paths: list[PathResult] = []

    for i in range(config.n_paths):
        streams = make_streams(config.seed + i)
        result = simulate_path(strategy, config, streams)
        pnl_arr[i] = result.pnl
        q_arr[i] = result.q_T

    return MCResult(
        strategy_name=strategy.name,
        pnl_array=pnl_arr,
        q_array=q_arr,
        spread_used=spread,
        config=config,
    )
