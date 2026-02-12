from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .models import ModelParams, spread_constant, spread_time_varying


SpreadModel = Literal["constant", "time-varying"]
StrategyName = Literal["inventory", "symmetric"]
PriceModel = Literal["rademacher", "gaussian"]
ExecutionModel = Literal["bernoulli", "poisson"]


@dataclass(frozen=True)
class SimConfig:
    gamma: float
    n_paths: int = 1000
    seed: int = 0
    spread_model: SpreadModel = "constant"
    price_model: PriceModel = "rademacher"
    execution_model: ExecutionModel = "bernoulli"

    # When True, quotes may cross the mid-price (delta may become negative).
    allow_cross: bool = True

    # When True, the intensity uses delta := max(delta, 0) (treating delta as a distance).
    floor_intensity_delta: bool = True

    # When using Bernoulli fills, cap p := min(1, lambda*dt).
    cap_prob: bool = True


@dataclass
class SimResult:
    profit: np.ndarray  # (n_paths,)
    qT: np.ndarray  # (n_paths,)

    def summary(self) -> dict[str, float]:
        return {
            "profit_mean": float(np.mean(self.profit)),
            "profit_std": float(np.std(self.profit, ddof=1)),
            "qT_mean": float(np.mean(self.qT)),
            "qT_std": float(np.std(self.qT, ddof=1)),
        }


def _price_increments(*, rng: np.random.Generator, n_paths: int, n_steps: int, sigma: float, dt: float, price_model: PriceModel) -> np.ndarray:
    if price_model == "rademacher":
        eps = rng.integers(0, 2, size=(n_paths, n_steps), dtype=np.int8)
        eps = 2 * eps - 1  # {-1, +1}
        return eps.astype(float) * sigma * math.sqrt(dt)
    if price_model == "gaussian":
        return rng.normal(loc=0.0, scale=sigma * math.sqrt(dt), size=(n_paths, n_steps))
    raise ValueError(f"Unknown price_model={price_model}")


def _spreads(*, gamma: float, params: ModelParams, n_steps: int, dt: float, spread_model: SpreadModel) -> np.ndarray:
    if spread_model == "constant":
        Sp = spread_constant(gamma=gamma, k=params.k)
        return np.full(n_steps, Sp, dtype=float)

    if spread_model == "time-varying":
        t_grid = np.arange(n_steps) * dt
        return np.array([spread_time_varying(t=float(t), params=params, gamma=gamma) for t in t_grid], dtype=float)

    raise ValueError(f"Unknown spread_model={spread_model}")


def simulate(
    *,
    params: ModelParams,
    config: SimConfig,
    strategy: StrategyName,
    price_incs: np.ndarray | None = None,
    u_bid: np.ndarray | None = None,
    u_ask: np.ndarray | None = None,
) -> SimResult:
    """Simulate terminal profit and inventory for one strategy.

    If `price_incs`/`u_*` are provided, they are used (common random numbers).
    Shapes:
      price_incs: (n_paths, n_steps)
      u_bid/u_ask: (n_paths, n_steps)
    """

    n_steps = int(round(params.T / params.dt))
    if not np.isclose(n_steps * params.dt, params.T):
        raise ValueError("T must be an integer multiple of dt")

    rng = np.random.default_rng(config.seed)

    if price_incs is None:
        price_incs = _price_increments(
            rng=rng,
            n_paths=config.n_paths,
            n_steps=n_steps,
            sigma=params.sigma,
            dt=params.dt,
            price_model=config.price_model,
        )

    if u_bid is None:
        u_bid = rng.random(size=(config.n_paths, n_steps))
    if u_ask is None:
        u_ask = rng.random(size=(config.n_paths, n_steps))

    spreads = _spreads(gamma=config.gamma, params=params, n_steps=n_steps, dt=params.dt, spread_model=config.spread_model)

    S = np.full(config.n_paths, params.s0, dtype=float)
    q = np.full(config.n_paths, params.q0, dtype=int)
    X = np.zeros(config.n_paths, dtype=float)

    for i in range(n_steps):
        t = i * params.dt
        Sp = float(spreads[i])
        half = Sp / 2.0

        if strategy == "symmetric":
            p_a = S + half
            p_b = S - half
        elif strategy == "inventory":
            # r(s,q,t) = s - q * gamma * sigma^2 * (T - t)
            r = S - q * config.gamma * (params.sigma**2) * (params.T - t)
            p_a = r + half
            p_b = r - half
        else:
            raise ValueError(f"Unknown strategy={strategy}")

        if not config.allow_cross:
            # Enforce non-negative *distances* to the mid (quotes do not cross).
            p_a = np.maximum(p_a, S)
            p_b = np.minimum(p_b, S)

        delta_a = p_a - S
        delta_b = S - p_b
        if not config.allow_cross:
            if np.any(delta_a < -1e-12) or np.any(delta_b < -1e-12):
                raise RuntimeError("Negative deltas encountered despite allow_cross=False")

        delta_a_int = np.maximum(delta_a, 0.0) if config.floor_intensity_delta else delta_a
        delta_b_int = np.maximum(delta_b, 0.0) if config.floor_intensity_delta else delta_b

        lam_a = params.A * np.exp(-params.k * delta_a_int)
        lam_b = params.A * np.exp(-params.k * delta_b_int)

        if config.execution_model == "bernoulli":
            p_fill_a = lam_a * params.dt
            p_fill_b = lam_b * params.dt
            if config.cap_prob:
                p_fill_a = np.minimum(1.0, p_fill_a)
                p_fill_b = np.minimum(1.0, p_fill_b)

            fill_a = u_ask[:, i] < p_fill_a
            fill_b = u_bid[:, i] < p_fill_b

            # Ask fill: sell 1 at p_a
            if np.any(fill_a):
                q[fill_a] -= 1
                X[fill_a] += p_a[fill_a]

            # Bid fill: buy 1 at p_b
            if np.any(fill_b):
                q[fill_b] += 1
                X[fill_b] -= p_b[fill_b]

        elif config.execution_model == "poisson":
            n_a = rng.poisson(lam=lam_a * params.dt)
            n_b = rng.poisson(lam=lam_b * params.dt)
            q -= n_a.astype(int)
            X += n_a * p_a
            q += n_b.astype(int)
            X -= n_b * p_b
        else:
            raise ValueError(f"Unknown execution_model={config.execution_model}")

        S = S + price_incs[:, i]

    profit = X + q * S
    return SimResult(profit=profit, qT=q.copy())


def simulate_pair(*, params: ModelParams, config: SimConfig) -> dict[str, SimResult]:
    """Simulate both strategies using common random numbers."""
    n_steps = int(round(params.T / params.dt))
    rng = np.random.default_rng(config.seed)

    price_incs = _price_increments(
        rng=rng,
        n_paths=config.n_paths,
        n_steps=n_steps,
        sigma=params.sigma,
        dt=params.dt,
        price_model=config.price_model,
    )
    u_bid = rng.random(size=(config.n_paths, n_steps))
    u_ask = rng.random(size=(config.n_paths, n_steps))

    inv = simulate(params=params, config=config, strategy="inventory", price_incs=price_incs, u_bid=u_bid, u_ask=u_ask)
    sym = simulate(params=params, config=config, strategy="symmetric", price_incs=price_incs, u_bid=u_bid, u_ask=u_ask)
    return {"inventory": inv, "symmetric": sym}
