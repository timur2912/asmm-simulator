"""Monte Carlo experiment runner with multi-strategy support."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from mm_sim.config import SimulationConfig
from mm_sim.metrics import SummaryStats
from mm_sim.simulator import MCResult, simulate_mc
from mm_sim.strategies.base import Strategy


@dataclass
class ExperimentResult:
    """Full experiment result across strategies."""

    config: SimulationConfig
    mc_results: dict[str, MCResult] = field(default_factory=dict)
    summaries: dict[str, SummaryStats] = field(default_factory=dict)
    elapsed_seconds: float = 0.0


def run_experiment(
    strategies: list[Strategy],
    config: SimulationConfig,
) -> ExperimentResult:
    """Run MC simulation for multiple strategies."""
    t0 = time.time()
    result = ExperimentResult(config=config)

    for strat in strategies:
        mc = simulate_mc(strat, config)
        result.mc_results[strat.name] = mc
        result.summaries[strat.name] = mc.summary

    result.elapsed_seconds = time.time() - t0
    return result


def run_gamma_sweep(
    gammas: list[float],
    base_config: SimulationConfig,
) -> dict[float, ExperimentResult]:
    """Run experiments across multiple gamma values."""
    from mm_sim.strategies import InventoryStrategy, SymmetricStrategy

    results = {}
    for gamma in gammas:
        cfg = base_config.model_copy(update={"gamma": gamma})
        strats = [InventoryStrategy(), SymmetricStrategy()]
        results[gamma] = run_experiment(strats, cfg)
    return results
