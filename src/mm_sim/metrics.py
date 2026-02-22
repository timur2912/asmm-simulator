"""Performance metrics: Π_T, summary statistics, Monte Carlo SE."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SummaryStats:
    """Summary statistics for a Monte Carlo experiment."""

    mean_profit: float
    std_profit: float
    mean_q: float
    std_q: float
    mc_se_mean_profit: float
    mc_se_std_profit: float
    mc_se_mean_q: float
    mc_se_std_q: float
    n_paths: int
    spread: float
    strategy_name: str


def compute_summary(
    pnl_array: np.ndarray,
    q_array: np.ndarray,
    spread: float = 0.0,
    strategy_name: str = "",
    ddof: int = 1,
) -> SummaryStats:
    """Compute summary statistics from MC arrays.

    Uses ddof=1 (sample std) by default, matching typical reporting.
    """
    n = len(pnl_array)
    mean_p = float(np.mean(pnl_array))
    std_p = float(np.std(pnl_array, ddof=ddof)) if n > 1 else 0.0
    mean_q = float(np.mean(q_array))
    std_q = float(np.std(q_array, ddof=ddof)) if n > 1 else 0.0

    # Monte Carlo standard errors
    mc_se_mean_p = std_p / np.sqrt(n)
    # SE of sample std: approx std / sqrt(2(n-1))
    mc_se_std_p = std_p / np.sqrt(2 * (n - 1)) if n > 1 else 0.0
    mc_se_mean_q = std_q / np.sqrt(n)
    mc_se_std_q = std_q / np.sqrt(2 * (n - 1)) if n > 1 else 0.0

    return SummaryStats(
        mean_profit=mean_p,
        std_profit=std_p,
        mean_q=mean_q,
        std_q=std_q,
        mc_se_mean_profit=mc_se_mean_p,
        mc_se_std_profit=mc_se_std_p,
        mc_se_mean_q=mc_se_mean_q,
        mc_se_std_q=mc_se_std_q,
        n_paths=n,
        spread=spread,
        strategy_name=strategy_name,
    )


def terminal_pnl(X: float, q: int, S: float) -> float:
    """Π_T = X_T + q_T · S_T"""
    return X + q * S
