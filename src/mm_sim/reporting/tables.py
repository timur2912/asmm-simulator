"""Generate Tables 1–3 style summary outputs."""

from __future__ import annotations

import pandas as pd

from mm_sim.experiments import ExperimentResult
from mm_sim.metrics import SummaryStats


def summary_to_row(s: SummaryStats) -> dict:
    return {
        "Strategy": s.strategy_name,
        "Spread": round(s.spread, 4),
        "Profit": round(s.mean_profit, 2),
        "std(Profit)": round(s.std_profit, 2),
        "Final q": round(s.mean_q, 2),
        "std(Final q)": round(s.std_q, 2),
    }


def experiment_table(result: ExperimentResult) -> pd.DataFrame:
    """Create a summary table for one experiment (one gamma)."""
    rows = [summary_to_row(s) for s in result.summaries.values()]
    return pd.DataFrame(rows)


def gamma_sweep_tables(
    results: dict[float, ExperimentResult],
) -> dict[float, pd.DataFrame]:
    """Create tables for each gamma value."""
    return {gamma: experiment_table(res) for gamma, res in results.items()}


def format_table_markdown(df: pd.DataFrame, gamma: float) -> str:
    """Format a table as markdown string."""
    header = f"## γ = {gamma}\n\n"
    return header + df.to_markdown(index=False) + "\n"


def print_tables(results: dict[float, ExperimentResult]) -> None:
    """Print all tables to stdout."""
    for gamma in sorted(results.keys()):
        df = experiment_table(results[gamma])
        print(f"\n{'='*60}")
        print(f"  γ = {gamma}")
        print(f"{'='*60}")
        print(df.to_string(index=False))
        print()
