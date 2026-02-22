"""Histogram overlays and path plots (Figures 2–4 style)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mm_sim.experiments import ExperimentResult
from mm_sim.simulator import MCResult


def plot_pnl_histogram(
    results: dict[str, MCResult],
    gamma: float,
    output_path: Path | str | None = None,
    bins: int = 50,
) -> None:
    """Overlay histogram of terminal P&L for inventory vs symmetric."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"inventory": "steelblue", "symmetric": "coral"}
    for name, mc in results.items():
        ax.hist(
            mc.pnl_array,
            bins=bins,
            alpha=0.5,
            label=f"{name} (μ={np.mean(mc.pnl_array):.1f}, σ={np.std(mc.pnl_array):.1f})",
            color=colors.get(name, None),
            density=True,
        )

    ax.set_xlabel("Terminal P&L (Π_T)")
    ax.set_ylabel("Density")
    ax.set_title(f"Terminal P&L Distribution — γ = {gamma}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_inventory_histogram(
    results: dict[str, MCResult],
    gamma: float,
    output_path: Path | str | None = None,
    bins: int = 50,
) -> None:
    """Overlay histogram of terminal inventory for inventory vs symmetric."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"inventory": "steelblue", "symmetric": "coral"}
    for name, mc in results.items():
        ax.hist(
            mc.q_array,
            bins=bins,
            alpha=0.5,
            label=f"{name} (μ={np.mean(mc.q_array):.2f}, σ={np.std(mc.q_array):.2f})",
            color=colors.get(name, None),
            density=True,
        )

    ax.set_xlabel("Terminal Inventory (q_T)")
    ax.set_ylabel("Density")
    ax.set_title(f"Terminal Inventory Distribution — γ = {gamma}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sample_path(
    mc_result: MCResult,
    path_index: int = 0,
    output_path: Path | str | None = None,
) -> None:
    """Plot a sample path: mid-price, reservation price, bid/ask quotes."""
    if not mc_result.paths or path_index >= len(mc_result.paths):
        return

    state = mc_result.paths[path_index].state
    if state is None:
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    t = state.t_history
    ax1 = axes[0]
    ax1.plot(t, state.S_history, "k-", label="Mid-price S", linewidth=1)
    if not all(np.isnan(v) for v in state.r_history):
        ax1.plot(t, state.r_history, "b--", label="Reservation r", linewidth=0.8, alpha=0.7)
    ax1.plot(t, state.ask_history, "r-", label="Ask", linewidth=0.5, alpha=0.6)
    ax1.plot(t, state.bid_history, "g-", label="Bid", linewidth=0.5, alpha=0.6)
    ax1.set_ylabel("Price")
    ax1.set_title(f"Sample Path — {mc_result.strategy_name}")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(t, state.q_history, "b-", linewidth=1)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Inventory q")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
