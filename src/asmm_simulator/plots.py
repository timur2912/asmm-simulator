from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_profit_hist(*, out_path: Path, profit_inventory: np.ndarray, profit_symmetric: np.ndarray, gamma: float, bins: int = 60) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.hist(profit_symmetric, bins=bins, alpha=0.55, label="Symmetric", density=True)
    plt.hist(profit_inventory, bins=bins, alpha=0.55, label="Inventory", density=True)
    plt.title(f"Terminal profit distribution (gamma={gamma})")
    plt.xlabel("Profit  (X_T + q_T S_T)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
