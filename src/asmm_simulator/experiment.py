from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .models import ModelParams, spread_constant
from .plots import plot_profit_hist
from .sim import SimConfig, simulate_pair


def reproduce(*, out_dir: Path, n_paths: int, seed: int, gammas: list[float] | None = None) -> pd.DataFrame:
    """Run the paper-style reproduction experiment and write CSV/PNG outputs."""

    out_dir.mkdir(parents=True, exist_ok=True)
    params = ModelParams()
    gammas = gammas or [0.01, 0.1, 0.5]

    rows: list[dict] = []

    for gamma in gammas:
        config = SimConfig(gamma=gamma, n_paths=n_paths, seed=seed)
        results = simulate_pair(params=params, config=config)

        Sp = spread_constant(gamma=gamma, k=params.k)

        for name, res in results.items():
            summ = res.summary()
            rows.append(
                {
                    "gamma": gamma,
                    "strategy": name,
                    "spread": Sp,
                    **summ,
                    **asdict(config),
                }
            )

        # Per-gamma artifacts
        df_gamma = pd.DataFrame([r for r in rows if r["gamma"] == gamma and r["spread_model"] == "constant"])
        df_gamma.to_csv(out_dir / f"gamma_{gamma}_table.csv", index=False)

        plot_profit_hist(
            out_path=out_dir / f"gamma_{gamma}_profit_hist.png",
            profit_inventory=results["inventory"].profit,
            profit_symmetric=results["symmetric"].profit,
            gamma=gamma,
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary.csv", index=False)
    return df
