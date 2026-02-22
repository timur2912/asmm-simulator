"""Export utilities: CSV, JSON, Markdown."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from mm_sim.experiments import ExperimentResult
from mm_sim.reporting.tables import experiment_table, format_table_markdown


def export_csv(df: pd.DataFrame, path: Path | str) -> None:
    df.to_csv(path, index=False)


def export_markdown(results: dict[float, ExperimentResult], path: Path | str) -> None:
    lines = ["# Simulation Results\n"]
    for gamma in sorted(results.keys()):
        df = experiment_table(results[gamma])
        lines.append(format_table_markdown(df, gamma))
    Path(path).write_text("\n".join(lines))


def export_json(results: dict[float, ExperimentResult], path: Path | str) -> None:
    data = {}
    for gamma, res in results.items():
        data[str(gamma)] = {
            name: {
                "mean_profit": s.mean_profit,
                "std_profit": s.std_profit,
                "mean_q": s.mean_q,
                "std_q": s.std_q,
                "spread": s.spread,
                "n_paths": s.n_paths,
            }
            for name, s in res.summaries.items()
        }
    Path(path).write_text(json.dumps(data, indent=2))


def export_config(config: dict, path: Path | str) -> None:
    Path(path).write_text(yaml.dump(config, default_flow_style=False))
