"""CLI entry point for the ASMM simulator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="asmm-simulator",
        description="Avellaneda-Stoikov Market Making Simulator",
    )
    parser.add_argument(
        "command",
        choices=["reproduce", "run", "sweep"],
        help="Command to execute",
    )
    parser.add_argument("--n-paths", type=int, default=1000, help="Number of MC paths")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gamma", type=float, nargs="+", default=[0.01, 0.1, 0.5])
    parser.add_argument("--output-dir", type=str, default="outputs/default")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument(
        "--write-snapshots", action="store_true", help="Write regression snapshots"
    )
    parser.add_argument(
        "--record-trajectories", action="store_true", help="Record full trajectories"
    )

    args = parser.parse_args(argv)

    if args.command == "reproduce":
        _reproduce(args)
    elif args.command == "run":
        _run_single(args)
    elif args.command == "sweep":
        _sweep(args)


def _reproduce(args: argparse.Namespace) -> None:
    """Reproduce paper Section 3.3 results."""
    from mm_sim.config import SimulationConfig
    from mm_sim.experiments import run_gamma_sweep
    from mm_sim.reporting.export import export_csv, export_config, export_json, export_markdown
    from mm_sim.reporting.plots import plot_inventory_histogram, plot_pnl_histogram, plot_sample_path
    from mm_sim.reporting.tables import experiment_table, print_tables

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    base = SimulationConfig(
        S0=100.0,
        T=1.0,
        dt=0.005,
        sigma=2.0,
        A=140.0,
        k=1.5,
        q0=0,
        X0=0.0,
        gamma=0.1,
        n_paths=args.n_paths,
        seed=args.seed,
        use_crn=True,
    )

    gammas = args.gamma
    print(f"Running gamma sweep: {gammas}, n_paths={args.n_paths}, seed={args.seed}")

    results = run_gamma_sweep(gammas, base)

    print_tables(results)

    # Export tables
    for gamma, res in results.items():
        df = experiment_table(res)
        export_csv(df, out / f"table_gamma_{gamma}.csv")

        # Histograms
        plot_pnl_histogram(res.mc_results, gamma, out / f"pnl_hist_gamma_{gamma}.png")
        plot_inventory_histogram(res.mc_results, gamma, out / f"inv_hist_gamma_{gamma}.png")

    export_markdown(results, out / "results.md")
    export_json(results, out / "results.json")
    export_config(base.model_dump(mode="json"), out / "run_config.yaml")

    # Write snapshots if requested
    if args.write_snapshots:
        snap_dir = Path("tests/regression/snapshots")
        snap_dir.mkdir(parents=True, exist_ok=True)
        import json

        for gamma, res in results.items():
            snap = {}
            for name, s in res.summaries.items():
                snap[name] = {
                    "mean_profit": s.mean_profit,
                    "std_profit": s.std_profit,
                    "mean_q": s.mean_q,
                    "std_q": s.std_q,
                    "spread": s.spread,
                }
            fname = f"expected_table_gamma_{str(gamma).replace('.', '_')}.json"
            (snap_dir / fname).write_text(json.dumps(snap, indent=2))
        print(f"Snapshots written to {snap_dir}")

    print(f"\nResults saved to {out}/")
    for gamma, res in results.items():
        print(f"  γ={gamma}: elapsed={res.elapsed_seconds:.2f}s")


def _run_single(args: argparse.Namespace) -> None:
    """Run a single experiment with given config."""
    from mm_sim.config import SimulationConfig, load_config
    from mm_sim.experiments import run_experiment
    from mm_sim.reporting.tables import experiment_table
    from mm_sim.strategies import InventoryStrategy, SymmetricStrategy

    if args.config:
        cfg = load_config(args.config, {"n_paths": args.n_paths, "seed": args.seed})
    else:
        cfg = SimulationConfig(n_paths=args.n_paths, seed=args.seed, gamma=args.gamma[0])

    strats = [InventoryStrategy(), SymmetricStrategy()]
    result = run_experiment(strats, cfg)
    df = experiment_table(result)
    print(df.to_string(index=False))


def _sweep(args: argparse.Namespace) -> None:
    """Run gamma sweep."""
    _reproduce(args)


if __name__ == "__main__":
    main()
