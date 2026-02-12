from __future__ import annotations

import argparse
from pathlib import Path

from .experiment import reproduce


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="asmm_simulator", description="ASMM (Avellaneda–Stoikov) synthetic market making simulator")
    p.add_argument("--n-paths", type=int, default=1000, help="Number of Monte Carlo paths")
    p.add_argument("--seed", type=int, default=0, help="RNG seed")
    p.add_argument("--out-dir", type=Path, default=Path("outputs"), help="Output directory")

    sub = p.add_subparsers(dest="cmd", required=True)

    rep = sub.add_parser("reproduce", help="Run paper-style reproduction (tables + histograms)")
    rep.add_argument("--spread-model", choices=["constant", "time-varying"], default="constant")
    rep.add_argument("--price-model", choices=["rademacher", "gaussian"], default="rademacher")
    rep.add_argument("--execution-model", choices=["bernoulli", "poisson"], default="bernoulli")
    rep.add_argument("--allow-cross", action=argparse.BooleanOptionalAction, default=True, help="Allow quotes to cross the mid-price")
    rep.add_argument(
        "--floor-intensity-delta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use max(delta, 0) when computing intensities",
    )
    rep.add_argument("--cap-prob", action=argparse.BooleanOptionalAction, default=True, help="Cap Bernoulli p=min(1, lambda*dt)")

    return p


def main(argv: list[str] | None = None) -> int:
    p = _build_parser()
    args = p.parse_args(argv)

    if args.cmd == "reproduce":
        reproduce(
            out_dir=args.out_dir,
            n_paths=args.n_paths,
            seed=args.seed,
            spread_model=args.spread_model,
            price_model=args.price_model,
            execution_model=args.execution_model,
            allow_cross=args.allow_cross,
            floor_intensity_delta=args.floor_intensity_delta,
            cap_prob=args.cap_prob,
        )
        return 0

    raise RuntimeError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
