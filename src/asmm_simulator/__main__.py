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

    sub.add_parser("reproduce", help="Run paper-style reproduction (tables + histograms)")

    return p


def main(argv: list[str] | None = None) -> int:
    p = _build_parser()
    args = p.parse_args(argv)

    if args.cmd == "reproduce":
        reproduce(out_dir=args.out_dir, n_paths=args.n_paths, seed=args.seed)
        return 0

    raise RuntimeError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
