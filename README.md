# ASMM Simulator (Avellaneda–Stoikov Market Making)

This repository reproduces the synthetic simulation study from the classic inventory-based market making model (Avellaneda–Stoikov style), matching the experiment blueprint described in the prompt.

## What it does

- Simulates mid-price paths (Rademacher approximation to Brownian motion):
  \(S_{t+dt} = S_t + \epsilon_t\,\sigma\sqrt{dt}\), \(\epsilon_t \in \{+1,-1\}\).
- Simulates executions on bid/ask quotes using Bernoulli arrivals with intensity
  \(\lambda(\delta) = A e^{-k\delta}\).
- Compares two strategies at the same spread:
  - **Inventory-based** quoting centered on the reservation (indifference) price
    \(r_t = S_t - q_t\gamma\sigma^2(T-t)\).
  - **Symmetric** quoting centered on the mid-price \(S_t\).

## Quickstart

```bash
cd /workspace/project
python -m pip install -e .
python -m pytest

# CLI help
python -m asmm_simulator --help

# Reproduce paper-style tables/plots (writes to ./outputs)
python -m asmm_simulator --n-paths 1000 reproduce
```

## Outputs

The `reproduce` command writes:

- `outputs/summary.csv` (all \(\gamma\) and both strategies)
- `outputs/gamma_<...>_table.csv`
- `outputs/gamma_<...>_profit_hist.png`

## Notes on implementation choices

Key modeling choices are configurable to reflect paper ambiguities:

- Quotes may cross the mid-price (`allow_cross=True`), but intensities treat \(\delta\) as a distance (`floor_intensity_delta=True`).
- Bernoulli fill probability is capped: \(p = \min(1, \lambda(\delta)\,dt)\).

See `python -m asmm_simulator reproduce --help` for flags.
