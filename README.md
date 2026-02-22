# ASMM Simulator — Avellaneda-Stoikov Market Making

A Python implementation reproducing the simulation study in **Section 3.3** of the Avellaneda-Stoikov optimal market-making paper. The simulator compares an **inventory-based** quoting strategy against a **symmetric benchmark** under various risk-aversion parameters.

## Key Hypotheses Tested

| # | Hypothesis | Validation |
|---|-----------|------------|
| H1 | Inventory-based quoting reduces risk | `std(P&L)` and `std(q_T)` lower for inventory strategy |
| H2 | Risk aversion γ controls inventory effect | Increasing γ → tighter inventory control; γ→0 → strategies converge |
| H3 | LOB execution probabilities matter | Fill rate decays with distance; optimal quoting balances fills vs risk |

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run the full paper reproduction (Tables 1–3, Figures 2–4)
python -m mm_sim --n-paths 1000 reproduce

# Run tests
pytest -q
```

## Project Structure

```
src/mm_sim/
  config.py              # SimulationConfig (pydantic)
  rng.py                 # Seed management, RNG streams, CRN
  price_process.py       # Rademacher / Gaussian mid-price models
  intensities.py         # λ(δ) = A·exp(-k·δ) + numerical stability
  execution.py           # Bernoulli / Poisson fill simulation
  state.py               # SimState: S, q, X, t tracking
  strategies/
    base.py              # Abstract Strategy interface
    symmetric.py         # Symmetric (naive) benchmark
    inventory.py         # Inventory-based reservation price strategy
  simulator.py           # Single-path + Monte Carlo simulation loops
  metrics.py             # Π_T, summary stats, MC standard errors
  experiments.py         # Multi-strategy experiment runner
  reporting/
    tables.py            # Tables 1–3 style outputs
    plots.py             # Histogram overlays + sample path plots
    export.py            # CSV / JSON / Markdown export

configs/                 # YAML configuration files
  base.yaml
  paper_section_3_3.yaml
  variants/              # Spread model, execution model, constraint variants

tests/
  unit/                  # Component-level correctness (86 tests)
  integration/           # System-level simulation invariants (29 tests)
  regression/            # Paper table reproduction (9 tests)
  statistical/           # Hypothesis robustness (8 tests)
  performance/           # Runtime budget (3 tests)

scripts/
  reproduce_paper.py     # Standalone reproduction script
  run_experiment.py      # Single-config experiment runner
  make_plots.py          # Plot generation

docs/
  assumptions.md         # Edge-case policies and implementation choices
  methodology.md         # Mathematical formulation
  reproduction_notes.md  # Notes on paper ambiguities
  test_plan.md           # Test ↔ hypothesis mapping
```

## Model Summary

### Mid-Price Dynamics
```
S_{t+dt} = S_t ± σ√dt   (Rademacher increments, p = 1/2)
```

### Execution Intensity
```
λ(δ) = A · exp(-k · δ)
```

### Reservation (Indifference) Price
```
r_t = S_t - q_t · γ · σ² · (T - t)
```

### Spread Formula
```
Sp(γ) = (2/γ) · ln(1 + γ/k)     [constant, table-consistent]
```

### Strategies
- **Inventory**: quotes centered around `r_t` with spread `Sp`
- **Symmetric**: quotes centered around `S_t` with same spread `Sp`

### Terminal P&L
```
Π_T = X_T + q_T · S_T
```

## Parameters (Paper Section 3.3)

| Parameter | Value |
|-----------|-------|
| S₀ | 100 |
| T | 1 |
| σ | 2 |
| dt | 0.005 |
| A | 140 |
| k | 1.5 |
| q₀ | 0 |
| γ | {0.01, 0.1, 0.5} |
| MC paths | 1000 |

## Sample Results (γ = 0.1)

| Strategy | Spread | Profit | std(Profit) | Final q | std(Final q) |
|----------|--------|--------|-------------|---------|--------------|
| inventory | 1.29 | ~64 | ~6 | ~0 | ~3 |
| symmetric | 1.29 | ~68 | ~13 | ~0 | ~9 |

The inventory strategy achieves **~50% lower P&L volatility** and **~65% lower inventory dispersion** at the cost of modestly lower mean profit.

## Configuration

Experiments are configured via YAML files or CLI arguments:

```bash
# Use a config file
python -m mm_sim --config configs/paper_section_3_3.yaml run

# Override parameters
python -m mm_sim --n-paths 500 --gamma 0.1 0.5 --seed 123 reproduce
```

## Testing

```bash
# Full suite (135 tests)
pytest -q

# By category
pytest tests/unit/ -q
pytest tests/integration/ -q
pytest tests/regression/ -q
pytest tests/statistical/ -q
pytest tests/performance/ -q

# With coverage
pytest --cov=mm_sim --cov-report=term-missing
```

## Spread Model Variants

The paper has an ambiguity in the spread formula. Both variants are implemented:

- **Variant A** (constant, table-consistent): `Sp = (2/γ)·ln(1 + γ/k)`
- **Variant B** (time-varying, theory-consistent): `Sp_t = γσ²(T-t) + (2/γ)·ln(1 + γ/k)`

The constant spread (Variant A) is used as the default, matching the paper's reported table values.

## License

MIT
