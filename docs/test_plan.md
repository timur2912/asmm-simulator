# Test Plan

## Hypothesis Mapping

### H1 — Inventory-based optimal quoting reduces risk

| Test | Location | Description |
|------|----------|-------------|
| B4   | `test_strategy_comparison_directional.py` | std(Profit) and std(q_T) lower for inventory strategy |
| D1   | `test_variance_reduction_significance.py` | Statistical significance of variance reduction |
| D2   | `test_distributional_shift.py` | Inventory q_T distribution more concentrated |

### H2 — Risk aversion controls inventory effect

| Test | Location | Description |
|------|----------|-------------|
| B5   | `test_gamma_monotonicity_effects.py` | Increasing γ reduces inventory dispersion |
| D3   | `test_convergence_gamma_to_zero.py` | γ→0 makes strategies converge |

### H3 — LOB execution probabilities matter

| Test | Location | Description |
|------|----------|-------------|
| B3   | `test_intensity_distance_effect.py` | Increasing δ reduces fill rate |
| A3   | `test_intensities.py` | λ(δ) decreases with δ |

## Test Categories

### Unit Tests (A1–A9)
- Component-level correctness
- Deterministic, fast, no MC simulation needed
- Cover: config validation, price process, intensities, execution, state, strategies, metrics, RNG, edge cases

### Integration Tests (B1–B7)
- System-level simulation correctness
- May require short MC runs (100–300 paths)
- Cover: path invariants, reproducibility, directional comparisons, monotonicity

### Regression Tests (C1–C2)
- Approximate reproduction of paper tables
- Compare against stored snapshots with MC tolerance
- Full 1000-path runs

### Statistical Tests (D1–D3)
- Formal hypothesis testing
- Bootstrap or parametric tests
- Fixed seed for CI stability

### Performance Tests (E1)
- Runtime budget enforcement
- Ensure simulation completes in reasonable time
