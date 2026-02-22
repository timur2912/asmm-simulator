# Reproduction Notes

## Paper Reference

Avellaneda, M. & Stoikov, S. (2008). "High-frequency trading in a limit order book."
Quantitative Finance, 8(3), 217–224.

## Section 3.3 Parameters

| Parameter | Value |
|-----------|-------|
| S₀        | 100   |
| T         | 1     |
| σ         | 2     |
| dt        | 0.005 |
| q₀        | 0     |
| A         | 140   |
| k         | 1.5   |
| N (paths) | 1000  |

## Expected Results (Tables 1–3)

### γ = 0.1, Spread ≈ 1.29
| Strategy  | Profit | std(Profit) | Final q | std(Final q) |
|-----------|--------|-------------|---------|--------------|
| Inventory | 62.94  | 5.89        | 0.10    | 2.80         |
| Symmetric | 67.21  | 13.43       | -0.018  | 8.66         |

### γ = 0.01, Spread ≈ 1.33
| Strategy  | Profit | std(Profit) | Final q | std(Final q) |
|-----------|--------|-------------|---------|--------------|
| Inventory | 66.78  | 8.76        | -0.02   | 4.70         |
| Symmetric | 67.36  | 13.40       | -0.31   | 8.65         |

### γ = 0.5, Spread ≈ 1.15
| Strategy  | Profit | std(Profit) | Final q | std(Final q) |
|-----------|--------|-------------|---------|--------------|
| Inventory | 33.92  | 4.72        | -0.02   | 1.88         |
| Symmetric | 66.20  | 14.53       | 0.25    | 9.06         |

## Key Observations

1. Inventory strategy consistently shows lower std(Profit) and std(Final q)
2. Higher γ → more aggressive inventory control → lower dispersion
3. Higher γ → lower mean profit (cost of risk management)
4. As γ → 0, strategies converge

## Ambiguities Resolved

- **Spread**: Using constant spread Sp(γ) = (2/γ)ln(1+γ/k), matching table values
- **Price increments**: Rademacher ±σ√dt as described in text
- **Execution**: Bernoulli with probability λ(δ)·dt, capped at 1.0
