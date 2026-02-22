# Assumptions and Implementation Choices

## Spread Model (Main Reproduction)

**Choice**: Constant spread (Variant A) is used for the main reproduction.

The spread is computed as:

```
Sp(γ) = (2/γ) · ln(1 + γ/k)
```

This matches the single spread values reported in Tables 1–3 of the paper:
- γ = 0.01 → Sp ≈ 1.33
- γ = 0.1  → Sp ≈ 1.29
- γ = 0.5  → Sp ≈ 1.15

The time-varying variant (Variant B) is available as a configuration option.

## Handling δ < 0 (Quotes Crossing Mid-Price)

**Default**: δ is NOT clipped. The inventory strategy may produce negative δ values
when inventory is large, causing λ(δ) > A. This is the literal interpretation of the
formulas.

**Alternative**: Set `constraints.clip_delta: true` to enforce δ ≥ 0.

## Handling λ·dt > 1

**Default**: Execution probability is clipped to `prob_cap` (default 1.0).
This prevents invalid Bernoulli probabilities.

With A=140, dt=0.005, and δ≈0: λ·dt ≈ 0.7, which is valid but not "small."
For δ < 0, λ·dt can exceed 1.0.

## Bernoulli vs Poisson Fills

**Default**: Bernoulli model, matching the paper's wording:
"With probability λ(δ)·dt, a market order arrives."

Poisson model is available as `execution_model: poisson`.

## Simultaneous Bid and Ask Fills

**Allowed**: Bid and ask fills are simulated as independent events.
Both can occur in the same dt interval. The probability of both occurring
is O(dt²), which is small for dt=0.005.

## Standard Deviation (ddof)

**Choice**: ddof=1 (sample standard deviation) is used for all reported statistics.
This is the conventional choice for sample statistics.

## Price Increments

**Default**: Rademacher increments ±σ√dt, matching the paper's description.
Gaussian increments N(0, σ²dt) are available as `price_model: gaussian`.

## Common Random Numbers (CRN)

**Default**: CRN is enabled (`use_crn: true`). Both strategies use the same
seed per path, producing the same price path and execution uniform draws.
This reduces variance in strategy comparisons.
