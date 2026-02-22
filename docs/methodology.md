# Methodology

## Overview

This simulator reproduces the simulation study in Section 3.3 of the
Avellaneda-Stoikov (2008) paper on high-frequency market making.

## Mid-Price Dynamics

The mid-price follows a driftless Brownian motion:

```
dS_u = σ dW_u
```

Discretized as:

```
S_{t+dt} = S_t + ε_t · σ · √dt,  ε_t ∈ {+1, -1} w.p. 1/2
```

## Reservation Price

The inventory-based reservation (indifference) price:

```
r(S, q, t) = S - q · γ · σ² · (T - t)
```

## Execution Intensity

Market order arrival intensity as a function of quote distance:

```
λ(δ) = A · exp(-k · δ)
```

## Strategies

### Inventory Strategy
- Center quotes around reservation price r_t
- p_ask = r_t + Sp/2, p_bid = r_t - Sp/2

### Symmetric Strategy (Benchmark)
- Center quotes around mid-price S_t
- p_ask = S_t + Sp/2, p_bid = S_t - Sp/2

## Spread Computation

Constant spread (main):
```
Sp(γ) = (2/γ) · ln(1 + γ/k)
```

Time-varying spread (alternative):
```
Sp_t(γ) = γ · σ² · (T - t) + (2/γ) · ln(1 + γ/k)
```

## Terminal P&L

```
Π_T = X_T + q_T · S_T
```

## Monte Carlo Procedure

1. For each path i = 1, ..., N:
   - Initialize S₀, q₀=0, X₀=0
   - For each time step t → t+dt:
     - Compute quotes
     - Simulate fills (Bernoulli with p = λ(δ)·dt)
     - Update inventory and cash
     - Update mid-price
   - Record Π_T and q_T
2. Compute summary statistics across N paths
