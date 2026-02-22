"""Configuration schemas and validation for the simulation."""

from __future__ import annotations

from enum import Enum
from typing import Optional

import yaml
from pydantic import BaseModel, Field, model_validator


class PriceModel(str, Enum):
    RADEMACHER = "rademacher"
    GAUSSIAN = "gaussian"


class ExecutionModel(str, Enum):
    BERNOULLI = "bernoulli"
    POISSON = "poisson"


class SpreadModel(str, Enum):
    CONSTANT = "constant"
    TIME_VARYING = "time_varying"


class ConstraintsConfig(BaseModel):
    clip_delta: bool = False
    clip_prob: bool = True
    prob_cap: float = Field(default=1.0, gt=0.0, le=1.0)


class SimulationConfig(BaseModel):
    """Full simulation configuration matching paper Section 3.3."""

    S0: float = Field(default=100.0, description="Initial mid-price")
    T: float = Field(default=1.0, gt=0.0, description="Time horizon")
    dt: float = Field(default=0.005, gt=0.0, description="Time step")
    sigma: float = Field(default=2.0, ge=0.0, description="Volatility")
    A: float = Field(default=140.0, gt=0.0, description="Intensity scale")
    k: float = Field(default=1.5, gt=0.0, description="Intensity decay")
    q0: int = Field(default=0, description="Initial inventory")
    X0: float = Field(default=0.0, description="Initial cash")
    gamma: float = Field(default=0.1, gt=0.0, description="Risk aversion")
    n_paths: int = Field(default=1000, ge=1, description="Monte Carlo paths")
    price_model: PriceModel = PriceModel.RADEMACHER
    execution_model: ExecutionModel = ExecutionModel.BERNOULLI
    spread_model: SpreadModel = SpreadModel.CONSTANT
    constraints: ConstraintsConfig = Field(default_factory=ConstraintsConfig)
    seed: int = Field(default=42)
    use_crn: bool = Field(default=True, description="Common random numbers")

    @model_validator(mode="after")
    def validate_dt_divides_t(self) -> "SimulationConfig":
        n_steps = round(self.T / self.dt)
        if abs(n_steps * self.dt - self.T) > 1e-12:
            raise ValueError(f"dt={self.dt} does not evenly divide T={self.T}")
        return self

    @property
    def n_steps(self) -> int:
        return round(self.T / self.dt)

    def constant_spread(self) -> float:
        """Sp(γ) = (2/γ) ln(1 + γ/k)"""
        import math

        return (2.0 / self.gamma) * math.log(1.0 + self.gamma / self.k)

    def time_varying_spread(self, t: float) -> float:
        """Sp_t(γ) = γσ²(T-t) + (2/γ) ln(1 + γ/k)"""
        return self.gamma * self.sigma**2 * (self.T - t) + self.constant_spread()


def load_config(path: str, overrides: Optional[dict] = None) -> SimulationConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    if overrides:
        data.update(overrides)
    return SimulationConfig(**data)
