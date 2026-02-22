"""A1: Config validation tests."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from mm_sim.config import ConstraintsConfig, SimulationConfig


class TestConfigValidation:
    def test_default_config_valid(self):
        cfg = SimulationConfig()
        assert cfg.S0 == 100.0
        assert cfg.n_steps == 200

    def test_dt_must_be_positive(self):
        with pytest.raises(ValidationError):
            SimulationConfig(dt=-0.001)

    def test_T_must_be_positive(self):
        with pytest.raises(ValidationError):
            SimulationConfig(T=0.0)

    def test_A_must_be_positive(self):
        with pytest.raises(ValidationError):
            SimulationConfig(A=0.0)

    def test_k_must_be_positive(self):
        with pytest.raises(ValidationError):
            SimulationConfig(k=-1.0)

    def test_sigma_non_negative(self):
        cfg = SimulationConfig(sigma=0.0)
        assert cfg.sigma == 0.0

    def test_sigma_negative_rejected(self):
        with pytest.raises(ValidationError):
            SimulationConfig(sigma=-1.0)

    def test_n_paths_at_least_one(self):
        cfg = SimulationConfig(n_paths=1)
        assert cfg.n_paths == 1
        with pytest.raises(ValidationError):
            SimulationConfig(n_paths=0)

    def test_gamma_must_be_positive(self):
        with pytest.raises(ValidationError):
            SimulationConfig(gamma=0.0)

    def test_dt_divides_T(self):
        cfg = SimulationConfig(T=1.0, dt=0.005)
        assert cfg.n_steps == 200

    def test_dt_not_dividing_T_raises(self):
        with pytest.raises(ValidationError, match="does not evenly divide"):
            SimulationConfig(T=1.0, dt=0.003)

    def test_prob_cap_in_range(self):
        c = ConstraintsConfig(prob_cap=0.5)
        assert c.prob_cap == 0.5
        with pytest.raises(ValidationError):
            ConstraintsConfig(prob_cap=0.0)
        with pytest.raises(ValidationError):
            ConstraintsConfig(prob_cap=1.5)

    def test_constant_spread_formula(self):
        cfg = SimulationConfig(gamma=0.1, k=1.5)
        sp = cfg.constant_spread()
        expected = (2.0 / 0.1) * math.log(1.0 + 0.1 / 1.5)
        assert abs(sp - expected) < 1e-10

    def test_time_varying_spread_at_t0(self):
        cfg = SimulationConfig(gamma=0.1, k=1.5, sigma=2.0, T=1.0)
        sp_tv = cfg.time_varying_spread(0.0)
        sp_const = cfg.constant_spread()
        # At t=0: sp_tv = γσ²T + sp_const
        expected = 0.1 * 4.0 * 1.0 + sp_const
        assert abs(sp_tv - expected) < 1e-10

    def test_time_varying_spread_at_T(self):
        cfg = SimulationConfig(gamma=0.1, k=1.5, sigma=2.0, T=1.0)
        sp_tv = cfg.time_varying_spread(1.0)
        sp_const = cfg.constant_spread()
        # At t=T: sp_tv = sp_const (time-dependent term vanishes)
        assert abs(sp_tv - sp_const) < 1e-10

    def test_n_steps_property(self):
        cfg = SimulationConfig(T=1.0, dt=0.005)
        assert cfg.n_steps == 200

    def test_price_model_enum(self):
        cfg = SimulationConfig(price_model="rademacher")
        assert cfg.price_model.value == "rademacher"
        cfg2 = SimulationConfig(price_model="gaussian")
        assert cfg2.price_model.value == "gaussian"

    def test_execution_model_enum(self):
        cfg = SimulationConfig(execution_model="bernoulli")
        assert cfg.execution_model.value == "bernoulli"
        cfg2 = SimulationConfig(execution_model="poisson")
        assert cfg2.execution_model.value == "poisson"
