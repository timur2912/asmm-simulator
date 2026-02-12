import math

import pytest

from asmm_simulator.models import ModelParams, reservation_price, spread_constant


def test_spread_constant_matches_paper_table_values_approximately():
    params = ModelParams(k=1.5)
    cases = [
        (0.1, 1.29),
        (0.01, 1.33),
        (0.5, 1.15),
    ]
    for gamma, approx in cases:
        Sp = spread_constant(gamma=gamma, k=params.k)
        assert math.isfinite(Sp)
        assert abs(Sp - approx) < 0.05


def test_reservation_price_signs():
    params = ModelParams(T=1.0, sigma=2.0)
    s = 100.0
    t = 0.2
    gamma = 0.1

    r0 = reservation_price(s=s, q=0, t=t, params=params, gamma=gamma)
    r_long = reservation_price(s=s, q=5, t=t, params=params, gamma=gamma)
    r_short = reservation_price(s=s, q=-5, t=t, params=params, gamma=gamma)

    assert r0 == pytest.approx(s)
    assert r_long < s
    assert r_short > s
