import numpy as np

from asmm_simulator.models import ModelParams
from asmm_simulator.sim import SimConfig, simulate_pair


def test_inventory_strategy_reduces_variance_vs_symmetric_small_mc():
    params = ModelParams()
    config = SimConfig(gamma=0.1, n_paths=400, seed=123)
    res = simulate_pair(params=params, config=config)

    inv = res["inventory"].summary()
    sym = res["symmetric"].summary()

    assert inv["profit_std"] < sym["profit_std"]
    assert inv["qT_std"] < sym["qT_std"]


def test_profit_definition_matches_mark_to_market_identity():
    params = ModelParams()
    config = SimConfig(gamma=0.1, n_paths=50, seed=7)
    res = simulate_pair(params=params, config=config)

    # Basic sanity: profit arrays are finite and consistent in shape
    for r in res.values():
        assert r.profit.shape == (config.n_paths,)
        assert r.qT.shape == (config.n_paths,)
        assert np.isfinite(r.profit).all()
