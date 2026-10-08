"""Property tests for the multi-objective reward (spec Section 4.4).

Checks that hold for any state:

- the three logged terms stay in their documented ranges
- the reward is clipped to [-10, 10]
- raising the trade-off weight w never increases the reward for the same state
- raising exposure never increases the reward for the same state
- the preference weight round-trips through the observation
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from flywaydqn.env.reward import RewardFunction, make_reward_fn

ALPHA = 1.0
GAMMA = 0.02
S_THR = 10.0
MAX_TX_MW = 100.0
MAX_REF_MBPS = 1000.0


def _reward_fn() -> RewardFunction:
    return make_reward_fn(
        alpha=ALPHA,
        gamma=GAMMA,
        kappa=5.0,
        s_thr=S_THR,
        max_tx_power_mw=MAX_TX_MW,
        max_throughput_ref_mbps=MAX_REF_MBPS,
        reward_clip=10.0,
    )


def _evaluate(
    fn: RewardFunction,
    *,
    throughput_mbps: float,
    power_mw: float,
    s_p95: float,
    log10_w: float,
) -> dict[str, Any]:
    return dict(
        fn(
            sum_throughput_mbps=throughput_mbps,
            tx_power_mw=power_mw,
            s_p95_w_per_m2=s_p95,
            observation={"log10_w": log10_w},
        )
    )


def test_reward_is_clipped_to_the_documented_range() -> None:
    fn = _reward_fn()
    for throughput in (0.0, 500.0, 5000.0):
        for s_p95 in (0.0, 50.0, 5000.0):
            for log10_w in (-1.0, 0.0, 2.0):
                result = _evaluate(
                    fn,
                    throughput_mbps=throughput,
                    power_mw=MAX_TX_MW,
                    s_p95=s_p95,
                    log10_w=log10_w,
                )
                assert -10.0 <= result["reward"] <= 10.0


def test_logged_terms_stay_in_range() -> None:
    fn = _reward_fn()
    result = _evaluate(fn, throughput_mbps=250.0, power_mw=50.0, s_p95=5.0, log10_w=0.0)
    terms = result["terms"]
    assert set(terms) == {"throughput", "exposure", "power", "w"}
    assert 0.0 <= terms["throughput"] <= 1.0
    assert 0.0 <= terms["power"] <= 1.0
    assert terms["exposure"] >= 0.0
    assert terms["w"] > 0.0


def test_terms_are_logged_separately_every_step() -> None:
    fn = _reward_fn()
    rng = np.random.default_rng(5)
    for _ in range(200):
        result = _evaluate(
            fn,
            throughput_mbps=float(rng.uniform(0.0, 1000.0)),
            power_mw=float(rng.uniform(0.0, MAX_TX_MW)),
            s_p95=float(rng.uniform(0.0, 100.0)),
            log10_w=float(rng.uniform(-1.0, 2.0)),
        )
        assert len(result["terms"]) == 4
        assert np.isfinite(result["reward"])


def test_weight_round_trips_from_the_observation() -> None:
    fn = _reward_fn()
    for log10_w in (-1.0, -0.5, 0.0, 1.0, 2.0):
        result = _evaluate(fn, throughput_mbps=100.0, power_mw=10.0, s_p95=1.0, log10_w=log10_w)
        assert result["terms"]["w"] == pytest.approx(10.0**log10_w, rel=1e-12)


def test_higher_weight_never_increases_reward_for_the_same_state() -> None:
    fn = _reward_fn()
    rewards = [
        _evaluate(fn, throughput_mbps=300.0, power_mw=80.0, s_p95=40.0, log10_w=log10_w)["reward"]
        for log10_w in (-1.0, 0.0, 0.5, 1.0, 2.0)
    ]
    # Adjacent-pair comparison: no strict=True here, the second sequence is
    # deliberately one element shorter.
    assert all(b <= a + 1e-12 for a, b in zip(rewards, rewards[1:], strict=False))


def test_higher_exposure_never_increases_reward_for_the_same_state() -> None:
    fn = _reward_fn()
    previous = None
    for s_p95 in np.linspace(0.0, 200.0, 50):
        reward = _evaluate(
            fn, throughput_mbps=300.0, power_mw=80.0, s_p95=float(s_p95), log10_w=1.0
        )["reward"]
        if previous is not None:
            assert reward <= previous + 1e-12
        previous = reward


def test_more_throughput_never_reduces_reward() -> None:
    fn = _reward_fn()
    previous = None
    for throughput in np.linspace(0.0, 1000.0, 50):
        reward = _evaluate(
            fn, throughput_mbps=float(throughput), power_mw=50.0, s_p95=3.0, log10_w=0.0
        )["reward"]
        if previous is not None:
            assert reward >= previous - 1e-12
        previous = reward


def test_more_power_never_increases_reward() -> None:
    fn = _reward_fn()
    previous = None
    for power in np.linspace(0.0, MAX_TX_MW, 50):
        reward = _evaluate(
            fn, throughput_mbps=100.0, power_mw=float(power), s_p95=3.0, log10_w=0.0
        )["reward"]
        if previous is not None:
            assert reward <= previous + 1e-12
        previous = reward


def test_unweighted_throughput_only_case_matches_the_formula() -> None:
    # With w = 0 (log10_w -> -inf is unreachable, use a tiny weight) and zero
    # exposure, the reward reduces to alpha*T_hat - gamma*P_hat.
    fn = _reward_fn()
    result = _evaluate(fn, throughput_mbps=500.0, power_mw=50.0, s_p95=0.0, log10_w=-12.0)
    expected = ALPHA * (500.0 / MAX_REF_MBPS) - GAMMA * (50.0 / MAX_TX_MW) - 0.0
    assert result["reward"] == pytest.approx(expected, abs=1e-6)


def test_reward_matches_the_spec_formula_exactly() -> None:
    from flywaydqn.exposure.thresholds import exposure_penalty_softplus

    fn = _reward_fn()
    result = _evaluate(fn, throughput_mbps=400.0, power_mw=25.0, s_p95=25.0, log10_w=0.5)
    w = 10.0**0.5
    t_hat = 400.0 / MAX_REF_MBPS
    p_hat = 25.0 / MAX_TX_MW
    e_hat = exposure_penalty_softplus(25.0, S_THR, kappa=5.0)
    expected = ALPHA * t_hat - ALPHA * w * e_hat - GAMMA * p_hat
    assert result["reward"] == pytest.approx(expected, rel=1e-12)
