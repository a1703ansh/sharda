"""Tests for the Gymnasium ``BeamEnv`` (spec Phase 3).

Covers the acceptance criteria in the build plan: ``check_env`` clean, a random
policy surviving 1000 episodes, seeded reproducibility, and the ``info`` dict
carrying every reward term and exposure metric. Two further tests pin the action
semantics the learned controller depends on -- the power factor must change
served throughput, and the null factor must reduce footprint exposure -- because
a trained agent is only worth reporting if those factors do what they claim.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Final, cast

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env
from pydantic import ValidationError

from flywaydqn.agents.baselines_demo import StaticBackoffPolicy
from flywaydqn.env import ENV_ID, BeamEnv, BeamEnvConfig, load_env_config, make_reward_fn
from flywaydqn.rf.sinr import max_throughput_reference_mbps
from flywaydqn.units import dbm_to_mw

# A small but structurally identical environment: same spaces, same physics,
# reduced horizon and footprint sampling so the suite stays fast.
FAST_STEPS: Final[int] = 40
FAST_EPISODES: Final[int] = 1000


def _fast_config(**overrides: Any) -> BeamEnvConfig:
    base = load_env_config()
    values = base.model_dump()
    values["episode_steps"] = FAST_STEPS
    values["exposure"] = {**values["exposure"], "n_cells": 16}
    values["observation"] = {"users_observed": 4}
    values.update(overrides)
    return BeamEnvConfig.model_validate(values)


def _action(env: BeamEnv, beam_index: int, power_index: int, null_active: bool) -> np.int64:
    # np.int64 because that is what `action_space.sample()` hands the env, and
    # what the declared ActType is.
    return np.int64(
        beam_index * (len(env.config.power_levels_dbm) * 2) + power_index * 2 + int(null_active)
    )


def _rollout(env: BeamEnv, actions: list[np.int64]) -> list[dict[str, Any]]:
    infos: list[dict[str, Any]] = []
    for action in actions:
        _, _, _, _, info = env.step(np.int64(action))
        infos.append(info)
    return infos


def test_shipped_env_yaml_loads_and_matches_the_schema() -> None:
    config = load_env_config()
    assert config.carrier_hz == 3.5e9
    assert config.reward.log10_w_min < config.reward.log10_w_max
    assert config.exposure.s_thr_w_m2 > 0.0
    assert len(config.power_levels_dbm) >= 2


def test_unknown_config_key_is_rejected() -> None:
    values = load_env_config().model_dump()
    values["surprise"] = 1
    with pytest.raises(ValidationError):
        BeamEnvConfig.model_validate(values)


def test_action_space_is_the_flattened_factorisation() -> None:
    """16 azimuth beams x 4 power levels x null on/off (spec Decision D4)."""
    env = BeamEnv(_fast_config(), seed=1)
    action_space = cast(spaces.Discrete, env.action_space)
    assert action_space.n == 16 * len(env.config.power_levels_dbm) * 2
    assert env.n_actions == action_space.n


def test_every_action_index_decodes_and_round_trips() -> None:
    env = BeamEnv(_fast_config(), seed=1)
    seen: set[tuple[int, int, bool]] = set()
    for action in range(env.n_actions):
        decoded = env.decode_action(np.int64(action))
        assert 0 <= decoded.beam_index < 16
        assert 0 <= decoded.power_index < len(env.config.power_levels_dbm)
        assert decoded.power_mw == dbm_to_mw(env.config.power_levels_dbm[decoded.power_index])
        seen.add((decoded.beam_index, decoded.power_index, decoded.null_active))
    assert len(seen) == env.n_actions


def test_decode_rejects_an_action_outside_the_space() -> None:
    env = BeamEnv(_fast_config(), seed=1)
    with pytest.raises(ValueError, match="outside action_space"):
        env.decode_action(np.int64(env.n_actions))


def test_check_env_passes_without_warnings() -> None:
    """Built through ``gymnasium.make`` so the checker sees the registered spec."""
    env = gym.make(ENV_ID, config=_fast_config(), seed=1).unwrapped
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_env(env)
    assert [str(item.message) for item in caught] == []


def test_reset_is_reproducible_for_a_seed() -> None:
    first = BeamEnv(_fast_config(), seed=11)
    second = BeamEnv(_fast_config(), seed=11)
    obs_a, info_a = first.reset(seed=11)
    obs_b, info_b = second.reset(seed=11)
    np.testing.assert_array_equal(obs_a, obs_b)
    assert info_a["log10_w"] == info_b["log10_w"]
    assert info_a["flock_bearing_deg"] == info_b["flock_bearing_deg"]
    actions = [np.int64(value) for value in (0, 5, 17, 63, 127 % first.n_actions)] * 8
    roll_a = _rollout(first, actions)
    roll_b = _rollout(second, actions)
    assert [info["flock_range_m"] for info in roll_a] == [info["flock_range_m"] for info in roll_b]
    assert [info["exposure"]["s_p95"] for info in roll_a] == [
        info["exposure"]["s_p95"] for info in roll_b
    ]


def test_different_seeds_draw_different_episodes() -> None:
    one = BeamEnv(_fast_config(), seed=1)
    two = BeamEnv(_fast_config(), seed=2)
    info_one = one.reset(seed=1)[1]
    info_two = two.reset(seed=2)[1]
    assert info_one["log10_w"] != info_two["log10_w"]
    assert info_one["flock_range_m"] != info_two["flock_range_m"]


def test_observation_is_inside_the_box_every_step() -> None:
    env = BeamEnv(_fast_config(), seed=3)
    observation, _ = env.reset(seed=3)
    assert env.observation_space.contains(observation)
    rng = np.random.default_rng(0)
    for _ in range(FAST_STEPS):
        observation, _, _, _, _ = env.step(np.int64(rng.integers(env.n_actions)))
        assert env.observation_space.contains(observation)
        assert np.all(np.isfinite(observation))


def test_step_returns_the_gymnasium_five_tuple_and_truncates_at_the_horizon() -> None:
    env = BeamEnv(_fast_config(), seed=4)
    env.reset(seed=4)
    for index in range(FAST_STEPS):
        observation, reward, terminated, truncated, info = env.step(np.int64(0))
        assert env.observation_space.contains(observation)
        assert isinstance(reward, float)
        assert math.isfinite(reward)
        assert terminated is False
        assert truncated is (index == FAST_STEPS - 1)
        assert info["step"] == index + 1


def test_random_policy_survives_a_thousand_episodes() -> None:
    env = BeamEnv(_fast_config(), seed=5)
    rng = np.random.default_rng(5)
    episodes = 0
    steps = 0
    while episodes < FAST_EPISODES:
        env.reset(seed=int(rng.integers(1 << 30)))
        truncated = False
        while not truncated:
            _, reward, terminated, truncated, info = env.step(np.int64(rng.integers(env.n_actions)))
            assert math.isfinite(reward)
            assert math.isfinite(info["exposure"]["s_p95"])
            assert not terminated
            steps += 1
        episodes += 1
    assert episodes == FAST_EPISODES
    assert steps == FAST_EPISODES * FAST_STEPS


def test_info_reward_terms_match_an_independently_built_reward_fn() -> None:
    config = _fast_config(episode_steps=8)
    env = BeamEnv(config, seed=6)
    env.reset(seed=6)
    reference = make_reward_fn(
        alpha=config.reward.alpha,
        gamma=config.reward.gamma,
        kappa=config.reward.kappa,
        s_thr=config.exposure.s_thr_w_m2,
        max_tx_power_mw=max(dbm_to_mw(level) for level in config.power_levels_dbm),
        max_throughput_ref_mbps=max(
            1.0,
            max_throughput_reference_mbps(
                config.episode.n_users,
                config.radio.bandwidth_mhz,
                config.radio.spectral_efficiency_cap,
            ),
        ),
        reward_clip=config.reward.reward_clip,
    )
    _, _, _, _, info = env.step(_action(env, 8, 3, False))
    expected = reference(
        sum_throughput_mbps=info["sum_throughput_mbps"],
        tx_power_mw=info["power_mw"],
        s_p95_w_per_m2=info["exposure"]["s_p95"],
        observation={"log10_w": info["log10_w"]},
    )
    for name in ("throughput", "exposure", "power"):
        assert info["reward_terms"][name] == pytest.approx(expected["terms"][name])
    assert info["reward_terms"]["w"] == pytest.approx(info["weight"])


def test_a_lower_power_level_reduces_served_throughput() -> None:
    """Same geometry, same beam, 9 dB less power: the rate must fall."""
    full = BeamEnv(_fast_config(), seed=7)
    low = BeamEnv(_fast_config(), seed=7)
    full.reset(seed=7)
    low.reset(seed=7)
    full_steps: list[np.int64] = []
    low_steps: list[np.int64] = []
    for _ in range(10):
        full_steps.append(_action(full, 8, 3, False))
        low_steps.append(_action(low, 8, 0, False))
    full_info = _rollout(full, full_steps)[-1]
    low_info = _rollout(low, low_steps)[-1]
    assert full_info["power_mw"] > low_info["power_mw"]
    assert full_info["flock_range_m"] == pytest.approx(low_info["flock_range_m"])
    assert full_info["sum_throughput_mbps"] > low_info["sum_throughput_mbps"]


def test_the_null_factor_reduces_footprint_exposure() -> None:
    """Paired runs: mobility is action-independent, so the geometry matches."""
    plain = BeamEnv(_fast_config(), seed=8)
    nulled = BeamEnv(_fast_config(), seed=8)
    plain.reset(seed=8)
    nulled.reset(seed=8)
    plain_actions: list[np.int64] = []
    null_actions: list[np.int64] = []
    for _ in range(12):
        plain_actions.append(_action(plain, 8, 3, False))
        null_actions.append(_action(nulled, 8, 3, True))
    plain_infos = _rollout(plain, plain_actions)
    null_infos = _rollout(nulled, null_actions)
    engaged = [index for index, info in enumerate(null_infos) if info["null_active"]]
    assert engaged, "the null factor never engaged; the env would not exercise it"
    for index in engaged:
        assert null_infos[index]["exposure"]["s_p95"] < plain_infos[index]["exposure"]["s_p95"]
        # The served-rate effect is *not* signed: a projection null redistributes
        # power, so it can raise gain toward some users (spec finding 3). What
        # must hold is that it is not a no-op on the link budgets.
        assert null_infos[index]["sum_throughput_mbps"] != pytest.approx(
            plain_infos[index]["sum_throughput_mbps"]
        )


def test_policy_state_hands_back_the_scene_power_not_the_transient_one() -> None:
    """The rule-based bridge must expose the sector's full power every step.

    A fixed-fraction back-off reads ``setup.tx_power_mw`` as full power and
    multiplies it down. Handing it the power transmitted on the previous step
    instead compounds the reduction every step and pins the baseline at the
    lowest level (200 W -> 100 W -> 50 W -> 25 W -> 25 W...), so the measured
    row would describe a policy nobody configured. The setup is the same one
    ``DemoRunner`` passes, so both surfaces score one policy at one power.
    """
    config = _fast_config()
    env = BeamEnv(config, seed=12)
    env.reset(seed=12)
    full_mw = dbm_to_mw(max(config.power_levels_dbm))

    backoff = StaticBackoffPolicy(backoff_range_m=1e15, backoff_power_frac=0.5)
    setups: list[float] = []
    powers: list[float] = []
    for _ in range(8):
        state, setup = env.policy_state()
        setups.append(setup.tx_power_mw)
        action = backoff.act(state, setup)
        index = env.action_index_for(action.beam.index, action.power_mw, action.null_active)
        powers.append(float(env.step(index)[4]["power_mw"]))

    # The power the policy reads as "full" never drifts...
    assert setups == pytest.approx([full_mw] * 8)
    # ...so the transmitted power is stable and sits at half of it, not at the
    # bottom of the grid.
    assert powers == pytest.approx([powers[0]] * 8)
    assert powers[0] == pytest.approx(full_mw / 2.0, rel=0.02)


def test_the_action_grid_cannot_express_a_sector_shutdown() -> None:
    """Pin the env surface's honest limit: its power grid has no off action.

    ``SectorShutdownPolicy`` asks for 0 mW; the nearest level on the energy
    grid is the *lowest* one, not zero. Anything reporting a shut-down sector
    must therefore be measured on the demo runner, which applies the action's
    power exactly. This test exists so a future change cannot quietly make the
    environment look like it can switch the carrier off.
    """
    env = BeamEnv(_fast_config(), seed=13)
    env.reset(seed=13)
    index = env.action_index_for(0, 0.0, False)
    assert env.decode_action(index).power_index == 0
    assert env.step(index)[4]["power_mw"] == pytest.approx(
        dbm_to_mw(min(env.config.power_levels_dbm))
    )


def test_the_flock_respawns_once_it_leaves_the_sector() -> None:
    """A fast flock crosses the recycle range repeatedly and is re-entered inbound."""
    config = _fast_config(episode_steps=200)
    config.episode.flock_min_range_m = 300.0
    config.episode.flock_max_range_m = 320.0
    config.episode.flock_recycle_range_m = 400.0
    config.episode.flock_speed_ms = 200.0
    # Without speed spread a single mobility step moves the centroid by at most
    # one `step_m`, so a larger change can only be the respawn path.
    config.episode.flock_speed_spread = 0.0
    env = BeamEnv(config, seed=9)
    env.reset(seed=9)
    ranges = [env.step(_action(env, 8, 3, False))[4]["flock_range_m"] for _ in range(199)]
    step_m = config.episode.flock_speed_ms * config.dt_seconds
    # The range can overshoot the recycle threshold by at most one drift step.
    assert max(ranges) <= config.episode.flock_recycle_range_m + step_m + 1e-6
    # A re-entry drops the range back into the spawn band in a single step.
    reentries = [
        (earlier, later)
        for earlier, later in zip(ranges, ranges[1:], strict=False)
        if earlier - later > step_m
    ]
    assert reentries, "the flock never left the sector, so the respawn path was not exercised"
    spawn_low = config.episode.flock_min_range_m
    spawn_high = config.episode.flock_max_range_m
    assert all(spawn_low - 1e-6 <= later <= spawn_high + 1e-6 for _, later in reentries)
