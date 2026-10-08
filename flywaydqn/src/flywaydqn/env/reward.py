"""Reward construction for BeamEnv.

Spec (Section 4.4):

    r_t = alpha * T_hat - beta * E_hat - gamma * P_hat

with ``alpha = 1`` by default, ``gamma`` per config, and ``beta = alpha * w``
where ``w`` is the preference weight sampled per episode (Decision D2).
``E_hat`` is the softplus hinge penalty from
:func:`flywaydqn.exposure.thresholds.exposure_penalty_softplus` (Decision D3).

All reward terms are normalized to comparable ranges. The three terms are logged
separately every step. Per-step reward is clipped to ``[-10, 10]``.

Units
-----
- throughput: megabit per second (Mbps) for the raw value; ``T_hat`` is a
  normalized ratio and therefore dimensionless
- power: milliwatt (mW) for the raw value; ``P_hat`` is a normalized ratio
- exposure: ``E_hat`` is dimensionless, built from a W/m^2 density over a
  W/m^2 threshold
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict

import numpy as np

from flywaydqn.exposure.thresholds import exposure_penalty_softplus


class RewardTerms(TypedDict):
    """The three separately-logged reward terms, plus the active weight."""

    throughput: float
    exposure: float
    power: float
    w: float


class RewardResult(TypedDict):
    """Return value of a configured reward function."""

    reward: float
    terms: RewardTerms


RewardFunction = Callable[..., RewardResult]


def make_reward_fn(
    *,
    alpha: float = 1.0,
    gamma: float = 0.02,
    kappa: float = 5.0,
    s_thr: float = 10.0,
    max_tx_power_mw: float = 100.0,
    max_throughput_ref_mbps: float = 1000.0,
    reward_clip: float = 10.0,
) -> RewardFunction:
    """Return a configured reward function.

    The returned function takes the raw per-step quantities plus the current
    observation carrying the preference weight, computes the clipped reward, and
    returns the reward together with the separately-logged terms.

    Parameters
    ----------
    alpha : float
        Throughput weight. Default 1.0.
    gamma : float
        Power weight. Default 0.02.
    kappa : float
        Softplus kappa for the exposure penalty. Default 5.0 (Decision D3).
    s_thr : float
        Exposure threshold in W/m^2. Default 10.0, the ICNIRP 2020
        general-public reference level for 2-300 GHz (a configurable anchor, not
        an avian-harm threshold: spec Section 2.2).
    max_tx_power_mw : float
        Normalization reference for transmit power, in mW.
    max_throughput_ref_mbps : float
        Normalization reference for sum throughput, in Mbps.
    reward_clip : float
        Per-step reward clip magnitude. Default 10.0.

    Returns
    -------
    RewardFunction
        Callable returning ``RewardResult``.
    """
    if alpha <= 0.0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    if gamma < 0.0:
        raise ValueError(f"gamma must be >= 0, got {gamma}")
    if max_tx_power_mw <= 0.0:
        raise ValueError(f"max_tx_power_mw must be > 0, got {max_tx_power_mw}")
    if max_throughput_ref_mbps <= 0.0:
        raise ValueError(f"max_throughput_ref_mbps must be > 0, got {max_throughput_ref_mbps}")

    def reward_fn(
        *,
        sum_throughput_mbps: float,
        tx_power_mw: float,
        s_p95_w_per_m2: float,
        observation: dict[str, float],
    ) -> RewardResult:
        """Compute the clipped reward and its terms for one step.

        Parameters
        ----------
        sum_throughput_mbps : float
            Sum cell throughput this step, in Mbps.
        tx_power_mw : float
            Transmit power this step, in mW.
        s_p95_w_per_m2 : float
            95th-percentile power density over flock cells, in W/m^2.
        observation : dict of str to float
            Observation carrying ``log10_w``, the preference weight on a log10
            scale (Decision D2).

        Returns
        -------
        RewardResult
            ``{"reward": float, "terms": RewardTerms}``.
        """
        log10_w = float(observation.get("log10_w", 0.0))
        w = 10.0**log10_w
        beta = alpha * w

        t_hat = sum_throughput_mbps / max_throughput_ref_mbps
        p_hat = tx_power_mw / max_tx_power_mw
        e_hat = exposure_penalty_softplus(s_p95_w_per_m2, s_thr, kappa=kappa)

        raw_reward = alpha * t_hat - beta * e_hat - gamma * p_hat
        reward = float(np.clip(raw_reward, -reward_clip, reward_clip))

        return RewardResult(
            reward=reward,
            terms=RewardTerms(
                throughput=float(t_hat),
                exposure=float(e_hat),
                power=float(p_hat),
                w=float(w),
            ),
        )

    return reward_fn


__all__ = ["RewardFunction", "RewardResult", "RewardTerms", "make_reward_fn"]
