"""DQN training and evaluation entry point (spec Phase 5).

Trains a preference-conditioned Double DQN on
:class:`~flywaydqn.env.beam_env.BeamEnv`, then scores the trained policy and the
rule-based baselines -- unconstrained, static back-off, sector shutdown and
analytic nulling -- on a shared set of evaluation seeds, so every number in
``metrics.csv`` comes from a measured rollout of the same physics.

Outputs, under ``experiments/checkpoints/<run_name>/``:

- ``dqn_policy.zip`` + ``meta.json`` per seed (see
  :mod:`flywaydqn.agents.policy_io`)
- ``episodes.csv``: per-episode return and length for the learning curve
- ``learning_curve.png``: mean return across seeds with the min/max band
- ``metrics.csv`` + ``summary.md``: the policy-versus-baseline comparison. Every
  trained seed gets its own row (``dqn (seed N)``) so the spread across seeds is
  visible instead of hiding behind a single best-of number.

Three disjoint seed sets are kept apart on purpose: ``seeds`` train, then
``selection_seeds`` choose which trained checkpoint is the one to quote, and
``eval_seeds`` are held out for the reporting table only. Selecting on the
training return would pick a checkpoint for how well it fit the runs it saw.

Run it with::

    python -m flywaydqn.agents.train_dqn
    python -m flywaydqn.agents.train_dqn --total-steps 200000 --seeds 1 2 3 4 5
    python -m flywaydqn.agents.train_dqn --eval-only   # refresh the table only

Units: throughput Mbit/s, power mW, power density W/m^2, dose W*s/m^2, time s.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from flywaydqn.agents.baselines_demo import (
    AnalyticNullingPolicy,
    BeamPolicy,
    StaticBackoffPolicy,
    UnconstrainedPolicy,
)
from flywaydqn.agents.policy_io import (
    CheckpointError,
    CheckpointMeta,
    load_model,
    save_checkpoint,
)
from flywaydqn.env import BeamEnv, BeamEnvConfig, load_env_config

CONFIG_ENV_VAR: Final[str] = "FLYWAYDQN_TRAIN_CONFIG"

# Metrics recorded per step and aggregated per episode.
_STEP_METRICS: Final[tuple[str, ...]] = (
    "sum_throughput_mbps",
    "p95_density_wm2",
    "dose_wsm2",
    "frac_over",
    "power_mw",
    "null_fraction",
)


class DQNHyperparameters(BaseModel):
    """Stable-Baselines3 DQN hyperparameters (spec Phase 5)."""

    model_config = ConfigDict(extra="forbid")

    learning_rate: float = Field(5e-4, gt=0.0)
    buffer_size: int = Field(50_000, ge=1000)
    learning_starts: int = Field(1000, ge=0)
    batch_size: int = Field(64, ge=8)
    gamma: float = Field(0.99, gt=0.0, le=1.0)
    tau: float = Field(1.0, gt=0.0, le=1.0)
    target_update_interval: int = Field(500, ge=1)
    train_freq: int = Field(4, ge=1)
    gradient_steps: int = Field(1, ge=1)
    exploration_fraction: float = Field(0.3, ge=0.0, le=1.0)
    exploration_final_eps: float = Field(0.05, ge=0.0, le=1.0)
    double_q: bool = True
    net_arch: tuple[int, ...] = (128, 128)


class TrainingConfig(BaseModel):
    """Everything one training run needs, mirroring ``configs/train/train.yaml``."""

    model_config = ConfigDict(extra="forbid")

    run_name: str = Field("dqn_first_pass", min_length=1)
    seeds: tuple[int, ...] = (1, 2, 3)
    total_steps: int = Field(40_000, ge=1)
    eval_seeds: tuple[int, ...] = (101, 102, 103, 104, 105, 106, 107, 108)
    selection_seeds: tuple[int, ...] = (11, 12, 13, 14)
    eval_episode_steps: int = Field(300, ge=1)
    checkpoint_root: Path = Path("experiments/checkpoints")
    device: str = "cpu"
    dqn: DQNHyperparameters = Field(default_factory=DQNHyperparameters)

    @model_validator(mode="after")
    def _splits_stay_disjoint(self) -> TrainingConfig:
        """Refuse a config whose training, selection and held-out seeds overlap."""
        overlap = sorted(set(self.seeds) & set(self.selection_seeds))
        if overlap:
            raise ValueError(f"selection_seeds overlap the training seeds: {overlap}")
        leaked = sorted(set(self.eval_seeds) & (set(self.seeds) | set(self.selection_seeds)))
        if leaked:
            raise ValueError(f"eval_seeds must be held out, but overlap another split: {leaked}")
        return self


def train_config_path() -> Path:
    """Return the training YAML: the override variable, then the repository copy."""
    override = os.environ.get(CONFIG_ENV_VAR)
    if override:
        return Path(override)
    from_source = Path(__file__).resolve().parents[3] / "configs" / "train" / "train.yaml"
    if from_source.is_file():
        return from_source
    return Path.cwd() / "configs" / "train" / "train.yaml"


def load_training_config(path: Path | None = None) -> TrainingConfig:
    """Load and validate the training config, defaulting to ``configs/train/train.yaml``."""
    target = path if path is not None else train_config_path()
    with target.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return TrainingConfig.model_validate(raw)


class EpisodeCurveCallback(BaseCallback):
    """Record each finished episode's return so the run has a learning curve."""

    def __init__(self) -> None:
        super().__init__()
        self.episode_returns: list[float] = []
        self.episode_lengths: list[int] = []
        self.episode_end_steps: list[int] = []
        self._running_return = 0.0
        self._running_length = 0

    def _on_step(self) -> bool:
        """Accumulate the step reward and close an episode when it ends."""
        rewards = self.locals.get("rewards")
        if rewards is not None:
            self._running_return += float(np.asarray(rewards).sum())
        self._running_length += 1
        dones = self.locals.get("dones")
        if dones is not None and bool(np.asarray(dones).any()):
            self.episode_returns.append(self._running_return)
            self.episode_lengths.append(self._running_length)
            self.episode_end_steps.append(int(self.num_timesteps))
            self._running_return = 0.0
            self._running_length = 0
        return True


@dataclass(frozen=True)
class EpisodeOutcome:
    """Aggregated metrics for one evaluated episode."""

    mean_throughput_mbps: float
    mean_p95_wm2: float
    max_p95_wm2: float
    total_dose_wsm2: float
    mean_frac_over: float
    mean_power_mw: float
    null_fraction: float
    mean_reward: float


def _observation_size(env: BeamEnv) -> int:
    """Return the environment's observation length (its ``Box`` always has one)."""
    shape = env.observation_space.shape
    assert shape is not None
    return int(np.prod(shape))


def _train_one(
    config: TrainingConfig,
    env_config: BeamEnvConfig,
    seed: int,
    directory: Path,
) -> tuple[DQN, EpisodeCurveCallback]:
    """Train one seeded Double DQN and write its checkpoint and curve."""
    env = BeamEnv(env_config, seed=seed)
    callback = EpisodeCurveCallback()
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=config.dqn.learning_rate,
        buffer_size=config.dqn.buffer_size,
        learning_starts=config.dqn.learning_starts,
        batch_size=config.dqn.batch_size,
        gamma=config.dqn.gamma,
        tau=config.dqn.tau,
        target_update_interval=config.dqn.target_update_interval,
        train_freq=config.dqn.train_freq,
        gradient_steps=config.dqn.gradient_steps,
        exploration_fraction=config.dqn.exploration_fraction,
        exploration_final_eps=config.dqn.exploration_final_eps,
        policy_kwargs={"net_arch": list(config.dqn.net_arch)},
        seed=seed,
        device=config.device,
        verbose=0,
    )
    model.learn(total_timesteps=config.total_steps, callback=callback, progress_bar=False)
    episodes: list[float] = callback.episode_returns
    meta = CheckpointMeta(
        run_name=config.run_name,
        seed=seed,
        total_steps=config.total_steps,
        observation_size=_observation_size(env),
        n_actions=env.n_actions,
        hyperparameters=config.dqn.model_dump(),
        train_metrics={
            "episodes": float(len(episodes)),
            "mean_episode_return": float(statistics.fmean(episodes)) if episodes else 0.0,
        },
        env_config=env_config.model_dump(mode="json"),
    )
    save_checkpoint(model, directory, meta)
    return model, callback


def _evaluate_policy(
    env_config: BeamEnvConfig,
    action_fn: ActionFn,
    seeds: Sequence[int],
    steps: int,
) -> EpisodeOutcome:
    """Roll one policy out over ``seeds`` and aggregate the per-step metrics."""
    env = BeamEnv(env_config, seed=int(seeds[0]))
    sums: dict[str, float] = dict.fromkeys(_STEP_METRICS, 0.0)
    counts = 0
    max_p95 = 0.0
    total_dose = 0.0
    reward_sum = 0.0
    for seed in seeds:
        observation, _ = env.reset(seed=int(seed))
        for _ in range(steps):
            action = action_fn(env, observation)
            observation, reward, _, truncated, info = env.step(action)
            reward_sum += float(reward)
            exposure = info["exposure"]
            sums["sum_throughput_mbps"] += float(info["sum_throughput_mbps"])
            sums["p95_density_wm2"] += float(exposure["s_p95"])
            sums["dose_wsm2"] += float(exposure["dose"])
            sums["frac_over"] += float(exposure["frac_over"])
            sums["power_mw"] += float(info["power_mw"])
            sums["null_fraction"] += 1.0 if info["null_active"] else 0.0
            max_p95 = max(max_p95, float(exposure["s_max"]))
            total_dose += float(exposure["dose"])
            counts += 1
            if truncated:
                break
    divisor = max(1, counts)
    return EpisodeOutcome(
        mean_throughput_mbps=sums["sum_throughput_mbps"] / divisor,
        mean_p95_wm2=sums["p95_density_wm2"] / divisor,
        max_p95_wm2=max_p95,
        total_dose_wsm2=total_dose,
        mean_frac_over=sums["frac_over"] / divisor,
        mean_power_mw=sums["power_mw"] / divisor,
        null_fraction=sums["null_fraction"] / divisor,
        mean_reward=reward_sum / divisor,
    )


ActionFn = Callable[[BeamEnv, npt.NDArray[np.float64]], np.int64]
"""Picks an action index from the environment's current observation."""


def _learned_action_fn(model: DQN) -> ActionFn:
    """Greedy policy over the trained Q-network."""

    def action_fn(_: BeamEnv, observation: npt.NDArray[np.float64]) -> np.int64:
        action, _state = model.predict(observation, deterministic=True)
        return np.int64(np.asarray(action).reshape(-1)[0])

    return action_fn


def _select_seed_by_validation(
    env_config: BeamEnvConfig,
    models: dict[int, DQN],
    config: TrainingConfig,
) -> tuple[int, dict[int, float]]:
    """Score every trained seed on the validation split and pick the best.

    The score is the environment's own mean reward over ``selection_seeds`` --
    seeds neither trained on nor used for the reported table -- so the checkpoint
    that gets quoted is chosen by measured performance on unseen scenes rather
    than by the return it achieved on its own training rollouts.
    """
    scores: dict[int, float] = {}
    for seed, model in sorted(models.items()):
        outcome = _evaluate_policy(
            env_config,
            _learned_action_fn(model),
            config.selection_seeds,
            config.eval_episode_steps,
        )
        scores[seed] = outcome.mean_reward
    # sorted() first, so a tie resolves to the lowest seed instead of to
    # dict-iteration order.
    best = max(sorted(scores), key=lambda candidate: scores[candidate])
    return best, scores


def _evaluation_rows(
    env_config: BeamEnvConfig,
    models: dict[int, DQN],
    config: TrainingConfig,
) -> list[dict[str, Any]]:
    """Score every trained seed and every baseline on the held-out seeds."""
    evaluations: list[tuple[str, ActionFn]] = [
        (f"dqn (seed {seed})", _learned_action_fn(model)) for seed, model in sorted(models.items())
    ]
    # ``sector-shutdown`` is deliberately absent. It transmits 0 mW, and this
    # environment's power grid has no off action: the bridge would snap zero
    # onto the *lowest* non-zero level and the row would read "sector shutdown"
    # while radiating 25 W. It is measured on the demo runner instead, which
    # applies ``BeamAction.power_mw`` exactly, by
    # ``experiments/run_baselines.py``.
    evaluations += [
        ("unconstrained", _rule_based_action_fn(UnconstrainedPolicy())),
        ("static-backoff", _rule_based_action_fn(StaticBackoffPolicy())),
        ("analytic-nulling", _rule_based_action_fn(AnalyticNullingPolicy())),
        ("random", _random_action_fn(0)),
    ]
    rows: list[dict[str, Any]] = []
    for name, action_fn in evaluations:
        outcome = _evaluate_policy(
            env_config, action_fn, config.eval_seeds, config.eval_episode_steps
        )
        rows.append({"policy": name, **outcome.__dict__})
        print(
            f"[eval] {name}: sum rate {outcome.mean_throughput_mbps:.0f} Mbps, "
            f"dose {outcome.total_dose_wsm2:.4f} W·s/m², "
            f"max S {outcome.max_p95_wm2:.5f} W/m²"
        )
    return rows


def _rule_based_action_fn(policy: BeamPolicy) -> ActionFn:
    """Wrap a rule-based policy so it acts through the environment's action space."""

    def action_fn(env: BeamEnv, _: npt.NDArray[np.float64]) -> np.int64:
        state, setup = env.policy_state()
        decision = policy.act(state, setup)
        return env.action_index_for(decision.beam.index, decision.power_mw, decision.null_active)

    return action_fn


def _random_action_fn(seed: int) -> ActionFn:
    """Uniformly random policy, as a floor on the comparison."""
    rng = np.random.default_rng(seed)

    def action_fn(env: BeamEnv, _: npt.NDArray[np.float64]) -> np.int64:
        return np.int64(rng.integers(env.n_actions))

    return action_fn


def _write_curve(directory: Path, curves: dict[int, EpisodeCurveCallback]) -> None:
    """Write the per-episode curve as CSV, and a mean/min/max PNG when possible."""
    with (directory / "episodes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "episode", "timestep", "return", "length"])
        for seed, callback in sorted(curves.items()):
            for index, (ret, length, step) in enumerate(
                zip(
                    callback.episode_returns,
                    callback.episode_lengths,
                    callback.episode_end_steps,
                    strict=False,
                )
            ):
                writer.writerow([seed, index, step, ret, length])

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - plotting is a dev extra
        (directory / "learning_curve.note").write_text(
            "matplotlib is not installed, so learning_curve.png was not written; "
            "episodes.csv holds the same series.\n",
            encoding="utf-8",
        )
        return

    max_len = max((len(c.episode_returns) for c in curves.values()), default=0)
    if max_len == 0:
        return
    matrix = np.full((len(curves), max_len), np.nan)
    for row, callback in enumerate(curves.values()):
        matrix[row, : len(callback.episode_returns)] = callback.episode_returns

    mean = np.nanmean(matrix, axis=0)
    low = np.nanmin(matrix, axis=0)
    high = np.nanmax(matrix, axis=0)
    figure, axes = plt.subplots(figsize=(7.0, 3.5), dpi=140)
    axes.fill_between(range(max_len), low, high, alpha=0.2, color="#0d9488", label="seed range")
    axes.plot(range(max_len), mean, color="#0f766e", linewidth=1.8, label="seed mean")
    axes.set_xlabel("episode")
    axes.set_ylabel("episode return")
    axes.set_title(f"Double DQN learning curve ({len(curves)} seeds)")
    axes.grid(alpha=0.25)
    axes.legend(loc="lower right", fontsize=8)
    figure.tight_layout()
    figure.savefig(directory / "learning_curve.png")
    plt.close(figure)


def _write_comparison(
    directory: Path,
    rows: list[dict[str, Any]],
    notes: Sequence[str] = (),
) -> None:
    """Write the policy-versus-baseline table as CSV and markdown."""
    fields = [
        "policy",
        "mean_throughput_mbps",
        "total_dose_wsm2",
        "max_p95_wm2",
        "mean_frac_over",
        "mean_power_mw",
        "null_fraction",
    ]
    with (directory / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})

    header = (
        "| policy | mean sum rate (Mbps) | total dose (W·s/m²) | max S (W/m²) "
        "| mean frac over | mean power (mW) | null fraction |"
    )
    divider = "|---|---|---|---|---|---|---|"
    lines = [header, divider]
    for row in rows:
        lines.append(
            f"| {row['policy']} | {row['mean_throughput_mbps']:.0f} | "
            f"{row['total_dose_wsm2']:.4f} | {row['max_p95_wm2']:.5f} | "
            f"{row['mean_frac_over']:.3f} | {row['mean_power_mw']:.0f} | "
            f"{row['null_fraction']:.3f} |"
        )
    (directory / "summary.md").write_text(
        "\n".join(lines + [""] + [f"- {note}" for note in notes]) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Train, evaluate and write every artifact; return a process exit code."""
    parser = argparse.ArgumentParser(description="Train and evaluate the FlywayDQN controller.")
    parser.add_argument("--config", type=Path, default=None, help="training YAML path")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--env-config", type=Path, default=None, help="env YAML path")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="skip training and rebuild metrics.csv/summary.md from stored checkpoints",
    )
    args = parser.parse_args(argv)

    config = load_training_config(args.config)
    if args.run_name:
        config = config.model_copy(update={"run_name": args.run_name})
    if args.total_steps:
        config = config.model_copy(update={"total_steps": args.total_steps})
    if args.seeds:
        config = config.model_copy(update={"seeds": tuple(args.seeds)})
    env_config = load_env_config(args.env_config)
    if env_config.episode_steps != 600:
        print(f"note: env episode horizon is {env_config.episode_steps} steps")

    directory = config.checkpoint_root / config.run_name
    directory.mkdir(parents=True, exist_ok=True)

    models: dict[int, DQN] = {}
    if args.eval_only:
        # The training is deterministic per seed, so re-training purely to refresh a
        # table would burn the same CPU for an identical network.
        probe = BeamEnv(env_config, seed=int(config.seeds[0]))
        for seed in config.seeds:
            seed_dir = directory / f"seed_{seed}"
            try:
                models[seed] = load_model(
                    seed_dir,
                    observation_size=_observation_size(probe),
                    n_actions=probe.n_actions,
                )
            except CheckpointError as error:
                print(f"[eval-only] cannot load seed {seed}: {error}")
                return 1
            print(f"[eval-only] loaded seed {seed} from {seed_dir}")
    else:
        (directory / "train_config.json").write_text(
            config.model_dump_json(indent=2), encoding="utf-8"
        )
        curves: dict[int, EpisodeCurveCallback] = {}
        for seed in config.seeds:
            print(f"[train] seed {seed}: {config.total_steps} steps")
            model, callback = _train_one(config, env_config, seed, directory / f"seed_{seed}")
            curves[seed] = callback
            models[seed] = model
            mean_return = (
                statistics.fmean(callback.episode_returns)
                if callback.episode_returns
                else -math.inf
            )
            print(
                f"[train] seed {seed}: episodes={len(callback.episode_returns)} "
                f"mean_return={mean_return:.3f}"
            )
        _write_curve(directory, curves)

    best_seed, scores = _select_seed_by_validation(env_config, models, config)
    for seed, score in sorted(scores.items()):
        print(f"[select] seed {seed}: validation mean reward {score:.3f}")
    print(
        f"[select] seed {best_seed} is the one to quote "
        f"(validation seeds {list(config.selection_seeds)})"
    )

    rows = _evaluation_rows(env_config, models, config)
    _write_comparison(
        directory,
        rows,
        notes=[
            f"Validation split {list(config.selection_seeds)} selected seed {best_seed} "
            f"(scores: {', '.join(f'seed {s} {v:.3f}' for s, v in sorted(scores.items()))}).",
            f"The {len(config.selection_seeds)} validation seeds and the "
            f"{len(config.eval_seeds)} reporting seeds are disjoint from the "
            f"{len(config.seeds)} training seeds.",
            "Every row is a measured rollout on the reporting seeds; the dqn rows are "
            "per-seed, so the spread across seeds is visible instead of a single best number.",
            "sector-shutdown is not in this table: it transmits 0 mW and the "
            "environment's power grid has no off action, so the bridge could only "
            "snap it onto the lowest non-zero level. Its measured row lives in "
            "experiments/results/baselines.md, taken on the demo runner.",
        ],
    )
    print(f"[done] artifacts under {directory}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
