"""Trained-policy artifacts: where a checkpoint lives and how it is loaded.

One trained agent is a directory:

```
experiments/checkpoints/<run>/
├── dqn_policy.zip    # Stable-Baselines3 checkpoint (the network + its spaces)
└── meta.json         # what produced it: seed, steps, spaces, hyperparameters
```

``meta.json`` is the contract: the service loads a checkpoint only when its
recorded observation and action sizes match the environment it is being served
into, so a checkpoint trained against a different action set fails loudly instead
of silently mis-indexing actions.

Units: none (this module moves files and reads metadata).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from stable_baselines3 import DQN

CHECKPOINT_FILENAME: Final[str] = "dqn_policy.zip"
META_FILENAME: Final[str] = "meta.json"


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is missing or incompatible with the environment."""


@dataclass(frozen=True)
class CheckpointMeta:
    """What one trained checkpoint records about itself."""

    run_name: str
    seed: int
    total_steps: int
    observation_size: int
    n_actions: int
    hyperparameters: dict[str, Any]
    train_metrics: dict[str, float]
    env_config: dict[str, Any]

    def to_json(self) -> str:
        """Serialize as JSON with a stable key order."""
        return json.dumps(
            {
                "run_name": self.run_name,
                "seed": self.seed,
                "total_steps": self.total_steps,
                "observation_size": self.observation_size,
                "n_actions": self.n_actions,
                "hyperparameters": self.hyperparameters,
                "train_metrics": self.train_metrics,
                "env_config": self.env_config,
            },
            indent=2,
            sort_keys=True,
        )


def save_checkpoint(model: DQN, directory: Path, meta: CheckpointMeta) -> Path:
    """Write ``model`` and ``meta`` under ``directory`` and return the directory."""
    directory.mkdir(parents=True, exist_ok=True)
    model.save(str(directory / CHECKPOINT_FILENAME))
    (directory / META_FILENAME).write_text(meta.to_json(), encoding="utf-8")
    return directory


def load_meta(directory: Path) -> CheckpointMeta:
    """Read a checkpoint's metadata, or raise :class:`CheckpointError`."""
    meta_path = directory / META_FILENAME
    if not meta_path.is_file():
        raise CheckpointError(f"no checkpoint metadata at {meta_path}")
    raw: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
    return CheckpointMeta(
        run_name=str(raw["run_name"]),
        seed=int(raw["seed"]),
        total_steps=int(raw["total_steps"]),
        observation_size=int(raw["observation_size"]),
        n_actions=int(raw["n_actions"]),
        hyperparameters=dict(raw["hyperparameters"]),
        train_metrics={key: float(value) for key, value in raw["train_metrics"].items()},
        env_config=dict(raw["env_config"]),
    )


def load_model(directory: Path | str, *, observation_size: int, n_actions: int) -> DQN:
    """Load a checkpoint, refusing one that does not match the live spaces.

    Parameters
    ----------
    directory : Path
        Checkpoint directory written by :func:`save_checkpoint`.
    observation_size : int
        Observation length of the environment the policy will be served into.
    n_actions : int
        Action count of that environment.

    Raises
    ------
    CheckpointError
        If the checkpoint is missing, or its recorded spaces do not match.
    """
    model_path = Path(directory) / CHECKPOINT_FILENAME
    if not model_path.is_file():
        raise CheckpointError(f"no checkpoint at {model_path}")
    meta = load_meta(Path(directory))
    if meta.observation_size != observation_size or meta.n_actions != n_actions:
        raise CheckpointError(
            f"checkpoint {directory} was trained for observation_size="
            f"{meta.observation_size}, n_actions={meta.n_actions}; the environment "
            f"expects {observation_size} and {n_actions}"
        )
    return DQN.load(str(model_path))


__all__ = [
    "CHECKPOINT_FILENAME",
    "META_FILENAME",
    "CheckpointError",
    "CheckpointMeta",
    "load_meta",
    "load_model",
    "save_checkpoint",
]
