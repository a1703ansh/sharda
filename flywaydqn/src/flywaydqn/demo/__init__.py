"""Canned demo scenarios and the deterministic simulator behind them.

The FastAPI service defaults to :func:`demo_runner_factory`, which loads one
of the YAML scenarios under ``configs/demo`` and steps it with the rule-based
policies from :mod:`flywaydqn.agents.baselines_demo`.
"""

from flywaydqn.demo.runner import (
    CONFIG_DIR_ENV_VAR,
    ArrayConfig,
    CodebookConfig,
    DemoRunner,
    ExposureConfig,
    FlockConfig,
    PolicyConfig,
    RadioConfig,
    RewardConfig,
    ScenarioConfig,
    UsersConfig,
    demo_config_dir,
    demo_runner_factory,
    list_demo_scenarios,
    load_scenario,
    scenario_path,
)

__all__ = [
    "CONFIG_DIR_ENV_VAR",
    "ArrayConfig",
    "CodebookConfig",
    "DemoRunner",
    "ExposureConfig",
    "FlockConfig",
    "PolicyConfig",
    "RadioConfig",
    "RewardConfig",
    "ScenarioConfig",
    "UsersConfig",
    "demo_config_dir",
    "demo_runner_factory",
    "list_demo_scenarios",
    "load_scenario",
    "scenario_path",
]
