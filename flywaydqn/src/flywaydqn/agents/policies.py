"""Policy registry: map names to BeamPolicy factories (spec WP-A1).

This module is the single place names are resolved. baselines_demo.py keeps the
policy implementations; the registry maps short names to factories that return
BeamPolicy instances configured from PolicyConfig.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from flywaydqn.agents.baselines_demo import (
    AnalyticNullingPolicy,
    BeamPolicy,
    SectorShutdownPolicy,
    StaticBackoffPolicy,
    UnconstrainedPolicy,
)

if TYPE_CHECKING:
    from flywaydqn.demo.runner import PolicyConfig

__all__ = [
    "PolicySpec",
    "UnknownPolicyError",
    "available_policies",
    "available_policy_names",
    "build_policy",
    "policy_name_for",
    "register_policy",
]


class PolicyFactory(Protocol):
    def __call__(self, config: PolicyConfig) -> BeamPolicy: ...


@dataclass(frozen=True)
class PolicySpec:
    name: str
    label: str
    description: str
    factory: PolicyFactory


class UnknownPolicyError(Exception):
    def __init__(self, name: str, known: tuple[str, ...]) -> None:
        msg = f"unknown policy name {name!r}; known: {', '.join(known)}"
        super().__init__(msg)
        self.name = name
        self.known = known


_registry: dict[str, PolicySpec] = {}


def register_policy(spec: PolicySpec) -> None:
    _registry[spec.name] = spec


def available_policies() -> tuple[PolicySpec, ...]:
    return tuple(_registry.values())


def available_policy_names() -> tuple[str, ...]:
    return tuple(sorted(_registry.keys()))


def build_policy(name: str, config: PolicyConfig) -> BeamPolicy:
    spec = _registry.get(name)
    if spec is None:
        raise UnknownPolicyError(name, available_policy_names())
    return spec.factory(config)


_reverse: dict[str, str] = {}


def _register_default() -> None:
    register_policy(
        PolicySpec(
            name="analytic-nulling",
            label="Analytic Null (baseline)",
            description="Guarded analytic nulling that targets the peak exposure cell.",
            factory=lambda cfg: AnalyticNullingPolicy(
                null_max_range_m=cfg.null_max_range_m,
                serving_guard_deg=cfg.serving_guard_deg,
                mirror_guard_deg=cfg.mirror_guard_deg,
                min_exposure_reduction_db=cfg.min_exposure_reduction_db,
                max_rate_loss_frac=cfg.max_rate_loss_frac,
            ),
        )
    )
    register_policy(
        PolicySpec(
            name="unconstrained",
            label="Baseline SON",
            description="Greedy serving beam with no mitigation.",
            factory=lambda _cfg: UnconstrainedPolicy(),
        )
    )
    register_policy(
        PolicySpec(
            name="static-backoff",
            label="Static Back-off",
            description=(
                "Fixed transmit-power reduction while the flock is in range; "
                "else full power. A blanket precautionary baseline."
            ),
            factory=lambda cfg: StaticBackoffPolicy(
                backoff_range_m=cfg.null_max_range_m,
                backoff_power_frac=0.5,
            ),
        )
    )
    register_policy(
        PolicySpec(
            name="sector-shutdown",
            label="Sector Shutdown",
            description=(
                "Transmitter off while the flock is in range; full power otherwise. "
                "A worst-case mitigation lower bound on exposure."
            ),
            factory=lambda cfg: SectorShutdownPolicy(
                shutdown_range_m=cfg.null_max_range_m,
            ),
        )
    )
    _reverse[AnalyticNullingPolicy.__name__] = "analytic-nulling"
    _reverse[UnconstrainedPolicy.__name__] = "unconstrained"
    _reverse[StaticBackoffPolicy.__name__] = "static-backoff"
    _reverse[SectorShutdownPolicy.__name__] = "sector-shutdown"


_register_default()


def policy_name_for(policy: BeamPolicy) -> str:
    name = _reverse.get(policy.__class__.__name__)
    if name is not None:
        return name
    return policy.__class__.__name__


def register_dqn(name: str, label: str, description: str, factory: PolicyFactory) -> None:
    """Register a trained DQN policy by name (WP-E)."""
    register_policy(PolicySpec(name=name, label=label, description=description, factory=factory))
