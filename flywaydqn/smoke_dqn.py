"""Ad-hoc live smoke test for the served DQN controller (not part of the suite)."""

import math

from flywaydqn.agents.policies import available_policy_names
from flywaydqn.demo import DemoRunner
from flywaydqn.service.app import create_app
from flywaydqn.service.schemas import ScenarioRequest


def _finite(value: object) -> bool:
    if isinstance(value, dict):
        return all(_finite(v) for v in value.values())
    if isinstance(value, list):
        return all(_finite(v) for v in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


create_app()
print("policies:", available_policy_names())

runner = DemoRunner(ScenarioRequest(key="scenario_002", seed=7, weight=1.0, policy="dqn"))
frames = [runner.initial_frame()] + [runner.step() for _ in range(40)]
for frame in frames[::10]:
    print(
        frame.step,
        frame.policy_name,
        f"{frame.inference_latency_ms:.2f} ms",
        "null" if frame.null_active else "-",
        (frame.explanation or "")[:120],
    )
dumped = [frame.model_dump(mode="json") for frame in frames]
print("all finite:", all(_finite(d) for d in dumped))
print("nulls seen:", sum(f.null_active for f in frames))
print("beams seen:", sorted({f.beam.index for f in frames}))
print("powers mW:", sorted({round(f.power_mw) for f in frames}))
