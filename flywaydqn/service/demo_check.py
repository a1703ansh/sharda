import sys, contextlib, io, json

sys.path.insert(0, "src")
import flywaydqn  # noqa: F401 - ensure package registered first so dataclasses can resolve DemoScene.__module__

import importlib.util

with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
    spec = importlib.util.spec_from_file_location("beam_demo", "service/beam_demo.py")
    be = importlib.util.module_from_spec(spec)
    # Register before exec: dataclasses resolves field types through
    # sys.modules[cls.__module__], which is None if the module is not registered.
    sys.modules["beam_demo"] = be
    spec.loader.exec_module(be)

s0 = be.DemoScene()
s1 = be._step_deterministic(s0, 0, be.DEMO_SEED)
o1 = be._scene_as_dict(s1)
print("beam", o1["servingBeamIndex"] + 1, round(o1["servingBeamAzimuthDeg"], 1), "gain", round(o1["servingBeamGainDbi"], 1))
print("flock", round(o1["flock"]["bearingDeg"], 1), "deg", round(o1["flock"]["rangeM"], 1), "m", o1["flock"]["nBirds"], "birds")
print("null", o1["nullActive"], round(o1["nullDepthDb"], 2) if o1["nullDepthDb"] else 0, "dB")
print("Smax", round(o1["maxDensity"], 4), "frac_over", round(o1["metrics"]["fracOver"], 4))
print("reward", round(o1["reward"]["reward"], 3), o1["reward"])
print("throughput", round(o1["throughputMbps"], 1), "baseline", round(o1["baselineThroughputMbps"], 1))
print("users", len(o1["users"]))
print("json bytes", len(json.dumps(o1)))
print()

s2 = be._step_deterministic(s1, 1, be.DEMO_SEED)
o2 = be._scene_as_dict(s2)
print("step2 flock range", round(o2["flock"]["rangeM"], 1), "m bearing", round(o2["flock"]["bearingDeg"], 1), "deg")
print("step2 Smax", round(o2["maxDensity"], 4), "reward", round(o2["reward"]["reward"], 3))
print("drift ok:", abs(o2["flock"]["rangeM"] - o1["flock"]["rangeM"]) > 0.001)
print("json roundtrip ok:", json.loads(json.dumps(o1)) == o1)
