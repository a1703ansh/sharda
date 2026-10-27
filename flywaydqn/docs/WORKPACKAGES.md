# FlywayDQN — Work Packages for a Judged, Zero-Fabrication Demo

**Purpose.** Split the remaining work into packages that can be handed to separate
agent threads. Every package below has one owner, a bounded file set, and an
acceptance check that can be run without the other packages.

**Rule that governs all of them.** Nothing displayed to a judge may be a literal
that is not produced by the running system. If a value cannot be computed yet,
the UI must show "not available", not a plausible number.

---

## 0. Dependency graph

```
WP-A1  wire contract (threshold, null depth, latency, policy)   <-- MUST LAND FIRST
  ├── WP-A2  dashboard honesty pass
  ├── WP-F   real timeline + stat tiles
  └── WP-B   Gymnasium BeamEnv
        ├── WP-C  baselines + measured table
        └── WP-D  DQN training -> checkpoint
              └── WP-E  serve trained policy + real inspector
WP-G  packaging/repro (independent of all of the above)
```

Only WP-A1 blocks others. WP-G can run at any time.

---

## 1. Frozen interfaces (do not change without updating this section)

These are the seams the packages communicate through. A change here invalidates
other packages, so they are frozen now.

### 1.1 `StateFrame` (wire → UI), `SCHEMA_VERSION = "3"`

Added in WP-A1, on top of the existing frame:

| Field | Type | Meaning |
|---|---|---|
| `metrics.sThrWm2` | `float` | The scenario's exposure threshold, in W/m², taken from the scenario config. This is the only threshold a consumer may display. |
| `nullDepthDb` | `float` | Measured exposure reduction the transmitted weights achieved against the same-scenario un-nulled beam: `10·log10(p95_un-nulled / p95_transmitted)`, same footprint cells, same step. Exactly `0.0` when no null is transmitted. |
| `inferenceLatencyMs` | `float` | Server-measured wall-clock time of the policy decision for this step, in ms. |

`SCHEMA_VERSION = "2"` then became `"3"` with:

| Field / route | Type | Meaning |
|---|---|---|
| `baseline` | `BaselineMetrics` | The same step scored *without* mitigation from the same live geometry (the scenario's `unconstrained` reference): `policyName`, `powerMw`, `sumThroughputMbps`, `p95DensityWm2`, `maxDensityWm2`, `fracOver`, `doseWsm2`. A consumer shows before/after from one stream, no second session to align. |
| `POST /sessions/{id}/pace` | route | Retunes a live stream's wall-clock pacing (presentation only; `dtS` and every measured value stay identical). |

`GET /policies/dqn/meta` (WP-E) is an additive read-only route returning the
checkpoint's `meta.json`; it does not change the frame contract, so the
version stays `"3"`.

Existing fields that stay authoritative: `beam.*`, `nullActive`, `powerMw`,
`flock.*`, `users[].{sinr,throughputMbps}`, `metrics.{sumThroughputMbps,
p95DensityWm2,maxDensityWm2,fracOver,doseWsm2,exposurePenalty,throughputTerm,
powerTerm,reward}`, `explanation`, `weight`.

The UI must delete its local constants for the threshold (`5.0`), null depth
(`-26.4`), and latency (`0`) and read these instead.

### 1.2 Session request (UI → service)

`ScenarioRequest` gains `policy: str | None = None`.

* `null` → the scenario YAML's own policy (unchanged behaviour).
* `"unconstrained"` → `UnconstrainedPolicy` (what the UI calls **Baseline SON**).
* `"analytic-nulling"` → `AnalyticNullingPolicy` (shown in the UI as **Analytic Null (baseline)**).
* Any other string → `422`/`404` with the allowed names in the message.

`GET /policies` returns `[{"name", "label", "description"}]` for the policy
registry, so the UI can populate the engine toggle from the server instead of
hardcoding two buttons.

### 1.3 Policy registry (backend)

One module, `flywaydqn/agents/policies.py`, maps `name -> BeamPolicy factory`.
`baselines_demo.py` keeps the policy implementations; the registry is the only
place names are resolved. `PolicyConfig.name` in the scenario YAML stays the
default when the request does not override.

### 1.4 Trained-policy artifact

`agents/policy_io.py` owns load/save (implemented). One run is a directory
`experiments/checkpoints/<run>/` holding per-seed subdirectories with
`dqn_policy.zip` (the Stable-Baselines3 checkpoint) and `meta.json`
(`{run_name, seed, total_steps, observation_size, n_actions, hyperparameters,
train_metrics, env_config}`), alongside `episodes.csv`, `learning_curve.png`,
`metrics.csv` and `summary.md` for the run as a whole.

`load_model` refuses a checkpoint whose recorded `observation_size`/`n_actions`
do not match the live environment, so a stale artifact fails loudly instead of
mis-indexing actions. WP-E adds the registry entry that serves it as policy name
`"dqn"`.

Note: Stable-Baselines3's `DQN` implements Double Q-learning but **not** a dueling
head, so anything described as "Dueling DQN" in the UI must either be implemented
explicitly or corrected to "Double DQN".

Also resolved while building this: the shipped `configs/env/env.yaml` used
`s_thr_w_m2: 10.0` (the ICNIRP anchor). At that threshold the modelled footprint
density (~0.03 W/m²) sits ~300x below the threshold, the softplus penalty is ~0
for every action, and the exposure objective does not exist. The env default is
now `0.03` — the same precautionary threshold the demo scenarios use — with the
sensitivity sweep over 0.01/0.1/1 mW/cm² (spec Section 8) still to be run.

---

## 2. Work packages

### WP-A1 — Wire contract: make the live numbers live
**Owner:** backend. **Files:** `src/flywaydqn/service/schemas.py`,
`src/flywaydqn/demo/runner.py`, `src/flywaydqn/service/app.py`,
`src/flywaydqn/agents/policies.py` (new), `tests/unit/test_demo.py`,
`tests/unit/test_contract.py` (new).

**Scope:** the three fields in 1.1, the `policy` field and `GET /policies` in
1.2, and the registry in 1.3. Null depth is measured in `DemoRunner._frame` from
`action.beam.weights` versus the transmitted `weights` over the same `cells`
(reuse `exposure_p95_w_m2`). Latency wraps `self._policy.act(...)` in
`time.perf_counter()`.

**Done when:**
1. `sThrWm2` equals the scenario YAML's `exposure.s_thr_w_m2` for all three scenarios.
2. `nullDepthDb` is `0.0` on every frame where `nullActive` is false, and on a
   null frame equals the policy's own reported reduction within `1e-6` dB.
3. `inferenceLatencyMs > 0` and finite on every frame.
4. `POST /sessions {"policy": "unconstrained"}` on `scenario_001` produces a run
   with `nullActive == False` for all 600 steps, while the default produces at
   least one `nullActive == True` step.
5. Unknown policy name → `404` naming the accepted values.
6. `SCHEMA_VERSION == "3"` (see 1.1; started at `"2"`, bumped when `baseline`
   and pace landed) and a contract test asserts the frame's field set
   exactly, so any future drift fails CI instead of silently reaching the UI.

### WP-A2 — Dashboard honesty pass
**Owner:** frontend. **Files:** `ui/src/types/index.ts`, `ui/src/lib/api.ts`,
`ui/src/hooks/useDemoSession.ts`, `ui/src/components/Layout/Sidebar.tsx`,
`ui/src/components/Screens/BeamSteeringConsole.tsx` (readouts only).

**Scope:** delete `EXPOSURE_LIMIT_WM2`, `NOMINAL_NULL_DEPTH_DB`, the `1.0`
badge tick, and the `inferenceLatencyMs` field that was never written. Read
threshold, null depth and latency from the frame. Drive the engine toggle from
`GET /policies` and send `policy` on `POST /sessions`.

**Done when:** `grep -n "5\.0\|-26\.4" ui/src/hooks/useDemoSession.ts` returns
nothing threshold/null-depth shaped; toggling the engine changes the streamed
`nullActive` behaviour; `tsc && vite build` passes.

### WP-B — Gymnasium `BeamEnv`
**Owner:** backend/RL. **Files:** `src/flywaydqn/env/beam_env.py` (new),
`src/flywaydqn/env/__init__.py`, `tests/unit/test_beam_env.py` (new).

**Scope:** wrap the existing RF/mobility/exposure modules (do not re-implement
them) into `reset(seed)`/`step(action) -> (obs, reward, terminated, truncated, info)`.
Observation and action spaces as specified in README §4.3 (discrete 16×4×2 = 128
flattened actions; observation carries per-UE summary, flock summary, current
beam/power, previous throughput/exposure, and `log10(w)`). `info` carries every
reward term, the exposure metrics, and `null_active`. Reuse
`flywaydqn.env.reward.make_reward_fn` unchanged.

**Done when:** `gymnasium.utils.env_checker.check_env(BeamEnv())` passes without
warnings; a random policy survives 1000 episodes; a seeded `reset` is
bit-reproducible; reward terms in `info` match `make_reward_fn` used directly.

### WP-C — Baselines and the measured comparison table
**Owner:** backend. **Files:** `src/flywaydqn/agents/baselines.py` (new;
static back-off and sector shutdown), `src/flywaydqn/agents/policies.py`,
`experiments/run_baselines.py` (new), `experiments/results/` (generated).

**Scope:** add **static back-off** (fixed power reduction while the flock is in
range) and **sector shutdown** (transmit off while in range), register them, and
write a script that runs all five policies (max-throughput SON, static back-off,
sector shutdown, analytic nulling, PPO) over ≥100 seeded scenarios and emits CSV
+ a markdown table with mean ± 95% CI for throughput, dose, `S_max`, fraction of
time over threshold, and mean power.

**Done when:** `python -m experiments.run_baselines` regenerates the table from a
clean checkout; every number in the table traces to the CSV; the analytic-nulling
baseline is present because the DQN has to beat it to mean anything.

### WP-D — Train a real DQN
**Owner:** RL. **Files:** `src/flywaydqn/agents/train_dqn.py` (new),
`configs/train/train.yaml`, `experiments/checkpoints/<run>/` (generated).

**Scope:** SB3 DQN (document Double/Dueling choices) on `BeamEnv`, preference
conditioned by sampling `log10(w)` per episode, hyperparameters from Hydra,
metrics logged to MLflow or a local CSV if MLflow is unavailable. **Before
starting, reconcile the environment:** the installed interpreter is Python 3.13
with numpy 2.5.3 while `pyproject.toml` pins `==3.11.*` and `numpy<2.0`; torch,
SB3, hydra, omegaconf, scipy, mlflow and optuna are not installed and `uv` is not
on PATH. Record whatever resolution is used and make the pins agree with it.

**Done when:** ≥3 seeds produce learning curves with confidence intervals; the
best checkpoint is selected on validation seeds and reported on held-out seeds;
the training command and seed are committed so the curve is reproducible; the
numbers go in the README table and nowhere as literals.

### WP-E — Serve the trained policy, show its real internals
**Owner:** RL + frontend. **Files:** `src/flywaydqn/agents/dqn_policy.py` (new,
implements `BeamPolicy`), `agents/policy_io.py`, `policies.py` (registry entry
`"dqn"`), `ui/src/components/Screens/DQNPolicyInspector.tsx`,
`ui/src/types/index.ts`.

**Scope (shipped):** a `BeamPolicy` that loads the checkpoint and scores the
128 actions through `BeamEnv.observation_from` — the trainer's own feature
construction — using the live `LiveState` plus the last applied action, so the
served decision describes the frame on screen. `create_app` registers it as
`"dqn"` at service startup (only when `experiments/checkpoints/dqn_first_pass/
seed_1` is readable; a checkout without the build artifact simply does not
offer the controller). `GET /policies/dqn/meta` serves the checkpoint's
`meta.json`; the inspector's Training State card renders it (steps, logged
return, spaces, hyperparameters), and its decision-trace row renders
`frame.explanation`, which carries the argmax action, decoded beam/power/null
and the Q-value the running network produced for that step.

**Deliberately not shipped:** the `?qvalues=1` per-frame Q-vector payload. It
would change the frozen frame contract (see 1.1), and the schema stays `"3"`;
the inspector says the Q-vector is not on the wire instead of drawing it.
(Use `q_net` directly if a debugging screen ever needs it.)

**Done when:** the inspector renders nothing that is not derivable from the
checkpoint or the live frame; with no checkpoint present the screen says so
explicitly instead of drawing a plot.

### WP-F — Timeline and stat tiles from real history
**Owner:** frontend. **Files:** `ui/src/hooks/useDemoSession.ts`,
`ui/src/components/Screens/BeamSteeringConsole.tsx` (chart + tiles),
`ui/src/lib/history.ts` (new).

**Scope (shipped):** `ui/src/lib/history.ts` accumulates every displayed frame
(`appendSample`, capped at 4 000) and derives `summarize()` aggregates; the
hook appends in `applyFrame` and freezes the outgoing run with
`startNewRun()` whenever a session is re-created. The chart plots the run's
real per-frame exposure series (the old code repeated the current value once
per log entry), and the tiles compute peak exposure, mean throughput,
observed azimuth span and measured decision rate from stored frames — the
`frameCount >= 60` gate (measured against a six-entry log, so it could never
open) is gone. A fifth "BEFORE — PREVIOUS RUN" tile plus the controller
banner keep the last run's controller name, peak exposure, null depth and TX
power visible after an engine switch.

**Deviation from the original scope, recorded deliberately:** the "baseline"
comparison uses a frozen snapshot of the previous run rather than
automatically fetching a second `policy: "unconstrained"` session; the chart
legend still says "Baseline not wired" because that second-session curve does
not exist. **Done when (met):** every number in the chart and tiles is a
function of stored frames, and with history too short to compute a value the
tile shows "—" rather than a plausible constant.

### WP-G — Packaging, reproducibility, truth pass
**Owner:** platform. **Files:** `docker/Dockerfile.service`,
`docker/Dockerfile.ui` (new; `docker/demo-compose.yaml` already references them),
`uv.lock` or pinned `requirements.txt` (new), `README.md`, `MODEL_CARD.md` (new),
`Makefile`.

**Scope:** the `docker compose up` quickstart currently fails because neither
Dockerfile exists. Add them, or remove the compose instructions — do not leave a
documented command that cannot run. Reconcile pins with the versions actually
tested, generate a lockfile, make `make check` runnable by the same steps CI
uses, correct the README progress table (it still says "Phase 8 — not started"
and "173 tests" while 210 tests pass), and write the model card: intended use,
scope, known failure modes, the precautionary framing of the threshold, and the
fact that Phase B (radar-derived tracks) was not executed.

---

## 3. How to hand these off

Each package is self-contained. To run one in its own agent thread, start from
the repository root and paste the package's paragraph above verbatim as the task,
plus: "Read `docs/WORKPACKAGES.md` first; the interfaces in section 1 are frozen —
if you need to change one, change this file in the same commit and say so."

**Order of dispatch:** WP-A1 first (it is the only blocker). Then WP-A2, WP-B and
WP-G in parallel. WP-C and WP-D after WP-B. WP-E after WP-D. WP-F after WP-A1.

**Collision map (keep these file sets disjoint):** WP-A1 owns `service/` +
`demo/runner.py`; WP-B owns `env/beam_env.py`; WP-C owns `agents/baselines.py` +
`experiments/`; WP-D owns `agents/train_dqn.py` + `configs/train/`; WP-E owns
`agents/dqn_policy.py` + the inspector screen; WP-F owns the console chart and
tiles; WP-G owns `docker/`, `README.md`, `MODEL_CARD.md`, lockfiles. Two packages
that both need `agents/policies.py` (WP-A1, WP-C, WP-E) must land sequentially —
A1 creates it, C and E only add rows to the registry.
