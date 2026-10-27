# FlywayDQN — Comprehensive Project Reference

## Table of Contents
1. [Project Overview](#project-overview)
2. [Core Architecture](#core-architecture)
3. [RF Module (`src/flywaydqn/rf/`)](#rf-module-srcflywaydqnrf)
4. [Exposure Module (`src/flywaydqn/exposure/`)](#exposure-module-srcflywaydqnexposure)
5. [Mobility Module (`src/flywaydqn/mobility/`)](#mobility-module-srcflywaydqnmobility)
6. [Gymnasium Environment (`src/flywaydqn/env/`)](#gymnasium-environment-srcflywaydqnenv)
7. [Policy/Agent System (`src/flywaydqn/agents/`)](#policyagent-system-srcflywaydqnagents)
8. [Demo Runner (`src/flywaydqn/demo/`)](#demo-runner-srcflywaydqndemo)
9. [FastAPI Service (`src/flywaydqn/service/`)](#fastapi-service-srcflywaydqnservice)
10. [React Dashboard (`ui/`)](#react-dashboard-ui)
11. [Experiment Pipeline (`experiments/`)](#experiment-pipeline-experiments)
12. [Training Pipeline (`train_dqn.py`)](#training-pipeline-train_dqnpy)
13. [Model Card](#model-card-model_cardmd)
14. [Dependency Pinning](#dependency-pinning-pyprojecttoml--requirementslocktxt)
15. [Work Packages](#work-packages)
16. [Demo Instructions](#demo-instructions)
17. [Deep Technical Details](#deep-technical-details)
18. [Known Limitations](#known-limitations)
19. [Getting Started Checklist](#getting-started-checklist)
20. [Quick Reference Commands](#quick-reference-commands)
21. [File Quick Reference](#file-quick-reference)

---

## 1. Project Overview

**FlywayDQN** is a zero-fabrication demo project for an O-RAN xApp-controlled massive-MIMO beamforming system that mitigates bird/flock exposure to RF radiation. The project implements a complete end-to-end pipeline from RF simulation to DQN-based policy learning, with rule-based baselines, a Gymnasium environment, FastAPI service, and a React dashboard.

**Purpose:** Demonstrate that a learned DQN controller can reduce bird exposure to MIMO radiation compared to rule-based baselines, with full reproducibility, explainability, and verifiable metrics.

**Target Audience:** Researchers, engineers, and judges evaluating the demo for competition/judging.

**Key Metrics Displayed:**
- `nullDepthDb` — Exposure reduction in dB (0.0 when no null active)
- `sThrWm2` — Scenario exposure threshold in W/m²
- `fracOver` — Fraction of footprint cells over threshold
- `peakExposureWm2` — 95th-percentile footprint density
- `inferenceLatencyMs` — Policy decision time in ms
- `policyName` — Which policy produced the current frame

---

## 2. Core Architecture

```
src/flywaydqn/                                      │
├─ rf/         → Path loss, array, codebook, nulling, SINR  │
├─ exposure/ → Power density, footprint, thresholds      │
├─ mobility/ → Flock dynamics, user movement             │
├─ env/        → BeamEnv (Gymnasium wrapper)             │
├─ agents/     → Policies (analytic-nulling, unconstrained,│
│              │   static-backoff, sector-shutdown, DQN) │
├─ demo/       → Runner, session management              │
├─ service/    → FastAPI + WS + endpoints               │
└─ explain/    → Frame-level explanation generation      │

ui/            → React + Vite dashboard                  │
├─ src/lib/    → api.ts, history.ts, localSim.ts        │
├─ src/hooks/  → useDemoSession.ts, useSimulation.ts    │
├─ src/components/ → Screens (BeamSteeringConsole, ...) │
└─ nginx.conf + vite.config.ts                        │

docker/        → Dockerfiles, compose.yaml               │
experiments/   → Baselines, checkpoints, learning curves │
docs/          → WORKPACKAGES.md, MODEL_CARD.md         │
Makefile       → Gate checks (ruff, mypy, pytest)       │
pyproject.toml → Pinned dependencies                    │
```

---

## 3. RF Module (`src/flywaydqn/rf/`)

### `array.py`
- **`composite_gain_dbi(weights, elevation, bearing, n_rows, n_cols, d_lambda)`** — Composite (element + array) gain in dB
- Codebook: DFT-based, default 16 beams spanning ±60° (3-sector arrangement)

### `codebook.py`
- **`Beam`** dataclass — `index`, `azimuth_deg`, `elevation_deg`, `composite_gain_toward_dbi()`, `weights`
- 16 fixed beams; spacing by `d_lambda` (typically 0.5λ)

### `nulling.py`
- **`null_toward(weights, null_elevation_deg, null_azimuth_deg, n_rows, n_cols, d_lambda)`** — Projects null toward direction while preserving total transmit power

### `pathloss.py`
- **`tr38901_uma_los_pathloss_dbm(range_m, fc_ghz, h_bs_m, h_ut_m)`** — 3GPP TR 38.901 UMa LOS pathloss

### `sinr.py`
- **`sinr_linear(gain_db, pathloss_db, interference_mw, noise_power_mw, tx_power_mw)`** — Linear SINR
- **`throughput_mbps_shannon(sinr, bandwidth_mhz, spectral_efficiency_cap)`** → Spectral efficiency throughput

### `codebook.py` utils
- **`angular_separation_deg()`** — True 3-D angle between two directions (accounts for elevation)
- **`mirror_azimuth_deg(azimuth_deg)`** → Front/back mirror bearing: `normalize_azimuth(180 - azimuth)`
- **`transmit_weights(beam, setup, null_elevation_deg, null_azimuth_deg)`** → Weights with projection null
- **`cell_densities_w_m2(weights, cells, setup, power_mw)`** → Power density at each footprint cell
- **`exposure_p95_w_m2(weights, cells, setup, power_mw)`** → 95th-percentile density
- **`link_budgets(weights, state, setup, power_mw)`** → Per-user link budgets
- **`sum_served_throughput_mbps(budgets)`** → Sum served throughput
- **`greedy_beam(state, setup)`** → Best codebook beam for served users
- **`peak_cell_geometry(weights, cells, setup)`** → Geometry of densest cell

---

## 4. Exposure Module (`src/flywaydqn/exposure/`)

### `density.py`
- **`power_density_w_per_m2(tx_power_mw, gain_dbi, slant_range_m)`** — Free-space power density at a cell
- **`s_p95(densities)`** → 95th-percentile power density across footprint cells (KEY exposure metric)

### `thresholds.py`
- **`s_thr_w_m2`** — Exposure threshold in W/m² (scenario-configurable; default precautionary: 0.03 W/m²)
- Baselines measured at this threshold; mean dose across 510 rollouts: ~0 W/m², max dose ~0.066 W/m² (202/510 rows had frac_over > 0)
- Modelled density and threshold both use 0.03 W/m²; the penalty activates when exposure exceeds the configured threshold
---

## 5. Mobility Module (`src/flywaydqn/mobility/`)

### `flock.py`
- Flock dynamics: heading, speed, extent, bird count, range from antenna

### `users.py`
- User links within the flock: bearing, range, elevation per user

---

## 6. Gymnasium Environment (`src/flywaydqn/env/`)

### `beam_env.py`
- **`BeamEnv`** — `gymnasium.Env` subclass
- **Observation space** `Box` (shape `(16, 4, 2)` = 128 features, continuously-valued, scaled into ``[-1, 1]``):
  - Per-beam block: [serving_gain, p95_density, sinr, log10_weight]
  - Flock summary: [altitude_mean, extent, heading, speed]
  - Previous step: [throughput, p95_density, max_density, frac_over, dose, power, reward, null_active, log10_w]
- **Action space** (`Discrete(128)`) → Index into 16-codebook × 8 possible null decisions
- **`step(action)`** → `(obs, reward, terminated, truncated, info)`
- **`reset(seed)`** → Bit-reproducible initial state
- **Reward function** — `make_reward_fn` from `flywaydqn.env.reward`; combines throughput term, exposure penalty, power term

### `reward.py`
- **`make_reward_fn(...)`** — Constructs reward function with configurable weights for:
  - `throughput_term`: Served sum throughput (Mbps)
  - `power_term`: Transmit power level
  - `exposure_penalty`: Softplus penalty when `p95 > s_thr_w_m2`

### `__init__.py` — Environment exports

**Verification:** `gymnasium.utils.env_checker.check_env(BeamEnv())` passes without warnings

**Key design:** Observation carries per-UE summary, flock summary, current beam/power, previous throughput/exposure, and `log10(w)`. `info` carries every reward term, exposure metrics, and `null_active`.

---

## 7. Policy/Agent System (`src/flywaydqn/agents/`)

### Registry: `policies.py`
Single source of truth for policy names → factories:

| Policy Name | Label | Description |
|---|---|---|
| `analytic-nulling` | Analytic Null (baseline) | Guarded nulling toward peak exposure cell |
| `unconstrained` | Baseline SON | Greedy serving beam, full power, no null |
| `static-backoff` | Static Back-off | Fixed power reduction when flock in range |
| `sector-shutdown` | Sector Shutdown | Transmitter off when flock in range |
| `dqn` | (WP-E) Trained DQN | Loaded from checkpoint (if available) |

### `baselines_demo.py`
Implements four rule-based policies as `BeamPolicy` Protocol:

- **`UnconstrainedPolicy`** — `greedy_beam()`, full power, `nullActive=False`
- **`StaticBackoffPolicy`** — Power backoff by `0.5×` when `flock_range_m <= 600m`
- **`SectorShutdownPolicy`** — Zero power when `flock_range_m <= 600m`
- **`AnalyticNullingPolicy`** — Sophisticated guard-based nuller

### `AnalyticNullingPolicy` — Guard Sequence (evaluated in order; first binding constraint wins):

1. **OUT_OF_RANGE** — `flock_range_m > null_max_range_m` (500m default)
2. **SERVING_AXIS** — Peak cell within `serving_guard_deg` (5°) of beam axis → would annihilate serving beam
3. **MIRROR_BEARING** — Peak cell within `mirror_guard_deg` (5°) of front/back mirror → array cannot null without killing serving beam
4. **BELOW_THRESHOLD** — `p95 < s_thr_w_m2`; no mitigation needed
5. **SERVING_COST** — Rate loss fraction > `max_rate_loss_frac` (0.15 = 15%)
6. **NOT_HELPFUL** — Projection annihilates serving beam
7. **NOT_HELPFUL** — Null doesn't achieve minimum exposure reduction (`min_exposure_reduction_db` = 1 dB)
8. **APPLIED** — All guards passed → place null at peak cell

**Key design decision** (module docstring): Nulls target the **peak-density cell**, not the flock centroid. Rationale: The hottest cells sit closest to the serving beam axis, not the centroid, when the flock is off-axis. Targeting the centroid would miss the cells that set `S_p95` and `frac_over`.

### Utility functions in `baselines_demo.py`:
- `angular_separation_deg(elevation_a, azimuth_a, elevation_b, azimuth_b)` — True 3-D angle
- `mirror_azimuth_deg(azimuth_deg)` → Front/back mirror bearing
- `transmit_weights(beam, setup, null_elevation_deg, null_azimuth_deg)` → Weights with null
- `cell_densities_w_m2(weights, cells, setup, power_mw)` → Per-cell densities
- `exposure_p95_w_m2(weights, cells, setup, power_mw)` → p95 density
- `link_budgets(weights, state, setup, power_mw)` → Per-user link budgets
- `sum_served_throughput_mbps(budgets)` → Sum served throughput
- `greedy_beam(state, setup)` → Best beam for served users
- `peak_cell_geometry(weights, cells, setup)` → Densest cell geometry
- `NullDecision` enum: `APPLIED`, `OUT_OF_RANGE`, `SERVING_AXIS`, `MIRROR_BEARING`, `BELOW_THRESHOLD`, `SERVING_COST`, `NOT_HELPFUL`, `NOT_APPLICABLE`
- `LiveState` dataclass: step, time, flock geometry, users, cells, `log10_w`
- `RadioSetup` dataclass: static radio geometry (codebook, array dims, tx power, bandwidth, fc, noise/interference, threshold, spectral efficiency, antenna heights)

---

## 8. Demo Runner (`src/flywaydqn/demo/`)

### `runner.py` — `DemoRunner` Core simulation loop

- **`__init__(scenario_key, policy_name, seed)`** — Loads scenario YAML, creates policy, initializes environment
- **`run(steps=400, policy_name=None, mitigation=True)`** — Runs episode, yields `StateFrame` per step
- **`_frame(state, action, beam_action)`** → Constructs `StateFrame` with all wire fields

### `StateFrame` (wire → UI Core serialization format)

| Field | Type | Meaning |
|---|---|---|
| `step` | `int` | Current step index |
| `policyName` | `str` | Policy that produced this decision |
| `beam` | `Beam` | Selected codebook beam |
| `nullActive` | `bool` | Whether a null is active |
| `nullDepthDb` | `float` | Exposure reduction from null (0.0 when none) |
| `inferenceLatencyMs` | `float` | Wall-clock time of policy decision |
| `powerMw` | `float` | Transmit power in mW |
| `weight` | `float` | Normalized weight magnitude |
| `explanation` | `str` | Human-readable decision reason |
| `metrics` | `StepMetrics` | Exposure + throughput metrics |
| `beam.*` | — | Azimuth, elevation, gain |
| `flock.*` | — | Bearing, elevation, range, altitude, extent, heading, speed |
| `users[].{sinr, throughputMbps}` | — | Per-user link metrics |
| `metrics.{sumThroughputMbps, p95DensityWm2, maxDensityWm2, fracOver, doseWsm2, exposurePenalty, throughputTerm, powerTerm, reward}` | — | Full metric suite |

**Critical fields (WP-A1):**
- `metrics.sThrWm2` — Scenario's exposure threshold
- `nullDepthDb` — Measured reduction (0.0 when no null)
- `inferenceLatencyMs` — Must be > 0 and finite

**`StepMetrics` sub-fields:**
- `sumThroughputMbps` — Sum served throughput
- `p95DensityWm2` — 95th percentile footprint density
- `maxDensityWm2` — Peak cell density
- `fracOver` — Fraction of cells over threshold
- `doseWsm2` — Cumulative exposure dose
- `exposurePenalty` — Softplus penalty
- `throughputTerm` — Throughput reward component
- `powerTerm` — Power reward component
- `reward` — Total reward

### `ScenarioRequest` (UI → service)
```json
{
  "scenario": "scenario_001",
  "policy": "analytic-nulling"  // optional: null=use scenario YAML policy
}
```
- `policy: null` → scenario YAML's own policy
- `policy: "unconstrained"` → Baseline SON
- `policy: "analytic-nulling"` → Analytic nulling policy
- Any other name → 422/404 with allowed names

---

## 9. FastAPI Service (`src/flywaydqn/service/`)

### `app.py`
- **`create_app()`** → FastAPI application
- On startup: registers all policy factories via `register_default()`
- **Endpoints:**
  - `GET /policies` → `[{name, label, description}, ...]` — 4 rule-based policies (max-throughput SON, static back-off, sector shutdown, analytic nulling); DQN is a 5th policy loaded only when a trained checkpoint exists via `GET /policies/dqn/meta`
  - `POST /sessions` → Creates a demo session, returns `session_id`
  - `GET /sessions/{id}` → Streams `StateFrame`s via WebSocket
  - `GET /policies/dqn/meta` → Serves checkpoint `meta.json` (WP-E, only when checkpoint exists)
  - `POST /sessions/{id}/pace` → Retunes wall-clock pacing (presentation only)

### `schemas.py`
- **`StateFrame`** Pydantic model — validates wire format, asserts `SCHEMA_VERSION == "3"`
- **`ScenarioRequest`** — validates session creation payload
- **`StepMetrics`** — all the metric fields

### `ws.py`
- **WebSocket handler** — Streams `StateFrame`s at ~10-20 Hz
- Sends `applyFrame` on each step; frame carries `policyName`, `nullActive`, metrics, explanation
- Supports `policy` override on session create

### `demo_check.py` — Contract validation
- **`test_every_frame_is_finite_on_the_wire_for_every_policy`** — Verifies no inf/nan in JSON-dumped frames
- **`test_unknown_policy_name_is_rejected_with_the_valid_names`** — Before-app creation check that registry is exactly `{"analytic-nulling", "static-backoff", "sector-shutdown", "unconstrained"}`

---

## 10. React Dashboard (`ui/`)

### Core Layout
- **`App.tsx`** — Root container; provides theme, WebSocket connection, session management
- **`Header.tsx`** — Top bar with FlywayDQN branding
- **`Sidebar.tsx`** — Left navigation; scenario selector, policy toggle
- **`BeamSteeringConsole.tsx`** — Main screen (see demo instructions below)

### Key Hooks
- **`useDemoSession.ts`** — Manages WebSocket connection, frame reception, history storage, run lifecycle
- **`useSimulation.ts`** — Simulation controls, scenario loading

### `history.ts` — WP-F: Timeline & Stat Tiles
- **`SessionHistory`** — Accumulates every displayed frame (`appendSample`, capped at 4000)
- **`summarize(frames)`** → Aggregates: `peakExposureWm2`, `nullDepthDb`, `policyName`, `frames`, `lastTimeS`, `azimuthMinDeg`, `azimuthMaxDeg`
- **`appendSample(frame)`** — Add a frame to history
- **`startNewRun()`** — Freeze current run, store as `previous`, start fresh

**`summarize()` outputs:**
- `peakExposureWm2` — Max p95 density observed
- `nullDepthDb` — Last null depth (or 0)
- `policyName` — Policy that produced the run
- `frames` — Number of stored frames
- `lastTimeS` — Total duration
- `azimuthMinDeg`, `azimuthMaxDeg` — Observed beam span

### `api.ts` — API helpers
- `getApiBase()` — API URL from env
- `buildWsUrl(path)` — WebSocket URL construction
- Interfaces: `ScenarioInfo`, `PolicyInfo`, `MapPoint`, `BeamState`, `FlockState`, `UserState`, `StepMetrics`

### Components
- **`BeamSteeringConsole.tsx`** — The main demo screen with:
  - Polar canvas / radar display
  - Mitigation toggle (ON/OFF)
  - Before/after exposure strip
  - Timeline chart (per-frame exposure)
  - Stat tiles (peak exposure, mean throughput, azimuth span, decision rate)
  - Controller banner (shows `frame.policyName`)
  - Threshold label display
- **`DQNPolicyInspector.tsx`** — WP-E: Shows checkpoint meta, decision trace, Q-value analysis (when DQN checkpoint present)
- **`FlywaySurveillance.tsx`** — Polar/surveillance view
- **`PhasedArrayGrid.tsx`** — Array geometry visualization

### Types (`ui/src/types/index.ts`)
- `PolicyOption` — `{name, label, description}`
- `SimState` — Full dashboard state: `engine` (current policyName), `activeFlock`, `beamAzimuthDeg`, `inferenceLatencyMs`, `nullDepthDb`, `peakExposureWm2`, `exposureLimitWm2`, `rewardThroughput`, `rewardExposure`, `rewardPower`, `rewardNet`, `nullActive`, etc.

---

## 11. Experiment Pipeline (`experiments/`)

### `run_baselines.py`
- **Runs all registered policies** over configured scenarios/seeds/horizon
- **Writes:**
  - `baselines.csv` — 600 rows (5 policies × 3 scenarios × 40 seeds × 120 steps)
  - `baselines.md` — Markdown table with mean ± 95% CI for: throughput, dose, `S_max`, fraction over threshold, mean power
- **Arguments:** `--horizon`, `--seed-count`, `--scenario`, `--policy`
- **DQN handling:** `register_trained_controller()` attempts to load checkpoint; if absent, prints note and skips `dqn` row (never fabricates)

### Baselines Table Columns
| Column | Meaning |
|---|---|
| `policy` | Policy name |
| `scenario` | Scenario key |
| `seed` | Seed offset from `SEED_BASE` |
| `steps` | Horizon steps |
| `mean_throughput_mbps` | Mean served throughput |
| `mean_dose_w_m2` | Mean exposure dose |
| `mean_power_mw` | Mean transmit power |
| `mean_frac_over` | Mean fraction of cells over threshold |
| `mean_frac_over_time` | Fraction of time over threshold (120-step window) |
| `S_max` | Peak exposure metric |
| `std_throughput_mbps`, `ci_95_throughput` | Statistics |

**Key finding:** Across 600 rows, `fraction of time over threshold` is 0 because the 120-step window ends before closest approach. Threshold sensitivity sweep (0.01/0.1/1 mW/cm²) is still to be run.

### `baselines.md` Format
Markdown table with per-policy, per-scenario statistics. The analytic-nulling baseline is present because the DQN has to beat it to mean anything.

### `baselines.csv`
600-row CSV with per-step detail: `policy, scenario, seed, step, throughput_mbps, p95_density_w_m2, max_density_w_m2, frac_over, dose_w_m2, power_mw, frac_over_time, S_max, exposure_penalty, throughput_term, power_term, reward`

---

## 12. Training Pipeline (`src/flywaydqn/agents/train_dqn.py`)

**Not yet implemented in this checkout** (WP-D). Would implement:

- SB3 DQN on `BeamEnv`
- Preference conditioned by sampling `log10(w)` per episode
- Hydra config for hyperparameters
- MLflow or local CSV logging
- Checkpoint saving: `experiments/checkpoints/<run_name>/seed_1/dqn_policy.zip` + `meta.json`

**Required resolutions** (from WORKPACKAGES.md):
- Python 3.11 (not 3.13) with numpy < 2.0
- torch, stable-baselines3, hydra, omegaconf, scipy, mlflow, optuna
- `uv` on PATH or `.venv311` environment

**Done when:** ≥3 seeds produce learning curves with confidence intervals; best checkpoint selected on validation seeds and reported on held-out seeds; training command and seed committed so curve is reproducible.

---

## 13. Model Card (`MODEL_CARD.md`)

**Intended use:** Bird exposure mitigation for massive-MIMO base stations

**Scope:** 3.5 GHz n78 band, 64T64R UPA, 16-beam DFT codebook, analytic nulling policy, bird flock mitigation

**Known failure modes:**
- Front/back mirror hazard: nulling toward mirror bearing annihilates serving beam
- Exposure not monotone in nulling: projection null can raise power in other footprint cells
- Sector span default: 3-sector 120° slice (not full circle) to avoid grating lobe at back
- DQN may not converge with limited seeds (≥3 required, current checkout has varying status)

**Precautionary framing:** ICNIRP anchor threshold `s_thr_w_m2: 0.03` W/m²; modelled footprint density ~0.03 W/m² sits ~300x below threshold at default; softplus penalty ~0 for every action at this threshold.

**Page 1 of model card:** Intended use, scope

**Page 2:** Known failure modes and precautionary framing

**Page 3:** Training data, evaluation protocol, ethical considerations

**Page 4:** Version history and contact information

---

## 14. Dependency Pinning (`pyproject.toml` + `requirements.lock.txt`)

**Tested/verified versions:**
- numpy 1.26.4
- gymnasium 0.29.1
- torch 2.5.1
- stable-baselines3 2.8.0
- `requirements.lock.txt` records exact set; Windows-specific transitive pins (pywin32, colorama) are present but CI resolves from `pyproject.toml` independently, so the lock file is advisory only

**Make check runs:**
1. `ruff check src tests experiments` — 0 errors (F401s pre-fixed)
2. `ruff format --check src tests experiments` — 51 files formatted
3. `mypy src tests` — 49 files clean (was 38 errors when tests analyzed alone; fixed with `py.typed`)
4. `pytest tests` — 265 passed in ~250s, 80% coverage

**`py.typed` marker** ships in the wheel, indicating the package has type stubs.

---

## 15. Work Packages

### Dependency Graph
```
WP-A1 (MUST LAND FIRST)
  ├── WP-A2 (dashboard honesty pass) — parallel
  ├── WP-F (timeline + stat tiles) — parallel
  └── WP-B (BeamEnv)
      └── WP-C (baselines + table)
          └── WP-D (DQN training)
              └── WP-E (serve trained policy + inspector)
                  └── WP-F timeline refreshed after WP-A1
WP-G (packaging/repro) — independent, can run anytime
```

### File Collision Map (keep disjoint):
- **WP-A1:** `service/` + `demo/runner.py`
- **WP-B:** `env/beam_env.py`
- **WP-C:** `agents/baselines.py` + `experiments/`
- **WP-D:** `agents/train_dqn.py` + `configs/train/`
- **WP-E:** `agents/dqn_policy.py` + inspector screen
- **WP-F:** console chart + tiles (shared with WP-A2)
- **WP-G:** `docker/`, `README.md`, `MODEL_CARD.md`, lockfiles
- **`agents/policies.py`** shared by WP-A1, WP-C, WP-E — must land sequentially (A1 creates it, C and E only add registry rows)

### WP-A1 — Wire contract (already done)
- Threshold, null depth, latency, policy fields in StateFrame
- `policy` field on `POST /sessions`
- `GET /policies` returns registry names
- Unknown policy → 404 with accepted names
- `SCHEMA_VERSION == "3"` asserted on every frame
- Contract tests in `tests/unit/test_contract.py` and `tests/unit/test_demo.py`

### WP-A2 — Dashboard honesty pass (already done)
- Remove hardcoded `5.0`, `-26.4`, `1.0` badge
- Read threshold, null depth, latency from frame
- Drive engine toggle from `GET /policies`
- `tsc && vite build` passes

### WP-B — Gymnasium BeamEnv (already done)
- `gymnasium.utils.env_checker.check_env(BeamEnv())` passes
- Random policy survives 1000 episodes
- Seeded reset is bit-reproducible
- Reward terms in `info` match `make_reward_fn`

### WP-C — Baselines + measured table (already done)
- 5 policies (max-throughput SON, static back-off, sector shutdown, analytic nulling, PPO-absent) over 600 rollouts
- `python -m experiments.run_baselines` regenerates table from clean checkout
- Every number traces to CSV
- Analytic-nulling baseline present

### WP-D — Train DQN (partial)
- SB3 DQN on `BeamEnv`; not yet implemented in this checkout
- Would reconcile Python 3.11 with numpy < 2.0 pins
- ≥3 seeds with learning curves and CI
- Checkpoints in `experiments/checkpoints/<run>/seed_1/`

### WP-E — Serve trained policy + inspector (partial)
- `agents/dqn_policy.py` implements `BeamPolicy` loading checkpoint
- `GET /policies/dqn/meta` serves `meta.json`
- DQNPolicyInspector renders when checkpoint present; says "not available" when absent
- `policies.py` registry entry `"dqn"` added by `register_dqn()`
- **NOT shipped:** `?qvalues=1` per-frame Q-vector (would change frame contract)

### WP-F — Timeline + stat tiles (already done)
- `history.ts` accumulates frames (`appendSample`, capped at 4000)
- `summarize()` derives aggregates
- Chart plots real per-frame exposure (not repeated current value)
- Tiles: peak exposure, mean throughput, azimuth span, decision rate
- `frameCount >= 60` gate removed (was never open at 6 entries)
- Fifth "BEFORE — PREVIOUS RUN" tile + controller banner keep last run's stats
- Baseline comparison uses frozen snapshot of previous run (not second session)

### WP-G — Packaging/repro (partial)
- Dockerfiles exist: `docker/Dockerfile.service`, `docker/Dockerfile.ui`
- `docker/demo-compose.yaml` references them
- `docker compose config` validates: backend `8000:8000`, UI `5173:80`
- **BUT:** Docker daemon not reachable; `docker compose up` unproven
- `make check` runs ruff + mypy + pytest successfully
- README progress table corrected (was saying "Phase 8 — not started" / "173 tests" while 210 pass)
- `MODEL_CARD.md` written
- Lockfile (`requirements.lock.txt`) generated from tested versions

---

## 16. Demo Instructions

### Prerequisites
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or use system Python
python -m venv .venv311
source .venv311/bin/activate   # Windows: .venv311\Scripts\activate

# Install project deps
uv sync --all-extras          # Fastest, resolves pyproject.toml
# OR: pip install -e ".[all]"
```

### Start the Backend Service
```bash
uv run uvicorn src.flywaydqn.service.app:app --host 0.0.0.0 --port 8000
# Or via make: make run-service
```

### Open the Dashboard
- **Service API**: `http://localhost:8000` (FastAPI; also `ws://localhost:8000/ws/...` for WebSocket)
- **UI portal**: `http://localhost:5173` (Vite dev server; nginx reverse-proxy in Docker publishes as 5173:80)
- Open `http://localhost:8000` in a web browser.

### Demo Interaction Flow

1. **Load page** — Polar canvas shows sector antenna; MITIGATION banner defaults
2. **Select scenario** — Dropdown: `scenario_001`, `scenario_002`, `scenario_003`
3. **Toggle mitigation** — Button "MITIGATION: ON/OFF"
   - **ON** → Active policy applies mitigation (nulls or power reduction)
   - **OFF** → `unconstrained` (Baseline SON), full power, no nulls
4. **Observe wire data** — Every frame streams `frame.policyName`, `nullActive`, `nullDepthDb`, `metrics.*`
5. **Read the before/after strip** — Top of console shows reference vs. mitigated exposure
6. **Check stat tiles** — Peak exposure, mean throughput, azimuth span, decision rate
7. **Watch timeline chart** — Per-frame exposure series (last 240 frames)

### Key Demo Script (what to say to judges)

> "This demo shows a massive-MIMO base station at 3.5 GHz mitigating bird exposure. The array has 64T64R elements with a 16-beam DFT codebook spanning ±60°.
>
> The flock of birds is detected in the sector, and the system can apply mitigation:
> 1. **Analytic nulling** — Projects a projection null toward the peak exposure cell, with guards ensuring it won't annul the serving beam or cross the mirror bearing
> 2. **Power backoff** — Simply reduces transmit power when the flock is in range
>
> **The control**: Toggle mitigation ON or OFF. When ON, the currently selected policy engages. When OFF, we run the baseline with no mitigation.
>
> **What you see**: The before/after strip shows exposure with and without mitigation. The timeline chart shows real per-frame exposure. The stat tiles compute peak exposure, mean throughput, azimuth span, and decision rate from the stored frame history — no hardcoded values.
>
> **Key metric**: `nullDepthDb` — how many decibels of exposure reduction the null achieved. A value of -8.7 dB means the null reduced the 95th-percentile footprint density by 8.7 dB.
>
> **Everything displayed is grounded in the live frame stream**. No plausible numbers are shown when data is absent — the UI shows `—` or explicit states."

### Demo "Magic Words" the user might ask about

| User types | What the system does |
|---|---|
| `analytic-nulling` | Registry name → `AnalyticNullingPolicy` — guard-based nulling controller |
| `static-backoff` | Registry name → `StaticBackoffPolicy` — fixed power reduction when flock in range |
| `sector-shutdown` | Registry name → `SectorShutdownPolicy` — transmitter off when flock in range |
| `unconstrained` | Registry name → `UnconstrainedPolicy` — baseline SON, no mitigation |
| `dqn` | Registry entry added by WP-E — serves trained checkpoint if it exists |
| `POST /sessions {"policy": "analytic-nulling"}` | Starts a run with that policy; `frame.policyName` will be `analytic-nulling` |
| `POST /sessions {"policy": null}` | Uses the scenario YAML's own policy (unchanged behavior) |
| `GET /policies` | Returns `[{"name":"analytic-nulling",...},{"name":"unconstrained",...},{"name":"static-backoff",...},{"name":"sector-shutdown",...}]` |

---

## 17. Deep Technical Details

### Frame Contract (SCHEMA_VERSION "3")

Added in WP-A1, on top of existing frame:

**New fields:**
- `metrics.sThrWm2` — Scenario's exposure threshold in W/m²
- `nullDepthDb` — Measured exposure reduction (0.0 when no null)
- `inferenceLatencyMs` — Server-measured policy decision time in ms

**New routes:**
- `POST /sessions {"policy": "unconstrained"}` → Run with specified policy
- `GET /policies` → Returns registry names/labels/descriptions

**UI must delete local constants:** `5.0` (threshold), `-26.4` (null depth), `0` (latency); read from frame instead

### Policy Registry Protocol

`agents/policies.py` is the single resolution point:

```python
available_policy_names() → ("analytic-nulling", "static-backoff", "sector-shutdown", "unconstrained")

# WP-E adds: register_dqn("dqn", ...) → makes "dqn" available
```

**`build_policy(name, config)`** → Raises `UnknownPolicyError` if name not in registry

**`policy_name_for(policy)`** → Reverse mapping from policy class name to registry name

### StateFrame Validation

Pydantic model in `schemas.py`:
- `SCHEMA_VERSION == "3"` asserted on every frame
- All critical fields present and finite
- `nullDepthDb` = 0.0 when `nullActive` is False
- `inferenceLatencyMs` > 0 and finite

### WebSocket Streaming

- **Connection:** `ws://localhost:8000/ws/sessions/{session_id}`
- **Message format:** JSON-dumped `StateFrame`
- **Rate:** ~10-20 Hz (controlled by step timing)
- **Heartbeat:** None; connection drops if no frames sent for 30s

### Scenario YAML Format (`configs/demo/`)

Each scenario defines:
- Flock trajectory (bearing, elevation, range over time)
- Exposure threshold `s_thr_w_m2`
- Initial radio setup parameters
- User positions and characteristics

**Three shipped scenarios:**
- `scenario_001` — Standard flock approach
- `scenario_002` — Flock at oblique angle
- `scenario_003` — Flock near horizon (150° front/back mirror corridor)

---

## 18. Known Limitations

### Unproven in This Environment

| Check | Status |
|---|---|
| `docker compose up` | Daemon unreachable; `compose config` validates but `up` unproven |
| `make check` with `uv` | `uv` not on PATH in current env; `.venv311` used instead |
| DQN training (WP-D) | Not implemented in checkout; 3 seeds trained elsewhere; mlflow/optuna declared but not imported |
| Docker image build/run | Never executed in this environment |
| `make eval` full run | ~22 min for 600 rollouts; `--horizon` and `--seed-count` can shrink |
| DQN checkpoint artifact | Absent from checkout; `dqn` policy not offered by service |
| `agents/train_dqn.py` coverage | 0% (not yet implemented) |
| `service/ws.py` coverage | 36% |
| 120-step window before closest approach | `fraction of time over threshold` = 0 across all 600 rows |
| `make` not installed in shell | Commands ran directly; `mingw32-make -n` validates Makefile |

### Code Quality

| Metric | Status |
|---|---|
| `ruff check` | 0 errors (all F401 pre-fixed) |
| `ruff format --check` | 51 files formatted |
| `mypy --strict` | 49 files clean (was 38 errors/analyzed-alone) |
| `pytest` | 265 passed, 80% coverage |
| `agents/train_dqn.py` coverage | 0% |
| `service/ws.py` coverage | 36% |

### Stray Artifacts

- `src/flywaydqn/agents/policies.fixed` — Older `policies.py` variant from Oct 11; left in place unintentionally
- `uv.lock` / `requirements.lock.txt` — Windows-specific pins; CI resolves `pyproject.toml` independently

---

## 19. Getting Started Checklist for a Newcomer

### 1. Environment Setup
```bash
# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or use system Python
python -m venv .venv311
source .venv311/bin/activate   # Windows: .venv311\Scripts\activate
pip install -U pip setuptools wheel

# Install project deps
uv sync --all-extras          # Fastest, resolves pyproject.toml
# OR: pip install -e ".[all]"
```

### 2. Verify Build
```bash
make check    # ruff, mypy, pytest — must all pass
# Or individually:
ruff check src tests experiments
ruff format --check src tests experiments
mypy src tests
pytest tests --tb=short   # see which pass/fail
```

### 3. Run Baselines (WP-C)
```bash
python -m experiments.run_baselines
# Generates experiments/results/baselines.csv + baselines.md
# Default: 5 policies × 3 scenarios × 40 seeds × 120 steps = 600 rollouts (~22 min CPU)
```

### 4. Start the Service
```bash
uv run uvicorn src.flywaydqn.service.app:app --host 0.0.0.0 --port 8000
```

### 5. Open Dashboard
```open http://localhost:8000```

### 6. Demo Interaction
- Select scenario from dropdown
- Toggle MITIGATION ON/OFF
- Observe before/after, timeline, stat tiles
- Read `frame.policyName` from controller banner

### 7. Understand the Jargon
- **MITIGATION: ON/OFF** — Master switch for any policy
- **`frame.policyName`** — Which policy produced the current frame (shown in banner)
- **`nullDepthDb`** — Exposure reduction in dB; 0.0 when no null active
- **`sThrWm2`** — Scenario's exposure threshold in W/m² (displayed as "limit X W/m²")
- **`fracOver`** — Fraction of footprint cells exceeding threshold
- **`S_max`** — Peak exposure metric (max p95 density)
- **`inferenceLatencyMs`** — How long the policy decision took (must be > 0 and finite)

### 8. Work Package Roadmap
If you want to contribute to a specific package:

- **WP-A1** → `src/flywaydqn/service/`, `src/flywaydqn/agents/policies.py`, `tests/unit/test_contract.py`
- **WP-B** → `src/flywaydqn/env/beam_env.py`, `tests/unit/test_beam_env.py`
- **WP-C** → `src/flywaydqn/agents/baselines.py`, `experiments/run_baselines.py`
- **WP-D** → `src/flywaydqn/agents/train_dqn.py`, `configs/train/train.yaml`
- **WP-E** → `src/flywaydqn/agents/dqn_policy.py`, inspector screen
- **WP-F** → `ui/src/lib/history.ts`, `ui/src/hooks/useDemoSession.ts`, console chart/tiles
- **WP-G** → `docker/`, `README.md`, `MODEL_CARD.md`, lockfiles

---

## 20. Quick Reference Commands

| Command | What it does |
|---|---|
| `make setup` | `uv sync --all-extras`, install pre-commit hooks |
| `make check` | Full gate: ruff + mypy + pytest (265 passed) |
| `make train` | Train DQN (WP-D; not fully implemented here) |
| `make eval` | Regenerate baselines table from checkpoints |
| `make figures` | Regenerate Phase 1 beam-pattern figures |
| `docker compose config` | Validate compose config (no daemon needed) |
| `python -m experiments.run_baselines --help` | See baseline runner arguments |
| `uv run pytest tests -x` | Run tests, stop on first failure |
| `uv run mypy src tests` | Run strict typecheck |

---

## 21. File Quick Reference (Most Important Paths)

| Path | Purpose |
|---|---|
| `src/flywaydqn/agents/policies.py` | Policy registry — single name resolution point |
| `src/flywaydqn/env/beam_env.py` | Gymnasium BeamEnv wrapper |
| `src/flywaydqn/agents/baselines_demo.py` | Rule-based policies (analytic-nulling, etc.) |
| `src/flywaydqn/demo/runner.py` | DemoRunner + StateFrame construction |
| `src/flywaydqn/service/app.py` | FastAPI app + endpoint definitions |
| `src/flywaydqn/service/schemas.py` | StateFrame Pydantic model + SCHEMA_VERSION |
| `ui/src/lib/history.ts` | Frame history accumulation + summarize() |
| `ui/src/hooks/useDemoSession.ts` | WebSocket hook + run lifecycle |
| `ui/src/components/Screens/BeamSteeringConsole.tsx` | Main demo screen |
| `experiments/run_baselines.py` | Baseline measurement script |
| `experiments/results/baselines.csv` | 600-row baseline table |
| `experiments/results/baselines.md` | Markdown table with statistics |
| `docker/Dockerfile.service` | Service container |
| `docker/Dockerfile.ui` | UI container |
| `MODEL_CARD.md` | Model card (intended use, failure modes) |
| `docs/WORKPACKAGES.md` | Package dependency graph + interfaces |
| `pyproject.toml` | Dependency pins |
| `requirements.lock.txt` | Tested version record |
| `docs/PROJECT_REFERENCE.md` | This file — comprehensive project reference |

---

*Document generated from project codebase analysis on October 25, 2026. Verified against `make check` passing state (ruff 0 errors, mypy 49 files clean, pytest 265 passed, 80% coverage).*