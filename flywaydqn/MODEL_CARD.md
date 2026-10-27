# Model Card — FlywayDQN controller (`dqn_first_pass`)

Every number below was produced by the commands recorded in this file and is
reproducible from `configs/train/train.yaml`. Nothing here is a target, an
aspiration, or a hand-typed result. The artifacts themselves live under
`experiments/checkpoints/dqn_first_pass/`, which `.gitignore` excludes as a build
artifact — that is why the numbers are also written down here.

## 1. What this is

A preference-conditioned **Double DQN** (Stable-Baselines3 2.8.0) that picks a
serving beam, a transmit power, and whether to place a projection null on the
flock, for one 8×8 planar array sector at 3.5 GHz. It is the `dqn` entry in the
policy registry (`flywaydqn/agents/policies.py`), served by
`flywaydqn/agents/dqn_policy.py`.

| Property | Value (recorded in `seed_*/meta.json`) |
|---|---|
| Action space | 128 discrete = 16 beams × 4 powers × 2 null states |
| Observation | 44 floats (8 observed users, flock summary, current beam/power, previous throughput/exposure, `log10(w)`) |
| Power levels | 44 / 47 / 50 / 53 dBm = 25119 / 50119 / 100000 / 199526 mW |
| Environment | `BeamEnv`, UMa path loss, 100 MHz, `dt = 0.1 s`, 600-step (60 s) episodes |
| Algorithm | Double DQN — **not** dueling: SB3's `DQN` implements Double Q-learning and has no dueling head |
| Network | MLP `[128, 128]` |
| Hyperparameters | lr 5e-4, buffer 50 000, batch 64, gamma 0.99, tau 1.0, target update 500, train_freq 4, gradient_steps 1, exploration 0.3 → 0.05, `double_q: true` |
| Reward | preference-conditioned: `log10(w)` resampled per episode in [−1, 2], `alpha 1.0`, `gamma 0.02`, `kappa 5.0`, clip 10 |

## 2. Intended use, and what this is not

**Intended use.** A research and teaching artifact for the FlywayDQN study: a
judge-facing demo of beam/power control under a precautionary exposure
constraint, and a starting point for longer training runs.

**Not intended for.** Deployment, regulatory compliance, or any claim about
real-world RF safety. It is not an exposure assessment tool.

**Scope limits, stated plainly:**

- **Phase A (synthetic) only.** The flock, its motion, and the served users are
  synthetic and deterministic per seed. **Phase B (radar-derived tracks) was
  never executed**, and the track-replay loader was not built, so no result here
  supports any claim of training on NEXRAD, eBird, or any real field data.
- **One sector, one array geometry, one carrier.** Nothing here demonstrates
  transfer to another geometry, band, or site.
- **The threshold is precautionary, not a safety limit.** Every scenario and the
  environment default use `s_thr_w_m2 = 0.03 W/m²`, which is far below the
  ICNIRP-style reference level of 10 W/m². It is a policy threshold chosen so the
  exposure objective is non-trivial at the modelled footprint density; the
  exposure penalty is a **smooth softplus hinge** on `p95 / s_thr`, not an
  exponential, and not a biological effect model.
- **Checkpoints are gitignored.** A fresh clone has no `dqn` controller at all —
  `create_app` registers it only when `seed_1` is readable, and
  `tests/unit/test_dqn_service.py` skips with a reason instead of failing.

## 3. The budget that is real, and why

Three numbers were floating around, and only one was ever executed:

| Recorded | Value | Status |
|---|---|---|
| `train_config.json` (interrupted first attempt) | 30 000 | stale, never produced a published artifact |
| `seed_1/meta.json` (the checkpoint that had been shipped) | 40 000 | **real — this is the budget** |
| `configs/train/train.yaml` (aspirational key) | 60 000 | no run ever executed it |

`configs/train/train.yaml`, `TrainingConfig.total_steps` and
`tests/unit/test_dqn_service.py` now all state **40 000** steps per seed, so a
re-run reproduces the artifacts instead of silently training something larger.

## 4. Training result (40 000 steps per seed, 3 seeds)

| Seed | Episodes completed | Mean episode return |
|---|---|---|
| 1 | 66 | 382.8748208157944 |
| 2 | 66 | 444.69009927863425 |
| 3 | 66 | 425.5482526082884 |

66 episodes × 600 steps = 39 600 of the 40 000 steps; the last episode does not
finish. The spread between the best and worst seed is **61.815** in mean return.
Per-episode series are in `episodes.csv`; the seed band and mean are in
`learning_curve.png`.

**Runtime, measured.** 428 s, 516 s and 318 s per seed (seeds 1–3, from artifact
timestamps), i.e. **21 min 3 s** for the three seeds and **24 min 28 s** for the
whole original run including its evaluation pass, on a 16-logical-core Windows
laptop (CPU only, `device: cpu`). The same 40 000-step budget measured in the
foreground on the same machine takes about 205 s, so background execution there
ran roughly 2× slower.

**Determinism.** Re-running seed 1 at 40 000 steps reproduced the previously
shipped `seed_1` checkpoint exactly in recorded metrics (`mean_episode_return`
382.8748208157944, 66 episodes) and in checkpoint file size. Content hashes were
not compared, so this is evidence of reproducibility, not a byte-identity proof.

## 5. Held-out evaluation

Measured with `--eval-only` on 8 reporting seeds (101–108, 300 steps each) that
are disjoint from both the training seeds and the validation seeds. Full
precision is in `metrics.csv`; the same table is `summary.md`.

| Policy | Mean sum rate (Mbps) | Total dose (W·s/m²) | Max p95 density (W/m²) | Mean fraction over threshold | Mean power (mW) | Null fraction |
|---|---|---|---|---|---|---|
| **dqn (seed 1)** — served, validation-selected | 4130.34 | **106.2338** | **0.061387** | **0.07454** | 197845.4 | 0.011 |
| dqn (seed 2) | 4122.98 | 6.6631 | 0.015501 | 0.00000 | 100000.0 | 1.000 |
| dqn (seed 3) | 4060.22 | 3.8431 | 0.064678 | 0.00066 | 116386.9 | 0.940 |
| unconstrained (no mitigation) | 4055.74 | 35.2838 | 0.044916 | 0.00098 | 199526.2 | 0.000 |
| static-backoff | 3782.59 | 17.8298 | 0.022511 | 0.00000 | 100912.3 | 0.000 |
| analytic-nulling | 4055.76 | 35.0470 | 0.044868 | 0.00066 | 199526.2 | 0.005 |
| random | 3716.38 | 12.9016 | 0.108733 | 0.00320 | 91747.1 | 0.506 |

`sector-shutdown` is absent from this table on purpose. It transmits **0 mW**, and
`BeamEnv`'s action space is a four-point **non-zero** power grid with no off
action, so the bridge could only snap it onto the lowest level (25 W) while the row
still read "shutdown". Its measured row (true 0 mW) lives in
`experiments/results/baselines.md`, taken on the demo runner, which applies
action power exactly.

**A correction, recorded rather than quietly applied.** An earlier revision of this
table scored the rule-based baselines through `BeamEnv.policy_state()`, which handed
the policy the power transmitted on the *previous* step as if it were the sector's
full power. A fixed-fraction back-off multiplied that down again every step —
199526 → 100000 → 50000 mW, pinning at the lowest level — which is why the earlier
`static-backoff` row read 27 051 mW and was nearly indistinguishable from the
shutdown row. `policy_state()` now returns the scene's configured setup, matching
`DemoRunner`, and every row above was re-measured with `--eval-only`.

Ratios below are arithmetic on the rows above:

- **Only 1 of 3 seeds reduces exposure.** Seed 2 cuts dose 5.3× (35.28 → 6.66)
  and peak density 2.9× (0.0449 → 0.0155) while *raising* mean sum rate 1.7%.
- **The served checkpoint is the worst of the three on exposure.** Seed 1 sits
  3.0× *above* the unconstrained baseline in dose, 1.37× above it in peak
  density, and is over the threshold on 7.5% of steps. Its null is on the air on
  1.1% of steps and it transmits at essentially full power (197 845 mW).
  Seed 3 also exceeds the baseline peak (1.44×) despite the lowest dose of any
  policy, because its null (on 94% of steps) is not landing on the peak cell.
- **Exposure behaviour spans 27.6× across seeds** (dose 3.84 → 106.23) and 4.2×
  in peak density (0.0155 → 0.0647). With three seeds, that spread is the
  headline: this controller's exposure result is not reproducible across seeds.
- **The best exposure reduction still belongs to the blanket baseline, and it is
  cheap.** Static back-off halves the unconstrained dose (35.2838 → 17.8298, 2.0×
  down) and its peak density (0.044916 → 0.022511, 2.0× down) for a 6.7% throughput
  cost (4056 → 3783 Mbps, mean power 100 912 mW = half of full). Sector shutdown
  goes further and is measured on the demo surface: across 600 rollouts of
  `experiments/results/baselines.md` it reaches a mean of 6.596e+04 mW and
  1414 Mbit/s, and 0 mW / 0 Mbit/s for the whole of `scenario_003`. A learned
  controller that only matches this is not a contribution.
- **`analytic-nulling` is effectively inert here.** Its guards decline on 99.5%
  of steps (null fraction 0.005) and it lands within 0.7% of doing nothing
  (35.047 vs 35.284). Beating it is therefore weak evidence of anything.
- `random` has the worst peak density of all policies (0.1087, 2.4× the
  unconstrained baseline).

## 6. Selection, and why it picked badly

The quoted checkpoint is chosen on a **validation split** (`selection_seeds =
[11, 12, 13, 14]`), scored by the environment's own mean reward, then reported on
the held-out seeds above — selecting on the training return would reward a
checkpoint for how well it fit the episodes it had already seen.

Validation mean reward: seed 1 **0.838**, seed 2 0.803, seed 3 0.787 → seed 1 is
quoted.

That selection picked the exposure-worst seed, which is a real finding and a
known failure mode: the reward is preference-conditioned and mixes throughput,
exposure and power, so a checkpoint can win on reward while being poor on the
exposure axis alone. **Do not read the validation ranking as an exposure
ranking.** A future run should select on the exposure metric it reports, or on
the reward at a fixed, high `w`.

## 7. Known failure modes

1.**Seed variance swamps the feature.** The spec's Phase 5 acceptance criterion asks for **≥5 seeds**; this run has **3** (seeds 1–3). Two additional seeds would be needed to fully satisfy the criterion; treat the exposure result as a first pass, not an estimate.
2. **The served seed is a poor default.** `dqn_policy.py` loads `seed_1`; on this
   evidence `seed_2` is the checkpoint that actually respects the constraint.
3. **Exposure and null usage are not the same thing.** Seed 3 nulls on 94% of
   steps and still exceeds the baseline peak — a null can miss the peak cell.
4. **Reward-based selection is a bad proxy for exposure** (section 6).
5. **The nulling baseline is a near no-op** in this environment (section 5), so
   this run does not establish that learning beats careful analytic mitigation.
6. **`null = 1` is a no-op when no flock is in range**, a known action-space
   redundancy that this run did not mask out (spec Decision D4).
7. **Not converged.** 40 000 steps is a laptop-CPU first pass — 66 episodes per
   seed, seed spread still 61.8 in return, no length sweep, no learning-rate or
   buffer study, and no result beyond the scenarios in `configs/demo/`.
8. **No run tracking service.** The plan's stack names MLflow (or W&B) for
   reward terms, exposure metrics and Q-value statistics. This run logs to
   `episodes.csv` and `metrics.csv` instead; `mlflow` and `optuna` are declared in
   `pyproject.toml` but nothing imports them, so there is no sweep either. The
   per-step series a tracking server would hold is in `episodes.csv`.
9. **No evaluation beyond one environment config.** The threshold sensitivity
   sweep (0.01 / 0.1 / 1 mW/cm²) has not been run, so nothing here says how the
   ranking behaves as the constraint moves.
10. **The baseline this controller has to beat was mis-measured until now.** An
   earlier revision scored the rule-based baselines through a bridge that handed
   them the previous step's power as "full power"; see section 5 for the
   correction. Section 5's rows are re-measured, and `sector-shutdown` is measured
   on the demo runner because this environment's power grid has no off action.

## 8. Reproducing

```bash
PYTHONPATH=src ./.venv311/Scripts/python.exe -m flywaydqn.agents.train_dqn
PYTHONPATH=src ./.venv311/Scripts/python.exe -m flywaydqn.agents.train_dqn --eval-only
```

`--eval-only` reloads the stored checkpoints, re-runs validation selection and
rewrites `metrics.csv` / `summary.md` without retraining (the training is
deterministic per seed, so retraining only to refresh a table would burn the same
CPU for an identical network). Use `.venv311` — it is the interpreter with
torch 2.5.1+cpu and SB3 2.8.0; `uv` is not on `PATH` here, so the `uv run`
targets in the `Makefile` do not work from this shell.

Splits: train `[1, 2, 3]`, validation `[11, 12, 13, 14]`, reporting
`[101, …, 108]`. `TrainingConfig` refuses any config where these overlap.
