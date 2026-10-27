"""Measure every controller on the demo surface and write the comparison table.

WP-C: the table the learned controller has to beat. Every row is a rollout of
the same :class:`~flywaydqn.demo.runner.DemoRunner` the dashboard streams, over
the repository's own scenario files.

Surface choice (deliberate)
---------------------------
The demo runner is used rather than
:class:`~flywaydqn.env.beam_env.BeamEnv` because the runner applies a policy's
:class:`~flywaydqn.agents.baselines_demo.BeamAction` *exactly* -- its
``power_mw`` (including ``0``) and its ``weights``. The environment's action
space is a four-point **non-zero** power grid with no off action, so a sector
shutdown routed through it could only be snapped onto the *lowest* non-zero
level (25 W) while the row still read "sector shutdown". Anything that claims a
sector is off has to be measured where it can actually be off.

The controller set is taken from the live registry, after the same
:func:`~flywaydqn.agents.dqn_policy.register_trained_controller` call the
service makes at startup, so the controllers measured here are the controllers
``GET /policies`` offers -- one list, not two.

Outputs, all under ``experiments/results/``:

- ``baselines.csv``: one row per (policy, scenario, seed), appended and flushed
  as each rollout finishes. Source of truth for every number in the markdown
  table. Deliberately excludes wall-clock time, so a clean re-run is
  byte-identical.
- ``baselines.md``: mean +/- 95% confidence interval per policy, a per-scenario
  breakdown, the controller coverage against the spec's baseline list, and the
  exact command that regenerated them.
- ``baselines_run.json``: the measured run's wall-clock time, command, and
  selection. Timing lives here rather than in the deterministic CSV.

Run it with::

    python -m experiments.run_baselines
    python -m experiments.run_baselines --seed-count 100 --horizon 300

Units: throughput Mbit/s, dose W*s/m^2, power density W/m^2, power mW, time s.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from flywaydqn.agents.dqn_policy import register_trained_controller
from flywaydqn.agents.policies import available_policy_names
from flywaydqn.demo.runner import (
    DemoRunner,
    ScenarioRequest,
    list_demo_scenarios,
    load_scenario,
)
from flywaydqn.service.schemas import StateFrame

RESULTS_DIR: Final[Path] = Path(__file__).resolve().parent / "results"
CSV_PATH: Final[Path] = RESULTS_DIR / "baselines.csv"
MD_PATH: Final[Path] = RESULTS_DIR / "baselines.md"
MANIFEST_PATH: Final[Path] = RESULTS_DIR / "baselines_run.json"

# A fixed, reproducible seed block. It is disjoint from the trainer's splits
# (training 1-3, validation 11-14, reporting 101-108) and from the shipped
# scenarios' own seeds, so refreshing this table never re-measures a scene the
# controller was fitted on.
SEED_BASE: Final[int] = 20_000

DEFAULT_SEED_COUNT: Final[int] = 40
DEFAULT_HORIZON: Final[int] = 120

# The spec's baseline list (README Phase 4), so the table can state plainly
# which entries exist and which are still missing instead of quietly shipping a
# short list.
SPEC_BASELINES: Final[tuple[tuple[str, str], ...]] = (
    ("unconstrained", "max-throughput SON: greedy beam, full power, no mitigation"),
    ("static-backoff", "static back-off: fixed power reduction while the flock is in range"),
    ("sector-shutdown", "sector shutdown: transmitter off while the flock is in range"),
    ("analytic-nulling", "analytic nulling: closed-form projection null on the peak cell"),
    ("ppo", "PPO trained on the same environment"),
)

# Columns of one CSV row, in file order. Aggregates in the markdown are means
# and confidence intervals over these rows and nothing else.
_ROW_FIELDS: Final[tuple[str, ...]] = (
    "policy",
    "scenario",
    "seed",
    "frames",
    "horizon_steps",
    "s_thr_wm2",
    "mean_throughput_mbps",
    "total_dose_wsm2",
    "max_s_wm2",
    "mean_s_p95_wm2",
    "mean_frac_over",
    "time_over_threshold",
    "mean_power_mw",
    "null_fraction",
    "mean_null_depth_db",
)

# Metrics aggregated per policy for the headline table.
_AGGREGATED: Final[tuple[str, ...]] = (
    "mean_throughput_mbps",
    "total_dose_wsm2",
    "max_s_wm2",
    "mean_frac_over",
    "time_over_threshold",
    "mean_power_mw",
)


@dataclass(frozen=True)
class InstanceRow:
    """One measured rollout: a policy on one scenario/seed pair.

    Every field is read off frames the runner actually produced. ``total_dose``
    and ``max_s`` are rollout aggregates (sum and maximum over the frames);
    the rest are means over the frames.
    """

    policy: str
    scenario: str
    seed: int
    frames: int
    horizon_steps: int
    s_thr_wm2: float
    mean_throughput_mbps: float
    total_dose_wsm2: float
    max_s_wm2: float
    mean_s_p95_wm2: float
    mean_frac_over: float
    time_over_threshold: float
    mean_power_mw: float
    null_fraction: float
    mean_null_depth_db: float
    wall_clock_s: float


def _roll_frames(
    policy_name: str, scenario_key: str, seed: int, horizon: int
) -> list[StateFrame]:
    """Roll one policy through one scenario/seed pair and keep every frame."""
    scenario = load_scenario(scenario_key)
    request = ScenarioRequest(
        key=scenario_key,
        seed=seed,
        weight=scenario.weight,
        policy=policy_name,
    )
    runner = DemoRunner(request)
    frames = [runner.initial_frame()]
    for _ in range(horizon):
        frames.append(runner.step())
    return frames


def _measure_instance(policy_name: str, scenario_key: str, seed: int, horizon: int) -> InstanceRow:
    """Roll one policy through one scenario/seed pair and reduce its frames.

    Raises rather than returning a partial row: a policy that dies mid-rollout
    would otherwise be averaged as if it had served the whole horizon.
    """
    started = time.perf_counter()
    frames = _roll_frames(policy_name, scenario_key, seed, horizon)
    wall_clock_s = time.perf_counter() - started

    count = len(frames)
    threshold = frames[0].metrics.s_thr_wm2
    throughput = [frame.metrics.sum_throughput_mbps for frame in frames]
    dose = [frame.metrics.dose_wsm2 for frame in frames]
    peak = [frame.metrics.max_density_wm2 for frame in frames]
    p95 = [frame.metrics.p95_density_wm2 for frame in frames]
    frac_over = [frame.metrics.frac_over for frame in frames]
    power = [frame.power_mw for frame in frames]
    null_active = [1.0 if frame.null_active else 0.0 for frame in frames]
    null_depth = [frame.null_depth_db for frame in frames]

    return InstanceRow(
        policy=policy_name,
        scenario=scenario_key,
        seed=seed,
        frames=count,
        horizon_steps=horizon,
        s_thr_wm2=threshold,
        mean_throughput_mbps=statistics.fmean(throughput),
        total_dose_wsm2=math.fsum(dose),
        max_s_wm2=max(peak),
        mean_s_p95_wm2=statistics.fmean(p95),
        mean_frac_over=statistics.fmean(frac_over),
        # "Fraction of time over threshold" (spec Section 8): the share of
        # frames whose peak footprint density exceeds the scenario threshold.
        time_over_threshold=sum(1 for value in peak if value > threshold) / count,
        mean_power_mw=statistics.fmean(power),
        null_fraction=statistics.fmean(null_active),
        # `null_depth_db` is exactly 0.0 whenever no null is on the air, so this
        # mean is already conditional on transmission and never implies a
        # measured depth that did not happen.
        mean_null_depth_db=statistics.fmean(null_depth),
        wall_clock_s=wall_clock_s,
    )


def _mean_ci(values: Sequence[float]) -> tuple[float, float]:
    """Return ``(mean, half-width of the 95% CI)`` for ``values``."""
    if not values:
        return math.nan, 0.0
    if len(values) < 2:
        return values[0], 0.0
    return statistics.fmean(values), 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def _number(value: float, digits: int = 4) -> str:
    """Format a number, using ``—`` for a value that could not be computed."""
    if value is None or not math.isfinite(value):
        return "—"
    if value == 0.0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1000.0 or magnitude < 1e-4:
        return f"{value:.{digits}g}"
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def _cell(pair: tuple[float, float], digits: int = 4) -> str:
    """Render ``mean ± ci`` for one aggregated metric."""
    mean, half_width = pair
    if not math.isfinite(mean):
        return "—"
    if half_width == 0.0:
        return _number(mean, digits)
    return f"{_number(mean, digits)} ± {_number(half_width, digits)}"


def _csv_record(row: InstanceRow) -> dict[str, object]:
    """The CSV projection of a row: wall-clock time is intentionally absent."""
    record = asdict(row)
    return {field: record[field] for field in _ROW_FIELDS}


def _write_csv(rows: Sequence[InstanceRow]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_ROW_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow(_csv_record(row))


def _write_manifest(
    *,
    wall_clock_s: float,
    policies: Sequence[str],
    scenarios: Sequence[str],
    seeds: Sequence[int],
    horizon: int,
    resumed: bool,
) -> None:
    """Timing and selection for the run, kept out of the deterministic CSV."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "command": (
            "python -m experiments.run_baselines"
            f" --horizon {horizon} --seed-count {len(seeds)}"
            + (" --resume" if resumed else "")
        ),
        "wall_clock_s": round(wall_clock_s, 3),
        "policies": list(policies),
        "scenarios": list(scenarios),
        "seed_first": min(seeds) if seeds else None,
        "seed_last": max(seeds) if seeds else None,
        "seed_count": len(seeds),
        "horizon_steps": horizon,
        "resumed": resumed,
        "csv": CSV_PATH.name,
        "md": MD_PATH.name,
    }
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _by_policy(rows: Sequence[InstanceRow]) -> dict[str, list[InstanceRow]]:
    grouped: dict[str, list[InstanceRow]] = {}
    for row in rows:
        grouped.setdefault(row.policy, []).append(row)
    return grouped


def _write_markdown(
    rows: Sequence[InstanceRow],
    *,
    policy_order: Sequence[str],
    scenarios: Sequence[str],
    seeds: Sequence[int],
    horizon: int,
    missing: Sequence[str],
) -> None:
    """Write the human-readable table; every cell comes from ``rows``."""
    grouped = _by_policy(rows)
    lines: list[str] = [
        "# Baselines — measured comparison table",
        "",
        "Generated by `python -m experiments.run_baselines`. Every number below is",
        "an aggregate of `experiments/results/baselines.csv`, which holds one row per",
        "policy × scenario × seed; the CSV is the source of truth.",
        "",
        "Surface: `flywaydqn.demo.runner.DemoRunner` — the same frame producer the",
        "dashboard streams. It applies each policy's action exactly (transmit power",
        "including a shut-down 0 mW, and the transmitted weights), which the",
        "environment's four-point **non-zero** power grid cannot represent.",
        "",
        "## Coverage of the spec's baseline list",
        "",
        "| baseline | present in this table | what it is |",
        "|---|---|---|",
    ]
    measured = set(grouped)
    for name, description in SPEC_BASELINES:
        mark = "yes" if name in measured else "**no**"
        lines.append(f"| `{name}` | {mark} | {description} |")
    for name in policy_order:
        if name not in {entry[0] for entry in SPEC_BASELINES}:
            lines.append(f"| `{name}` | yes | trained controller served by the service |")

    lines += [
        "",
        f"- Horizon: {horizon} steps advanced per rollout (+1 initial frame), dt = 0.1 s",
        f"- Scenarios ({len(scenarios)}): {', '.join(f'`{key}`' for key in scenarios)}",
        f"- Seeds ({len(seeds)}): `{SEED_BASE}`–`{SEED_BASE + len(seeds) - 1}`",
        f"- Measured rollouts (rows): {len(rows)} "
        f"({len(policy_order)} policies × {len(scenarios)} scenarios × {len(seeds)} seeds)",
        "",
        "## Mean ± 95% CI over all rollouts",
        "",
        "| policy | mean sum rate (Mbit/s) | total dose (W·s/m²) | max S (W/m²) "
        "| mean frac over | fraction of time over threshold | mean power (mW) |",
        "|---|---|---|---|---|---|---|",
    ]
    for policy in policy_order:
        group = grouped.get(policy)
        if not group:
            continue
        cells = [_cell(_mean_ci([getattr(row, metric) for row in group])) for metric in _AGGREGATED]
        lines.append(f"| `{policy}` | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Per-scenario breakdown (mean, no CI)",
        "",
        "| policy | scenario | n | mean sum rate (Mbit/s) | total dose (W·s/m²) "
        "| max S (W/m²) | fraction of time over threshold | mean power (mW) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for policy in policy_order:
        for scenario in scenarios:
            group = [row for row in grouped.get(policy, []) if row.scenario == scenario]
            if not group:
                continue
            cells = [
                _number(statistics.fmean([getattr(row, metric) for row in group]))
                for metric in ("mean_throughput_mbps", "total_dose_wsm2", "max_s_wm2")
            ]
            over = statistics.fmean([row.time_over_threshold for row in group])
            power = statistics.fmean([row.mean_power_mw for row in group])
            lines.append(
                f"| `{policy}` | `{scenario}` | {len(group)} | "
                + " | ".join(cells)
                + f" | {_number(over)} | {_number(power)} |"
            )

    lines += ["", "## Notes", ""]
    lines.append(
        "- Rollouts are paired: every policy sees the same scenario/seed pair, so the "
        "comparison is a paired one rather than two independent sample sets."
    )
    lines.append(
        "- `null_fraction` and `mean_null_depth_db` are per-row columns of the CSV; "
        "`null_depth_db` is exactly 0 on frames with no null on the air."
    )
    lines.append(
        "- The DQN row is the checkpoint the service serves "
        "(`FLYWAYDQN_CHECKPOINT_DIR` overrides it), not a re-trained or re-selected one."
    )
    lines.append(
        "- The CSV excludes wall-clock time so a clean re-run is byte-identical; "
        "the run's timing and exact selection are in `baselines_run.json`."
    )
    lines.append(
        "- `null_fraction` is the check on whether a window measured mitigation: if it "
        "is 0, the horizon ended before the null engaged and those exposure columns "
        "are a no-null result, not a nulling result."
    )
    # State the threshold outcome from the rows rather than leaving a run of
    # zeroes to be read as a missing column.
    crossed = max((row.time_over_threshold for row in rows), default=0.0)
    threshold = rows[0].s_thr_wm2 if rows else math.nan
    if crossed == 0.0 and rows:
        worst_p95 = max(row.mean_s_p95_wm2 for row in rows)
        lines.append(
            f"- **No rollout crossed the scenario threshold.** The largest mean footprint "
            f"p95 anywhere in the table is {_number(worst_p95, 3)} W/m² against a "
            f"{_number(threshold, 3)} W/m² threshold, so `fraction of time over threshold` "
            f"is 0 on every row by fact, not by omission. A {horizon}-step window is "
            f"{horizon * 0.1:.0f} s of simulated time, which ends before the flocks reach "
            f"closest approach; the exposure axis still separates the policies through "
            f"`total dose` and `max S`. Raise `--horizon` to move the table into the "
            f"threshold-crossing regime."
        )
    for name in missing:
        lines.append(f"- **Not measured:** `{name}` — not implemented yet, so absent by fact.")

    lines += [
        "",
        "## Command",
        "",
        "```bash",
        "cd flywaydqn",
        f".venv311/Scripts/python.exe -m experiments.run_baselines --horizon {horizon} "
        f"--seed-count {len(seeds)}",
        "```",
        "",
    ]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure the demo controllers and write experiments/results/baselines.*"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help=f"steps advanced per rollout (default {DEFAULT_HORIZON})",
    )
    parser.add_argument(
        "--seed-count",
        type=int,
        default=DEFAULT_SEED_COUNT,
        help=f"seeds per scenario (default {DEFAULT_SEED_COUNT})",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        help="scenario key to measure; repeatable (default: every shipped scenario)",
    )
    parser.add_argument(
        "--policy",
        action="append",
        default=None,
        help="policy name to measure; repeatable (default: the whole registry)",
    )
    parser.add_argument(
        "--from-csv",
        action="store_true",
        help=(
            "skip measuring and rebuild baselines.md from the existing "
            "baselines.csv, so a presentation change does not cost a re-run"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "keep the rows already in baselines.csv and measure only the missing "
            "(policy, scenario, seed) pairs; requires the same horizon"
        ),
    )
    return parser.parse_args(argv)


def _csv_header() -> tuple[str, ...]:
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        return tuple(next(csv.reader(handle)))


def _read_csv(*, allow_empty: bool = False) -> list[InstanceRow]:
    """Load the stored rows, so the markdown can be rebuilt without re-running."""
    if not CSV_PATH.is_file():
        raise SystemExit(f"cannot read {CSV_PATH}: it does not exist")
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: list[InstanceRow] = []
        for record in reader:
            rows.append(
                InstanceRow(
                    policy=record["policy"],
                    scenario=record["scenario"],
                    seed=int(record["seed"]),
                    frames=int(record["frames"]),
                    horizon_steps=int(record["horizon_steps"]),
                    s_thr_wm2=float(record["s_thr_wm2"]),
                    mean_throughput_mbps=float(record["mean_throughput_mbps"]),
                    total_dose_wsm2=float(record["total_dose_wsm2"]),
                    max_s_wm2=float(record["max_s_wm2"]),
                    mean_s_p95_wm2=float(record["mean_s_p95_wm2"]),
                    mean_frac_over=float(record["mean_frac_over"]),
                    time_over_threshold=float(record["time_over_threshold"]),
                    mean_power_mw=float(record["mean_power_mw"]),
                    null_fraction=float(record["null_fraction"]),
                    mean_null_depth_db=float(record["mean_null_depth_db"]),
                    wall_clock_s=float(record.get("wall_clock_s") or 0.0),
                )
            )
    if not rows and not allow_empty:
        raise SystemExit(f"{CSV_PATH} has no rows; nothing to rebuild")
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    """Measure every selected controller and write both artifacts."""
    args = _parse_args(argv)
    if args.horizon < 1:
        raise SystemExit("--horizon must be at least 1")
    if args.seed_count < 1:
        raise SystemExit("--seed-count must be at least 1")
    if args.from_csv and args.resume:
        raise SystemExit("--from-csv and --resume are mutually exclusive")

    if args.from_csv:
        # Rebuild only the presentation: the CSV keeps the measurements, the
        # markdown is a view over them.
        stored = _read_csv()
        policies_in_csv = sorted({row.policy for row in stored}, key=lambda name: name)
        scenarios_in_csv = sorted({row.scenario for row in stored})
        seeds_in_csv = sorted({row.seed for row in stored})
        horizon = stored[0].horizon_steps
        missing = [name for name, _ in SPEC_BASELINES if name not in set(policies_in_csv)]
        _write_markdown(
            stored,
            policy_order=policies_in_csv,
            scenarios=scenarios_in_csv,
            seeds=seeds_in_csv,
            horizon=horizon,
            missing=missing,
        )
        print(f"[baselines] rebuilt {MD_PATH} from {CSV_PATH} ({len(stored)} rows)")
        return 0

    started = time.perf_counter()

    # Same registration the service performs at startup, so the measured
    # controllers are exactly the served ones. Absent checkpoint -> the `dqn`
    # row is skipped with a printed reason, never fabricated.
    has_dqn = register_trained_controller()
    registry = list(available_policy_names())

    if args.policy:
        unknown = sorted(set(args.policy) - set(registry))
        if unknown:
            raise SystemExit(f"unknown policy name(s) {unknown}; registry offers {registry}")
        policies = [name for name in registry if name in set(args.policy)]
    else:
        policies = registry

    shipped = list_demo_scenarios()
    known_scenarios = {info.key for info in shipped}
    scenarios = list(args.scenario) if args.scenario else sorted(known_scenarios)
    unknown_scenarios = sorted(set(scenarios) - known_scenarios)
    if unknown_scenarios:
        raise SystemExit(
            f"unknown scenario key(s) {unknown_scenarios}; shipped: {sorted(known_scenarios)}"
        )

    seeds = [SEED_BASE + offset for offset in range(args.seed_count)]
    rollouts = len(policies) * len(scenarios) * len(seeds)
    print(
        f"[baselines] {len(policies)} policies x {len(scenarios)} scenarios x {len(seeds)} seeds "
        f"= {rollouts} rollouts of {args.horizon} steps"
    )
    print(f"[baselines] policies: {', '.join(policies)}")
    if not has_dqn:
        print("[baselines] note: no readable checkpoint, so the 'dqn' row is absent")

    stored_rows: list[InstanceRow] = []
    done: set[tuple[str, str, int]] = set()
    if args.resume and CSV_PATH.is_file():
        if _csv_header() != _ROW_FIELDS:
            raise SystemExit(
                f"{CSV_PATH} uses an older column set; rerun without --resume "
                "or move it aside"
            )
        stored_rows = _read_csv(allow_empty=True)
        horizons = {row.horizon_steps for row in stored_rows}
        if stored_rows and horizons != {args.horizon}:
            raise SystemExit(
                f"--resume needs horizon {sorted(horizons)} from the stored rows, "
                f"not {args.horizon}"
            )
        done = {(row.policy, row.scenario, row.seed) for row in stored_rows}
        if done:
            print(f"[baselines] resume: {len(done)} rows already stored")

    new_rows: list[InstanceRow] = []
    # Open before the first rollout and flush after each row: an interrupted
    # run leaves every finished measurement behind, not an empty file.
    mode = "a" if stored_rows else "w"
    with CSV_PATH.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_ROW_FIELDS))
        if mode == "w":
            writer.writeheader()
            handle.flush()
        for policy in policies:
            policy_started = time.perf_counter()
            measured = 0
            for scenario in scenarios:
                for seed in seeds:
                    if (policy, scenario, seed) in done:
                        continue
                    row = _measure_instance(policy, scenario, seed, args.horizon)
                    writer.writerow(_csv_record(row))
                    handle.flush()
                    new_rows.append(row)
                    measured += 1
            elapsed = time.perf_counter() - policy_started
            print(
                f"[baselines] {policy}: {measured} measured"
                f" ({len(scenarios) * len(seeds) - measured} already stored)"
                f" in {elapsed:.1f}s"
            )

    rows = stored_rows + new_rows
    if not rows:
        # Never leave an empty table behind: an empty CSV next to a "mean ± CI"
        # header is exactly the kind of artifact this rule exists to prevent.
        raise SystemExit("no rollouts were measured; nothing written")

    measured_policies = [name for name in policies if any(row.policy == name for row in rows)]
    measured_policies += sorted(
        {row.policy for row in rows} - set(measured_policies)
    )
    measured_scenarios = [name for name in scenarios if any(row.scenario == name for row in rows)]
    measured_scenarios += sorted(
        {row.scenario for row in rows} - set(measured_scenarios)
    )
    measured_seeds = sorted({row.seed for row in rows})
    missing = [name for name, _ in SPEC_BASELINES if name not in set(measured_policies)]

    wall = time.perf_counter() - started
    _write_manifest(
        wall_clock_s=wall,
        policies=measured_policies,
        scenarios=measured_scenarios,
        seeds=measured_seeds,
        horizon=args.horizon,
        resumed=args.resume,
    )
    _write_markdown(
        rows,
        policy_order=measured_policies,
        scenarios=measured_scenarios,
        seeds=measured_seeds,
        horizon=args.horizon,
        missing=missing,
    )

    print(f"[baselines] wrote {CSV_PATH} ({len(rows)} rows) and {MD_PATH}")
    print(f"[baselines] wall clock {wall:.1f}s")
    if missing:
        print(f"[baselines] spec baselines not measured (not implemented): {', '.join(missing)}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
