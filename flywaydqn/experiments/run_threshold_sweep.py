"""Threshold sensitivity sweep for the measured controllers.

README Section 8 asks for a threshold sensitivity table over 0.01/0.1/1 mW/cm^2
(= 0.1/1/10 W/m^2). In this simulation the exposure threshold does not change
the radio or the policy's action; it only moves the reference line used to
count frames whose peak footprint density is over threshold. The sweep therefore
rolls each policy/scenario/seed **once**, keeps the per-frame peak density, and
evaluates every threshold on those same frames. Nothing is re-simulated per
threshold, so a row can be traced to the same rollout as `baselines.csv`.

Outputs, all under ``experiments/results/``:

- ``threshold_sweep.csv``: one row per (policy, scenario, seed, threshold),
  written incrementally. The threshold columns are deterministic; wall-clock
  time lives in the manifest, not here.
- ``threshold_sweep.md``: mean +/- 95% CI per policy and threshold.
- ``threshold_sweep_run.json``: wall-clock time, command, and selection.

Run it with::

    python -m experiments.run_threshold_sweep
    python -m experiments.run_threshold_sweep --seed-count 10 --horizon 500
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

from experiments.run_baselines import (
    SEED_BASE,
    _cell,
    _mean_ci,
    _number,
    _roll_frames,
)
from flywaydqn.agents.dqn_policy import register_trained_controller
from flywaydqn.agents.policies import available_policy_names
from flywaydqn.demo.runner import list_demo_scenarios

RESULTS_DIR: Final[Path] = Path(__file__).resolve().parent / "results"
CSV_PATH: Final[Path] = RESULTS_DIR / "threshold_sweep.csv"
MD_PATH: Final[Path] = RESULTS_DIR / "threshold_sweep.md"
MANIFEST_PATH: Final[Path] = RESULTS_DIR / "threshold_sweep_run.json"

# 0.01/0.1/1 mW/cm^2 expressed in W/m^2 (1 mW/cm^2 = 10 W/m^2).
SPEC_THRESHOLDS_WM2: Final[tuple[float, ...]] = (0.1, 1.0, 10.0)
DEFAULT_SEED_COUNT: Final[int] = 10
DEFAULT_HORIZON: Final[int] = 500

_ROW_FIELDS: Final[tuple[str, ...]] = (
    "policy",
    "scenario",
    "seed",
    "threshold_wm2",
    "scenario_s_thr_wm2",
    "frames",
    "horizon_steps",
    "mean_throughput_mbps",
    "total_dose_wsm2",
    "max_s_wm2",
    "time_over_threshold",
    "mean_power_mw",
    "null_fraction",
    "mean_null_depth_db",
)


@dataclass(frozen=True)
class ThresholdRow:
    """One rollout evaluated at one threshold on that rollout's own frames."""

    policy: str
    scenario: str
    seed: int
    threshold_wm2: float
    scenario_s_thr_wm2: float
    frames: int
    horizon_steps: int
    mean_throughput_mbps: float
    total_dose_wsm2: float
    max_s_wm2: float
    time_over_threshold: float
    mean_power_mw: float
    null_fraction: float
    mean_null_depth_db: float


def _measure_sweep(
    policy_name: str,
    scenario_key: str,
    seed: int,
    horizon: int,
    thresholds: Sequence[float],
) -> list[ThresholdRow]:
    """Roll one policy/scenario/seed pair and evaluate every threshold on it."""
    frames = _roll_frames(policy_name, scenario_key, seed, horizon)
    count = len(frames)
    throughput = [frame.metrics.sum_throughput_mbps for frame in frames]
    dose = [frame.metrics.dose_wsm2 for frame in frames]
    peak = [frame.metrics.max_density_wm2 for frame in frames]
    power = [frame.power_mw for frame in frames]
    null_active = [1.0 if frame.null_active else 0.0 for frame in frames]
    null_depth = [frame.null_depth_db for frame in frames]

    return [
        ThresholdRow(
            policy=policy_name,
            scenario=scenario_key,
            seed=seed,
            threshold_wm2=threshold,
            scenario_s_thr_wm2=frames[0].metrics.s_thr_wm2,
            frames=count,
            horizon_steps=horizon,
            mean_throughput_mbps=statistics.fmean(throughput),
            total_dose_wsm2=math.fsum(dose),
            max_s_wm2=max(peak),
            time_over_threshold=sum(1 for value in peak if value > threshold) / count,
            mean_power_mw=statistics.fmean(power),
            null_fraction=statistics.fmean(null_active),
            mean_null_depth_db=statistics.fmean(null_depth),
        )
        for threshold in thresholds
    ]


def _csv_record(row: ThresholdRow) -> dict[str, object]:
    record = asdict(row)
    return {field: record[field] for field in _ROW_FIELDS}


def _write_csv(rows: Sequence[ThresholdRow]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_ROW_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow(_csv_record(row))


def _read_csv() -> list[ThresholdRow]:
    if not CSV_PATH.is_file():
        raise SystemExit(f"cannot read {CSV_PATH}: it does not exist")
    rows: list[ThresholdRow] = []
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            rows.append(
                ThresholdRow(
                    policy=record["policy"],
                    scenario=record["scenario"],
                    seed=int(record["seed"]),
                    threshold_wm2=float(record["threshold_wm2"]),
                    scenario_s_thr_wm2=float(record["scenario_s_thr_wm2"]),
                    frames=int(record["frames"]),
                    horizon_steps=int(record["horizon_steps"]),
                    mean_throughput_mbps=float(record["mean_throughput_mbps"]),
                    total_dose_wsm2=float(record["total_dose_wsm2"]),
                    max_s_wm2=float(record["max_s_wm2"]),
                    time_over_threshold=float(record["time_over_threshold"]),
                    mean_power_mw=float(record["mean_power_mw"]),
                    null_fraction=float(record["null_fraction"]),
                    mean_null_depth_db=float(record["mean_null_depth_db"]),
                )
            )
    if not rows:
        raise SystemExit(f"{CSV_PATH} has no rows; nothing to rebuild")
    return rows


def _write_markdown(
    rows: Sequence[ThresholdRow],
    *,
    policy_order: Sequence[str],
    scenarios: Sequence[str],
    seeds: Sequence[int],
    horizon: int,
    thresholds: Sequence[float],
    missing: Sequence[str],
) -> None:
    grouped: dict[tuple[str, float], list[ThresholdRow]] = {}
    for row in rows:
        grouped.setdefault((row.policy, row.threshold_wm2), []).append(row)

    lines: list[str] = [
        "# Threshold sensitivity — measured table",
        "",
        "Generated by `python -m experiments.run_threshold_sweep`. Every number below",
        "is an aggregate of `experiments/results/threshold_sweep.csv`; the CSV is the",
        "source of truth.",
        "",
        "Each policy/scenario/seed pair is rolled once. The exposure threshold does",
        "not change the radio or the policy action in this simulation, so the",
        "throughput, dose, max S, power, and null columns are identical across",
        "thresholds for the same rollout; only `fraction of frames over threshold`",
        "moves. That fraction uses the per-frame peak footprint density.",
        "",
        "## Coverage",
        "",
        "| threshold (W/m²) | threshold (mW/cm²) | what it is |",
        "|---|---|---|",
    ]
    for threshold in thresholds:
        lines.append(
            f"| {_number(threshold)} | {_number(threshold / 10.0)} | "
            + ("spec sensitivity point" if threshold in SPEC_THRESHOLDS_WM2 else "custom")
            + " |"
        )

    lines += [
        "",
        f"- Horizon: {horizon} steps advanced per rollout (+1 initial frame), dt = 0.1 s",
        f"- Scenarios ({len(scenarios)}): {', '.join(f'`{key}`' for key in scenarios)}",
        f"- Seeds ({len(seeds)}): `{min(seeds)}`–`{max(seeds)}`",
        f"- Rows: {len(rows)} "
        f"({len(policy_order)} policies × {len(scenarios)} scenarios × {len(seeds)} seeds "
        f"× {len(thresholds)} thresholds)",
        "",
        "## Mean ± 95% CI over all rollouts",
        "",
        "| policy | threshold (W/m²) | mean sum rate (Mbit/s) | total dose (W·s/m²) "
        "| max S (W/m²) | fraction of frames over threshold | mean power (mW) |",
        "|---|---|---|---|---|---|---|",
    ]
    for policy in policy_order:
        for threshold in thresholds:
            group = grouped.get((policy, threshold))
            if not group:
                continue
            cells = [
                _cell(_mean_ci([getattr(row, metric) for row in group]))
                for metric in (
                    "mean_throughput_mbps",
                    "total_dose_wsm2",
                    "max_s_wm2",
                    "time_over_threshold",
                    "mean_power_mw",
                )
            ]
            lines.append(f"| `{policy}` | {_number(threshold)} | " + " | ".join(cells) + " |")

    lines += ["", "## Notes", ""]
    crossing = max((row.time_over_threshold for row in rows), default=0.0)
    scenario_thresholds = sorted({row.scenario_s_thr_wm2 for row in rows})
    if crossing == 0.0 and rows:
        lowest = min(thresholds)
        lines.append(
            f"- **No measured frame reaches the lowest sweep threshold ({_number(lowest)} "
            "W/m²).** This is a finding, not a missing column: the modelled peak "
            "footprint densities stay below every spec sensitivity point."
        )
    lines.append(
        "- The demo scenarios use their own configurable threshold "
        f"({', '.join(_number(value) + ' W/m²' for value in scenario_thresholds)}); "
        "the baseline table measures crossings at that threshold."
    )
    lines.append(
        "- The cell-level fraction (`mean_frac_over`) needs the full footprint grid "
        "and is not recomputed here; `baselines.csv` carries it at the scenario "
        "threshold."
    )
    lines.append(
        "- `null_fraction` is the check on whether the window measured mitigation: if "
        "it is 0, the horizon ended before the null engaged."
    )
    lines.append(
        "- Wall-clock time and the exact command are in `threshold_sweep_run.json`, "
        "not in the deterministic CSV."
    )
    for name in missing:
        lines.append(f"- **Not measured:** `{name}` — not implemented yet, so absent by fact.")

    lines += [
        "",
        "## Command",
        "",
        "```",
        "python -m experiments.run_threshold_sweep"
        f" --horizon {horizon} --seed-count {len(seeds)}",
        "```",
        "",
    ]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def _write_manifest(
    *,
    wall_clock_s: float,
    thresholds: Sequence[float],
    policies: Sequence[str],
    scenarios: Sequence[str],
    seeds: Sequence[int],
    horizon: int,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "command": (
            "python -m experiments.run_threshold_sweep"
            f" --horizon {horizon} --seed-count {len(seeds)}"
            + "".join(f" --threshold {_number(value)}" for value in thresholds)
        ),
        "wall_clock_s": round(wall_clock_s, 3),
        "thresholds_wm2": list(thresholds),
        "thresholds_mw_cm2": [value / 10.0 for value in thresholds],
        "policies": list(policies),
        "scenarios": list(scenarios),
        "seed_first": min(seeds) if seeds else None,
        "seed_last": max(seeds) if seeds else None,
        "seed_count": len(seeds),
        "horizon_steps": horizon,
        "csv": CSV_PATH.name,
        "md": MD_PATH.name,
    }
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep exposure thresholds over the measured controllers' frames."
    )
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--seed-count", type=int, default=DEFAULT_SEED_COUNT)
    parser.add_argument("--scenario", action="append", default=None)
    parser.add_argument("--policy", action="append", default=None)
    parser.add_argument(
        "--threshold",
        type=float,
        action="append",
        default=None,
        help=(
            "exposure threshold in W/m²; repeatable "
            f"(default: {', '.join(_number(value) for value in SPEC_THRESHOLDS_WM2)})"
        ),
    )
    parser.add_argument(
        "--from-csv",
        action="store_true",
        help="rebuild threshold_sweep.md from an existing CSV without re-running",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.horizon < 1:
        raise SystemExit("--horizon must be at least 1")
    if args.seed_count < 1:
        raise SystemExit("--seed-count must be at least 1")

    if args.from_csv:
        stored = _read_csv()
        policies = sorted({row.policy for row in stored})
        scenarios = sorted({row.scenario for row in stored})
        seeds = sorted({row.seed for row in stored})
        stored_thresholds = sorted({row.threshold_wm2 for row in stored})
        horizon = stored[0].horizon_steps
        _write_markdown(
            stored,
            policy_order=policies,
            scenarios=scenarios,
            seeds=seeds,
            horizon=horizon,
            thresholds=stored_thresholds,
            missing=[],
        )
        print(f"[threshold-sweep] rebuilt {MD_PATH} from {CSV_PATH} ({len(stored)} rows)")
        return 0

    thresholds = tuple(args.threshold) if args.threshold else SPEC_THRESHOLDS_WM2
    if any(not math.isfinite(value) or value <= 0.0 for value in thresholds):
        raise SystemExit("--threshold values must be finite and positive")
    thresholds = tuple(dict.fromkeys(thresholds))

    started = time.perf_counter()
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
        f"[threshold-sweep] {len(policies)} policies x {len(scenarios)} scenarios x "
        f"{len(seeds)} seeds = {rollouts} rollouts, {len(thresholds)} thresholds each"
    )
    print(f"[threshold-sweep] policies: {', '.join(policies)}")
    if not has_dqn:
        print("[threshold-sweep] note: no readable checkpoint, so the 'dqn' rows are absent")

    rows: list[ThresholdRow] = []
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_ROW_FIELDS))
        writer.writeheader()
        handle.flush()
        for policy in policies:
            policy_started = time.perf_counter()
            for scenario in scenarios:
                for seed in seeds:
                    measured = _measure_sweep(policy, scenario, seed, args.horizon, thresholds)
                    for row in measured:
                        writer.writerow(_csv_record(row))
                    handle.flush()
                    rows.extend(measured)
            elapsed = time.perf_counter() - policy_started
            print(
                f"[threshold-sweep] {policy}: "
                f"{len(scenarios) * len(seeds)} rollouts in {elapsed:.1f}s"
            )

    if not rows:
        raise SystemExit("no rollouts were measured; nothing written")

    measured_policies = [name for name in policies if any(row.policy == name for row in rows)]
    measured_scenarios = [name for name in scenarios if any(row.scenario == name for row in rows)]
    wall = time.perf_counter() - started
    _write_manifest(
        wall_clock_s=wall,
        thresholds=thresholds,
        policies=measured_policies,
        scenarios=measured_scenarios,
        seeds=seeds,
        horizon=args.horizon,
    )
    _write_markdown(
        rows,
        policy_order=measured_policies,
        scenarios=measured_scenarios,
        seeds=seeds,
        horizon=args.horizon,
        thresholds=thresholds,
        missing=[],
    )
    print(f"[threshold-sweep] wrote {CSV_PATH} ({len(rows)} rows) and {MD_PATH}")
    print(f"[threshold-sweep] wall clock {wall:.1f}s")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
