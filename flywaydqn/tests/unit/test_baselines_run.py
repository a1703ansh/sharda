"""Unit tests for the WP-C baseline runner's artifacts and resume path."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest

from experiments import run_baselines as runner


def _patch_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(runner, "CSV_PATH", tmp_path / "baselines.csv")
    monkeypatch.setattr(runner, "MD_PATH", tmp_path / "baselines.md")
    monkeypatch.setattr(runner, "MANIFEST_PATH", tmp_path / "baselines_run.json")


def _fast_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "register_trained_controller", lambda: False)
    monkeypatch.setattr(runner, "available_policy_names", lambda: ["unconstrained"])


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_measurement_is_deterministic_and_csv_omits_wall_clock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_paths(monkeypatch, tmp_path)

    first = runner._measure_instance("unconstrained", "scenario_003", 20_000, 2)
    second = runner._measure_instance("unconstrained", "scenario_003", 20_000, 2)

    assert first.frames == 3
    assert first.horizon_steps == 2
    assert replace(first, wall_clock_s=0.0) == replace(second, wall_clock_s=0.0)

    runner._write_csv([first])
    header = (tmp_path / "baselines.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "wall_clock_s" not in header

    loaded = runner._read_csv()
    assert len(loaded) == 1
    assert loaded[0].wall_clock_s == 0.0
    assert replace(loaded[0], wall_clock_s=0.0) == replace(first, wall_clock_s=0.0)


def test_main_writes_artifacts_and_resume_does_not_duplicate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_paths(monkeypatch, tmp_path)
    _fast_registry(monkeypatch)

    args = [
        "--seed-count",
        "1",
        "--horizon",
        "2",
        "--scenario",
        "scenario_003",
        "--policy",
        "unconstrained",
    ]
    assert runner.main(args) == 0

    rows = _rows(tmp_path / "baselines.csv")
    assert len(rows) == 1
    assert "wall_clock_s" not in rows[0]
    markdown = (tmp_path / "baselines.md").read_text(encoding="utf-8")
    assert "| `unconstrained` |" in markdown
    assert "null_fraction" in markdown
    manifest = json.loads((tmp_path / "baselines_run.json").read_text(encoding="utf-8"))
    assert manifest["seed_count"] == 1
    assert manifest["horizon_steps"] == 2

    assert runner.main([*args, "--resume"]) == 0
    assert len(_rows(tmp_path / "baselines.csv")) == 1


def test_interrupted_run_keeps_finished_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_paths(monkeypatch, tmp_path)
    _fast_registry(monkeypatch)

    real_measure = runner._measure_instance
    calls = {"count": 0}

    def flaky_measure(policy: str, scenario: str, seed: int, horizon: int) -> runner.InstanceRow:
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("simulated interruption")
        return real_measure(policy, scenario, seed, horizon)

    monkeypatch.setattr(runner, "_measure_instance", flaky_measure)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        runner.main(
            [
                "--seed-count",
                "2",
                "--horizon",
                "1",
                "--scenario",
                "scenario_003",
                "--policy",
                "unconstrained",
            ]
        )

    rows = _rows(tmp_path / "baselines.csv")
    assert len(rows) == 1
    assert not (tmp_path / "baselines.md").exists()
    assert not (tmp_path / "baselines_run.json").exists()
