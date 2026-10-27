"""Unit tests for the threshold-sensitivity artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from experiments import run_threshold_sweep as sweep


def _patch_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sweep, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(sweep, "CSV_PATH", tmp_path / "threshold_sweep.csv")
    monkeypatch.setattr(sweep, "MD_PATH", tmp_path / "threshold_sweep.md")
    monkeypatch.setattr(sweep, "MANIFEST_PATH", tmp_path / "threshold_sweep_run.json")


def _fast_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sweep, "register_trained_controller", lambda: False)
    monkeypatch.setattr(sweep, "available_policy_names", lambda: ["unconstrained"])


def _args() -> list[str]:
    return [
        "--seed-count",
        "1",
        "--horizon",
        "2",
        "--scenario",
        "scenario_003",
        "--policy",
        "unconstrained",
    ]


def test_sweep_reuses_frames_and_writes_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_paths(monkeypatch, tmp_path)
    _fast_registry(monkeypatch)
    monkeypatch.setattr(sweep, "SPEC_THRESHOLDS_WM2", (0.01, 0.03))

    first = sweep._measure_sweep("unconstrained", "scenario_003", 20_000, 2, (0.01, 0.03))
    second = sweep._measure_sweep("unconstrained", "scenario_003", 20_000, 2, (0.01, 0.03))
    assert first == second
    assert len(first) == 2
    assert all(0.0 <= row.time_over_threshold <= 1.0 for row in first)
    assert first[0].mean_throughput_mbps == first[1].mean_throughput_mbps
    assert first[0].total_dose_wsm2 == first[1].total_dose_wsm2

    assert sweep.main(_args()) == 0
    with (tmp_path / "threshold_sweep.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert {float(row["threshold_wm2"]) for row in rows} == {0.01, 0.03}
    markdown = (tmp_path / "threshold_sweep.md").read_text(encoding="utf-8")
    assert "# Threshold sensitivity" in markdown
    assert "`unconstrained`" in markdown
    manifest = json.loads((tmp_path / "threshold_sweep_run.json").read_text(encoding="utf-8"))
    assert manifest["thresholds_wm2"] == [0.01, 0.03]
    assert manifest["seed_count"] == 1

    assert sweep.main(["--from-csv"]) == 0
    assert len((tmp_path / "threshold_sweep.csv").read_text(encoding="utf-8").splitlines()) == 3
