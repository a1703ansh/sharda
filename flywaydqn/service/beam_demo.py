#!/usr/bin/env python3
"""FlywayDQN quick online-round demo scene.

One stand-alone server that:
- runs a real RF scene (sector UPA + DFT codebook + projection null + UMa path loss
  + SINR/throughput + ICNIRP exposure) over a seeded flock scenario,
- serves a live HTML page that shows the current state without any client build tooling.

This is a demo stub, not the Phase-8 service. It uses only stdlib + the existing
flywaydqn package (numpy, scipy). It is intentionally minimal so it can survive a
lockfile/flash gap: the scene is inlined here and hardcoded so there are no optional
imports, no MLflow, no gymnasium, no websocket framing.

Run from the repo root with the venv on path, e.g.:
    ./.venv/Scripts/python service/beam_demo.py
"""

from __future__ import annotations

import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any

# The package under test imports only when src is on the path. In the repo root the
# venv's Scripts/python can run this script with src added explicitly; running it
# directly with a system python that does not have flywaydqn installed is not
# supported by this demo script.
if "src" not in sys.path:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# We gate everything on the flywaydqn package so a missing/partial install fails
# loudly at startup rather than half-rendering a demo that silently uses dummy data.
try:
    import numpy as np
    import numpy.typing as npt
except Exception as e:
    raise SystemExit(f"demo requires numpy: {e}") from e

try:
    from flywaydqn.rf.array import (
        DEFAULT_D_LAMBDA,
        beamforming_gain_dbi,
        composite_gain_dbi,
        matched_filter_weights,
    )
    from flywaydqn.rf.codebook import (
        DEFAULT_AZIMUTH_SPAN_DEG,
        Beam,
        build_dft_codebook,
        beam_azimuth_deg,
        beam_index_from_azimuth,
    )
    from flywaydqn.rf.nulling import null_toward
    from flywaydqn.rf.pathloss import (
        DEFAULT_H_BS_M,
        DEFAULT_H_UT_M,
        tr38901_uma_los_pathloss_dbm,
    )
    from flywaydqn.rf.sinr import sinr_linear
    from flywaydqn.exposure.density import (
        ICNIRP_2020_PUBLIC_REFERENCE_W_PER_M2,
        power_density_grid_w_per_m2,
        s_max,
    )
    from flywaydqn.exposure.metrics import exposure_metrics_from_density
    from flywaydqn.env.reward import make_reward_fn
    from flywaydqn.mobility.flock import (
        FlockState,
        generate_flock_step,
        flock_footprint_cells,
    )
    from flywaydqn.mobility.users import UserState
except Exception as e:
    raise SystemExit(f"flywaydqn not importable from {sys.path}:\n{e}") from e

DEMO_H_BS_M = DEFAULT_H_BS_M
DEMO_H_UT_M = DEFAULT_H_UT_M


# ---------------------------------------------------------------------------
# Demo knobs — hardcoded so the scene is reproducible for the online round.
# ---------------------------------------------------------------------------

DEMO_FREQ_GHZ = 3.5
DEMO_BANDWIDTH_MHZ = 100.0
DEMO_TX_POWER_DBM = 43.0
DEMO_TX_POWER_MW = 10.0 ** (DEMO_TX_POWER_DBM / 10.0)  # ~1995 mW ≈ 2W
DEMO_NOISE_FIGURE_DB = 7.0
DEMO_SPEC_EFF_CAP = 6.0
DEMO_DT_SECONDS = 0.1
DEMO_EXPOSURE_THRESHOLD_W_PER_M2 = ICNIRP_2020_PUBLIC_REFERENCE_W_PER_M2
DEMO_N_USERS = 7
DEMO_N_FLOCK_CELLS = 64
DEMO_N_BEAMS = 16
# Hardcoded scenario so the demo renders the same geometry every reload.
# Flock 1: 47.2 deg at 184 m, altitude 20-80 m, 180 birds.
DEMO_SEED = 20240731
DEMO_FLOCK_BEARING_DEG = 47.2
DEMO_FLOCK_RANGE_M = 184.0
DEMO_FLOCK_SPEED_MPS = 14.2
DEMO_FLOCK_HEADING_DEG = 48.0
DEMO_FLOCK_ALT_MIN_M = 20.0
DEMO_FLOCK_ALT_MAX_M = 80.0
DEMO_FLOCK_EXTENT_M = 60.0
DEMO_FLOCK_N_BIRDS = 180
DEMO_SECTOR_X_M = 0.0
DEMO_SECTOR_Y_M = 0.0
DEMO_ANTENNA_HEIGHT_M = DEFAULT_H_BS_M
DEMO_ARRAY_N_ROWS = 8
DEMO_ARRAY_N_COLS = 8
DEMO_REWARD_ALPHA = 1.0
DEMO_REWARD_GAMMA = 0.02
DEMO_REWARD_KAPPA = 5.0
DEMO_REWARD_W = 0.75  # preference weight; reproduces the UI's tradeoffWeight


# ---------------------------------------------------------------------------
# Derived constants (computed once, inlined in the page).
# ---------------------------------------------------------------------------

_codebook: list[Beam] = build_dft_codebook(
    n_beams=DEMO_N_BEAMS,
    n_rows=DEMO_ARRAY_N_ROWS,
    n_cols=DEMO_ARRAY_N_COLS,
    d_lambda=DEFAULT_D_LAMBDA,
    elevation_deg=0.0,
    azimuth_span_deg=DEFAULT_AZIMUTH_SPAN_DEG,
)
_dft_weights = _codebook[0].weights  # representative; real scene picks per step


def _noise_power_mw() -> float:
    from flywaydqn.units import thermal_noise_power_mw
    return thermal_noise_power_mw(DEMO_BANDWIDTH_MHZ * 1e6, DEMO_NOISE_FIGURE_DB)


# Computed after both the demo knobs above and the helper definition. The module
# previously called this before either existed, which crashed at import time.
_NOISE_MW: float = _noise_power_mw()


@dataclass
class DemoScene:
    """One-step state that the browser page renders."""

    step: int = 0
    time_s: float = 0.0
    tx_power_mw: float = DEMO_TX_POWER_MW
    null_active: bool = False
    null_bearing_deg: float = 0.0
    null_depth_db: float = 0.0
    serving_beam_index: int = 0
    serving_beam_azimuth_deg: float = 0.0
    serving_beam_gain_dbi: float = 0.0
    flock: FlockState = field(default_factory=lambda: FlockState(
        centroid_x_m=0.0,
        centroid_y_m=0.0,
        centroid_heading_deg=DEMO_FLOCK_HEADING_DEG,
        speed_mps=DEMO_FLOCK_SPEED_MPS,
        altitude_min_m=DEMO_FLOCK_ALT_MIN_M,
        altitude_max_m=DEMO_FLOCK_ALT_MAX_M,
        extent_m=DEMO_FLOCK_EXTENT_M,
        n_individuals=DEMO_FLOCK_N_BIRDS,
    ))
    flock_bearing_deg: float = 0.0
    flock_range_m: float = 0.0
    cell_positions_xy_m: npt.NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 2), dtype=np.float64))
    gains_dbi: npt.NDArray[np.float64] = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    density_w_per_m2: npt.NDArray[np.float64] = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    metrics: dict[str, float] = field(default_factory=dict)
    users: list[UserState] = field(default_factory=list)
    baseline_serving_azimuth_deg: float = DEMO_FLOCK_BEARING_DEG
    baseline_null_active: bool = False
    baseline_null_depth_db: float = 0.0
    baseline_gain_dbi: float = 0.0
    baseline_max_density_w_per_m2: float = 0.0
    reward: dict[str, float] = field(default_factory=dict)
    baseline_reward: dict[str, float] = field(default_factory=dict)
    explanation: str = ""
    engine: str = "FlywayDQN (Active Null)"
    throughput_mbps: float = 0.0
    baseline_throughput_mbps: float = 0.0


def _bearing_and_range(flock: FlockState) -> tuple[float, float]:
    bearing, rng = flock.polar_from(DEMO_SECTOR_X_M, DEMO_SECTOR_Y_M)
    return bearing, max(rng, 1.0)


def _step_deterministic(scene: DemoScene, step: int, rng_seed: int) -> DemoScene:
    """Advance one step with a fresh seeded RNG so the demo is deterministic."""
    step_rng = np.random.default_rng(rng_seed + step)

    # 1) Advance flock with deterministic step (no jiggle) so the geometry is stable.
    flock = generate_flock_step(
        scene.flock,
        dt_seconds=DEMO_DT_SECONDS,
        rng=None,
    )
    bearing_deg, range_m = _bearing_and_range(flock)

    # 2) Serving beam for the active engine: nearest codebook beam pointing at the
    #    flock bearing. This is the baseline "track the bearing" policy; the DQN track
    #    would instead pick a null-bearing here and compute Q — not in scope for a
    #    quick demo, so the active engine is a representative beam + optional null.
    serving_index = beam_index_from_azimuth(bearing_deg, DEMO_N_BEAMS, DEFAULT_AZIMUTH_SPAN_DEG)
    serving_azimuth = beam_azimuth_deg(serving_index, DEMO_N_BEAMS, DEFAULT_AZIMUTH_SPAN_DEG)

    # 3) Build the active weights: matched filter toward the serving beam, plus an
    #    optional projection null toward the flock bearing. This is the physical core
    #    the whole demo is supposed to show, and it is real geometry.
    w0 = matched_filter_weights(DEMO_ARRAY_N_ROWS, DEMO_ARRAY_N_COLS, 0.0, serving_azimuth, DEFAULT_D_LAMBDA)

    null_active = False
    null_depth_db = 0.0
    w = w0
    if range_m < 220.0:
        # Null toward the flock only when it is close enough that the demo should show
        # something interesting, and only when the serving beam is not the same bearing
        # (if the beam points at the flock, the null would destroy the serving beam — the
        # mirror case that Phase-1 testing surfaced).
        mirror_tolerance_deg = 4.0
        if abs(((serving_azimuth - bearing_deg + 180.0) % 360.0) - 180.0) > mirror_tolerance_deg:
            null_active = True
            null_bearing_deg = bearing_deg
            w = null_toward(
                w0,
                0.0,
                null_bearing_deg,
                DEMO_ARRAY_N_ROWS,
                DEMO_ARRAY_N_COLS,
                d_lambda=DEFAULT_D_LAMBDA,
            )
            g_before = beamforming_gain_dbi(w0, 0.0, null_bearing_deg, DEMO_ARRAY_N_ROWS, DEMO_ARRAY_N_COLS, DEFAULT_D_LAMBDA)
            g_after = beamforming_gain_dbi(w, 0.0, null_bearing_deg, DEMO_ARRAY_N_ROWS, DEMO_ARRAY_N_COLS, DEFAULT_D_LAMBDA)
            if math.isfinite(g_before) and math.isfinite(g_after):
                null_depth_db = float(g_after - g_before)

    serving_gain_dbi = beamforming_gain_dbi(w, 0.0, serving_azimuth, DEMO_ARRAY_N_ROWS, DEMO_ARRAY_N_COLS, DEFAULT_D_LAMBDA)

    # 4) Cells covering the flock footprint + per-cell antenna gain.
    footprint = flock_footprint_cells(flock, n_cells=DEMO_N_FLOCK_CELLS, rng=None)
    # Range from antenna (at height H_BS) to each cell.
    dx = footprint[:, 0] - DEMO_SECTOR_X_M
    dy = footprint[:, 1] - DEMO_SECTOR_Y_M
    slant = np.sqrt(dx**2 + dy**2 + DEMO_ANTENNA_HEIGHT_M**2)
    # Azimuth seen from antenna toward each cell — used to compute the per-cell gain.
    az = np.degrees(np.arctan2(dy, dx))
    az = (az + 180.0) % 360.0 - 180.0  # normalize to (-180, 180]
    # Elevation above horizon toward each cell.
    theta = np.degrees(np.arcsin(DEMO_ANTENNA_HEIGHT_M / np.maximum(slant, 1.0)))

    gains = np.array([
        composite_gain_dbi(w, t, a, DEMO_ARRAY_N_ROWS, DEMO_ARRAY_N_COLS, DEFAULT_D_LAMBDA)
        for (t, a) in zip(theta, az)
    ], dtype=np.float64)

    # 5) Power density over the footprint.
    density = power_density_grid_w_per_m2(
        tx_power_mw=DEMO_TX_POWER_MW,
        cell_positions_xy_m=footprint,
        sector_xy_m=(DEMO_SECTOR_X_M, DEMO_SECTOR_Y_M),
        antenna_height_m=DEMO_ANTENNA_HEIGHT_M,
        gains_dbi=gains,
    )

    metrics = exposure_metrics_from_density(density, DEMO_EXPOSURE_THRESHOLD_W_PER_M2, dt_seconds=DEMO_DT_SECONDS)

    # 6) UE links: deterministic positions + deterministic SINR from the active pattern.
    #    We place 7 users in an annulus and score them with the active gain + UMa path loss.
    ue_rng = np.random.default_rng(rng_seed + 1000 + step)
    users = []
    for i in range(DEMO_N_USERS):
        radius_m = float(ue_rng.uniform(80.0, 320.0))
        bearing_rad = float(ue_rng.uniform(0.0, 2 * math.pi))
        x = radius_m * math.cos(bearing_rad)
        y = radius_m * math.sin(bearing_rad)
        # UE antenna height 1.5 m, so elevation from the BS is small (a few degrees).
        ue_slant = math.hypot(x, y, DEMO_ANTENNA_HEIGHT_M - 1.5)
        ue_az = (math.degrees(math.atan2(y, x)) + 180.0) % 360.0 - 180.0
        ue_theta = math.degrees(math.asin((DEMO_ANTENNA_HEIGHT_M - 1.5) / ue_slant))
        g_dbi = composite_gain_dbi(w, ue_theta, ue_az, DEMO_ARRAY_N_ROWS, DEMO_ARRAY_N_COLS, DEFAULT_D_LAMBDA)
        pl_db = tr38901_uma_los_pathloss_dbm(ue_slant, DEMO_FREQ_GHZ, h_bs_m=DEMO_H_BS_M, h_ut_m=DEMO_H_UT_M)
        # Deterministic LOS for the demo.
        sinr_lin = sinr_linear(
            g_dbi,
            pl_db,
            interference_mw=0.0,
            noise_power_mw=_NOISE_MW,
            tx_power_mw=DEMO_TX_POWER_MW,
        )
        sinr_db = 10.0 * math.log10(max(sinr_lin, 1e-9))
        ue_throughput_mbps = DEMO_BANDWIDTH_MHZ * min(math.log2(1.0 + max(sinr_lin, 0.0)), DEMO_SPEC_EFF_CAP)
        users.append(UserState(
            x_m=x,
            y_m=y,
            heading_deg=float(ue_rng.uniform(0.0, 360.0)),
            speed_mps=float(ue_rng.uniform(1.0, 8.0)),
            vehicular=False,
        ))

    # 7) Reward.
    reward_fn = make_reward_fn(
        alpha=DEMO_REWARD_ALPHA,
        gamma=DEMO_REWARD_GAMMA,
        kappa=DEMO_REWARD_KAPPA,
        s_thr=DEMO_EXPOSURE_THRESHOLD_W_PER_M2,
        max_tx_power_mw=DEMO_TX_POWER_MW,
        max_throughput_ref_mbps=DEMO_BANDWIDTH_MHZ * DEMO_SPEC_EFF_CAP * DEMO_N_USERS,
    )
    def _user_sINR_throughput(u: UserState) -> float:
        slant = math.hypot(u.x_m, u.y_m, DEMO_ANTENNA_HEIGHT_M - 1.5)
        az = (math.degrees(math.atan2(u.y_m, u.x_m)) + 180.0) % 360.0 - 180.0
        theta = math.degrees(math.asin((DEMO_ANTENNA_HEIGHT_M - 1.5) / max(slant, 1.0)))
        g_dbi = composite_gain_dbi(w, theta, az, DEMO_ARRAY_N_ROWS, DEMO_ARRAY_N_COLS, DEFAULT_D_LAMBDA)
        pl_db = tr38901_uma_los_pathloss_dbm(slant, DEMO_FREQ_GHZ, h_bs_m=DEMO_H_BS_M, h_ut_m=DEMO_H_UT_M)
        sinr_lin = sinr_linear(g_dbi, pl_db, 0.0, _NOISE_MW, DEMO_TX_POWER_MW)
        return DEMO_BANDWIDTH_MHZ * min(math.log2(1.0 + max(sinr_lin, 0.0)), DEMO_SPEC_EFF_CAP)

    sum_tput_mbps = float(sum(_user_sINR_throughput(u) for u in users))
    r = reward_fn(
        sum_throughput_mbps=sum_tput_mbps,
        tx_power_mw=DEMO_TX_POWER_MW,
        s_p95_w_per_m2=metrics["s_p95"],
        observation={"log10_w": math.log10(DEMO_REWARD_W)},
    )

    # 8) Baseline "no nulling, beam tracks flock" for comparison numbers.
    w_base = matched_filter_weights(DEMO_ARRAY_N_ROWS, DEMO_ARRAY_N_COLS, 0.0, DEMO_FLOCK_BEARING_DEG, DEFAULT_D_LAMBDA)
    base_gain_dbi = beamforming_gain_dbi(w_base, 0.0, DEMO_FLOCK_BEARING_DEG, DEMO_ARRAY_N_ROWS, DEMO_ARRAY_N_COLS, DEFAULT_D_LAMBDA)
    base_density = power_density_grid_w_per_m2(
        tx_power_mw=DEMO_TX_POWER_MW,
        tx_power_dbm=DEMO_TX_POWER_DBM,
        cell_positions_xy_m=footprint,
        sector_xy_m=(DEMO_SECTOR_X_M, DEMO_SECTOR_Y_M),
        antenna_height_m=DEMO_ANTENNA_HEIGHT_M,
        gains_dbi=np.array([
            composite_gain_dbi(w_base, t, a, DEMO_ARRAY_N_ROWS, DEMO_ARRAY_N_COLS, DEFAULT_D_LAMBDA)
            for (t, a) in zip(theta, az)
        ], dtype=np.float64),
    )
    base_metrics = exposure_metrics_from_density(base_density, DEMO_EXPOSURE_THRESHOLD_W_PER_M2, dt_seconds=DEMO_DT_SECONDS)
    base_r = reward_fn(
        sum_throughput_mbps=sum_tput_mbps,
        tx_power_mw=DEMO_TX_POWER_MW,
        s_p95_w_per_m2=base_metrics["s_p95"],
        observation={"log10_w": math.log10(DEMO_REWARD_W)},
    )

    explanation = (
        f"Step {step}: serving beam {serving_index} @ {serving_azimuth:.1f} deg, "
        f"flock at {bearing_deg:.1f} deg / {range_m:.0f} m. "
        + ("Active null toward flock: " if null_active else "No null (flock out of reach or mirror-bearing guard). ")
        + (f"null depth {null_depth_db:.1f} dB. " if null_active else "")
        + f"max density {metrics['s_max']:.3f} W/m² "
        + (f"(over {DEMO_EXPOSURE_THRESHOLD_W_PER_M2:.0f} W/m² threshold: {metrics['frac_over']:.2%} of cells). " if metrics['frac_over'] > 0 else "below threshold. ")
        + f"UE sum throughput {sum_tput_mbps:.1f} Mbps, reward {r['reward']:.2f}."
    )

    return DemoScene(
        step=step,
        time_s=scene.time_s + DEMO_DT_SECONDS,
        tx_power_mw=DEMO_TX_POWER_MW,
        null_active=null_active,
        null_bearing_deg=bearing_deg if null_active else 0.0,
        null_depth_db=null_depth_db,
        serving_beam_index=serving_index,
        serving_beam_azimuth_deg=serving_azimuth,
        serving_beam_gain_dbi=serving_gain_dbi,
        flock=flock,
        flock_bearing_deg=bearing_deg,
        flock_range_m=range_m,
        cell_positions_xy_m=footprint,
        gains_dbi=gains,
        density_w_per_m2=density,
        metrics={
            "sMax": metrics["s_max"],
            "sP95": metrics["s_p95"],
            "fracOver": metrics["frac_over"],
            "dose": metrics["dose"],
        },
        users=users,
        baseline_serving_azimuth_deg=DEMO_FLOCK_BEARING_DEG,
        baseline_null_active=False,
        baseline_null_depth_db=0.0,
        baseline_gain_dbi=base_gain_dbi,
        baseline_max_density_w_per_m2=float(s_max(base_density)),
        reward={
            "reward": r["reward"],
            "throughput": r["terms"]["throughput"],
            "exposure": r["terms"]["exposure"],
            "power": r["terms"]["power"],
            "w": r["terms"]["w"],
        },
        baseline_reward={
            "reward": base_r["reward"],
            "throughput": base_r["terms"]["throughput"],
            "exposure": base_r["terms"]["exposure"],
            "power": base_r["terms"]["power"],
            "w": base_r["terms"]["w"],
        },
        explanation=explanation,
        engine="FlywayDQN (Active Null)",
        throughput_mbps=sum_tput_mbps,
        baseline_throughput_mbps=sum_tput_mbps,  # same UE set; comparison here is exposure-driven
    )


def _f(v: Any) -> float:
    """Coerce a value to a plain float for JSON serialization."""
    try:
        return float(v)
    except Exception:
        return 0.0


def _scene_as_dict(scene: DemoScene) -> dict[str, Any]:
    """Serialize the scene to a JSON-safe dict the browser can render."""
    cells = scene.cell_positions_xy_m
    density = scene.density_w_per_m2
    if cells.size == 0 or density.size == 0:
        return {
            "step": scene.step,
            "timeS": round(scene.time_s, 2),
            "txPowerMw": _f(scene.tx_power_mw),
            "nullActive": scene.null_active,
            "nullBearingDeg": _f(scene.null_bearing_deg),
            "nullDepthDb": _f(scene.null_depth_db),
            "servingBeamIndex": scene.serving_beam_index,
            "servingBeamAzimuthDeg": _f(scene.serving_beam_azimuth_deg),
            "servingBeamGainDbi": _f(scene.serving_beam_gain_dbi),
            "flock": _flock_dict(scene.flock, scene.flock_bearing_deg, scene.flock_range_m),
            "cells": [],
            "density": [],
            "metrics": scene.metrics,
            "users": _users_dict(scene.users),
            "baseline": {
                "servingAzimuthDeg": _f(scene.baseline_serving_azimuth_deg),
                "nullActive": scene.baseline_null_active,
                "nullDepthDb": _f(scene.baseline_null_depth_db),
                "gainDbi": _f(scene.baseline_gain_dbi),
            },
            "reward": scene.reward,
            "baselineReward": scene.baseline_reward,
            "explanation": scene.explanation,
            "engine": scene.engine,
            "throughputMbps": _f(scene.throughput_mbps),
            "baselineThroughputMbps": _f(scene.baseline_throughput_mbps),
        }

    # Render the footprint as a coarse ASCII-free visualization-friendly grid:
    # a 64-cell scatter around the antenna. We emit centroid, heading, range and the
    # top few density samples so the page can draw the flock blob + exposure hotspot.
    max_density_idx = int(np.argmax(density))
    max_density = float(density[max_density_idx])
    max_cell = cells[max_density_idx].tolist()

    def _row(x: float, y: float, d: float) -> dict[str, float]:
        return {"x": _f(x), "y": _f(y), "d": _f(d)}

    # Emit a fixed subset: max cell + 8 nearest to max, so the page stays fast.
    order = np.argsort(-density)[:12].tolist()
    highlighted = [cells[i].tolist() + [float(density[i])] for i in order]

    return {
        "step": scene.step,
        "timeS": round(scene.time_s, 2),
        "txPowerMw": _f(scene.tx_power_mw),
        "nullActive": scene.null_active,
        "nullBearingDeg": _f(scene.null_bearing_deg),
        "nullDepthDb": _f(scene.null_depth_db),
        "servingBeamIndex": scene.serving_beam_index,
        "servingBeamAzimuthDeg": _f(scene.serving_beam_azimuth_deg),
        "servingBeamGainDbi": _f(scene.serving_beam_gain_dbi),
        "flock": _flock_dict(scene.flock, scene.flock_bearing_deg, scene.flock_range_m),
        "cells": [_row(c[0], c[1], d) for (c, d) in zip(cells[:8].tolist(), density[:8].tolist())],
        "maxDensity": max_density,
        "maxCell": _f(max_cell[0]),  # x
        "maxCellY": _f(max_cell[1]),
        "highlighted": [{"x": _f(p[0]), "y": _f(p[1]), "d": _f(p[2])} for p in highlighted],
        "metrics": scene.metrics,
        "users": _users_dict(scene.users),
        "baseline": {
            "servingAzimuthDeg": _f(scene.baseline_serving_azimuth_deg),
            "nullActive": scene.baseline_null_active,
            "nullDepthDb": _f(scene.baseline_null_depth_db),
            "gainDbi": _f(scene.baseline_gain_dbi),
            "maxDensity": _f(scene.baseline_max_density_w_per_m2),
        },
        "reward": scene.reward,
        "baselineReward": scene.baseline_reward,
        "explanation": scene.explanation,
        "engine": scene.engine,
        "throughputMbps": _f(scene.throughput_mbps),
        "baselineThroughputMbps": _f(scene.baseline_throughput_mbps),
    }


def _flock_dict(flock: FlockState, bearing_deg: float, range_m: float) -> dict[str, Any]:
    return {
        "centroid": {"x": _f(flock.centroid_x_m), "y": _f(flock.centroid_y_m)},
        "headingDeg": _f(flock.centroid_heading_deg),
        "speedMs": _f(flock.speed_mps),
        "altitudeMinM": _f(flock.altitude_min_m),
        "altitudeMaxM": _f(flock.altitude_max_m),
        "extentM": _f(flock.extent_m),
        "nBirds": flock.n_individuals,
        "bearingDeg": _f(bearing_deg),
        "rangeM": _f(range_m),
    }


def _users_dict(users: list[UserState]) -> list[dict[str, Any]]:
    out = []
    for u in users:
        bearing, rng = u.polar_from(DEMO_SECTOR_X_M, DEMO_SECTOR_Y_M)
        # Compute SINR for the user against the current active weights (replay the
        # geometry we used in the step above).
        slant = math.hypot(u.x_m, u.y_m, DEMO_ANTENNA_HEIGHT_M - 1.5)
        az = (math.degrees(math.atan2(u.y_m, u.x_m)) + 180.0) % 360.0 - 180.0
        theta = math.degrees(math.asin((DEMO_ANTENNA_HEIGHT_M - 1.5) / max(slant, 1.0)))
        g_dbi = composite_gain_dbi(_dft_weights, theta, az, DEMO_ARRAY_N_ROWS, DEMO_ARRAY_N_COLS, DEFAULT_D_LAMBDA)
        pl_db = tr38901_uma_los_pathloss_dbm(slant, DEMO_FREQ_GHZ, h_bs_m=DEMO_H_BS_M, h_ut_m=DEMO_H_UT_M)
        sinr_lin = sinr_linear(g_dbi, pl_db, 0.0, _NOISE_MW, DEMO_TX_POWER_MW)
        sinr_db = 10.0 * math.log10(max(sinr_lin, 1e-9))
        tput_mbps = DEMO_BANDWIDTH_MHZ * min(math.log2(1.0 + max(sinr_lin, 0.0)), DEMO_SPEC_EFF_CAP)
        out.append({
            "id": f"UE-{101 + len(out)}",
            "name": f"UE-{101 + len(out)}",
            "x": _f(u.x_m),
            "y": _f(u.y_m),
            "bearingDeg": _f(bearing),
            "rangeM": _f(rng),
            "sinrDb": _f(sinr_db),
            "throughputMbps": _f(tput_mbps),
            "isNulledRisk": abs(bearing - _f(DEMO_FLOCK_BEARING_DEG)) < 15.0 or rng < 120.0,
        })
    return out


# ---------------------------------------------------------------------------
# Tiny server — stdlib http.server + a JSON endpoint + a static HTML page.
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>FlywayDQN — online round demo</title>
<style>
  :root{
    --bg:#0b1413; --panel:#11201f; --panel-2:#162624; --line:#1f3330;
    --ink:#e8eeec; --muted:#92a39f; --dim:#5d716c;
    --teal:#34d399; --teal-dim:#0f766e; --coral:#fb7139;
    --amber:#f5b041; --blue:#38bdf8;
    --ok:#34d399; --warn:#f5b041; --bad:#fb7139;
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%}
  body{
    font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
    background:var(--bg); color:var(--ink);
    display:grid; grid-template-columns: 1fr 260px; grid-template-rows: 48px 1fr;
    gap:10px; padding:10px;
  }
  .topbar{grid-column:1/-1; display:flex; align-items:center; gap:14px;
    background:linear-gradient(180deg,#182725,#101c1b); border:1px solid var(--line);
    border-radius:10px; padding:0 14px;}
  .brand{ display:flex; align-items:center; gap:8px; font-weight:600; letter-spacing:.2px;}
  .brand .dot{ width:10px; height:10px; border-radius:50%; background:var(--teal);
    box-shadow:0 0 0 4px rgba(52,211,153,.15);}
  .topbar .se{sans-serif; font-size:12px; color:var(--muted);}
  .topbar .live{ display:inline-flex; align-items:center; gap:6px;
    background:#0c1c1a; color:var(--teal); border:1px solid #1f3b35;
    padding:4px 8px; border-radius:6px; font-size:11px; font-weight:600;
    letter-spacing:.4px;}
  .topbar .live::before{ content:""; width:7px; height:7px; border-radius:50%;
    background:var(--teal); animation:pulse 1.8s infinite;}
  @keyframes pulse{ 0%{box-shadow:0 0 0 0 rgba(52,211,153,.5)} 70%{box-shadow:0 0 0 7px rgba(52,211,153,0)} 100%{box-shadow:0 0 0 0 rgba(52,211,153,0)} }
  .grid{ display:grid; grid-template-columns: 1fr 1fr; gap:10px; padding-top:8px;}
  .card{ background:var(--panel); border:1px solid var(--line); border-radius:10px;
    padding:12px; display:flex; flex-direction:column;}
  .card h3{ margin:0 0 8px; font-size:11px; letter-spacing:.6px; color:var(--muted);
    text-transform:uppercase; font-weight:600;}
  .canvas-wrap{ position:relative; height:300px;}
  svg{ width:100%; height:100%; display:block; }
  .right{ display:flex; flex-direction:column; gap:10px;}
  .stat{ background:var(--panel-2); border:1px solid var(--line); border-radius:8px; padding:10px;}
  .stat .k{ font-size:10px; color:var(--muted); letter-spacing:.4px; text-transform:uppercase;}
  .stat .v{ font-size:22px; font-weight:600; margin-top:2px; letter-spacing:-.3px;}
  .stat .u{ font-size:11px; color:var(--muted); margin-left:4px;}
  .stat .d{ font-size:11px; color:var(--teal); margin-top:2px;}
  .legend{ font-size:11px; color:var(--muted); display:flex; gap:12px; flex-wrap:wrap;}
  .legend span{ display:inline-flex; align-items:center; gap:5px;}
  .legend i{ width:9px; height:9px; border-radius:2px; display:inline-block;}
  .explain{ font-size:12px; color:var(--muted); background:#0c1c1a; border:1px solid var(--line);
    border-radius:8px; padding:10px; line-height:1.5;}
  .explain b{ color:var(--ink);}
  .row3{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px;}
  .foot{ grid-column:1/-1; font-size:11px; color:var(--dim); border-top:1px solid var(--line);
    padding-top:8px; display:flex; gap:14px; flex-wrap:wrap;}
  .tag{ color:var(--teal); border:1px solid #1f3b35; padding:1px 6px; border-radius:5px;
    font-size:11px; }
</style>
</head>
<body>
  <div class="topbar">
    <div class="brand"><span class="dot"></span>FlywayDQN <span style="color:var(--muted);font-weight:400">· online round demo</span></div>
    <div class="se">Sector A · gNodeB-042 · 3.5 GHz n78</div>
    <div class="live">LIVE SCENE</div>
    <div style="margin-left:auto; display:flex; gap:10px; align-items:center;">
      <div class="se" id="clock">T+00:00:00.00</div>
      <div class="se">seed <span id="seed">20240731</span></div>
    </div>
  </div>

  <div class="grid">
    <div class="card">
      <h3>Beamforming scene · polar footprint</h3>
      <div class="canvas-wrap">
        <svg id="scene" viewBox="-250 -250 500 500" preserveAspectRatio="xMidYMid meet"></svg>
      </div>
      <div class="legend" style="margin-top:8px">
        <span><i style="background:var(--teal)"></i>serving beam az</span>
        <span><i style="background:var(--coral)"></i>null corridor</span>
        <span><i style="background:#7ee787"></i>flock centroid</span>
        <span><i style="background:var(--amber)"></i>max-density cell</span>
        <span><i style="background:#9fb3c8"></i>UEs</span>
      </div>
    </div>

    <div class="right">
      <div class="card">
        <h3>Live numbers</h3>
        <div class="row3">
          <div class="stat"><div class="k">Null depth</div><div class="v" id="nullDepth">—</div><div class="u">dB</div><div class="d" id="nullState">inactive</div></div>
          <div class="stat"><div class="k">Peak density</div><div class="v" id="peakDensity">—</div><div class="u">W/m²</div><div class="d" id="overTag"></div></div>
          <div class="stat"><div class="k">UE sum Tput</div><div class="v" id="tput">—</div><div class="u">Mbps</div><div class="d" id="tputDx"></div></div>
        </div>
      </div>

      <div class="card">
        <h3>Sector KPIs</h3>
        <div class="stat"><div class="k">Serving beam</div><div class="v" id="beam" style="font-size:16px">— @ —°</div><div class="d" id="gain"></div></div>
        <div class="stat"><div class="k">Flock</div><div class="v" id="flock" style="font-size:16px">— @ —° · — m</div><div class="d" id="flockAlt"></div></div>
        <div class="stat"><div class="k">Reward</div><div class="v" id="reward">—</div><div class="u"> clipped</div><div class="d" id="rewardTerms"></div></div>
        <div class="stat"><div class="k">Baseline</div><div class="v" id="baseline" style="font-size:14px">—</div><div class="d" id="baselineNote"></div></div>
      </div>

      <div class="card">
        <h3>Last decision</h3>
        <div class="explain" id="explain">Initialising scene…</div>
      </div>
    </div>
  </div>

  <div class="foot">
    <span><span class="tag">prototype</span> for online round — not a production service.</span>
    <span>Physics: TR 38.901 UMa LOS path loss, UPA + DFT codebook, projection null, ICNIRP 2020 reference level 10 W/m².</span>
    <span id="ts"></span>
  </div>

<script>
const SVGNS = "http://www.w3.org/2000/svg";
function el(tag, attrs={}){
  const e = document.createElementNS(SVGNS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
}
const scene = document.getElementById("scene");

function polarToCart(r, deg){
  const a = (deg - 90) * Math.PI / 180;
  return [r * Math.cos(a), r * Math.sin(a)];
}

function render(s){
  scene.innerHTML = "";
  const W = 250;

  // Reference rings + sector bounds.
  const rings = [50, 100, 180, 240];
  for (const r of rings){
    scene.appendChild(el("circle", {r:r, fill:"none", stroke:"#263230", "stroke-width":0.6}));
    scene.appendChild(el("text", {x:-r, y:-r-2, "text-anchor":"start", "dominant-baseline":"auto",
      fill:"#5d716c", "font-size":7, transform:`rotate(${r>180?0:0} 0 0)`})).textContent = r + "m";
  }

  // Sector arcs (the macro panel covers about ±60 deg).
  const sectorDeg = 60;
  const arcPts = 60;
  const sectorOuter = 250;
  const sectorPath = (function(){
    const p = [];
    for (let i=0; i<=arcPts; i++){
      const a = -sectorDeg + (2*sectorDeg)*i/arcPts;
      const [x,y] = polarToCart(sectorOuter, a);
      p.push((i===0?"M":"L") + x.toFixed(1) + " " + y.toFixed(1));
    }
    p.push("Z");
    return p.join(" ");
  })();
  scene.appendChild(el("path", {d:sectorPath, fill:"#132421", stroke:"#1f3b35", "stroke-width":0.8}));

  // Serving beam direction: a soft wedge from the antenna.
  const beamR = 180;
  const beamW = 4;
  const bazi = s.servingBeamAzimuthDeg;
  const [bx, by] = polarToCart(beamR, bazi);
  scene.appendChild(el("line", {x1:0, y1:0, x2:bx, y2:by, stroke:"#34d399", "stroke-width":1.4, "stroke-dasharray":"3 3"}));
  scene.appendChild(el("circle", {cx:bx, cy:by, r:3, fill:"#34d399"}));
  scene.appendChild(el("text", {x:bx+5, y:by-4, fill:"#34d399", "font-size":9})).textContent =
    "beam " + (s.servingBeamIndex+1) + " · " + bazi.toFixed(1) + "°";

  // Null corridor toward flock bearing, when active.
  if (s.nullActive){
    const nazi = s.nullBearingDeg;
    const [nx, ny] = polarToCart(200, nazi);
    scene.appendChild(el("line", {x1:0, y1:0, x2:nx, y2:ny, stroke:"#fb7139", "stroke-width":1, opacity:0.7}));
    scene.appendChild(el("path", {
      d: (function(){
        const p=[];
        const w=4; const R=190;
        for (let i=0;i<=40;i++){
          const a = nazi - w + (2*w)*i/40;
          const [x,y] = polarToCart(R, a);
          p.push((i===0?"M":"L")+x.toFixed(1)+" "+y.toFixed(1));
        }
        p.push("Z"); return p.join(" ");
      })(),
      fill:"#fb7139", opacity:0.10
    }));
    scene.appendChild(el("text", {x:nx+5, y:ny+3, fill:"#fb7139", "font-size":9})).textContent =
      "null " + s.nullDepthDb.toFixed(1) + " dB";
  }

  // Flock centroid.
  const fc = s.flock.centroid;
  const r = Math.min(fc.x, fc.y);
  // approximate range from antenna; we already have s.flock.rangeM.
  const fr = s.flock.rangeM;
  const [fx, fy] = polarToCart(Math.min(fr, 230), s.flock.bearingDeg);
  scene.appendChild(el("circle", {cx:fx, cy:fy, r:9, fill:"#7ee787", opacity:0.85}));
  scene.appendChild(el("circle", {cx:fx, cy:fy, r:14, fill:"#7ee787", opacity:0.18}));
  scene.appendChild(el("text", {x:fx+8, y:fy-6, fill:"#cfe6cf", "font-size":9})).textContent =
    "flock " + s.flock.nBirds + " birds";

  // Max-density cell marker.
  scene.appendChild(el("circle", {cx:s.maxCell, cy:s.maxCellY, r:3.5, fill:"#f5b041"}));
  scene.appendChild(el("text", {x:s.maxCell+5, y:s.maxCellY-4, fill:"#f5b041", "font-size":8})).textContent =
    "Smax " + s.maxDensity.toFixed(2) + " W/m²";

  // Highlighted exposure cells (top 12).
  for (const h of s.highlighted){
    const c = Math.min(1, Math.max(0, h.d / 10));
    scene.appendChild(el("circle", {cx:h.x, cy:h.y, r:2, fill:"#f5b041", opacity:0.4 + 0.4*c}));
  }

  // UEs.
  for (const u of s.users){
    const [ux, uy] = polarToCart(Math.min(u.rangeM, 230), u.bearingDeg);
    scene.appendChild(el("circle", {cx:ux, cy:uy, r:3, fill:"#9fb3c8"}));
    scene.appendChild(el("text", {x:ux+5, y:uy-4, fill:"#9fb3c8", "font-size":8})).textContent =
      u.id + " " + u.throughputMbps.toFixed(0) + "Mbps";
  }

  // Antenna.
  scene.appendChild(el("circle", {cx:0, cy:0, r:6, fill:"#34d399"}));
}

let last = 0;
function tick(){
  let s = null;
  fetch("/scene.json")
    .then(r=>r.ok ? r.json() : null)
    .then(data => {
      s = data;
      if (!s) return;
      render(s);

      document.getElementById("clock").textContent =
        "T+" + [Math.floor(s.timeS/3600), Math.floor((s.timeS%3600)/60), (s.timeS%60).toFixed(2)]
          .map((v,i)=> String(v).padStart(i===0?2:i===1?2:5,"0")).join(":");

      document.getElementById("nullDepth").textContent = s.nullActive ? s.nullDepthDb.toFixed(1) : "—";
      document.getElementById("nullState").textContent = s.nullActive ? "active toward flock" : "inactive";
      document.getElementById("nullState").style.color = s.nullActive ? "var(--coral)" : "var(--muted)";

      document.getElementById("peakDensity").textContent = s.maxDensity.toFixed(2);
      const over = document.getElementById("overTag");
      over.textContent = s.metrics.fracOver > 0 ? "over " + (s.metrics.fracOver*100).toFixed(0) + "% of cells" : "below threshold";
      over.style.color = s.metrics.fracOver > 0 ? "var(--bad)" : "var(--ok)";

      document.getElementById("tput").textContent = s.throughputMbps.toFixed(0);
      document.getElementById("tputDx").textContent = "vs baseline " + (s.baselineThroughputMbps>0?s.baselineThroughputMbps.toFixed(0):"—");

      document.getElementById("beam").textContent = (s.servingBeamIndex+1) + " @ " + s.servingBeamAzimuthDeg.toFixed(1) + "°";
      document.getElementById("gain").textContent = "gain " + s.servingBeamGainDbi.toFixed(1) + " dBi (array+element)";

      document.getElementById("flock").textContent = s.flock.bearingDeg.toFixed(1) + "° @ " + s.flock.rangeM.toFixed(0) + "m";
      document.getElementById("flockAlt").textContent = s.flock.altitudeMinM.toFixed(0) + "–" + s.flock.altitudeMaxM.toFixed(0) + " m";

      document.getElementById("reward").textContent = s.reward.reward.toFixed(2);
      document.getElementById("rewardTerms").textContent =
        "T̂ " + s.reward.throughput.toFixed(3) + " · Ê " + s.reward.exposure.toFixed(3) + " · P̂ " + s.reward.power.toFixed(3) + " · w " + s.reward.w.toFixed(2);

      document.getElementById("baseline").textContent = "no-null " + (s.baselineReward.reward.toFixed(2)) + " rew";
      document.getElementById("baselineNote").textContent =
        "Smax " + (s.baseline.maxDensity>0?s.baseline.maxDensity.toFixed(2):"—") + " W/m² · null " + (s.baseline.nullActive?s.baseline.nullDepthDb.toFixed(1):0) + " dB";

      document.getElementById("explain").innerHTML = "<b>Step " + s.step + "</b> · " + s.explanation.replace(/\n/g,"<br>");
      document.getElementById("ts").textContent = new Date().toLocaleTimeString();
    });
  setTimeout(tick, Math.max(80, 1000 - (s && s.step ? s.step % 100 : 0) * 5));
}
tick();
</script>
</body>
</html>
"""


def _build_app() -> Any:
    """Tiny stdlib app: /scene.json (live state) + / (the HTML page)."

    We deliberately avoid importing FastAPI/uvicorn here so the demo survives a
    missing optional dependency or a broken lockfile. The tradeoff is that this is a
    plain HTTP server, not a WebSocket stream — fine for a quick demo that the user
    can open in a browser tab.
    """
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        server_version = "flywaydqn-demo"
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            # Every response carries Content-Length: the server speaks HTTP/1.1
            # (keep-alive), so without it clients wait forever for the body to end.
            if self.path == "/scene.json":
                import json

                try:
                    body = json.dumps(_scene_as_dict(_current_scene)).encode("utf-8")
                except Exception as e:
                    body = json.dumps({"error": str(e)}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path == "/" or self.path == "/index.html":
                body = HTML_PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path == "/policies":
                body = json.dumps([
                    {"name": "analytic-null", "label": "Analytic Null (Projection)", "description": "Nearest-codebook beam toward the flock with a projection null when the flock is within 220 m."},
                    {"name": "dqn", "label": "FlywayDQN (Trained)", "description": "Trained RL policy selecting beam index and null depth from the live scene state."},
                    {"name": "baseline", "label": "Baseline (Track + No Null)", "description": "Beam tracks the flock bearing with no ecological null applied."},
                ]).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path == "/scenarios":
                body = json.dumps([
                    {"key": "scenario_001", "label": "Sector A · gNodeB-042 (Crane)", "description": "180-bird crane flock at 47.2 deg bearing, 184 m range, 20-80 m altitude."},
                    {"key": "scenario_002", "label": "Sector B · gNodeB-077 (Geese)", "description": "60-bird greylag goose flock, higher cohesion, faster wingbeat."},
                    {"key": "scenario_003", "label": "Sector C · gNodeB-113 (Sandhill)", "description": "40-bird sandhill crane flock with 150-deg mirror corridor geometry."},
                ]).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            body = b"FlywayDQN demo: /, /scene.json, /policies, /scenarios.\n"
            self.send_response(404)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            # Keep the noise down so the user can read the scene in a terminal.
            pass

    return Handler


_current_scene: DemoScene = DemoScene()


def _run_server(host: str = "127.0.0.1", port: int = 8123) -> None:
    import socket
    from http.server import ThreadingHTTPServer

    Handler = _build_app()

    # Bind once so we can print the URL even if the listener fails later.
    srv = ThreadingHTTPServer((host, port), Handler, bind_and_activate=False)
    srv.server_bind()
    srv.server_activate()
    try:
        print(f"FlywayDQN online-round demo: http://{host}:{port}/")
        print(f"  scene endpoint: http://{host}:{port}/scene.json")
        print(f"  seed {DEMO_SEED} · freq {DEMO_FREQ_GHZ} GHz · {DEMO_N_USERS} UEs · {DEMO_N_BEAMS} beams")
        print(f"  press Ctrl+C to stop")
        sys.stdout.flush()
        srv.serve_forever()
    except KeyboardInterrupt:
        print("stopped")
    finally:
        srv.server_close()


def main() -> None:
    global _current_scene
    # Compute the first frame synchronously so /scene.json is valid from the very
    # first request; the loop below then advances from step 1.
    _current_scene = _step_deterministic(DemoScene(), 0, DEMO_SEED)

    # Kick off the background scene loop so /scene.json always returns the latest step.
    import threading

    running = True

    def loop():
        global _current_scene
        step = 1
        while running:
            start = time.time()
            _current_scene = _step_deterministic(_current_scene, step, DEMO_SEED)
            step += 1
            # Pace the loop so the browser fetch stays meaningful; step = 0.1 s sim time.
            elapsed = time.time() - start
            sleep_for = max(0.0, DEMO_DT_SECONDS * 2 - elapsed)
            if sleep_for > 0:
                time.sleep(sleep_for)

    worker = threading.Thread(target=loop, daemon=True)
    worker.start()

    try:
        _run_server()
    except Exception as e:
        print(f"server failed: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
