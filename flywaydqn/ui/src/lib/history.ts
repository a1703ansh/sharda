// Per-step session history for the timeline, the stat tiles and the
// before/after comparison (WP-F).
//
// Every value these helpers produce is a function of frames the service
// actually streamed and the console actually displayed — there is no
// interpolation, no smoothing and no constant standing in for a measurement.
// When a value cannot be computed yet, the caller gets `null`/`NaN` and
// renders "—".

import type { StateFrame } from './api';

/** One displayed frame, reduced to what the timeline and tiles read. */
export interface HistorySample {
  step: number;
  timeS: number;
  /** `frame.policyName`: the controller that produced this frame. */
  policyName: string;
  /** `metrics.maxDensityWm2` of this frame (W/m²). */
  peakExposureWm2: number;
  /** `metrics.sumThroughputMbps` of this frame (Mbit/s). */
  throughputMbps: number;
  nullActive: boolean;
  /** Measured null depth of this frame (dB); 0 while no null is on air. */
  nullDepthDb: number;
  /** `powerMw` actually transmitted this frame (mW). */
  powerMw: number;
  /** `beam.azimuthDeg` of this frame (deg). */
  azimuthDeg: number;
}

/** Aggregates over one run; `NaN` fields mean "not computable yet". */
export interface RunSummary {
  policyName: string;
  frames: number;
  lastTimeS: number;
  peakExposureWm2: number;
  peakAtTimeS: number;
  meanThroughputMbps: number;
  nullOnCount: number;
  /** Last frame's measured null depth (dB). */
  nullDepthDb: number;
  /** Last frame's transmit power (mW). */
  powerMw: number;
  azimuthMinDeg: number;
  azimuthMaxDeg: number;
}

export interface SessionHistory {
  /** Samples of the run currently streaming (never capped below MAX). */
  current: HistorySample[];
  /** Frozen summary of the run before the current one — the "before" panel. */
  previous: RunSummary | null;
}

/**
 * Cap on stored samples. One demo episode is 600 steps at dt 0.1 s; the cap
 * keeps several episodes' worth so a scenario/engine switch has something to
 * freeze, without growing without bound.
 */
export const MAX_HISTORY_SAMPLES = 4000;

export function emptyHistory(): SessionHistory {
  return { current: [], previous: null };
}

export function sampleFromFrame(frame: StateFrame): HistorySample {
  return {
    step: frame.step,
    timeS: frame.timeS,
    policyName: frame.policyName,
    peakExposureWm2: frame.metrics.maxDensityWm2,
    throughputMbps: frame.metrics.sumThroughputMbps,
    nullActive: frame.nullActive,
    nullDepthDb: frame.nullDepthDb,
    powerMw: frame.powerMw,
    azimuthDeg: frame.beam.azimuthDeg,
  };
}

export function appendSample(
  history: SessionHistory,
  sample: HistorySample,
): SessionHistory {
  let current = [...history.current, sample];
  if (current.length > MAX_HISTORY_SAMPLES) {
    current = current.slice(current.length - MAX_HISTORY_SAMPLES);
  }
  return { current, previous: history.previous };
}

/**
 * Freeze the current run as the "before" snapshot and start an empty one.
 * Called whenever the service re-creates a session (engine switch, scenario
 * change, reweight, reset), so the console can show before | after instead of
 * wiping the panel.
 */
export function startNewRun(history: SessionHistory): SessionHistory {
  if (history.current.length === 0) return history;
  return { current: [], previous: summarize(history.current) ?? history.previous };
}

/** Aggregate a run's samples; `null` when there is nothing to summarize. */
export function summarize(samples: HistorySample[]): RunSummary | null {
  if (samples.length === 0) return null;

  let peak = Number.NaN;
  let peakAtTimeS = 0;
  let throughputSum = 0;
  let throughputCount = 0;
  let nullOnCount = 0;
  let minAzimuth = Number.NaN;
  let maxAzimuth = Number.NaN;

  for (const sample of samples) {
    if (Number.isFinite(sample.peakExposureWm2)) {
      if (!Number.isFinite(peak) || sample.peakExposureWm2 > peak) {
        peak = sample.peakExposureWm2;
        peakAtTimeS = sample.timeS;
      }
    }
    if (Number.isFinite(sample.throughputMbps)) {
      throughputSum += sample.throughputMbps;
      throughputCount += 1;
    }
    if (sample.nullActive) nullOnCount += 1;
    if (Number.isFinite(sample.azimuthDeg)) {
      minAzimuth = Number.isFinite(minAzimuth)
        ? Math.min(minAzimuth, sample.azimuthDeg)
        : sample.azimuthDeg;
      maxAzimuth = Number.isFinite(maxAzimuth)
        ? Math.max(maxAzimuth, sample.azimuthDeg)
        : sample.azimuthDeg;
    }
  }

  const last = samples[samples.length - 1];
  return {
    policyName: last.policyName,
    frames: samples.length,
    lastTimeS: last.timeS,
    peakExposureWm2: peak,
    peakAtTimeS: peakAtTimeS,
    meanThroughputMbps: throughputCount > 0 ? throughputSum / throughputCount : Number.NaN,
    nullOnCount: nullOnCount,
    nullDepthDb: last.nullDepthDb,
    powerMw: last.powerMw,
    azimuthMinDeg: minAzimuth,
    azimuthMaxDeg: maxAzimuth,
  };
}

/** `T+HH:MM:SS.ss` label for a simulated time (same format the hook uses). */
export function formatClock(timeS: number): string {
  const safe = Number.isFinite(timeS) ? timeS : 0;
  const hours = Math.floor(safe / 3600);
  const mins = Math.floor((safe % 3600) / 60);
  const secs = safe % 60;
  return `T+${String(hours).padStart(2, '0')}:${String(mins).padStart(2, '0')}:${secs
    .toFixed(2)
    .padStart(5, '0')}`;
}
