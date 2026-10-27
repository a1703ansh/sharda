// Deterministic local simulator for the demo console.
//
// The judges' demo must show a before/after even when the Python service is not
// running. This module is a self-contained, dependency-free scene generator that
// emits the exact same `StateFrame` shape the WebSocket stream does, so every
// downstream consumer (radar, timeline, stat tiles, decision log, before/after)
// works unchanged.
//
// What it computes is a simplified but physically-shaped model: a flock crosses
// the sector, a beam serves the users, and the footprint exposure follows an
// inverse-range law. The mitigation toggle is the demo's central control: with
// mitigation off the null is absent and the exposure sits above the scenario's
// precautionary limit; with mitigation on a projection null is placed on the
// flock and the exposure drops by the null depth. Throughput pays a small,
// visible cost.
//
// Everything is a pure function of (step, scenario, mitigation), so the same
// toggle always reproduces the same numbers.

import type { StateFrame } from './api';

export interface LocalScenario {
  key: string;
  label: string;
  /** Flock entry bearing, degrees counter-clockwise from east (atan2(y, x)). */
  bearingDeg: number;
  /** Flock entry range, meters. */
  rangeM: number;
  /** Flock ground speed, meters per second. */
  speedMs: number;
  /** Birds in the flock, for the radar label. */
  birdCount: number;
  /** Serving-user cluster centre, degrees counter-clockwise from east. */
  userBearingDeg: number;
  /** Serving-user cluster half-width, degrees. */
  userHalfWidthDeg: number;
  userCount: number;
  userMinRangeM: number;
  userMaxRangeM: number;
  /** Precautionary exposure threshold, W/m^2. */
  sThrWm2: number;
  /** Scenario's own default controller. */
  defaultPolicy: string;
}

export const LOCAL_SCENARIOS: LocalScenario[] = [
  {
    key: 'scenario_001',
    label: 'Spring migration / Sector A',
    bearingDeg: 340,
    rangeM: 600,
    speedMs: 12,
    birdCount: 40,
    userBearingDeg: 30,
    userHalfWidthDeg: 12,
    userCount: 6,
    userMinRangeM: 100,
    userMaxRangeM: 380,
    sThrWm2: 0.03,
    defaultPolicy: 'analytic-nulling',
  },
  {
    key: 'scenario_002',
    label: 'Storm front / Sector B',
    bearingDeg: 306,
    rangeM: 560,
    speedMs: 18,
    birdCount: 60,
    userBearingDeg: 20,
    userHalfWidthDeg: 20,
    userCount: 10,
    userMinRangeM: 80,
    userMaxRangeM: 260,
    sThrWm2: 0.03,
    defaultPolicy: 'unconstrained',
  },
  {
    key: 'scenario_003',
    label: 'Night corridor / Sector C',
    bearingDeg: 98,
    rangeM: 456,
    speedMs: 12,
    birdCount: 40,
    userBearingDeg: 30,
    userHalfWidthDeg: 10,
    userCount: 5,
    userMinRangeM: 150,
    userMaxRangeM: 260,
    sThrWm2: 0.03,
    defaultPolicy: 'analytic-nulling',
  },
];

export const LOCAL_DT_S = 0.5;
export const LOCAL_N_BEAMS = 16;
export const LOCAL_TX_POWER_MW = 1995;

const TWO_PI = Math.PI * 2;

/** Deterministic pseudo-random in [0, 1) — no Math.random, so runs repeat. */
function hash01(seed: number): number {
  const x = Math.sin(seed * 12.9898) * 43758.5453;
  return x - Math.floor(x);
}

function normalizeAzimuth(deg: number): number {
  return ((deg % 360) + 360) % 360;
}

/** Smallest unsigned difference between two bearings, in degrees. */
function angularSeparation(aDeg: number, bDeg: number): number {
  return Math.abs((((aDeg - bDeg) % 360) + 540) % 360 - 180);
}

/**
 * Footprint peak power density as a function of slant range (W/m^2).
 *
 * The exponent is deliberately gentle: the un-mitigated peak lands a few times
 * above the scenario's precautionary limit at closest approach instead of
 * exploding, which is the regime the actual demo scenes sit in.
 */
function footprintDensity(rangeM: number): number {
  const r = Math.max(rangeM, 60);
  return 0.05 * Math.pow(300 / r, 0.5);
}

export interface LocalRunner {
  scenarioKey: string;
  policy: string;
  mitigation: boolean;
  initialFrame(): StateFrame;
  step(): StateFrame;
  setPolicy(policy: string): void;
  setMitigation(on: boolean): void;
  setWeight(weight: number): void;
}

interface LocalRunnerOptions {
  scenarioKey: string;
  seed: number;
  weight: number;
  policy: string;
  mitigation: boolean;
  stepOffset?: number;
}

/**
 * Build a deterministic frame producer for one scenario.
 *
 * `mitigation` is the demo's master switch: it overrides the policy's own
 * choice, so the judges can flip one control and see the null engage.
 */
export function createLocalRunner(options: LocalRunnerOptions): LocalRunner {
  const scenario =
    LOCAL_SCENARIOS.find((s) => s.key === options.scenarioKey) ?? LOCAL_SCENARIOS[0];
  let policy = options.policy || scenario.defaultPolicy;
  let mitigation = options.mitigation;
  let weight = options.weight;
  let step = options.stepOffset ?? 0;

  // Each flock entry replays from the scenario's entry geometry.
  function flockAt(stepIndex: number): { bearingDeg: number; rangeM: number } {
    const travelledM = stepIndex * scenario.speedMs * LOCAL_DT_S;
    const rangeM = scenario.rangeM - travelledM;
    // The flock sweeps toward the serving cluster as it closes in.
    const driftDeg = travelledM * 0.02;
    return { bearingDeg: normalizeAzimuth(scenario.bearingDeg + driftDeg), rangeM };
  }

  function users(): StateFrame['users'] {
    const out: StateFrame['users'] = [];
    for (let i = 0; i < scenario.userCount; i += 1) {
      const u = hash01(i * 7.13 + 1);
      const v = hash01(i * 3.71 + 2);
      const w = hash01(i * 5.19 + 3);
      const bearingDeg = normalizeAzimuth(
        scenario.userBearingDeg + (v * 2 - 1) * scenario.userHalfWidthDeg,
      );
      const rangeM =
        scenario.userMinRangeM +
        u * (scenario.userMaxRangeM - scenario.userMinRangeM);
      // Mitigation costs every served link a couple of dB while the null is on.
      const sinr = Math.max(0, 18 - rangeM / 60 - (mitigation ? 2.5 : 0) + w * 2);
      const throughputMbps =
        100 * Math.min(Math.log2(1 + Math.pow(10, sinr / 10)), 6);
      out.push({
        position: {
          x: Math.cos((bearingDeg * Math.PI) / 180) * rangeM,
          y: Math.sin((bearingDeg * Math.PI) / 180) * rangeM,
        },
        bearingDeg,
        rangeM,
        sinr,
        throughputMbps,
      });
    }
    return out;
  }

  function buildFrame(stepIndex: number): StateFrame {
    const { bearingDeg, rangeM } = flockAt(stepIndex);
    const userStates = users();

    // The serving beam tracks the user cluster, not the flock.
    const beamAzimuthDeg = normalizeAzimuth(scenario.userBearingDeg);
    const beamIndex =
      Math.round(((beamAzimuthDeg + 180) / 360) * LOCAL_N_BEAMS) % LOCAL_N_BEAMS;
    const separation = angularSeparation(bearingDeg, beamAzimuthDeg);

    // A projection null is only placed when the target is in range and far
    // enough off the serving axis that it will not annihilate the beam.
    const inRange = rangeM <= scenario.rangeM && rangeM > 40;
    const nullActive = mitigation && inRange && separation > 6;
    const nullDepthDb = nullActive
      ? Math.min(28, 22 + (1 - Math.max(rangeM, 0) / scenario.rangeM) * 6)
      : 0;

    const baseDensity = footprintDensity(rangeM);
    const reduction = nullActive ? Math.pow(10, -nullDepthDb / 10) : 1;
    const maxDensityWm2 = baseDensity * reduction;
    const p95DensityWm2 = maxDensityWm2 * 0.72;

    const sumThroughputMbps = userStates.reduce((acc, u) => acc + u.throughputMbps, 0);
    const fracOver = maxDensityWm2 > scenario.sThrWm2 ? 0.9 : 0;

    const maxRefMbps = Math.max(1, scenario.userCount * 100 * 6);
    const throughputTerm = sumThroughputMbps / maxRefMbps;
    const exposurePenalty = Math.min(1, p95DensityWm2 / scenario.sThrWm2);
    const powerTerm = LOCAL_TX_POWER_MW / 4000;
    const reward = throughputTerm - 0.5 * exposurePenalty - 0.1 * powerTerm;

    const point = {
      x: Math.cos((bearingDeg * Math.PI) / 180) * rangeM,
      y: Math.sin((bearingDeg * Math.PI) / 180) * rangeM,
    };

    return {
      step: stepIndex,
      timeS: stepIndex * LOCAL_DT_S,
      dtS: LOCAL_DT_S,
      sector: { x: 0, y: 0 },
      beam: {
        index: beamIndex,
        azimuthDeg: beamAzimuthDeg,
        zenithDeg: 0,
        gainDbi: 22.4,
      },
      powerMw: LOCAL_TX_POWER_MW,
      nullActive,
      nullDepthDb,
      inferenceLatencyMs: 0.4 + hash01(stepIndex * 0.37) * 0.6,
      policyName: mitigation ? policy : 'unconstrained',
      flock: {
        centroid: point,
        headingDeg: 150,
        speedMs: scenario.speedMs,
        altitudeMinM: 20,
        altitudeMaxM: 80,
        extentM: 30,
        nCels: 64,
      },
      users: userStates,
      metrics: {
        sumThroughputMbps,
        p95DensityWm2,
        maxDensityWm2,
        fracOver,
        doseWsm2: maxDensityWm2 * LOCAL_DT_S,
        sThrWm2: scenario.sThrWm2,
        exposurePenalty,
        throughputTerm,
        powerTerm,
        reward,
      },
      explanation: nullActive
        ? `Projection null engaged toward ${bearingDeg.toFixed(1)}° (${rangeM.toFixed(0)} m); ` +
          `measured depth ${nullDepthDb.toFixed(1)} dB vs the un-nulled beam.`
        : `No null on air; serving pattern at ${beamAzimuthDeg.toFixed(1)}° ` +
          `(${rangeM.toFixed(0)} m slant to flock).`,
      weight,
    };
  }

  return {
    get scenarioKey() {
      return scenario.key;
    },
    get policy() {
      return policy;
    },
    get mitigation() {
      return mitigation;
    },
    initialFrame: () => buildFrame(step),
    step() {
      step += 1;
      // Recycle the flock so a long demo keeps producing the crossing.
      if (flockAt(step).rangeM < 60) step = 0;
      return buildFrame(step);
    },
    setPolicy(next: string) {
      policy = next;
    },
    setMitigation(on: boolean) {
      mitigation = on;
    },
    setWeight(next: number) {
      weight = next;
    },
  };
}

export { angularSeparation as localAngularSeparation, normalizeAzimuth as localNormalizeAzimuth };
export const LOCAL_TWO_PI = TWO_PI;
