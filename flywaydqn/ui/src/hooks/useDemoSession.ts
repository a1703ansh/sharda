import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { SimState, PolicyOption, FlockTarget, UEData, DecisionLogEntry, RadarEventEntry, ScreenId } from '../types';
import {
  buildWsUrl,
  createSession,
  fetchPolicies,
  fetchScenarios,
  setSessionPace,
  type ScenarioInfo,
  type ScenarioRequest,
  type StateFrame,
  type UserState as ApiUserState,
} from '../lib/api';
import {
  appendSample,
  emptyHistory,
  formatClock,
  sampleFromFrame,
  startNewRun,
  type SessionHistory,
} from '../lib/history';
import { LOCAL_DT_S, LOCAL_SCENARIOS, createLocalRunner, type LocalRunner } from '../lib/localSim';

interface UseDemoSessionOptions {
  scenarioKey?: string;
  seed?: number;
  initialWeight?: number;
  autoConnect?: boolean;
}

export interface UseDemoSessionResult {
  currentScreen: ScreenId;
  setCurrentScreen: (s: ScreenId) => void;
  simState: SimState;
  connected: boolean;
  error: string | null;
  connect: () => Promise<void>;
  disconnect: () => void;
  play: () => void;
  pause: () => void;
  togglePlay: () => void;
  step: () => void;
  setSpeed: (speed: 1 | 2 | 4) => void;
  setTradeoffWeight: (w: number) => void;
  selectScenario: (key: string, seed?: number) => Promise<void>;
  selectFlock: (flockId: string) => void;
  /** Select a controller by registered name; recreates the session to apply it. */
  setEngine: (policyName: string) => void;
  /** Whether the FlywayDQN mitigation is engaged — the demo's master switch. */
  mitigationEnabled: boolean;
  /** Flip the mitigation; off shows the un-mitigated exposure, on shows the null. */
  setMitigationEnabled: (on: boolean) => void;
  /** True while the built-in local scene is driving the console (no service). */
  usingLocalSource: boolean;
  resetSim: () => void;
  /** Current session id, if one is active. */
  sessionId: string | null;
  /** Scenario list reported by `GET /scenarios` (empty until the first connect). */
  scenarios: ScenarioInfo[];
  /** Key of the scenario the live session is running (drives selectors by key). */
  scenarioKey: string;
  /** Selectable controllers reported by `GET /policies` (empty until connect). */
  policies: PolicyOption[];
  /**
   * Per-step history of the current run plus the frozen summary of the run
   * before it (WP-F). Timeline, tiles and the before/after panel read it.
   */
  history: SessionHistory;
}

/**
 * Live demo console state.
 *
 * The FastAPI service at `VITE_API_BASE` (default http://localhost:8000) owns
 * the simulation: `POST /sessions` creates a session and
 * `WS /sessions/{id}/stream` pushes one `StateFrame` per step at the frame's own
 * `dt_s` cadence. The socket is receive-only — the server ignores inbound text —
 * so playback controls (play/pause/step/speed) are applied client-side over the
 * buffered frames, and anything that changes the physics (scenario, seed,
 * trade-off weight, reset) creates a fresh session instead of sending a command.
 *
 * The wire frame carries the physics; a few console fields are presentation
 * only and are marked `NOT ON THE WIRE` below. They are constant per scenario
 * rather than invented per frame.
 */

/** Exposure threshold is read from `metrics.sThrWm2`; there is no display ceiling. */
/** Display tick: one buffered frame per 100 ms is real time at speed 1. */
const DISPLAY_TICK_MS = 100;
/** Frames buffered ahead of the display loop before the oldest are dropped. */
const MAX_BUFFERED_FRAMES = 60;
/** Debounce for session re-creation while the Pareto slider is dragged. */
const REWEIGHT_DEBOUNCE_MS = 300;
/** Decision/radar log depth kept in view. */
const MAX_LOG_ENTRIES = 6;

/** NOT ON THE WIRE: species/count metadata per scenario, mirroring configs/demo. */
interface ScenarioProfile {
  telemetryId: string;
  species: string;
  commonName: string;
  birdCount: number;
  wingbeatHz: number;
  cohesionScore: number;
  thermalC: number;
}

const DEFAULT_PROFILE: ScenarioProfile = {
  telemetryId: '#409',
  species: 'Grus grus',
  commonName: 'Common Crane',
  birdCount: 40,
  wingbeatHz: 3.6,
  cohesionScore: 0.94,
  thermalC: 39.8,
};

const SCENARIO_PROFILES: Record<string, ScenarioProfile> = {
  scenario_001: DEFAULT_PROFILE,
  scenario_002: {
    telemetryId: '#412',
    species: 'Anser anser',
    commonName: 'Greylag Geese',
    birdCount: 60,
    wingbeatHz: 4.2,
    cohesionScore: 0.88,
    thermalC: 40.1,
  },
  scenario_003: {
    telemetryId: '#388',
    species: 'Antigone canadensis',
    commonName: 'Sandhill Crane',
    birdCount: 40,
    wingbeatHz: 3.4,
    cohesionScore: 0.91,
    thermalC: 39.5,
  },
};

/** NOT ON THE WIRE: friendly device names for the UE list, by index. */
const UE_LABELS = [
  '4K Mobile',
  'Fixed CPE',
  'Vehicular',
  'IoT Gateway',
  'Smart Grid',
  'AR Headset',
  'Telemetry Hub',
  'Mobile Node',
];

function normalizeAzimuth(deg: number): number {
  return ((deg % 360) + 360) % 360;
}

function signedAngleDelta(aDeg: number, bDeg: number): number {
  return Math.abs(((aDeg - bDeg + 180) % 360) - 180);
}

function riskLevelFor(distanceM: number): FlockTarget['riskLevel'] {
  if (distanceM < 240) return 'CRITICAL';
  if (distanceM < 400) return 'ALERT';
  return 'NOMINAL';
}

/** NOT ON THE WIRE: CQI/BLER are derived from the reported SINR for display. */
function cqiFromSinrDb(sinrDb: number): number {
  return Math.min(15, Math.max(1, Math.round((sinrDb - 3) / 1.7)));
}

function blerFromSinrDb(sinrDb: number): number {
  if (sinrDb >= 20) return 0.0;
  if (sinrDb >= 13) return 0.01;
  if (sinrDb >= 7) return 0.02;
  if (sinrDb >= 3) return 0.04;
  return 0.08;
}

function mean(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((acc, v) => acc + v, 0) / values.length;
}

/** Console readouts are formatted here; components print values verbatim. */
function round(value: number, decimals: number): number {
  if (!Number.isFinite(value)) return 0;
  const factor = 10 ** decimals;
  return Math.round(value * factor) / factor;
}

function flockTargetFromFrame(frame: StateFrame, profile: ScenarioProfile): FlockTarget {
  const distanceM = Math.hypot(frame.flock.centroid.x, frame.flock.centroid.y);
  const azimuthDeg = normalizeAzimuth(
    (Math.atan2(frame.flock.centroid.y, frame.flock.centroid.x) * 180) / Math.PI,
  );
  const speedMs = Math.max(frame.flock.speedMs, 0.1);
  return {
    id: profile.telemetryId,
    species: profile.species,
    commonName: profile.commonName,
    birdCount: profile.birdCount,
    azimuthDeg: round(azimuthDeg, 1),
    distanceM: round(distanceM, 0),
    altitudeM: round((frame.flock.altitudeMinM + frame.flock.altitudeMaxM) / 2, 0),
    speedMs: round(frame.flock.speedMs, 1),
    headingDeg: frame.flock.headingDeg,
    riskLevel: riskLevelFor(distanceM),
    wingbeatHz: profile.wingbeatHz,
    cohesionScore: profile.cohesionScore,
    thermalC: profile.thermalC,
    // Time to reach the sector origin at the current range and ground speed.
    estEntrySec: Math.max(0, Math.round(distanceM / speedMs)),
  };
}

function userFromFrame(
  user: ApiUserState,
  index: number,
  nullActive: boolean,
  flockBearingDeg: number,
): UEData {
  const id = `UE-${101 + index}`;
  const label = UE_LABELS[index % UE_LABELS.length];
  const sinrDb = Number.isFinite(user.sinr) ? user.sinr : 0;
  return {
    id,
    name: `${id} (${label})`,
    azimuthDeg: round(normalizeAzimuth(user.bearingDeg), 1),
    distanceM: round(user.rangeM, 0),
    sinrDb: round(sinrDb, 1),
    throughputMbps: round(user.throughputMbps, 0),
    blerPercent: blerFromSinrDb(sinrDb),
    cqi: cqiFromSinrDb(sinrDb),
    isNulledRisk: nullActive && signedAngleDelta(user.bearingDeg, flockBearingDeg) < 15,
  };
}

/** Baseline console state, shown until the first frame of a session arrives. */
function initialState(scenarioLabel: string, weight: number): SimState {
  const profile = SCENARIO_PROFILES.scenario_001 ?? DEFAULT_PROFILE;
  const placeholder: FlockTarget = {
    id: profile.telemetryId,
    species: profile.species,
    commonName: profile.commonName,
    birdCount: profile.birdCount,
    wingbeatHz: profile.wingbeatHz,
    cohesionScore: profile.cohesionScore,
    thermalC: profile.thermalC,
    azimuthDeg: 340,
    distanceM: 600,
    altitudeM: 50,
    speedMs: 12,
    headingDeg: 150,
    riskLevel: 'NOMINAL',
    estEntrySec: 50,
  };
  return {
    step: 0,
    timeSec: 0,
    timeString: formatClock(0),
    isPlaying: true,
    speed: 1,
    scenario: scenarioLabel,
    // Empty until the first frame reports which controller actually ran.
    engine: '',
    tradeoffWeight: weight,

    peakExposureWm2: 0,
    exposureLimitWm2: 0,
    timeOverLimitSec: 0,
    nullDepthDb: 0,
    nullActive: false,
    cellThroughputGbps: 0,
    throughputFraction: 0,
    baselineThroughputGbps: 0,
    meanSinrDb: 0,
    meanBlerPercent: 0,
    txRfPowerW: 0,
    inferenceLatencyMs: 0,

    beamIndex: 0,
    beamAzimuthDeg: 0,
    beamZenithDeg: 0,
    beamGainDbi: 0,
    beam: null,

    rewardThroughput: 0,
    rewardExposure: 0,
    rewardPower: 0,
    rewardNet: 0,

    activeFlock: placeholder,
    allFlocks: [placeholder],
    users: [],

    decisionLogs: [],
    radarEvents: [],
    explanation: null,
  };
}

export function useDemoSession(options: UseDemoSessionOptions = {}): UseDemoSessionResult {
  const {
    scenarioKey: initialScenarioKey,
    seed: initialSeed = 1,
    initialWeight = 0.75,
    autoConnect = true,
  } = options;

  const [currentScreen, setCurrentScreen] = useState<ScreenId>('beam-steering');
  const [simState, setSimState] = useState<SimState>(() =>
    initialState('Sector A [gNodeB-042]', initialWeight),
  );
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [scenarios, setScenarios] = useState<ScenarioInfo[]>([]);
  const [policies, setPolicies] = useState<PolicyOption[]>([]);
  // Every displayed frame appends here (WP-F): the timeline, the stat tiles
  // and the before/after panel read from it, not from the six-entry log.
  const [history, setHistory] = useState<SessionHistory>(emptyHistory);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [scenarioKey, setScenarioKey] = useState<string>(
    initialScenarioKey ?? 'scenario_001',
  );
  // The demo's master switch: whether the FlywayDQN mitigation runs. With it
  // off the console shows the un-mitigated exposure; flipping it on engages the
  // ecological null and the numbers change. It is a first-class control so the
  // judges can flip one thing and see the before/after.
  const [mitigationEnabled, setMitigationEnabledState] = useState(true);
  const [usingLocalSource, setUsingLocalSource] = useState(false);

  const socketRef = useRef<WebSocket | null>(null);
  const bufferRef = useRef<StateFrame[]>([]);
  const generationRef = useRef(0);
  const sessionIdRef = useRef<string | null>(null);
  const scenarioKeyRef = useRef<string | undefined>(initialScenarioKey);
  const seedRef = useRef<number>(initialSeed);
  const weightRef = useRef<number>(initialWeight);
  const playingRef = useRef(true);
  const speedRef = useRef<1 | 2 | 4>(1);
  const reweightTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const scenariosRef = useRef<ScenarioInfo[]>([]);
  const policiesRef = useRef<PolicyOption[]>([]);
  // Registered policy the user chose; undefined means "keep the scenario's own".
  const policyRef = useRef<string | undefined>(undefined);

  const beamIndexRef = useRef<number | null>(null);
  const nullActiveRef = useRef(false);
  const logSeqRef = useRef(0);
  // Deterministic local scene used when the Python service is unavailable.
  const localRunnerRef = useRef<LocalRunner | null>(null);
  const localTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const mitigationRef = useRef(true);

  const labelForKey = useCallback((key: string | undefined): string | undefined => {
    if (!key) return undefined;
    return scenariosRef.current.find((s) => s.key === key)?.label;
  }, []);

  /** Project one wire frame onto the console state, appending derived events. */
  const applyFrame = useCallback(
    (frame: StateFrame, isFirstOfSession: boolean) => {
      const key = scenarioKeyRef.current ?? scenariosRef.current[0]?.key ?? 'scenario_001';
      const profile = SCENARIO_PROFILES[key] ?? DEFAULT_PROFILE;
      const flock = flockTargetFromFrame(frame, profile);
      const users = frame.users.map((u, i) =>
        userFromFrame(u, i, frame.nullActive, flock.azimuthDeg),
      );

      // Event derivation happens here, outside the state updater: React may run
      // an updater twice (StrictMode), so the updater must stay pure and these
      // transition refs must advance exactly once per frame.
      const timestamp = formatClock(frame.timeS);
      const newDecisions: DecisionLogEntry[] = [];
      const newRadarEvents: RadarEventEntry[] = [];

      const pushDecision = (
        actionType: DecisionLogEntry['actionType'],
        description: string,
        details: string,
      ) => {
        logSeqRef.current += 1;
        newDecisions.unshift({
          id: `log-${logSeqRef.current}-${actionType}`,
          timestamp,
          step: frame.step,
          actionType,
          description,
          details,
        });
      };

      if (isFirstOfSession) {
        pushDecision(
          'STATE_OBSERVE',
          `Stream connected — ${labelForKey(key) ?? key} (seed ${seedRef.current}).`,
          `step ${frame.step} · reward ${frame.metrics.reward.toFixed(2)}`,
        );
        logSeqRef.current += 1;
        newRadarEvents.unshift({
          id: `ev-${logSeqRef.current}-entry`,
          timestamp,
          sensorSource: 'Ku-Radar',
          eventType: 'CORRIDOR_ENTRY',
          eventBadgeClass: 'coral',
          details: `Flock ${flock.id} in corridor at ${frame.flock.centroid.x.toFixed(
            0,
          )} m x, ${flock.distanceM.toFixed(0)} m range, ${Math.round(flock.altitudeM)} m AGL`,
          actionTriggered: 'Null-DQN-Prep',
        });
      }

      if (frame.nullActive !== nullActiveRef.current) {
        nullActiveRef.current = frame.nullActive;
        pushDecision(
          frame.nullActive ? 'NULL_DEEPEN' : 'RESTORE_CARRIER',
          frame.nullActive
            ? `Ecological null engaged toward ${flock.azimuthDeg.toFixed(1)}° (${flock.distanceM.toFixed(0)} m).`
            : `Null released — serving pattern restored at ${frame.beam.azimuthDeg.toFixed(1)}°.`,
          frame.explanation ?? `reward ${frame.metrics.reward.toFixed(2)}`,
        );
        if (frame.nullActive) {
          logSeqRef.current += 1;
          newRadarEvents.unshift({
            id: `ev-${logSeqRef.current}-null`,
            timestamp,
            sensorSource: 'LiDAR-3D',
            eventType: 'RF_PROXIMITY_WARN',
            eventBadgeClass: 'teal',
            details: `Projection null locked ${flock.azimuthDeg.toFixed(1)}° · beam ${
              frame.beam.index
            } serving ${frame.beam.azimuthDeg.toFixed(1)}°`,
            actionTriggered: 'Null-DQN-Applied',
          });
        }
      }

      if (frame.beam.index !== beamIndexRef.current) {
        beamIndexRef.current = frame.beam.index;
        pushDecision(
          'LOBE_SPLIT',
          `Serving beam → ${frame.beam.index} at ${frame.beam.azimuthDeg.toFixed(1)}° (gain ${frame.beam.gainDbi.toFixed(1)} dBi).`,
          `sum rate ${frame.metrics.sumThroughputMbps.toFixed(0)} Mbps`,
        );
      }

      setSimState((prev) => ({
        ...prev,
        step: frame.step,
        timeSec: frame.timeS,
        timeString: timestamp,
        scenario: labelForKey(key) ?? prev.scenario,
        tradeoffWeight: frame.weight,

        engine: frame.policyName,
        peakExposureWm2: round(frame.metrics.maxDensityWm2, 5),
        exposureLimitWm2: frame.metrics.sThrWm2,
        timeOverLimitSec: round(
          frame.metrics.fracOver > 0 ? prev.timeOverLimitSec + frame.dtS : prev.timeOverLimitSec,
          1,
        ),
        nullDepthDb: round(frame.nullDepthDb, 2),
        nullActive: frame.nullActive,
        cellThroughputGbps: round(frame.metrics.sumThroughputMbps / 1000, 2),
        throughputFraction: round(frame.metrics.throughputTerm, 3),
        // The true same-seed unconstrained baseline is WP-F; until then this is
        // not comparable, so keep it empty instead of inventing a number.
        baselineThroughputGbps: Number.NaN,
        meanSinrDb: round(mean(users.map((u) => u.sinrDb)), 1),
        meanBlerPercent: round(mean(users.map((u) => u.blerPercent)), 2),
        txRfPowerW: round(frame.powerMw / 1000, 1),
        inferenceLatencyMs: round(frame.inferenceLatencyMs, 2),

        beamIndex: frame.beam.index,
        beamAzimuthDeg: round(frame.beam.azimuthDeg, 1),
        beamZenithDeg: round(frame.beam.zenithDeg, 1),
        beamGainDbi: round(frame.beam.gainDbi, 1),
        beam: {
          index: frame.beam.index,
          azimuthDeg: round(frame.beam.azimuthDeg, 1),
          zenithDeg: round(frame.beam.zenithDeg, 1),
          gainDbi: round(frame.beam.gainDbi, 1),
        },

        rewardThroughput: round(frame.metrics.throughputTerm, 3),
        rewardExposure: round(frame.metrics.exposurePenalty, 3),
        rewardPower: round(frame.metrics.powerTerm, 3),
        rewardNet: round(frame.metrics.reward, 3),

        activeFlock: flock,
        allFlocks: [flock],
        users,
        explanation: frame.explanation ?? null,

        decisionLogs: [...newDecisions, ...prev.decisionLogs].slice(0, MAX_LOG_ENTRIES),
        radarEvents: [...newRadarEvents, ...prev.radarEvents].slice(0, MAX_LOG_ENTRIES),
      }));
      setHistory((prev) => appendSample(prev, sampleFromFrame(frame)));
    },
    [labelForKey],
  );

  const closeSocket = useCallback(() => {
    const socket = socketRef.current;
    socketRef.current = null;
    if (socket) {
      socket.onopen = null;
      socket.onmessage = null;
      socket.onclose = null;
      socket.onerror = null;
      if (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING) {
        socket.close();
      }
    }
  }, []);

  const openStream = useCallback(
    (streamPath: string, generation: number) => {
      closeSocket();
      const socket = new WebSocket(buildWsUrl(streamPath));
      socketRef.current = socket;

      socket.onopen = () => {
        if (generation !== generationRef.current) return;
        setConnected(true);
        setError(null);
      };
      socket.onmessage = (event: MessageEvent<string>) => {
        if (generation !== generationRef.current) return;
        let frame: StateFrame;
        try {
          frame = JSON.parse(event.data) as StateFrame;
        } catch {
          return;
        }
        const buffer = bufferRef.current;
        buffer.push(frame);
        if (buffer.length > MAX_BUFFERED_FRAMES) {
          buffer.splice(0, buffer.length - MAX_BUFFERED_FRAMES);
        }
      };
      socket.onerror = () => {
        if (generation !== generationRef.current) return;
        setError('Stream error — is the FlywayDQN service running?');
      };
      socket.onclose = () => {
        if (generation !== generationRef.current) return;
        setConnected(false);
      };
    },
    [closeSocket],
  );

  const stopLocal = useCallback(() => {
    if (localTimerRef.current !== null) {
      clearInterval(localTimerRef.current);
      localTimerRef.current = null;
    }
    localRunnerRef.current = null;
  }, []);

  /**
   * Run the deterministic local scene, pushing one frame per display tick.
   *
   * This is the demo's safety net: when the Python service is not running, the
   * console still gets real frames, a moving radar and a before/after, instead
   * of sitting offline.
   */
  const startLocal = useCallback(
    (opts: { key?: string; policy?: string; mitigation?: boolean } = {}) => {
      stopLocal();
      closeSocket();
      const key = opts.key ?? scenarioKeyRef.current ?? LOCAL_SCENARIOS[0].key;
      const policy =
        opts.policy ?? policyRef.current ?? LOCAL_SCENARIOS.find((s) => s.key === key)?.defaultPolicy ?? 'analytic-nulling';
      const mitigation = opts.mitigation ?? mitigationRef.current;
      mitigationRef.current = mitigation;
      policyRef.current = policy;

      scenariosRef.current = LOCAL_SCENARIOS.map((s) => ({
        key: s.key,
        label: s.label,
        description: `${s.birdCount}-bird flock crossing at ${s.rangeM} m.`,
      }));
      setScenarios(scenariosRef.current);
      if (policiesRef.current.length === 0) {
        policiesRef.current = [
          {
            name: 'analytic-nulling',
            label: 'Analytic Null (baseline)',
            description: 'Greedy serving beam plus a projection null on the flock footprint.',
          },
          {
            name: 'unconstrained',
            label: 'Unconstrained (No Mitigation)',
            description: 'Greedy serving beam at full power with no ecological null.',
          },
        ];
        setPolicies(policiesRef.current);
      }

      const runner = createLocalRunner({
        scenarioKey: key,
        seed: seedRef.current,
        weight: weightRef.current,
        policy,
        mitigation,
      });
      localRunnerRef.current = runner;
      setUsingLocalSource(true);
      setConnected(true);
      setError(null);
      setSessionId(null);
      sessionIdRef.current = null;
      beamIndexRef.current = null;
      nullActiveRef.current = false;
      setScenarioKey(key);
      scenarioKeyRef.current = key;

      applyFrame(runner.initialFrame(), true);
      localTimerRef.current = setInterval(() => {
        if (!playingRef.current) return;
        const local = localRunnerRef.current;
        if (!local) return;
        const frames: StateFrame[] = [];
        for (let i = 0; i < speedRef.current; i += 1) frames.push(local.step());
        applyFrame(frames[frames.length - 1], false);
      }, DISPLAY_TICK_MS);
    },
    [applyFrame, closeSocket, stopLocal],
  );

  const startSession = useCallback(
    async (opts: { key?: string; seed?: number; weight?: number; policy?: string } = {}) => {
      const generation = generationRef.current + 1;
      generationRef.current = generation;
      bufferRef.current = [];
      beamIndexRef.current = null;
      nullActiveRef.current = false;
      closeSocket();
      setConnected(false);
      setError(null);
      // A re-created session is a new run: freeze the old run's summary so the
      // console can show before | after instead of wiping the panel.
      setHistory(startNewRun);

      if (opts.key !== undefined) {
        scenarioKeyRef.current = opts.key;
        setScenarioKey(opts.key);
      }
      if (opts.seed !== undefined) seedRef.current = opts.seed;
      if (opts.weight !== undefined) weightRef.current = opts.weight;
      if (opts.policy !== undefined) policyRef.current = opts.policy;

      try {
        let list = scenariosRef.current;
        if (list.length === 0) {
          list = await fetchScenarios();
          if (generation !== generationRef.current) return;
          scenariosRef.current = list;
          setScenarios(list);
        }
        if (policiesRef.current.length === 0) {
          const available = await fetchPolicies();
          if (generation !== generationRef.current) return;
          policiesRef.current = available;
          setPolicies(available);
        }
        const key = scenarioKeyRef.current ?? list[0]?.key;
        if (!key) throw new Error('The service reported no demo scenarios.');

        const request: ScenarioRequest = {
          key,
          seed: seedRef.current,
          weight: weightRef.current,
          // Only sent once the user has chosen; otherwise the scenario's own
          // policy applies and the frame reports which one that was.
          policy: policyRef.current ?? null,
        };
        const created = await createSession(request);
        if (generation !== generationRef.current) return;

        sessionIdRef.current = created.id;
        setSessionId(created.id);
        // A fresh session starts at pace 1.0. Re-apply the speed the console is
        // already showing so the buttons stay honest after a policy/scenario
        // switch. Pacing is presentation only, so a failure here must not fall
        // back to the local scene.
        if (speedRef.current !== 1) {
          try {
            await setSessionPace(created.id, speedRef.current);
          } catch {
            /* keep the live stream; it simply runs at 1x */
          }
          if (generation !== generationRef.current) return;
        }
        applyFrame(created.frame, true);
        openStream(created.streamPath, generation);
      } catch (e) {
        if (generation !== generationRef.current) return;
        // The service is unreachable: fall back to the deterministic local
        // scene so the demo still runs. The console shows the same frames and
        // the before/after panel still works, with no Python needed.
        const message = e instanceof Error ? e.message : String(e);
        setError(`Service unreachable (${message}) — running the built-in demo scene.`);
        startLocal({ key: scenarioKeyRef.current, policy: policyRef.current });
      }
    },
    [applyFrame, closeSocket, openStream, startLocal],
  );

  const connect = useCallback(() => startSession(), [startSession]);

  const disconnect = useCallback(() => {
    generationRef.current += 1;
    closeSocket();
    bufferRef.current = [];
    sessionIdRef.current = null;
    setSessionId(null);
    setConnected(false);
  }, [closeSocket]);

  const pause = useCallback(() => {
    playingRef.current = false;
    setSimState((prev) => ({ ...prev, isPlaying: false }));
  }, []);

  const play = useCallback(() => {
    playingRef.current = true;
    setSimState((prev) => ({ ...prev, isPlaying: true }));
  }, []);

  const togglePlay = useCallback(() => {
    if (playingRef.current) {
      pause();
    } else {
      play();
    }
  }, [pause, play]);

  const step = useCallback(() => {
    const frame = bufferRef.current.shift();
    if (frame) applyFrame(frame, false);
  }, [applyFrame]);

  const setSpeed = useCallback((speed: 1 | 2 | 4) => {
    speedRef.current = speed;
    setSimState((prev) => ({ ...prev, speed }));
    // Drop any backlog beyond one display tick. Without this, lowering the
    // speed after the buffer has filled (up to 60 frames) leaves the console
    // permanently behind the wire, because production and consumption match.
    bufferRef.current = bufferRef.current.slice(-speed);
    const sessionId = sessionIdRef.current;
    if (sessionId !== null) {
      // The local scene already consumes `speed` frames per display tick; only
      // the live stream needs the server-side pace retuned.
      void setSessionPace(sessionId, speed).catch((e) => {
        console.warn('FlywayDQN: could not retune the stream pace', e);
      });
    }
  }, []);

  const setEngine = useCallback(
    (policyName: string) => {
      // The controller changes the physics, so it needs a new session; the
      // selection is kept across later reweights and resets.
      policyRef.current = policyName;
      // Switching to a mitigation controller turns the mitigation on; the
      // unconstrained controller is the "without our project" reference.
      const mitigation = policyName !== 'unconstrained';
      mitigationRef.current = mitigation;
      setMitigationEnabledState(mitigation);
      setSimState((prev) => {
        const label = policiesRef.current.find((p) => p.name === policyName)?.label ?? policyName;
        const entry: DecisionLogEntry = {
          id: `log-${prev.step}-policy-${Date.now()}`,
          timestamp: prev.timeString,
          step: prev.step,
          actionType: 'STATE_OBSERVE',
          description: `Controller → ${label}.`,
          details: `new session requested with policy ${policyName}`,
        };
        return {
          ...prev,
          engine: policyName,
          decisionLogs: [entry, ...prev.decisionLogs].slice(0, MAX_LOG_ENTRIES),
        };
      });
      if (localRunnerRef.current !== null) {
        // Local scene: freeze the old run and start a fresh one so the
        // before/after panel gets a real reference to compare against.
        setHistory(startNewRun);
        startLocal({ policy: policyName, mitigation });
        return;
      }
      void startSession({ policy: policyName });
    },
    [startLocal, startSession],
  );

  /**
   * Flip the mitigation on or off.
   *
   * This is the demo's one control: off is the "without our project" case, on
   * is FlywayDQN. In the local scene it simply retunes the runner; against a
   * live service it re-creates the session with the matching controller.
   */
  const setMitigationEnabled = useCallback(
    (on: boolean) => {
      mitigationRef.current = on;
      setMitigationEnabledState(on);
      const policy = on ? 'analytic-nulling' : 'unconstrained';
      policyRef.current = policy;
      if (localRunnerRef.current !== null) {
        // Freeze the run that is ending so the console shows before | after.
        setHistory(startNewRun);
        startLocal({ policy, mitigation: on });
        return;
      }
      void startSession({ policy });
    },
    [startLocal, startSession],
  );

  const setTradeoffWeight = useCallback(
    (w: number) => {
      weightRef.current = w;
      setSimState((prev) => ({ ...prev, tradeoffWeight: w }));
      if (reweightTimerRef.current !== null) clearTimeout(reweightTimerRef.current);
      reweightTimerRef.current = setTimeout(() => {
        reweightTimerRef.current = null;
        // The reward depends on log10(w), which the server reads at session
        // creation, so a new weight means a new session.
        void startSession({ weight: w });
      }, REWEIGHT_DEBOUNCE_MS);
    },
    [startSession],
  );

  const selectScenario = useCallback(
    async (key: string, seed?: number) => {
      await startSession({ key, seed });
    },
    [startSession],
  );

  const selectFlock = useCallback((flockId: string) => {
    setSimState((prev) => {
      const found = prev.allFlocks.find((f) => f.id === flockId);
      if (!found) return prev;
      return { ...prev, activeFlock: found };
    });
  }, []);

  const resetSim = useCallback(() => {        setSimState((prev) => ({
          ...initialState(prev.scenario, weightRef.current),
          isPlaying: playingRef.current,
          speed: speedRef.current,
          engine: prev.engine,
          beamIndex: prev.beamIndex,
          beamAzimuthDeg: prev.beamAzimuthDeg,
          beamZenithDeg: prev.beamZenithDeg,
          beamGainDbi: prev.beamGainDbi,
          beam: prev.beam,
        }));
    void startSession();
  }, [startSession]);

  // Display loop: consume buffered frames at the selected speed while playing.
  useEffect(() => {
    const timer = setInterval(() => {
      if (!playingRef.current) return;
      const buffer = bufferRef.current;
      if (buffer.length === 0) return;
      let latest: StateFrame | undefined;
      for (let i = 0; i < speedRef.current; i += 1) {
        const next = buffer.shift();
        if (!next) break;
        latest = next;
      }
      if (latest) applyFrame(latest, false);
    }, DISPLAY_TICK_MS);
    return () => clearInterval(timer);
  }, [applyFrame]);

  useEffect(() => {
    if (!autoConnect) return;
    void connect();
    return () => {
      generationRef.current += 1;
      closeSocket();
      bufferRef.current = [];
    };
    // Connect once per mount; use selectScenario/connect to change the request.
  }, [autoConnect, connect, closeSocket]);

  useEffect(
    () => () => {
      if (reweightTimerRef.current !== null) clearTimeout(reweightTimerRef.current);
      if (localTimerRef.current !== null) clearInterval(localTimerRef.current);
      localTimerRef.current = null;
    },
    [],
  );

  const outSessionId = sessionIdRef.current;

  return useMemo(
    () => ({
      currentScreen,
      setCurrentScreen,
      simState,
      connected,
      error,
      connect,
      disconnect,
      play,
      pause,
      togglePlay,
      step,
      setSpeed,
      setTradeoffWeight,
      selectScenario,
      selectFlock,
      setEngine,
      resetSim,
      sessionId: outSessionId,
      scenarios,
      scenarioKey,
      policies,
      history,
      mitigationEnabled,
      setMitigationEnabled,
      usingLocalSource,
    }),
    [
      currentScreen,
      simState,
      connected,
      error,
      connect,
      disconnect,
      play,
      pause,
      togglePlay,
      step,
      setSpeed,
      setTradeoffWeight,
      selectScenario,
      selectFlock,
      setEngine,
      resetSim,
        outSessionId,
      scenarios,
      scenarioKey,
      policies,
      history,
      mitigationEnabled,
      setMitigationEnabled,
      usingLocalSource,
    ],
  );
}
