const DEFAULT_API_BASE = 'http://localhost:8000';

export function getApiBase(): string {
  const envBase = (import.meta.env?.VITE_API_BASE as string | undefined) ?? undefined;
  if (envBase && envBase.trim().length > 0) {
    return envBase.trim().replace(/\/$/, '');
  }
  return DEFAULT_API_BASE;
}

export function buildWsUrl(path: string): string {
  const base = getApiBase();
  const wsBase = base.startsWith('https://')
    ? base.replace(/^https:\/\//, 'wss://')
    : base.replace(/^http:\/\//, 'ws://');
  return `${wsBase}${path.startsWith('/') ? path : `/${path}`}`;
}

export interface ScenarioInfo {
  key: string;
  label: string;
  description: string;
}

/** One selectable controller, as reported by `GET /policies`. */
export interface PolicyInfo {
  name: string;
  label: string;
  description: string;
}

export interface MapPoint {
  x: number;
  y: number;
}

export interface BeamState {
  index: number;
  azimuthDeg: number;
  zenithDeg: number;
  gainDbi: number;
}

export interface FlockState {
  centroid: MapPoint;
  headingDeg: number;
  speedMs: number;
  altitudeMinM: number;
  altitudeMaxM: number;
  extentM: number;
  nCels: number;
}

export interface UserState {
  position: MapPoint;
  bearingDeg: number;
  rangeM: number;
  sinr: number;
  throughputMbps: number;
}

export interface StepMetrics {
  sumThroughputMbps: number;
  p95DensityWm2: number;
  maxDensityWm2: number;
  fracOver: number;
  doseWsm2: number;
  /** The scenario's exposure threshold, i.e. the penalty's denominator. */
  sThrWm2: number;
  exposurePenalty: number;
  throughputTerm: number;
  powerTerm: number;
  reward: number;
}

export interface StateFrame {
  step: number;
  timeS: number;
  dtS: number;
  sector: MapPoint;
  beam: BeamState;
  powerMw: number;
  nullActive: boolean;
  /** Measured exposure reduction against the same scenario's un-nulled beam. */
  nullDepthDb: number;
  /** Server-measured policy-decision latency for this step, in ms. */
  inferenceLatencyMs: number;
  /** Controller that produced this frame. */
  policyName: string;
  flock: FlockState;
  users: UserState[];
  metrics: StepMetrics;
  explanation?: string | null;
  weight: number;
}

export interface ScenarioRequest {
  key: string;
  seed: number;
  weight: number;
  /** Registered policy name; omitted means the scenario file's own policy. */
  policy?: string | null;
}

export interface SessionCreated {
  id: string;
  frame: StateFrame;
  streamPath: string;
}

export async function fetchScenarios(): Promise<ScenarioInfo[]> {
  const res = await fetch(`${getApiBase()}/scenarios`, {
    headers: { Accept: 'application/json' },
  });
  if (!res.ok) {
    throw new Error(`Failed to fetch scenarios: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as ScenarioInfo[];
}

export async function fetchPolicies(): Promise<PolicyInfo[]> {
  const res = await fetch(`${getApiBase()}/policies`, {
    headers: { Accept: 'application/json' },
  });
  if (!res.ok) {
    throw new Error(`Failed to fetch policies: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as PolicyInfo[];
}

/** Run metadata of the checkpoint behind the `dqn` controller (WP-E). */
export interface DQNMetaInfo {
  runName: string;
  seed: number;
  totalSteps: number;
  observationSize: number;
  nActions: number;
  hyperparameters: Record<string, unknown>;
  trainMetrics: Record<string, number>;
}

/**
 * Read the checkpoint's `meta.json` through `GET /policies/dqn/meta`.
 * Rejects (rather than substituting defaults) when no checkpoint is present.
 */
export async function fetchDQNMeta(): Promise<DQNMetaInfo> {
  const res = await fetch(`${getApiBase()}/policies/dqn/meta`, {
    headers: { Accept: 'application/json' },
  });
  if (!res.ok) {
    throw new Error(`Failed to fetch DQN metadata: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as DQNMetaInfo;
}

export async function createSession(req: ScenarioRequest): Promise<SessionCreated> {
  const res = await fetch(`${getApiBase()}/sessions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`Failed to create session: ${res.status} ${res.statusText} ${text}`.trim());
  }
  return (await res.json()) as SessionCreated;
}

/**
 * Retune a live session's wall-clock pacing.
 *
 * Presentation only: the service changes how often it pushes a frame, never
 * the simulated timestep a frame reports. Returns void (204 No Content).
 */
export async function setSessionPace(sessionId: string, pace: number): Promise<void> {
  const res = await fetch(`${getApiBase()}/sessions/${encodeURIComponent(sessionId)}/pace`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
    body: JSON.stringify({ pace }),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`Failed to set session pace: ${res.status} ${res.statusText} ${text}`.trim());
  }
}
