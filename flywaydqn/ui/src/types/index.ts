// FlywayDQN shared types.
// Synchronized with FastAPI Pydantic schemas and UI components.

export type ScreenId = 'beam-steering' | 'surveillance' | 'dqn-inspector' | 'phased-array';

/**
 * One selectable controller, as reported by `GET /policies`.
 *
 * The engine selector is built from this list rather than from a hardcoded
 * union, so a controller that the server does not offer cannot be shown as a
 * choice.
 */
export interface PolicyOption {
  name: string;
  label: string;
  description: string;
}

export interface UEData {
  id: string;
  name: string;
  azimuthDeg: number;
  distanceM: number;
  sinrDb: number;
  throughputMbps: number;
  blerPercent: number;
  cqi: number;
  isNulledRisk?: boolean;
}

export interface FlockTarget {
  id: string;
  species: string;
  commonName: string;
  birdCount: number;
  azimuthDeg: number;
  distanceM: number;
  altitudeM: number;
  speedMs: number;
  headingDeg: number;
  riskLevel: 'NOMINAL' | 'ALERT' | 'CRITICAL';
  wingbeatHz: number;
  cohesionScore: number;
  thermalC: number;
  estEntrySec: number;
}

export interface DecisionLogEntry {
  id: string;
  timestamp: string;
  step: number;
  actionType: 'NULL_DEEPEN' | 'LOBE_SPLIT' | 'STATE_OBSERVE' | 'POWER_THROTTLE' | 'RESTORE_CARRIER';
  description: string;
  details: string;
}

export interface RadarEventEntry {
  id: string;
  timestamp: string;
  sensorSource: 'Ku-Radar' | 'Acoustic FLIR' | 'LiDAR-3D' | 'Thermal Array';
  eventType: 'CORRIDOR_ENTRY' | 'SPECIES_CONFIRM' | 'TRAJECTORY_UPD' | 'RF_PROXIMITY_WARN';
  eventBadgeClass: string;
  details: string;
  actionTriggered: string;
}

export interface SimState {
  step: number;
  timeSec: number;
  timeString: string;
  isPlaying: boolean;
  speed: 1 | 2 | 4;
  scenario: string;
  /** Registered name of the controller that produced the current frame. */
  engine: string;
  tradeoffWeight: number; // w = beta / alpha (0 to 5.0)
  
  // RF Metrics
  peakExposureWm2: number;
  /** The scenario's own exposure threshold, straight off the wire. */
  exposureLimitWm2: number;
  timeOverLimitSec: number;
  /** Null depth actually achieved this step, in dB; 0 while no null is on air. */
  nullDepthDb: number;
  /** True while the frame reports a null on the air (never inferred from depth). */
  nullActive: boolean;
  cellThroughputGbps: number;
  /** Sum rate as a share of the scenario's theoretical max (the reward's T̂). */
  throughputFraction: number;
  /** Same-seed unconstrained baseline is WP-F; empty until that is wired. */
  baselineThroughputGbps: number | typeof Number.NaN;
  meanSinrDb: number;
  meanBlerPercent: number;
  txRfPowerW: number;
  inferenceLatencyMs: number;

  // Serving beam from the live frame
  beamIndex: number;
  beamAzimuthDeg: number;
  beamZenithDeg: number;
  beamGainDbi: number;
  beam: {
    index: number;
    azimuthDeg: number;
    zenithDeg: number;
    gainDbi: number;
  } | null;

  // Reward terms
  rewardThroughput: number;
  rewardExposure: number;
  rewardPower: number;
  rewardNet: number;

  // Active Flock
  activeFlock: FlockTarget;
  allFlocks: FlockTarget[];  // Users
  users: UEData[];

  // Logs
  decisionLogs: DecisionLogEntry[];
  radarEvents: RadarEventEntry[];

  /**
   * `frame.explanation` of the most recent displayed frame: the runner's
   * full-text account of the decision (including, for the DQN controller, the
   * argmax action and the Q-value the network produced). Null until the first
   * frame arrives; never fabricated.
   */
  explanation: string | null;
}

/**
 * One simulation step emitted over the WebSocket stream.
 */
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
  explanation?: string;
  weight: number;
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
