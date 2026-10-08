// FlywayDQN shared types.
// Synchronized with FastAPI Pydantic schemas and UI components.

export type ScreenId = 'beam-steering' | 'surveillance' | 'dqn-inspector' | 'phased-array';

export type SteeringEngine = 'FlywayDQN (Active Null)' | 'Baseline SON' | 'Sector Shutdown' | 'Static Back-off';

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
  engine: SteeringEngine;
  tradeoffWeight: number; // w = beta / alpha (0 to 5.0)
  
  // RF Metrics
  peakExposureWm2: number;
  exposureLimitWm2: number;
  timeOverLimitSec: number;
  nullDepthDb: number;
  cellThroughputGbps: number;
  baselineThroughputGbps: number;
  meanSinrDb: number;
  meanBlerPercent: number;
  txRfPowerW: number;
  inferenceLatencyMs: number;
  
  // Reward terms
  rewardThroughput: number;
  rewardExposure: number;
  rewardPower: number;
  rewardNet: number;

  // Active Flock
  activeFlock: FlockTarget;
  allFlocks: FlockTarget[];
  
  // Users
  users: UEData[];
  
  // Logs
  decisionLogs: DecisionLogEntry[];
  radarEvents: RadarEventEntry[];
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
  exposurePenalty: number;
  throughputTerm: number;
  powerTerm: number;
  reward: number;
}

export interface ScenarioRequest {
  key: string;
  seed: number;
  weight: number;
}

export interface SessionCreated {
  id: string;
  frame: StateFrame;
  streamPath: string;
}
