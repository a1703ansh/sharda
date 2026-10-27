import React from 'react';
import { Wind, Feather, Cpu, ShieldAlert, Wifi, Sliders, Zap } from 'lucide-react';
import { SimState, PolicyOption, UEData, DecisionLogEntry } from '../../types';
import { formatClock, summarize, type SessionHistory } from '../../lib/history';

export interface ScenarioOption {
  key: string;
  label: string;
}

interface BeamSteeringConsoleProps {
  simState: SimState;
  /** Selectable controllers reported by `GET /policies`. */
  policies: PolicyOption[];
  onSetEngine: (policyName: string) => void;
  onSetTradeoffWeight: (w: number) => void;
  /** Scenario list from `GET /scenarios`; empty until the first connect. */
  scenarios?: ScenarioOption[];
  /** Key of the live session's scenario (selects the dropdown by key). */
  scenarioKey?: string;
  onSelectScenario?: (key: string) => void;
  /**
   * Per-step history of the current run plus the frozen summary of the run
   * before it (WP-F). The timeline, the stat tiles and the before/after
   * panel are computed from it — never from the six-entry decision log.
   */
  history: SessionHistory;
  /** Whether the FlywayDQN mitigation is engaged — the demo's master switch. */
  mitigationEnabled?: boolean;
  /** Flip the mitigation on/off (the before/after control the judges use). */
  onSetMitigation?: (on: boolean) => void;
}

// Helpers for scalar display values used across the console.
function nanOr(value: number, fallback: string): string {
  return Number.isFinite(value) ? String(value) : fallback;
}

/** Same as `nanOr` but pinned to a sensible number of decimals. */
function nanFixed(value: number, decimals: number, fallback = '—'): string {
  return Number.isFinite(value) ? value.toFixed(decimals) : fallback;
}

export const BeamSteeringConsole: React.FC<BeamSteeringConsoleProps> = ({
  simState,
  policies,
  onSetEngine,
  onSetTradeoffWeight,
  scenarios = [],
  scenarioKey,
  onSelectScenario,
  history,
  mitigationEnabled = true,
  onSetMitigation,
}) => {
  const scenarioOptions: ScenarioOption[] =
    scenarios.length > 0 ? scenarios : [{ key: simState.scenario, label: simState.scenario }];
  const currentScenarioKey =
    scenarioKey ??
    scenarioOptions.find((option) => option.label === simState.scenario)?.key ??
    scenarioOptions[0]?.key ??
    'scenario_001';
  // What is on the air comes from the measured null depth, not from which
  // button happens to look selected.
  const nullOnAir = simState.nullDepthDb !== 0;
  const overThreshold =
    Number.isFinite(simState.peakExposureWm2) && Number.isFinite(simState.exposureLimitWm2)
      ? simState.peakExposureWm2 > simState.exposureLimitWm2
      : false;
  // The scenario's own threshold; '—' until the first frame reports it.
  const thresholdLabel =
    Number.isFinite(simState.exposureLimitWm2) && simState.exposureLimitWm2 > 0
      ? `${simState.exposureLimitWm2.toFixed(3)} W/m²`
      : '—';

  // Timeline + stat tiles are functions of every frame this run has
  // displayed (WP-F): the service streamed them, the console rendered them,
  // history.ts stored them. `showHistory` turns true with the first frame —
  // the old `frameCount >= 60` gate could never open, because the decision
  // log it measured is capped at six entries.
  const summary = summarize(history.current);
  const previousRun = history.previous;
  const showHistory = summary !== null;
  const sessionDecisionRate =
    summary && summary.lastTimeS > 0 ? summary.frames / summary.lastTimeS : Number.NaN;
  // Serving-beam azimuth span observed over the run (every stored frame).
  const sessionMinAzimuth = summary?.azimuthMinDeg ?? Number.NaN;
  const sessionMaxAzimuth = summary?.azimuthMaxDeg ?? Number.NaN;
  // Label for a registered controller name; the name itself comes off the
  // wire (frame.policyName), so the banner shows what actually ran.
  const labelForPolicy = (name: string): string =>
    policies.find((policy) => policy.name === name)?.label ?? name;

  // The before/after pair, identified by the controller that produced each run
  // rather than by which one happened to be displayed first. `unconstrained` is
  // the "without our project" reference; anything else engaged the mitigation.
  const isReferencePolicy = (name: string): boolean => name === 'unconstrained';
  const referencePeak =
    previousRun != null && isReferencePolicy(previousRun.policyName)
      ? {
          peak: previousRun.peakExposureWm2,
          depth: previousRun.nullDepthDb,
          policyName: previousRun.policyName,
        }
      : summary != null && isReferencePolicy(summary.policyName)
        ? {
            peak: summary.peakExposureWm2,
            depth: summary.nullDepthDb,
            policyName: summary.policyName,
          }
        : null;
  const mitigatedPeak =
    previousRun != null && !isReferencePolicy(previousRun.policyName)
      ? {
          peak: previousRun.peakExposureWm2,
          depth: previousRun.nullDepthDb,
          policyName: previousRun.policyName,
        }
      : summary != null && !isReferencePolicy(summary.policyName)
        ? {
            peak: summary.peakExposureWm2,
            depth: summary.nullDepthDb,
            policyName: summary.policyName,
          }
        : null;
  const mitigatedLabel =
    mitigatedPeak != null ? labelForPolicy(mitigatedPeak.policyName) : 'the controller';
  // Headline change, stated in whichever direction the numbers actually moved.
  const exposureDelta =
    referencePeak != null &&
    mitigatedPeak != null &&
    Number.isFinite(referencePeak.peak) &&
    Number.isFinite(mitigatedPeak.peak) &&
    referencePeak.peak > 0
      ? (() => {
          const dropWm2 = referencePeak.peak - mitigatedPeak.peak;
          const pct = (dropWm2 / referencePeak.peak) * 100;
          const shown = Math.abs(pct) >= 99.5 ? `${pct.toFixed(1)}` : `${pct.toFixed(0)}`;
          return dropWm2 >= 0
            ? `change ` +
                `−${dropWm2.toFixed(4)} W/m² (${shown}% lower with ${mitigatedLabel})`
            : `change ` +
                `+${Math.abs(dropWm2).toFixed(4)} W/m² (${Math.abs(pct).toFixed(0)}% higher with ${mitigatedLabel})`;
        })()
      : null;

  // Reward terms for the current step.
  const frameRewardThroughput = simState.rewardThroughput;
  const frameRewardExposure = simState.rewardExposure;
  const frameRewardPower = simState.rewardPower;
  const frameRewardNet = simState.rewardNet;

  // Chart geometry: simple linear mapping once enough history exists.
  const chartWidth = 600;
  const chartHeight = 80;
  const paddingX = 8;
  const paddingRight = 12;
  const chartXMax = chartWidth - paddingRight;
  const chartXMin = paddingX;
  const chartSpan = chartXMax - chartXMin;
  // The curve is this run's real per-frame exposure series (last 240 frames),
  // not the current value repeated once per stored log entry.
  const sessionExposureHistory = history.current
    .slice(-240)
    .map((sample) => sample.peakExposureWm2);
  const maxHistory = sessionExposureHistory.length;
  const exposureMax = Math.max(1e-6, ...sessionExposureHistory.filter((v) => Number.isFinite(v)));

  function timelineX(index: number): number {
    return chartXMin + (index / Math.max(1, maxHistory - 1)) * chartSpan;
  }

  function exposureY(value: number): number {
    if (!Number.isFinite(value) || value <= 0) return chartHeight - 1;
    const span = Math.max(value, exposureMax);
    const safeSpan = Math.max(span, 1e-6);
    return chartHeight - 4 - ((value / Math.max(span, 1e-6)) * (chartHeight - 8));
  }

  // Radar placement. Frame bearings are atan2(y, x): 0° = east, 90° = north,
  // and the DFT codebook spans ±60° about boresight 0°. A full circle keeps
  // the flock visible in the 150° front/back mirror corridor (scenario_003)
  // as well as the sector. 0.317 px/m puts the 600 m ring on the 190 px edge.
  const RADAR_RANGE_M = 600;
  const RADAR_RADIUS_PX = 190;
  const radarRadiusPx = (rangeM: number): number =>
    (Math.min(Math.max(rangeM, 0), RADAR_RANGE_M) / RADAR_RANGE_M) * RADAR_RADIUS_PX;
  const radarPoint = (azimuthDeg: number, rangeM: number): { x: number; y: number } => {
    const radians = (azimuthDeg * Math.PI) / 180;
    const radius = radarRadiusPx(rangeM);
    return { x: Math.cos(radians) * radius, y: -Math.sin(radians) * radius };
  };
  const flockAzimuthDeg = simState.activeFlock.azimuthDeg;
  const flockRangeM = simState.activeFlock.distanceM;
  const flockPoint = radarPoint(flockAzimuthDeg, flockRangeM);
  // Bearings are normalised to (-180, 180]; show them as 0–360 on the plot.
  const flockAzimuthLabel = ((flockAzimuthDeg % 360) + 360) % 360;
  const beamPoint = radarPoint(simState.beamAzimuthDeg, RADAR_RANGE_M);
  // The flock silhouette shrinks as the flock closes in on the antenna.
  const flockScale = Math.max(0.45, Math.min(1.15, flockRangeM / 480));

  return (
    <div className="content-viewport">
      {/* Subheader Banner */}
      <div className="sub-header-strip">
        <div className="sub-header-left">
          <div className="sub-header-title">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 inline-block animate-pulse" />
            <span>Sector A Beamform Matrix</span>
          </div>
          <span className="sub-header-meta">
            LIVE — 3.5 GHz n78 massive-MIMO (64T64R) // 8×8 UPA, 16-beam DFT codebook
          </span>
        </div>

        <div className="sub-header-chips">
          <div className="info-chip">
            <Wind size={12} className="text-teal-600" />
            <span>WINDS 6.2 kt NE</span>
          </div>
          <div className="info-chip active-teal">
            <Feather size={12} className="text-teal-700" />
            <span>FLOCK: {simState.activeFlock.birdCount} BIRDS DETECTED</span>
          </div>
          <div className="info-chip">
            <Cpu size={12} className="text-blue-600" />
            <span>DECISION LATENCY: {simState.inferenceLatencyMs.toFixed(2)} ms</span>
          </div>

          {/* The demo's one control: mitigation off is "without our project",
              mitigation on is FlywayDQN. Flipping it re-runs the scene and the
              before/after numbers below change. */}
          <button
            type="button"
            onClick={() => onSetMitigation?.(!mitigationEnabled)}
            title={
              mitigationEnabled
                ? 'Mitigation is ON — click to show the un-mitigated exposure'
                : 'Mitigation is OFF — click to engage the active controller'
            }
            className={`info-chip ${mitigationEnabled ? 'active-teal' : ''}`}
            style={{ fontWeight: 700, cursor: 'pointer' }}
          >
            <span
              className="inline-block rounded-full"
              style={{
                width: 8,
                height: 8,
                backgroundColor: mitigationEnabled ? '#0d9488' : '#ef4444',
              }}
            />
            <span>MITIGATION: {mitigationEnabled ? 'ON' : 'OFF'}</span>
          </button>
        </div>
      </div>

      {/* Before/after strip: the headline number for the judges.
          The two sides are identified by the controller that produced them
          (frame.policyName), so the panel reads the same whichever direction
          the mitigation toggle was flipped. */}
      {referencePeak != null && mitigatedPeak != null && (
        <div
          className="mx-4 mt-3 flex flex-wrap items-center gap-x-5 gap-y-1 rounded border border-teal-300 bg-teal-50 px-3 py-2 font-mono text-[11px] text-teal-900"
          role="status"
        >
          <span className="font-bold uppercase">Before / after</span>
          <span>
            <span className="text-gray-600">{labelForPolicy(referencePeak.policyName)} </span>
            <strong style={{ color: '#b91c1c' }}>
              {referencePeak.peak.toFixed(4)} W/m²
            </strong>
            <span className="text-gray-500"> (no null)</span>
          </span>
          <span>→</span>
          <span>
            <span className="text-gray-600">with {mitigatedLabel} </span>
            <strong style={{ color: '#0d9488' }}>
              {mitigatedPeak.peak.toFixed(4)} W/m²
            </strong>
            <span className="text-gray-500">
              {mitigatedPeak.depth > 0 ? ` (null ${mitigatedPeak.depth.toFixed(1)} dB)` : ''}
            </span>
          </span>
          {exposureDelta != null && <span className="font-bold">{exposureDelta}</span>}
          <span className="text-gray-600">limit {thresholdLabel}</span>
        </div>
      )}

      {/* Controller banner: what the wire says actually ran. Driven by
          frame.policyName only — button highlights and this strip cannot
          disagree, because both read the same field off the frame. */}
      {simState.engine !== '' && (
        <div
          className="mx-4 mt-3 flex flex-wrap items-center gap-2 rounded border border-teal-300 bg-teal-50 px-3 py-2 font-mono text-[11px] text-teal-900"
          role="status"
        >
          <span className="h-2 w-2 rounded-full bg-teal-600 inline-block animate-pulse" />
          <span>
            ACTIVE CONTROLLER — frame.policyName = <strong>{simState.engine}</strong>{' '}
            → {labelForPolicy(simState.engine)}
          </span>
          {previousRun && (
            <span className="text-teal-700">
              · before: {labelForPolicy(previousRun.policyName)} — peak{' '}
              {Number.isFinite(previousRun.peakExposureWm2)
                ? previousRun.peakExposureWm2.toFixed(3)
                : '—'}{' '}
              W/m², null depth{' '}
              {Number.isFinite(previousRun.nullDepthDb)
                ? previousRun.nullDepthDb.toFixed(2)
                : '—'}{' '}
              dB, TX{' '}
              {Number.isFinite(previousRun.powerMw) ? (previousRun.powerMw / 1000).toFixed(1) : '—'}{' '}
              W
            </span>
          )}
        </div>
      )}

      {/* Main Grid: Polar Canvas on Left, Metrics on Right */}
      <div className="beam-screen-grid">
        {/* Left Card: Beamforming Matrix & Polar Canvas */}
        <div className="tech-card steering-canvas-card">
          <div className="radar-header-controls">
            <div className="flex items-center gap-3 flex-wrap">
              <span className="font-mono text-xs text-gray-500 font-semibold uppercase">SCENARIO:</span>
              <select
                className="scenario-select"
                value={currentScenarioKey}
                onChange={(e) => onSelectScenario?.(e.target.value)}
                title={`${scenarioOptions.length} scenario(s) from GET /scenarios`}
              >
                {scenarioOptions.map((option) => (
                  <option key={option.key} value={option.key}>
                    {option.label} ({option.key})
                  </option>
                ))}
              </select>
              <span className="font-mono text-[10px] text-gray-400">
                ({scenarioOptions.length} available)
              </span>
              <span className="font-mono text-xs text-gray-500 font-semibold uppercase">STEERING ENGINE:</span>
              <div className="engine-toggle-group">
                {policies.length === 0 && (
                  <span className="font-mono text-xs text-gray-400">awaiting /policies</span>
                )}
                {policies.map((policy: PolicyOption) => {
                  const active = simState.engine === policy.name;
                  return (
                    <button
                      key={policy.name}
                      title={policy.description}
                      className={`engine-btn ${active ? 'active-dqn' : ''}`}
                      onClick={() => onSetEngine(policy.name)}
                      disabled={active}
                    >
                      <span className="w-2 h-2 rounded-full bg-emerald-400 inline-block" />
                      <span>{policy.label}</span>
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="radar-legend">
              <div className="legend-item">
                <span className="legend-square" style={{ backgroundColor: '#0f766e' }} />
                <span>UE Main Beams</span>
              </div>
              <div className="legend-item">
                <span className="legend-square" style={{ backgroundColor: '#99f6e4' }} />
                <span>Ecological Null Corridor</span>
              </div>
              <div className="legend-item">
                <span className="legend-square" style={{ backgroundColor: '#fed7aa' }} />
                <span>Buffer Zone</span>
              </div>
            </div>
          </div>

          {/* Polar Radar Visualizer with Live Radar Sweep */}
          <div className="polar-radar-container">
            <svg viewBox="0 0 700 440" className="polar-radar-svg">
              <defs>
                {/* Gradient for UE beams */}
                <linearGradient id="beamGrad" x1="0%" y1="100%" x2="50%" y2="0%">
                  <stop offset="0%" stopColor="#0d9488" stopOpacity="0.45" />
                  <stop offset="70%" stopColor="#2dd4bf" stopOpacity="0.18" />
                  <stop offset="100%" stopColor="#2dd4bf" stopOpacity="0.0" />
                </linearGradient>

                {/* Gradient for Radar Sweep Line */}
                <linearGradient id="sweepGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="#0d9488" stopOpacity="0.6" />
                  <stop offset="70%" stopColor="#2dd4bf" stopOpacity="0.15" />
                  <stop offset="100%" stopColor="#2dd4bf" stopOpacity="0.0" />
                </linearGradient>

                {/* Gradient for Null Corridor */}
                <linearGradient id="nullGrad" x1="0%" y1="100%" x2="0%" y2="0%">
                  <stop offset="0%" stopColor="#99f6e4" stopOpacity="0.4" />
                  <stop offset="100%" stopColor="#ccfbf1" stopOpacity="0.08" />
                </linearGradient>

                {/* Pattern for Buffer zone */}
                <pattern id="diagonalHatch" width="8" height="8" patternTransform="rotate(45 0 0)" patternUnits="userSpaceOnUse">
                  <line x1="0" y1="0" x2="0" y2="8" stroke="#fdba74" strokeWidth="1.5" strokeOpacity="0.4" />
                </pattern>
              </defs>

              {/* Polar grid: full circle, 0° east / 90° north. Frame bearings
                  come from atan2(y, x), matching these labels. */}
              <g transform="translate(365, 220)">
                {/* Range rings: 150 / 300 / 450 / 600 m at 0.317 px/m. */}
                <circle r="47.5" fill="none" stroke="#d1fae5" strokeWidth="1" strokeDasharray="3 3" />
                <circle r="95" fill="none" stroke="#d1fae5" strokeWidth="1" strokeDasharray="3 3" />
                <circle r="142.5" fill="none" stroke="#a7f3d0" strokeWidth="1.2" />
                <circle r="190" fill="none" stroke="#cbd5e1" strokeWidth="1" />

                {/* Distance labels along the boresight ray. */}
                <text x="52" y="-4" fill="#94a3b8" fontSize="10" fontFamily="JetBrains Mono">150m</text>
                <text x="100" y="-4" fill="#94a3b8" fontSize="10" fontFamily="JetBrains Mono">300m</text>
                <text x="147" y="-4" fill="#94a3b8" fontSize="10" fontFamily="JetBrains Mono">450m</text>
                <text x="194" y="-4" fill="#94a3b8" fontSize="10" fontFamily="JetBrains Mono">600m</text>

                {/* Cardinal rays. */}
                <line x1="0" y1="0" x2="0" y2="-190" stroke="#e2e8f0" strokeWidth="1" />
                <text x="5" y="-181" fill="#94a3b8" fontSize="10" fontFamily="JetBrains Mono">090° N</text>

                <line x1="0" y1="0" x2="0" y2="190" stroke="#e2e8f0" strokeWidth="1" />
                <text x="5" y="187" fill="#94a3b8" fontSize="10" fontFamily="JetBrains Mono">270° S</text>

                <line x1="0" y1="0" x2="-190" y2="0" stroke="#e2e8f0" strokeWidth="1" />
                <text x="-185" y="14" fill="#94a3b8" fontSize="10" fontFamily="JetBrains Mono">180° W</text>

                <line x1="0" y1="0" x2="190" y2="0" stroke="#0f766e" strokeWidth="1.2" />
                <text x="194" y="14" fill="#0f766e" fontSize="10" fontFamily="JetBrains Mono" fontWeight="600">000° E · boresight</text>

                {/* Codebook cover: ±60° about east. */}
                <line x1="0" y1="0" x2="95" y2="-164.5" stroke="#0f766e" strokeWidth="1.2" />
                <text x="99" y="-158" fill="#0f766e" fontSize="10" fontFamily="JetBrains Mono" fontWeight="600">060°</text>

                <line x1="0" y1="0" x2="95" y2="164.5" stroke="#0f766e" strokeWidth="1.2" />
                <text x="99" y="168" fill="#0f766e" fontSize="10" fontFamily="JetBrains Mono" fontWeight="600">300°</text>

                {/* Front/back mirror corridor (scenario_003). */}
                <line x1="0" y1="0" x2="-164.5" y2="-95" stroke="#d97706" strokeWidth="1" strokeDasharray="3 3" />
                <text x="-122" y="-72" fill="#b45309" fontSize="9" fontFamily="JetBrains Mono" fontWeight="600">150° mirror</text>

                {/* DFT coverage wedge: the 16 beams span 120° about east. */}
                <path
                  d="M 0,0 L 95,164.5 A 190 190 0 0 0 95,-164.5 Z"
                  fill="url(#beamGrad)"
                />

                {/* Continuous radar sweep. */}
                <g className="sector-radar-sweep">
                  <path
                    d="M 0,0 L 190,0 A 190 190 0 0 0 183,-51 Z"
                    fill="url(#sweepGrad)"
                  />
                  <line x1="0" y1="0" x2="190" y2="0" stroke="#10b981" strokeWidth="2" opacity="0.8" />
                </g>

                {/* Serving beam: live codebook azimuth, 0° = boresight. */}
                <line
                  x1="0"
                  y1="0"
                  x2={beamPoint.x.toFixed(1)}
                  y2={beamPoint.y.toFixed(1)}
                  stroke="#0d9488"
                  strokeWidth="2.2"
                  strokeDasharray="8 4"
                  opacity="0.75"
                />
                <circle cx={beamPoint.x.toFixed(1)} cy={beamPoint.y.toFixed(1)} r="3.5" fill="#0d9488" />

                {/* Ecological null corridor: drawn toward the flock bearing and
                    only while a null is measured on air. The wedge is built
                    pointing north; SVG rotation is clockwise from north, while
                    frame azimuths are counter-clockwise from east, so the
                    rotation is 90° - bearing. It therefore tracks the flock. */}
                {nullOnAir && (
                  <g
                    className="null-shimmer"
                    transform={`rotate(${(90 - flockAzimuthDeg).toFixed(1)})`}
                  >
                    {/* Buffer zone background */}
                    <path
                      d="M 0,0 L -31,-183 A 185 185 0 0 1 31,-183 Z"
                      fill="url(#diagonalHatch)"
                    />
                    {/* Core deep null wedge */}
                    <path
                      d="M 0,0 L -16,-189 A 190 190 0 0 1 16,-189 Z"
                      fill="url(#nullGrad)"
                      stroke="#0d9488"
                      strokeWidth="1.5"
                      strokeDasharray="4 2"
                    />
                  </g>
                )}

                {/* Flock contour at the live bearing/range. The outer group
                    carries the data transform; the inner class only adds a
                    small hover, so CSS cannot override the live position. */}
                <g
                  transform={`translate(${flockPoint.x.toFixed(1)}, ${flockPoint.y.toFixed(1)}) scale(${flockScale.toFixed(2)})`}
                >
                  <g className="flock-hover">
                  {/* Radar Sonar Echo Ripples */}
                  <circle cx="0" cy="0" r="10" fill="none" stroke={overThreshold ? '#ef4444' : '#0d9488'} className="target-ping-1" />
                  <circle cx="0" cy="0" r="10" fill="none" stroke={overThreshold ? '#f87171' : '#14b8a6'} className="target-ping-2" />

                  {/* Outer Protection Bubble */}
                  <ellipse
                    cx="0"
                    cy="0"
                    rx="52"
                    ry="28"
                    transform="rotate(-40)"
                    fill={overThreshold ? 'rgba(239, 68, 68, 0.18)' : 'rgba(13, 148, 136, 0.12)'}
                    stroke={overThreshold ? '#ef4444' : '#0d9488'}
                    strokeWidth="1.5"
                    strokeDasharray={overThreshold ? 'none' : '4 3'}
                  />

                  {/* Bird icons flock representation */}
                  <g fill={overThreshold ? '#b91c1c' : '#0f766e'}>
                    <path d="M -15,-5 Q -10,-10 -5,-5 Q 0,-10 5,-5" stroke={overThreshold ? '#b91c1c' : '#0f766e'} strokeWidth="1.5" fill="none" />
                    <path d="M -5,5 Q 0,0 5,5 Q 10,0 15,5" stroke={overThreshold ? '#b91c1c' : '#0f766e'} strokeWidth="1.5" fill="none" />
                    <path d="M -22,8 Q -18,4 -14,8 Q -10,4 -6,8" stroke={overThreshold ? '#b91c1c' : '#0f766e'} strokeWidth="1.2" fill="none" />
                    <circle cx="0" cy="0" r="3" fill={overThreshold ? '#ef4444' : '#0d9488'} />
                  </g>

                  {/* Null Depth Tag */}
                  <rect
                    x="-65"
                    y="-40"
                    width="130"
                    height="18"
                    rx="4"
                    fill="#ffffff"
                    stroke={overThreshold ? '#ef4444' : '#0d9488'}
                    strokeWidth="1"
                  />
                  <text
                    x="0"
                    y="-28"
                    textAnchor="middle"
                    fill={nullOnAir ? '#0f766e' : overThreshold ? '#b91c1c' : '#475569'}
                    fontSize="9"
                    fontFamily="JetBrains Mono"
                    fontWeight="700"
                  >
                    {nullOnAir
                      ? `NULL: ${simState.nullDepthDb.toFixed(1)} dB @ ${Math.round(
                          flockAzimuthLabel,
                        )}° AZ`
                      : overThreshold
                        ? `OVER THRESHOLD (${simState.peakExposureWm2.toFixed(3)} W/m²)`
                        : 'NULL OFF'}
                  </text>
                  </g>
                </g>

                {/* UE Targets with Animated Propagating Beam Rays */}
                {(simState.users ?? []).map((ue: UEData) => {
                  const point = radarPoint(ue.azimuthDeg, ue.distanceM);

                  return (
                    <g key={ue.id} transform={`translate(${point.x.toFixed(1)}, ${point.y.toFixed(1)})`}>
                      {/* Animated carrier wave ray toward UE */}
                      <line
                        x1={(-point.x).toFixed(1)}
                        y1={(-point.y).toFixed(1)}
                        x2="0"
                        y2="0"
                        stroke="#0d9488"
                        strokeWidth="1.8"
                        className="rf-wave-ray"
                      />
                      <circle cx="0" cy="0" r="4.5" fill="#0d9488" stroke="#ffffff" strokeWidth="1.5" />
                      <text
                        x="7"
                        y="3"
                        fill="#334155"
                        fontSize="8.5"
                        fontFamily="JetBrains Mono"
                        fontWeight="600"
                      >
                        {ue.id}
                      </text>
                    </g>
                  );
                })}

                {/* gNodeB Base Station Origin with Concentric Ping */}
                <circle cx="0" cy="0" r="14" fill="none" stroke="#0d9488" strokeWidth="1.5" className="target-ping-1" />
                <circle cx="0" cy="0" r="8" fill="#0f766e" stroke="#ffffff" strokeWidth="2" />
                <text x="-28" y="24" fill="#0f766e" fontSize="10" fontFamily="JetBrains Mono" fontWeight="700">
                  gNodeB-042
                </text>
              </g>
            </svg>

            {/* Live track overlay: the same numbers the markers are drawn from. */}
            <div className="absolute top-3 left-3 bg-white/95 backdrop-blur-sm border border-emerald-200 rounded-md p-2.5 font-mono text-xs shadow-sm">
              <div className="text-gray-500 text-[10px] uppercase font-bold">LIVE TRACK</div>
              <div className="text-teal-700 text-sm font-bold">
                {nullOnAir ? `Null ${simState.nullDepthDb.toFixed(2)} dB` : 'Null off'}
              </div>
              <div className="text-gray-600 text-[10px]">
                Flock {flockAzimuthLabel.toFixed(0)}° / {flockRangeM.toFixed(0)} m
              </div>
              <div className="text-gray-600 text-[10px]">
                Beam {simState.beamIndex} @ {simState.beamAzimuthDeg.toFixed(1)}°
              </div>
              <div className="text-gray-500 text-[10px]">
                Threshold {thresholdLabel} · {overThreshold ? 'exceeded' : 'respected'}
              </div>
            </div>
          </div>

          {/* Canvas Bottom Legend & Metrics */}
          <div className="radar-bottom-bar">
            <div>
              <span>POWER DENSITY SPECTRUM (EIRP): </span>
              <span className="inline-block w-16 h-2 bg-gradient-to-r from-teal-500 via-amber-400 to-red-500 rounded mx-2 align-middle" />
              <span className="text-gray-400">&lt;-30dB | -20dB(Null) | 0dB | &gt;+10dB</span>
            </div>
            <div>
              <span>DECISION LATENCY: <strong>{simState.inferenceLatencyMs.toFixed(2)} ms</strong></span>
              <span className="mx-2 text-gray-300">|</span>
              <span>CONTROLLER: <strong>{simState.engine || '—'}</strong></span>
            </div>
          </div>
        </div>

        {/* Right Column: Key Telemetry Cards */}
        <div className="metrics-column">
          {/* Card 1: Peak Flock Exposure */}
          <div className="tech-card stat-big-card">
            <div className="stat-header">
              <span className="flex items-center gap-1.5 text-teal-700">
                <ShieldAlert size={14} />
                <span>PEAK FLOCK EXPOSURE</span>
              </span>
              <span className={`badge-pill ${overThreshold ? 'coral' : 'teal'}`}>
                {overThreshold ? 'THRESHOLD EXCEEDED' : 'BELOW THRESHOLD'}
              </span>
            </div>

            <div className="stat-value-row">
              <div>
                <span className="stat-big-val">{nanOr(simState.peakExposureWm2, '—')}</span>
                <span className="stat-unit">W/m²</span>
              </div>
              <div className="text-right text-xs font-mono text-gray-500">
                Limit: <strong>{thresholdLabel}</strong>
              </div>
            </div>

            <div className="stat-sub-grid">
              <div className="stat-sub-item">
                <span className="stat-sub-label">Time Over Limit</span>
                <span className="stat-sub-val">{simState.timeOverLimitSec.toFixed(1)} s</span>
              </div>
              <div className="stat-sub-item">
                <span className="stat-sub-label">Null Depth</span>
                <span className="stat-sub-val text-teal-600">
                  {nullOnAir ? `${simState.nullDepthDb.toFixed(2)} dB` : '—'}
                </span>
              </div>
              <div className="stat-sub-item">
                <span className="stat-sub-label">Distance</span>
                <span className="stat-sub-val">{simState.activeFlock.distanceM} m</span>
              </div>
            </div>
          </div>

          {/* Card 2: Cell Throughput */}
          <div className="tech-card stat-big-card">
            <div className="stat-header">
              <span className="flex items-center gap-1.5 text-slate-700">
                <Wifi size={14} />
                <span>CELL THROUGHPUT</span>
              </span>
              <span className="badge-pill blue">{simState.users.length} UEs CONNECTED</span>
            </div>

            <div className="stat-value-row">
              <div>
                <span className="stat-big-val">{nanOr(simState.cellThroughputGbps, '—')}</span>
                <span className="stat-unit">Gbps</span>
              </div>
              <div className="text-right text-xs font-mono text-gray-500">
                T̂ {nanOr(simState.throughputFraction, '—')} of per-scenario max
              </div>
            </div>

            <div className="stat-sub-grid">
              <div className="stat-sub-item">
                <span className="stat-sub-label">Mean SINR</span>
                <span className="stat-sub-val">{simState.meanSinrDb} dB</span>
              </div>
              <div className="stat-sub-item">
                <span className="stat-sub-label">Mean BLER</span>
                <span className="stat-sub-val">{simState.meanBlerPercent}%</span>
              </div>
              <div className="stat-sub-item">
                <span className="stat-sub-label">TX RF Power</span>
                <span className="stat-sub-val">{simState.txRfPowerW} W</span>
              </div>
            </div>
          </div>

          {/* Card 3: Trade-off Pareto Weight Dial */}
          <div className="tech-card tradeoff-card">
            <div className="stat-header">
              <span className="flex items-center gap-1.5 text-slate-700">
                <Sliders size={14} />
                <span>TRADE-OFF PARETO WEIGHT</span>
              </span>
              <span className="badge-pill teal">w = {nanOr(simState.tradeoffWeight, '—')}</span>
            </div>

            <div className="dial-slider-container">
              <input
                type="range"
                min="0.0"
                max="5.0"
                step="0.05"
                value={simState.tradeoffWeight}
                onChange={(e) => onSetTradeoffWeight(parseFloat(e.target.value))}
                className="dial-slider"
              />
              <div className="dial-labels">
                <span>Prioritize QoS (w=0.0)</span>
                <span>Balanced (w=1.0)</span>
                <span>Max Ecological (w=5.0)</span>
              </div>
            </div>

            <div className="formula-box">
              <div className="text-[10px] text-gray-400 font-bold mb-1">REWARD TERMS THIS STEP</div>
              <div className="text-teal-900 font-semibold text-xs">
                T {nanOr(frameRewardThroughput, '—')} · E {nanOr(frameRewardExposure, '—')} · P {nanOr(frameRewardPower, '—')}
              </div>
              <div className="text-[10px] text-gray-500 mt-1">
                net {nanOr(frameRewardNet, '—')} · w {nanOr(simState.tradeoffWeight, '—')}
              </div>
              <div className="text-[10px] text-gray-400 mt-1">
                terms from the live frame; w = tradeoff weight (reward config)
              </div>
            </div>
          </div>

          {/* Card 4: Live Decision Stream */}
          <div className="tech-card">
            <div className="card-title-row">
              <span className="card-title">
                <Zap size={13} className="text-teal-600" />
                <span>LIVE DECISION STREAM</span>
              </span>
              <span className="live-pulse-dot" />
            </div>

            <div className="decision-stream-box">
              {simState.decisionLogs.length === 0 ? (
                <div className="text-xs text-gray-500">Awaiting first decision from the controller.</div>
              ) : (
                simState.decisionLogs.map((log: DecisionLogEntry) => (
                  <div key={log.id} className="decision-log-item">
                    <div className="decision-log-header">
                      <span>{log.timestamp} · step {log.step}</span>
                      <span className="decision-log-action">{log.actionType}</span>
                    </div>
                    <div className="decision-log-desc">{log.description}</div>
                    {log.details && <div className="decision-log-sub">{log.details}</div>}
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Row: timeline & stat tiles from the live session */}
      <div className="bottom-timeline-row">
        {/* Episode Timeline */}
        <div className="tech-card">
          <div className="card-title-row">
            <span className="card-title">
              <span>Episode Timeline</span>
            </span>
            <span className="badge-pill teal">live session</span>
          </div>            <div className="flex flex-wrap items-center gap-x-6 gap-y-1 font-mono text-xs mb-2">
            {previousRun != null && (
              <span className="flex items-center gap-2">
                <span className="w-4 h-0.5 bg-coral border-t border-coral" />
                <span className="text-gray-700 font-semibold">
                  Previous run — {labelForPolicy(previousRun.policyName)}
                </span>
              </span>
            )}
            <span className="flex items-center gap-2">
              <span className="w-4 h-0.5 bg-teal-600 border-t border-teal-600" />
              <span className="text-teal-800 font-semibold">
                Current run — {simState.engine !== '' ? labelForPolicy(simState.engine) : 'live session'}
              </span>
            </span>
          </div>

          {/* Area Chart SVG */}
          <div className="h-28 w-full bg-slate-50/70 border border-slate-200 rounded p-2 relative">
            <svg viewBox="0 0 600 80" className="w-full h-full">
              {/* Scenario exposure limit from the live frame's metrics. */}
              {Number.isFinite(simState.exposureLimitWm2) && simState.exposureLimitWm2 > 0 && (
                <>
                  <line x1="0" y1={exposureY(simState.exposureLimitWm2)} x2="600" y2={exposureY(simState.exposureLimitWm2)} stroke="#f87171" strokeWidth="1" strokeDasharray="3 3" />
                  <text x="590" y={(exposureY(simState.exposureLimitWm2) ?? 20) - 4} textAnchor="end" fill="#ef4444" fontSize="8" fontFamily="JetBrains Mono">
                    exposure limit {simState.exposureLimitWm2.toFixed(3)} W/m²
                  </text>
                </>
              )}

              {/* Before/after pair: the previous run is the "without" reference, drawn
                  at the same scale as the live run so the judge can read the gap off the
                  chart instead of memorizing two numbers. */}
              {previousRun != null &&
                Number.isFinite(previousRun.peakExposureWm2) &&
                previousRun.peakExposureWm2 > 0 && (
                  <>
                    <line
                      x1="0"
                      y1={exposureY(previousRun.peakExposureWm2)}
                      x2="600"
                      y2={exposureY(previousRun.peakExposureWm2)}
                      stroke="#fca5a5"
                      strokeWidth="1"
                      strokeDasharray="5 3"
                    />
                    <text
                      x="590"
                      y={(exposureY(previousRun.peakExposureWm2) ?? 20) - 4}
                      textAnchor="end"
                      fill="#c2410c"
                      fontSize="8"
                      fontFamily="JetBrains Mono"
                    >
                      previous ({previousRun.policyName}): {previousRun.peakExposureWm2.toFixed(3)} W/m²
                    </text>
                  </>
                )}

              {/* Live session exposure curve: one dot per streamed/displayed frame. */}
              {showHistory &&
                Array.from({ length: maxHistory }, (_, i) => (
                  <circle
                    key={i}
                    cx={timelineX(i)}
                    cy={exposureY(sessionExposureHistory[i] ?? 0)}
                    r="1.4"
                    fill="#0d9488"
                  />
                ))}

              {/* Decision ticks: only meaningful once enough live frames exist. */}
              {showHistory &&
                Array.from({ length: maxHistory }, (_, i) => (
                  <circle
                    key={i}
                    cx={timelineX(i)}
                    cy={exposureY(sessionExposureHistory[i] ?? 0)}
                    r="1"
                    fill="#0d9488"
                  />
                ))}
            </svg>
            <div className="absolute top-2 left-1/2 -translate-x-12 bg-white/90 px-2 py-0.5 rounded border border-teal-300 font-mono text-[10px] text-teal-800">
              {showHistory && summary ? (
                <>
                  {formatClock(summary.lastTimeS)} · live peak{' '}
                  {nanFixed(summary.peakExposureWm2, 4)} W/m² · {summary.frames} frames
                  {exposureDelta != null && <> · {exposureDelta}</>}
                </>
              ) : (
                <span className="text-gray-500">collecting frames…</span>
              )}
            </div>
          </div>
        </div>

        {/* Stat tiles: every value is computed from stored frames (WP-F);
            the fifth tile freezes the previous run, so switching controllers
            shows before | after instead of wiping the panel. */}
        {showHistory && summary ? (
          <div className="stat-tiles-row">
            {/* Live session — what is on air now, from every displayed frame. */}
            <div className="stat-tile">
              <span className="stat-tile-title">LIVE PEAK EXPOSURE</span>
              <span className="stat-tile-val text-teal-700">
                {nanFixed(summary.peakExposureWm2, 4)} W/m²
              </span>
              <span className="stat-tile-sub">
                peak at {formatClock(summary.peakAtTimeS)} · current frame{' '}
                {nanFixed(simState.peakExposureWm2, 4)} W/m²
              </span>
            </div>

            {/* Same-run throughput so the judge sees throughput held across the mitigation. */}
            <div className="stat-tile">
              <span className="stat-tile-title">LIVE THROUGHPUT</span>
              <span className="stat-tile-val text-slate-800">
                {nanFixed(summary.meanThroughputMbps / 1000, 2)} Gbps
              </span>
              <span className="stat-tile-sub">
                mean over {summary.frames} frames ·{' '}
                {nanFixed(summary.lastTimeS, 1)} s elapsed
              </span>
            </div>

            {/* Null engagement is the mitigation signal that matters on the radar. */}
            <div className="stat-tile">
              <span className="stat-tile-title">NULL ENGAGEMENT</span>
              <span className="stat-tile-val text-teal-700">
                {summary.nullOnCount > 0
                  ? `${summary.nullOnCount} / ${summary.frames} steps`
                  : '0 / ' + summary.frames}
              </span>
              <span className="stat-tile-sub">
                {summary.nullOnCount > 0
                  ? `${(
                      (summary.nullOnCount / summary.frames) *
                      100
                    ).toFixed(0)}% of run · last null depth ${nanFixed(
                      summary.nullDepthDb,
                      2,
                    )} dB`
                  : 'null off for all observed steps'}
              </span>
            </div>

            {/* The before/after delta is the key number for the judges. */}
            {previousRun != null &&
              Number.isFinite(previousRun.peakExposureWm2) &&
              Number.isFinite(summary.peakExposureWm2) && (
                <div className="stat-tile">
                  <span className="stat-tile-title">EXPOSURE REDUCTION vs BEFORE</span>                <span className={`stat-tile-val ${(summary.peakExposureWm2 <= previousRun.peakExposureWm2) ? 'text-teal-700' : 'text-rose-700'}`}>
                    {(previousRun.peakExposureWm2 - summary.peakExposureWm2).toFixed(4)}
                  </span>
                  <span className="stat-tile-sub">
                    {previousRun.peakExposureWm2.toFixed(4)} →{' '}
                    {summary.peakExposureWm2.toFixed(4)} W/m²
                    {summary.peakExposureWm2 <= previousRun.peakExposureWm2
                      ? ' · current run lower'
                      : ' · current run higher'}
                  </span>
                </div>
              )}

            {/* Frozen previous run, whichever controller produced it. */}
            {previousRun != null && (
              <div className="stat-tile">
                <span className="stat-tile-title">PREVIOUS RUN</span>
                <span className="stat-tile-val text-slate-800" style={{ fontSize: '1rem' }}>
                  {labelForPolicy(previousRun.policyName)}
                </span>
                <span className="stat-tile-sub">
                  peak {nanFixed(previousRun.peakExposureWm2, 4)} W/m² · null{' '}
                  {nanFixed(previousRun.nullDepthDb, 2)} dB · TX{' '}
                  {nanFixed(previousRun.powerMw / 1000, 2)} W
                </span>
              </div>
            )}
          </div>
        ) : null}
      </div>
    </div>
  );
};
