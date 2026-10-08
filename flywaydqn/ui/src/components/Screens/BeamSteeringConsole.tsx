import React from 'react';
import { Wind, Feather, Cpu, ShieldAlert, Wifi, Sliders, Zap } from 'lucide-react';
import { SimState, SteeringEngine, UEData, DecisionLogEntry } from '../../types';

interface BeamSteeringConsoleProps {
  simState: SimState;
  onSetEngine: (engine: SteeringEngine) => void;
  onSetTradeoffWeight: (w: number) => void;
}

export const BeamSteeringConsole: React.FC<BeamSteeringConsoleProps> = ({
  simState,
  onSetEngine,
  onSetTradeoffWeight,
}) => {
  const isDqn = simState.engine === 'FlywayDQN (Active Null)';

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
            ONLINE — 28 GHz mmWave MIMO (64T64R) // COORD: 37.7749° N, 122.4194° W // AZ: 045.0°
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
            <span>INFERENCE: {simState.inferenceLatencyMs} ms</span>
          </div>
        </div>
      </div>

      {/* Main Grid: Polar Canvas on Left, Metrics on Right */}
      <div className="beam-screen-grid">
        {/* Left Card: Beamforming Matrix & Polar Canvas */}
        <div className="tech-card steering-canvas-card">
          <div className="radar-header-controls">
            <div className="flex items-center gap-3">
              <span className="font-mono text-xs text-gray-500 font-semibold uppercase">STEERING ENGINE:</span>
              <div className="engine-toggle-group">
                <button
                  className={`engine-btn ${!isDqn ? 'active-baseline' : ''}`}
                  onClick={() => onSetEngine('Baseline SON')}
                >
                  <span className="w-2 h-2 rounded-full bg-red-500 inline-block" />
                  <span>Baseline SON</span>
                </button>
                <button
                  className={`engine-btn ${isDqn ? 'active-dqn' : ''}`}
                  onClick={() => onSetEngine('FlywayDQN (Active Null)')}
                >
                  <span className="w-2 h-2 rounded-full bg-emerald-400 inline-block" />
                  <span>FlywayDQN (Active Null)</span>
                </button>
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

              {/* Polar Grid Lines from origin at (60, 390) */}
              <g transform="translate(60, 390)">
                {/* Range Rings */}
                <path d="M 0,-100 A 100 100 0 0 1 100,0" fill="none" stroke="#d1fae5" strokeWidth="1" strokeDasharray="3 3" />
                <path d="M 0,-200 A 200 200 0 0 1 200,0" fill="none" stroke="#d1fae5" strokeWidth="1" strokeDasharray="3 3" />
                <path d="M 0,-300 A 300 300 0 0 1 300,0" fill="none" stroke="#a7f3d0" strokeWidth="1.2" />
                <path d="M 0,-380 A 380 380 0 0 1 380,0" fill="none" stroke="#cbd5e1" strokeWidth="1" />

                {/* Distance Labels */}
                <text x="105" y="-5" fill="#94a3b8" fontSize="10" fontFamily="JetBrains Mono">100m</text>
                <text x="205" y="-5" fill="#94a3b8" fontSize="10" fontFamily="JetBrains Mono">200m</text>
                <text x="305" y="-5" fill="#94a3b8" fontSize="10" fontFamily="JetBrains Mono">300m</text>
                <text x="385" y="-5" fill="#94a3b8" fontSize="10" fontFamily="JetBrains Mono">400m</text>

                {/* Radial Azimuth Rays */}
                <line x1="0" y1="0" x2="0" y2="-390" stroke="#e2e8f0" strokeWidth="1" />
                <text x="5" y="-380" fill="#94a3b8" fontSize="10" fontFamily="JetBrains Mono">90° N</text>

                <line x1="0" y1="0" x2="275" y2="-275" stroke="#cbd5e1" strokeWidth="1" strokeDasharray="2 2" />
                <text x="280" y="-270" fill="#0f766e" fontSize="11" fontFamily="JetBrains Mono" fontWeight="600">045° (Boresight)</text>

                <line x1="0" y1="0" x2="390" y2="0" stroke="#e2e8f0" strokeWidth="1" />
                <text x="395" y="-5" fill="#94a3b8" fontSize="10" fontFamily="JetBrains Mono">000° E</text>

                <line x1="0" y1="0" x2="195" y2="-337" stroke="#e2e8f0" strokeWidth="1" strokeDasharray="2 2" />
                <text x="200" y="-340" fill="#94a3b8" fontSize="10" fontFamily="JetBrains Mono">060°</text>

                <line x1="0" y1="0" x2="337" y2="-195" stroke="#e2e8f0" strokeWidth="1" strokeDasharray="2 2" />
                <text x="342" y="-195" fill="#94a3b8" fontSize="10" fontFamily="JetBrains Mono">030°</text>

                {/* Active Sector 3dB Beam Envelope */}
                <path
                  d="M 0,0 L 80,-350 A 360 360 0 0 1 350,-80 Z"
                  fill="url(#beamGrad)"
                />

                {/* Dynamic Oscillating Sector Radar Sweep Beam */}
                <g className="sector-radar-sweep">
                  <path
                    d="M 0,0 L 380,0 A 380 380 0 0 0 360,-80 Z"
                    fill="url(#sweepGrad)"
                  />
                  <line x1="0" y1="0" x2="380" y2="0" stroke="#10b981" strokeWidth="2" opacity="0.8" />
                </g>

                {/* Ecological Null Corridor Cone (When DQN is enabled) */}
                {isDqn && (
                  <g className="null-shimmer">
                    {/* Buffer zone background */}
                    <path
                      d="M 0,0 L 220,-280 A 350 350 0 0 1 290,-220 Z"
                      fill="url(#diagonalHatch)"
                    />
                    {/* Core deep null wedge */}
                    <path
                      d="M 0,0 L 235,-265 A 350 350 0 0 1 270,-235 Z"
                      fill="url(#nullGrad)"
                      stroke="#0d9488"
                      strokeWidth="1.5"
                      strokeDasharray="4 2"
                    />
                  </g>
                )}

                {/* Animated Flock Contour & Centroid Position */}
                <g className="animated-flock-group">
                  {/* Radar Sonar Echo Ripples */}
                  <circle cx="0" cy="0" r="10" fill="none" stroke={isDqn ? '#0d9488' : '#ef4444'} className="target-ping-1" />
                  <circle cx="0" cy="0" r="10" fill="none" stroke={isDqn ? '#14b8a6' : '#f87171'} className="target-ping-2" />

                  {/* Outer Protection Bubble */}
                  <ellipse
                    cx="0"
                    cy="0"
                    rx="52"
                    ry="28"
                    transform="rotate(-40)"
                    fill={isDqn ? 'rgba(13, 148, 136, 0.12)' : 'rgba(239, 68, 68, 0.18)'}
                    stroke={isDqn ? '#0d9488' : '#ef4444'}
                    strokeWidth="1.5"
                    strokeDasharray={isDqn ? '4 3' : 'none'}
                  />

                  {/* Bird icons flock representation */}
                  <g fill={isDqn ? '#0f766e' : '#b91c1c'}>
                    <path d="M -15,-5 Q -10,-10 -5,-5 Q 0,-10 5,-5" stroke={isDqn ? '#0f766e' : '#b91c1c'} strokeWidth="1.5" fill="none" />
                    <path d="M -5,5 Q 0,0 5,5 Q 10,0 15,5" stroke={isDqn ? '#0f766e' : '#b91c1c'} strokeWidth="1.5" fill="none" />
                    <path d="M -22,8 Q -18,4 -14,8 Q -10,4 -6,8" stroke={isDqn ? '#0f766e' : '#b91c1c'} strokeWidth="1.2" fill="none" />
                    <circle cx="0" cy="0" r="3" fill={isDqn ? '#0d9488' : '#ef4444'} />
                  </g>

                  {/* Null Depth Tag */}
                  <rect
                    x="-65"
                    y="-40"
                    width="130"
                    height="18"
                    rx="4"
                    fill="#ffffff"
                    stroke={isDqn ? '#0d9488' : '#ef4444'}
                    strokeWidth="1"
                  />
                  <text
                    x="0"
                    y="-28"
                    textAnchor="middle"
                    fill={isDqn ? '#0f766e' : '#b91c1c'}
                    fontSize="9"
                    fontFamily="JetBrains Mono"
                    fontWeight="700"
                  >
                    {isDqn ? 'NULL: -26.4 dB (047° AZ)' : 'HIGH EXPOSURE (4.85 W/m²)'}
                  </text>
                </g>

                {/* UE Targets with Animated Propagating Beam Rays */}
                {simState.users.map((ue: UEData) => {
                  const rad = ((90 - ue.azimuthDeg) * Math.PI) / 180;
                  const x = Math.cos(rad) * ue.distanceM;
                  const y = -Math.sin(rad) * ue.distanceM;

                  return (
                    <g key={ue.id} transform={`translate(${x}, ${y})`}>
                      {/* Animated carrier wave ray toward UE */}
                      <line
                        x1={-x}
                        y1={-y}
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

            {/* Spatial Null Depth Overlay Top-Left */}
            <div className="absolute top-3 left-3 bg-white/95 backdrop-blur-sm border border-emerald-200 rounded-md p-2.5 font-mono text-xs shadow-sm">
              <div className="text-gray-500 text-[10px] uppercase font-bold">SPATIAL NULL DEPTH</div>
              <div className="text-teal-700 text-sm font-bold">{simState.nullDepthDb} dB EIRP</div>
              <div className="text-gray-500 text-[10px]">Corridor Width: 18.2°</div>
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
              <span>NULL BAND: <strong>180 MHz</strong></span>
              <span className="mx-2 text-gray-300">|</span>
              <span>BEAM STEER LATENCY: <strong>10ms</strong></span>
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
              <span className={`badge-pill ${simState.peakExposureWm2 < 1.0 ? 'teal' : 'coral'}`}>
                {simState.peakExposureWm2 < 1.0 ? 'NOMINAL SAFE' : 'THRESHOLD EXCEEDED'}
              </span>
            </div>

            <div className="stat-value-row">
              <div>
                <span className="stat-big-val">{simState.peakExposureWm2}</span>
                <span className="stat-unit">W/m²</span>
              </div>
              <div className="text-right text-xs font-mono text-gray-500">
                Limit: <strong>{simState.exposureLimitWm2.toFixed(2)} W/m²</strong>
              </div>
            </div>

            {/* Mini Sparkline Chart */}
            <div className="h-6 w-full bg-slate-50 rounded flex items-center px-1 overflow-hidden">
              <svg viewBox="0 0 200 24" className="w-full h-full">
                <path
                  d="M 0,16 Q 30,14 60,18 T 120,12 T 160,15 T 200,14"
                  fill="none"
                  stroke="#0d9488"
                  strokeWidth="2"
                />
              </svg>
            </div>

            <div className="stat-sub-grid">
              <div className="stat-sub-item">
                <span className="stat-sub-label">Time Over Limit</span>
                <span className="stat-sub-val">{simState.timeOverLimitSec.toFixed(1)} s (0%)</span>
              </div>
              <div className="stat-sub-item">
                <span className="stat-sub-label">Null Depth</span>
                <span className="stat-sub-val text-teal-600">{simState.nullDepthDb} dB</span>
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
              <span className="badge-pill blue">8 UEs CONNECTED</span>
            </div>

            <div className="stat-value-row">
              <div>
                <span className="stat-big-val">{simState.cellThroughputGbps}</span>
                <span className="stat-unit">Gbps</span>
              </div>
              <div className="stat-delta bg-emerald-50 text-emerald-700">
                -3.1% vs Baseline
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
              <span className="badge-pill teal">w = {simState.tradeoffWeight.toFixed(2)}</span>
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
              <div className="text-[10px] text-gray-400 font-bold mb-1">ACTIVE REWARD FUNCTION</div>
              <div className="text-teal-900 font-semibold text-xs">
                R = α·R<sub>TP</sub>({simState.rewardThroughput.toFixed(2)}) - β·E<sub>Flock</sub>({simState.rewardExposure.toFixed(2)}) - γ·P<sub>TX</sub>({simState.rewardPower.toFixed(2)})
              </div>
              <div className="text-[10px] text-gray-500 mt-1">
                α = 1.0 (QoS) | β = {(simState.tradeoffWeight * 3.3).toFixed(1)} (Eco Null) | γ = 0.1 (Energy)
              </div>
            </div>
          </div>

          {/* Card 4: DQN Decision Stream */}
          <div className="tech-card">
            <div className="card-title-row">
              <span className="card-title">
                <Zap size={13} className="text-teal-600" />
                <span>DQN DECISION STREAM</span>
              </span>
              <span className="live-pulse-dot" />
            </div>

            <div className="decision-stream-box">
              {simState.decisionLogs.map((log: DecisionLogEntry) => (
                <div key={log.id} className="decision-log-item">
                  <div className="decision-log-header">
                    <span>{log.timestamp} — STEP #{log.step}</span>
                    <span className="decision-log-action">ACTION: {log.actionType}</span>
                  </div>
                  <div className="decision-log-desc">{log.description}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Row: 15-min Timeline & Stats */}
      <div className="bottom-timeline-row">
        {/* Migration Pass Timeline */}
        <div className="tech-card">
          <div className="card-title-row">
            <span className="card-title">
              <span>15-min Migration Pass Episode Timeline</span>
            </span>
            <span className="badge-pill teal">PASS #084-A</span>
          </div>

          <div className="flex items-center gap-6 font-mono text-xs mb-2">
            <span className="flex items-center gap-2">
              <span className="w-4 h-0.5 bg-red-400 border-t border-dashed border-red-500" />
              <span className="text-gray-600">Baseline SON Exposure</span>
            </span>
            <span className="flex items-center gap-2">
              <span className="w-4 h-0.5 bg-teal-600" />
              <span className="text-teal-800 font-semibold">FlywayDQN Active Dose</span>
            </span>
          </div>

          {/* Area Chart SVG */}
          <div className="h-28 w-full bg-slate-50/70 border border-slate-200 rounded p-2 relative">
            <svg viewBox="0 0 600 80" className="w-full h-full">
              {/* Threshold line */}
              <line x1="0" y1="20" x2="600" y2="20" stroke="#f87171" strokeWidth="1" strokeDasharray="3 3" />
              <text x="590" y="16" textAnchor="end" fill="#ef4444" fontSize="8" fontFamily="JetBrains Mono">
                REGULATORY THRESHOLD (5.00 W/m²)
              </text>

              {/* Baseline bell curve exposure */}
              <path
                d="M 0,70 Q 150,68 250,30 Q 300,10 350,30 Q 450,68 600,70"
                fill="none"
                stroke="#fca5a5"
                strokeWidth="1.5"
                strokeDasharray="4 2"
              />

              {/* FlywayDQN suppressed dose curve */}
              <path
                d="M 0,72 Q 150,71 250,55 Q 300,50 350,55 Q 450,71 600,72 L 600,78 L 0,78 Z"
                fill="rgba(13, 148, 136, 0.15)"
              />
              <path
                d="M 0,72 Q 150,71 250,55 Q 300,50 350,55 Q 450,71 600,72"
                fill="none"
                stroke="#0d9488"
                strokeWidth="2"
              />

              {/* Current scrubber needle at T=08:14 */}
              <line x1="280" y1="0" x2="280" y2="80" stroke="#0f766e" strokeWidth="1.5" />
              <circle cx="280" cy="52" r="3" fill="#0f766e" />
            </svg>
            <div className="absolute top-2 left-1/2 -translate-x-12 bg-white/90 px-2 py-0.5 rounded border border-teal-300 font-mono text-[10px] text-teal-800">
              T+08:14 · Exposure: <strong>{simState.peakExposureWm2} W/m²</strong>
            </div>
          </div>
        </div>

        {/* 4 Stat Tiles */}
        <div className="stat-tiles-row">
          <div className="stat-tile">
            <span className="stat-tile-title">CUMULATIVE DOSE REDUCTION</span>
            <span className="stat-tile-val text-teal-700">-89.4%</span>
            <span className="stat-tile-sub">vs. SON Static Pattern</span>
          </div>

          <div className="stat-tile">
            <span className="stat-tile-title">THROUGHPUT DELIVERED</span>
            <span className="stat-tile-val text-slate-800">96.9%</span>
            <span className="stat-tile-sub">2.84 Gbps of 2.93 Gbps target</span>
          </div>

          <div className="stat-tile">
            <span className="stat-tile-title">SAFE CORRIDOR AZIMUTH SPAN</span>
            <span className="stat-tile-val text-slate-800">038° — 056°</span>
            <span className="stat-tile-sub">18° null opening</span>
          </div>

          <div className="stat-tile">
            <span className="stat-tile-title">AGENT ACTION FREQUENCY</span>
            <span className="stat-tile-val text-teal-700">100 Hz</span>
            <span className="stat-tile-sub">10ms Phase State Steps</span>
          </div>
        </div>
      </div>
    </div>
  );
};
