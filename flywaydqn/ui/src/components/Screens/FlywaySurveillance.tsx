import React, { useState } from 'react';
import { Lock, Radio, Compass } from 'lucide-react';
import { SimState, FlockTarget, RadarEventEntry } from '../../types';

interface FlywaySurveillanceProps {
  simState: SimState;
  onSelectFlock: (id: string) => void;
}

/**
 * Radar scope geometry.
 *
 * The map is drawn in the same azimuth convention the physics uses everywhere
 * else: 0 deg is panel boresight (east), +90 deg is north, and range is the
 * flock/user distance from the sector origin. Every marker is projected from
 * the live frame, so the scope moves exactly as the simulation does.
 */
const RADAR = {
  width: 620,
  height: 400,
  cx: 310,
  cy: 200,
  radius: 225,
  maxRangeM: 750,
} as const;

const RADAR_SCALE = RADAR.radius / RADAR.maxRangeM;
const RING_RANGES_M = [250, 500, 750] as const;

/**
 * Project an (azimuth, range) pair onto the radar SVG, clamping to the scope.
 * Markers keep a small margin inside the outer ring so their symbols and
 * labels never spill outside the 620x400 viewBox at the scope edge.
 */
const MARKER_MARGIN_PX = 20;

function polarToSvg(
  azimuthDeg: number,
  rangeM: number,
  marginPx: number = 0,
): { x: number; y: number } {
  const clamped = Math.max(0, Math.min(rangeM, RADAR.maxRangeM));
  const radiusPx = Math.min(clamped * RADAR_SCALE, RADAR.radius - marginPx);
  const angle = (azimuthDeg * Math.PI) / 180;
  return {
    x: RADAR.cx + radiusPx * Math.cos(angle),
    y: RADAR.cy - radiusPx * Math.sin(angle),
  };
}

/** Eight-point label for an azimuth measured counter-clockwise from east. */
function compassPoint(azimuthDeg: number): string {
  const names = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE'];
  const index = Math.round((((azimuthDeg % 360) + 360) % 360) / 45) % 8;
  return names[index];
}

/** The +-60 deg sector the codebook covers, as a filled wedge. */
function sectorWedgePath(halfSpanDeg = 60): string {
  const points: string[] = [];
  for (let angle = -halfSpanDeg; angle <= halfSpanDeg; angle += 10) {
    const p = polarToSvg(angle, RADAR.maxRangeM);
    points.push(`L ${p.x.toFixed(1)} ${p.y.toFixed(1)}`);
  }
  return `M ${RADAR.cx} ${RADAR.cy} ${points.join(' ')} Z`;
}

/** A short trailing wedge behind the sweep line. */
function sweepTrailPath(angleDeg: number): string {
  const points = [angleDeg, angleDeg - 12, angleDeg - 24].map((angle) => {
    const p = polarToSvg(angle, RADAR.maxRangeM);
    return `L ${p.x.toFixed(1)} ${p.y.toFixed(1)}`;
  });
  return `M ${RADAR.cx} ${RADAR.cy} ${points.join(' ')} Z`;
}

function riskColor(level: FlockTarget['riskLevel']): string {
  if (level === 'CRITICAL') return '#dc2626';
  if (level === 'ALERT') return '#ea580c';
  return '#0d9488';
}

export const FlywaySurveillance: React.FC<FlywaySurveillanceProps> = ({
  simState,
  onSelectFlock,
}) => {
  const [activeBuffer, setActiveBuffer] = useState<'500m' | '300m' | '100m'>('300m');
  const [filterType, setFilterType] = useState<'ALL' | 'NULL' | 'ENTRY'>('ALL');

  const flock = simState.activeFlock;
  const flockPos = polarToSvg(flock.azimuthDeg, flock.distanceM, MARKER_MARGIN_PX);
  const flockColor = riskColor(flock.riskLevel);
  const offScope = flock.distanceM > RADAR.maxRangeM;
  // Flip the tag below the crosshair when the flock is near the top edge, so
  // the label stays inside the SVG instead of being clipped.
  const labelBelow = flockPos.y < 80;

  const headingRad = (flock.headingDeg * Math.PI) / 180;
  const headingFrom = { x: Math.cos(headingRad) * 13, y: -Math.sin(headingRad) * 13 };
  const headingTo = { x: Math.cos(headingRad) * 54, y: -Math.sin(headingRad) * 54 };

  const beamEnd = polarToSvg(simState.beamAzimuthDeg, RADAR.maxRangeM);
  const beamLabel = polarToSvg(simState.beamAzimuthDeg, RADAR.maxRangeM * 0.82);
  const nullEnd = polarToSvg(flock.azimuthDeg, RADAR.maxRangeM);

  const sweepDeg = (simState.timeSec * 40) % 360;
  const sweepEnd = polarToSvg(sweepDeg, RADAR.maxRangeM);

  const bufferM = Number(activeBuffer.replace('m', ''));
  const bufferRadiusPx = bufferM * RADAR_SCALE;

  const peak = simState.peakExposureWm2;
  const limit = simState.exposureLimitWm2;
  const exposureRatio = limit > 0 ? Math.min(1.4, peak / limit) : 0;
  const overLimit = limit > 0 && peak > limit;

  const filteredEvents = simState.radarEvents.filter((ev: RadarEventEntry) => {
    if (filterType === 'NULL') return ev.actionTriggered.includes('Null');
    if (filterType === 'ENTRY') return ev.eventType === 'CORRIDOR_ENTRY';
    return true;
  });

  return (
    <div className="content-viewport" style={{ gap: '14px', maxWidth: '1720px' }}>
      {/* 1. Subheader Strip */}
      <div className="flex-row-between" style={{ flexWrap: 'wrap', gap: '8px', fontSize: '0.75rem', fontFamily: 'var(--font-mono)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#1e293b' }}>
            <span style={{ width: '9px', height: '9px', borderRadius: '50%', backgroundColor: '#10b981', display: 'inline-block' }} />
            <strong style={{ letterSpacing: '0.04em' }}>SURVEILLANCE MODE: PACIFIC FLYWAY SEC-04</strong>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#005951', fontWeight: 600 }}>
            <Radio size={14} style={{ color: '#0d9488' }} />
            <span>LIVE TRACK FUSION · RADAR + 5G RF</span>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontFamily: 'var(--font-mono)', fontSize: '0.72rem' }}>
          <span className="pill-badge mint">
            <span style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: '#0d9488' }} />
            <span>{simState.timeString}</span>
          </span>
          <span className="pill-badge blue">FRAME {simState.step.toLocaleString('en-US')}</span>
        </div>
      </div>

      {/* 2. Top 4 Live Stat Cards */}
      <div className="surveillance-top-grid">
        {/* Card 1: tracked flocks, from the frame */}
        <div className="surveillance-card">
          <div className="flex-row-between">
            <span style={{ fontSize: '0.68rem', fontFamily: 'var(--font-mono)', color: '#94a3b8', fontWeight: 700, letterSpacing: '0.06em' }}>
              ACTIVE TRACKED FLOCKS
            </span>
            <span className="pill-badge mint">RADAR 3D</span>
          </div>
          <div style={{ margin: '6px 0', display: 'flex', alignItems: 'baseline', gap: '8px' }}>
            <span style={{ fontSize: '1.9rem', fontWeight: 800, fontFamily: 'var(--font-sans)', color: '#0f172a', lineHeight: 1 }}>{simState.allFlocks.length}</span>
            <span style={{ fontSize: '1.05rem', fontWeight: 700, color: '#1e293b' }}>Flocks</span>
            <span style={{ fontSize: '0.75rem', color: '#94a3b8', fontFamily: 'var(--font-mono)' }}>{flock.birdCount} Birds</span>
          </div>
          <div style={{ fontSize: '0.75rem', color: '#005951', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
            {flock.id} {flock.commonName} · {flock.distanceM} m · {flock.azimuthDeg.toFixed(1)}°
          </div>
        </div>

        {/* Card 2: live exposure against the scenario's own limit */}
        <div className="surveillance-card">
          <div className="flex-row-between">
            <span style={{ fontSize: '0.68rem', fontFamily: 'var(--font-mono)', color: '#94a3b8', fontWeight: 700, letterSpacing: '0.06em' }}>
              PEAK FLOCK EXPOSURE
            </span>
            <span className="pill-badge mint">
              <span style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: overLimit ? '#dc2626' : '#0d9488' }} />
              <span>{overLimit ? 'OVER LIMIT' : 'RESPECTED'}</span>
            </span>
          </div>
          <div style={{ margin: '6px 0', display: 'flex', alignItems: 'baseline', gap: '8px' }}>
            <span style={{ fontSize: '1.6rem', fontWeight: 800, fontFamily: 'var(--font-sans)', color: overLimit ? '#dc2626' : '#005951', lineHeight: 1 }}>{peak.toFixed(5)}</span>
            <span style={{ fontSize: '0.82rem', fontFamily: 'var(--font-mono)', color: '#1e293b', fontWeight: 600 }}>W/m²</span>
          </div>
          <div style={{ height: '4px', width: '100%', backgroundColor: '#e2e8f0', borderRadius: '2px', overflow: 'hidden' }}>
            <div style={{ height: '100%', width: `${Math.min(100, exposureRatio * 100)}%`, backgroundColor: overLimit ? '#dc2626' : '#0d9488', borderRadius: '2px' }} />
          </div>
          <div style={{ fontSize: '0.68rem', fontFamily: 'var(--font-mono)', color: '#64748b', marginTop: '4px' }}>
            Limit {limit > 0 ? `${limit.toFixed(3)} W/m²` : '—'}
          </div>
        </div>

        {/* Card 3: live migration vector */}
        <div className="surveillance-card">
          <div className="flex-row-between">
            <span style={{ fontSize: '0.68rem', fontFamily: 'var(--font-mono)', color: '#94a3b8', fontWeight: 700, letterSpacing: '0.06em' }}>
              MIGRATION VECTOR
            </span>
            <Compass size={14} style={{ color: '#94a3b8' }} />
          </div>
          <div style={{ margin: '6px 0', display: 'flex', alignItems: 'baseline', gap: '6px' }}>
            <span style={{ fontSize: '1.9rem', fontWeight: 800, fontFamily: 'var(--font-sans)', color: '#0f172a', lineHeight: 1 }}>
              {Math.round(flock.headingDeg).toString().padStart(3, '0')}°
            </span>
            <span style={{ fontSize: '1rem', fontWeight: 700, color: '#1e293b' }}>{compassPoint(flock.headingDeg)}</span>
            <span style={{ fontSize: '0.78rem', color: '#64748b', fontFamily: 'var(--font-mono)' }}>@ {flock.speedMs.toFixed(1)} m/s</span>
          </div>
          <div style={{ fontSize: '0.72rem', color: '#64748b', fontFamily: 'var(--font-mono)', display: 'flex', justifyContent: 'space-between' }}>
            <span>ALT AGL: {flock.altitudeM} m</span>
            <span style={{ color: '#005951', fontWeight: 600 }}>Entry T-{flock.estEntrySec}s</span>
          </div>
        </div>

        {/* Card 4: live controller state (replaces invented sensor health) */}
        <div className="surveillance-card">
          <div className="flex-row-between">
            <span style={{ fontSize: '0.68rem', fontFamily: 'var(--font-mono)', color: '#94a3b8', fontWeight: 700, letterSpacing: '0.06em' }}>
              ACTIVE CONTROLLER
            </span>
            <span className="pill-badge mint">{simState.engine || 'AWAITING'}</span>
          </div>
          <div style={{ margin: '6px 0', display: 'flex', alignItems: 'baseline', gap: '8px' }}>
            <span style={{ fontSize: '1.15rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: simState.nullActive ? '#005951' : '#0f172a', lineHeight: 1.2 }}>
              {simState.nullActive ? 'NULL ON AIR' : 'NULL OFF'}
            </span>
          </div>
          <div style={{ fontSize: '0.72rem', color: '#64748b', fontFamily: 'var(--font-mono)', display: 'flex', justifyContent: 'space-between' }}>
            <span>{simState.nullActive ? `${simState.nullDepthDb.toFixed(2)} dB depth` : 'no notch transmitted'}</span>
            <span>Latency: {simState.inferenceLatencyMs.toFixed(2)} ms</span>
          </div>
        </div>
      </div>

      {/* 3. Main Row: Live Radar (Left) & Flock Target Detail (Right) */}
      <div className="surveillance-main-grid">
        {/* Left: Radar Card */}
        <div className="surveillance-card">
          <div>
            {/* Header */}
            <div className="flex-row-between" style={{ flexWrap: 'wrap', gap: '8px', paddingBottom: '8px' }}>
              <h2 style={{ fontFamily: 'var(--font-sans)', fontWeight: 800, fontSize: '0.92rem', color: '#0f172a', letterSpacing: '0.04em' }}>
                RADAR-01 // AIRSPACE CORRIDOR TRAJECTORY
              </h2>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontFamily: 'var(--font-mono)', fontSize: '0.72rem' }}>
                <span className="pill-badge purple">SECTOR_ALPHA</span>
                <span style={{ color: '#64748b', fontSize: '0.68rem' }}>Sector A: 0148-09</span>
                <span style={{ display: 'flex', alignItems: 'center', gap: '4px', color: overLimit ? '#dc2626' : '#0d9488', fontWeight: 700, fontSize: '0.68rem' }}>
                  <span style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: overLimit ? '#dc2626' : '#0d9488' }} />
                  <span>{overLimit ? 'EXPOSURE LIMIT EXCEEDED' : 'EXPOSURE LIMIT RESPECTED'}</span>
                </span>
                <span className="pill-badge blue">Range: {flock.distanceM} m</span>
              </div>
            </div>

            {/* In-Radar Legend */}
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '16px', fontSize: '0.72rem', fontFamily: 'var(--font-mono)', color: '#475569', padding: '4px 0 8px', flexWrap: 'wrap' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#ea580c' }} />
                <span>Flock (live)</span>
              </span>
              <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#0284c7' }} />
                <span>Users (live)</span>
              </span>
              <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span style={{ width: '10px', height: '2px', backgroundColor: '#005951' }} />
                <span>Beam</span>
              </span>
              <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span style={{ width: '10px', height: '0', borderTop: '2px dashed #0f766e' }} />
                <span>Null directs at flock</span>
              </span>
            </div>

            {/* Radar Circular Scope Container */}
            <div className="surveillance-radar-wrapper">
              <svg viewBox="0 0 620 400" style={{ width: '100%', height: '100%' }}>
                <defs>
                  <linearGradient id="survRadialSweep" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#0d9488" stopOpacity="0.36" />
                    <stop offset="100%" stopColor="#2dd4bf" stopOpacity="0.0" />
                  </linearGradient>
                  <marker id="survArrow" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
                    <path d="M0,0 L7,3 L0,6 Z" fill="#c2410c" />
                  </marker>
                </defs>

                {/* Codebook sector (+-60 deg about boresight) */}
                <path d={sectorWedgePath(60)} fill="#ecfdf5" stroke="#99f6e4" strokeWidth="1" />

                {/* Range rings */}
                {RING_RANGES_M.map((meters) => {
                  const radiusPx = meters * RADAR_SCALE;
                  return (
                    <g key={meters}>
                      <circle cx={RADAR.cx} cy={RADAR.cy} r={radiusPx} fill="none" stroke="#cbd5e1" strokeWidth="1" strokeDasharray={meters === RADAR.maxRangeM ? '0' : '3 3'} />
                      <text x={RADAR.cx + 4} y={RADAR.cy - radiusPx + 12} fill="#94a3b8" fontSize="9" fontFamily="JetBrains Mono">
                        {meters} m
                      </text>
                    </g>
                  );
                })}

                {/* Cardinal axes, same convention as the beam console: 0 E, 90 N */}
                <line x1={RADAR.cx} y1={RADAR.cy} x2={RADAR.cx + RADAR.radius} y2={RADAR.cy} stroke="#cbd5e1" strokeWidth="1" />
                <line x1={RADAR.cx} y1={RADAR.cy} x2={RADAR.cx - RADAR.radius} y2={RADAR.cy} stroke="#cbd5e1" strokeWidth="1" />
                <line x1={RADAR.cx} y1={RADAR.cy} x2={RADAR.cx} y2={RADAR.cy - RADAR.radius} stroke="#cbd5e1" strokeWidth="1" />
                <line x1={RADAR.cx} y1={RADAR.cy} x2={RADAR.cx} y2={RADAR.cy + RADAR.radius} stroke="#cbd5e1" strokeWidth="1" />
                <text x={RADAR.cx + RADAR.radius - 34} y={RADAR.cy - 6} fill="#94a3b8" fontSize="9" fontFamily="JetBrains Mono">000° E</text>
                <text x={RADAR.cx + 6} y={RADAR.cy - RADAR.radius + 10} fill="#008075" fontSize="9" fontFamily="JetBrains Mono" fontWeight="700">090° N</text>
                <text x={RADAR.cx - RADAR.radius + 4} y={RADAR.cy - 6} fill="#94a3b8" fontSize="9" fontFamily="JetBrains Mono">180° W</text>
                <text x={RADAR.cx + 6} y={RADAR.cy + RADAR.radius - 6} fill="#94a3b8" fontSize="9" fontFamily="JetBrains Mono">270° S</text>

                {/* Proximity buffer chosen on the right-hand card */}
                <circle cx={RADAR.cx} cy={RADAR.cy} r={bufferRadiusPx} fill="none" stroke="#f59e0b" strokeWidth="1.2" strokeDasharray="6 4" opacity="0.75" />
                <text x={RADAR.cx + bufferRadiusPx * 0.72} y={RADAR.cy + 12} fill="#b45309" fontSize="8.5" fontFamily="JetBrains Mono">
                  {activeBuffer} BUFFER
                </text>

                {/* Radar sweep, driven by simulation time so it pauses with the sim */}
                <path d={sweepTrailPath(sweepDeg)} fill="url(#survRadialSweep)" />
                <line x1={RADAR.cx} y1={RADAR.cy} x2={sweepEnd.x} y2={sweepEnd.y} stroke="#0d9488" strokeWidth="1.6" opacity="0.75" />

                {/* Serving beam direction */}
                <line x1={RADAR.cx} y1={RADAR.cy} x2={beamEnd.x} y2={beamEnd.y} stroke="#005951" strokeWidth="2" />
                <text x={beamLabel.x + 6} y={beamLabel.y - 6} fill="#005951" fontSize="8.5" fontFamily="JetBrains Mono" fontWeight="700">
                  BEAM #{simState.beamIndex} @ {simState.beamAzimuthDeg.toFixed(1)}°
                </text>

                {/* Null direction (toward the flock peak) */}
                {simState.nullActive && (
                  <g>
                    <line x1={RADAR.cx} y1={RADAR.cy} x2={nullEnd.x} y2={nullEnd.y} stroke="#0f766e" strokeWidth="1.6" strokeDasharray="5 4" />
                    <text x={nullEnd.x - 96} y={nullEnd.y + 16} fill="#0f766e" fontSize="8.5" fontFamily="JetBrains Mono" fontWeight="700">
                      NULL {simState.nullDepthDb.toFixed(2)} dB
                    </text>
                  </g>
                )}

                {/* User terminals, live positions */}
                {simState.users.map((user, index) => {
                  const pos = polarToSvg(user.azimuthDeg, user.distanceM, 8);
                  const letter = String.fromCharCode(65 + (index % 26));
                  return (
                    <g key={user.id}>
                      <circle cx={pos.x} cy={pos.y} r={4.5} fill="#e0f2fe" stroke="#0284c7" strokeWidth="1.5" />
                      <text x={pos.x} y={pos.y + 3} textAnchor="middle" fill="#0369a1" fontSize="7.5" fontFamily="JetBrains Mono" fontWeight="bold">
                        {letter}
                      </text>
                    </g>
                  );
                })}

                {/* Flock marker, live bearing/range, heading vector */}
                <g transform={`translate(${flockPos.x}, ${flockPos.y})`}>
                  <g className="flock-hover">
                    <circle r={20} fill="none" stroke={flockColor} strokeWidth="1.2" strokeDasharray="3 3" opacity="0.75" />
                    <circle r={9} fill="#fff7ed" stroke={flockColor} strokeWidth="2" />
                    <line x1={-14} y1={0} x2={14} y2={0} stroke={flockColor} strokeWidth="1.5" />
                    <line x1={0} y1={-14} x2={0} y2={14} stroke={flockColor} strokeWidth="1.5" />
                    <circle r={3} fill={flockColor} />
                  </g>
                  <line x1={headingFrom.x} y1={headingFrom.y} x2={headingTo.x} y2={headingTo.y} stroke="#c2410c" strokeWidth="2" markerEnd="url(#survArrow)" />
                  <rect x={-78} y={labelBelow ? 22 : -52} width={156} height={18} rx={3} fill="#ffffff" stroke={flockColor} strokeWidth="1.2" />
                  <text x={0} y={labelBelow ? 34 : -40} textAnchor="middle" fill="#9a3412" fontSize="9" fontFamily="JetBrains Mono" fontWeight="700">
                    {flock.id} {flock.distanceM} m{offScope ? ' (off-scope)' : ''}
                  </text>
                </g>

                {/* gNodeB-042 Base Station */}
                <circle cx={RADAR.cx} cy={RADAR.cy} r={16} fill="none" stroke="#0d9488" strokeWidth="1.2" className="target-ping-1" />
                <rect x={RADAR.cx - 10} y={RADAR.cy - 12} width="20" height="22" rx="4" fill="#005951" stroke="#ffffff" strokeWidth="2" />
                <path d={`M ${RADAR.cx - 5},${RADAR.cy + 4} L ${RADAR.cx},${RADAR.cy - 6} L ${RADAR.cx + 5},${RADAR.cy + 4} Z`} fill="#ffffff" />
                <line x1={RADAR.cx} y1={RADAR.cy - 6} x2={RADAR.cx} y2={RADAR.cy + 4} stroke="#005951" strokeWidth="1.5" />
                <circle cx={RADAR.cx} cy={RADAR.cy - 7} r="1.5" fill="#34d399" />
                <text x={RADAR.cx} y={RADAR.cy + 26} textAnchor="middle" fill="#005951" fontSize="10" fontFamily="JetBrains Mono" fontWeight="700">
                  gNodeB-042
                </text>
              </svg>

              {/* Live readout overlay */}
              <div
                style={{
                  position: 'absolute',
                  top: '10px',
                  left: '12px',
                  padding: '6px 9px',
                  borderRadius: '4px',
                  backgroundColor: 'rgba(255,255,255,0.92)',
                  border: '1px solid #e2e8f0',
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.66rem',
                  color: '#0f172a',
                  lineHeight: 1.55,
                  pointerEvents: 'none',
                }}
              >
                <div style={{ fontWeight: 800, color: '#005951' }}>LIVE TRACK</div>
                <div>{flock.id} · {flock.distanceM} m · {Math.round(flock.azimuthDeg)}° {compassPoint(flock.azimuthDeg)}</div>
                <div>BEAM #{simState.beamIndex} @ {simState.beamAzimuthDeg.toFixed(1)}° · {simState.nullActive ? 'NULL ON' : 'NULL OFF'}</div>
                <div style={{ color: '#64748b' }}>{simState.timeString} · step {simState.step}</div>
              </div>
            </div>
          </div>

          {/* Bottom live telemetry bar */}
          <div style={{ borderTop: '1px solid #f1f5f9', paddingTop: '10px', marginTop: '10px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', fontFamily: 'var(--font-mono)', color: '#64748b', flexWrap: 'wrap', gap: '8px' }}>
              <span>RANGE: <strong style={{ color: '#0f172a' }}>{flock.distanceM} m</strong>  BEARING: <strong style={{ color: '#0f172a' }}>{flock.azimuthDeg.toFixed(1)}°</strong>  ALT: <strong style={{ color: '#0f172a' }}>{flock.altitudeM} m AGL</strong></span>
              <span>PEAK: <strong style={{ color: overLimit ? '#dc2626' : '#0f172a' }}>{peak.toFixed(5)} W/m²</strong>  LIMIT: <strong style={{ color: '#0f172a' }}>{limit > 0 ? `${limit.toFixed(3)} W/m²` : '—'}</strong></span>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
              <span style={{ fontSize: '0.68rem', fontFamily: 'var(--font-mono)', color: '#94a3b8', fontWeight: 700, letterSpacing: '0.06em' }}>LIVE TARGETS:</span>
              {simState.allFlocks.map((target: FlockTarget) => {
                const isActive = simState.activeFlock.id === target.id;
                return (
                  <button
                    key={target.id}
                    onClick={() => onSelectFlock(target.id)}
                    style={{
                      padding: '4px 12px',
                      borderRadius: '9999px',
                      fontSize: '0.75rem',
                      fontFamily: 'var(--font-mono)',
                      border: isActive ? '2px solid #a5b4fc' : '1px solid #e2e8f0',
                      backgroundColor: isActive ? '#ffffff' : '#f8fafc',
                      color: isActive ? '#0f172a' : '#64748b',
                      fontWeight: isActive ? 700 : 500,
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px',
                      boxShadow: isActive ? '0 1px 3px rgba(0,0,0,0.08)' : 'none',
                    }}
                  >
                    <span style={{ width: '7px', height: '7px', borderRadius: '50%', backgroundColor: isActive ? '#ea580c' : '#0d9488' }} />
                    <span>{target.id} {target.commonName} {target.birdCount} birds</span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Right: Flock Target Inspector */}
        <div className="surveillance-card">
          <div>
            {/* Header */}
            <div className="flex-row-between">
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <h3 style={{ fontFamily: 'var(--font-sans)', fontWeight: 900, fontSize: '1.2rem', color: '#0f172a' }}>
                  FLOCK {flock.id}
                </h3>
                <span className="pill-badge orange">{flock.riskLevel}</span>
              </div>
              <div style={{ color: '#94a3b8', cursor: 'pointer' }}>
                <Lock size={15} />
              </div>
            </div>
            <div style={{ fontSize: '0.78rem', color: '#94a3b8', fontFamily: 'var(--font-sans)', fontStyle: 'italic', marginBottom: '14px' }}>
              {flock.species} / {flock.commonName}
            </div>

            {/* 2x2 Metric Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', marginBottom: '14px' }}>
              <div style={{ padding: '10px', backgroundColor: '#f8fafc', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
                <div style={{ fontSize: '0.65rem', fontFamily: 'var(--font-mono)', color: '#94a3b8', fontWeight: 700 }}>FLOCK COUNT</div>
                <div style={{ fontSize: '1.15rem', fontWeight: 800, color: '#0f172a', fontFamily: 'var(--font-sans)' }}>{flock.birdCount} Birds</div>
              </div>
              <div style={{ padding: '10px', backgroundColor: '#f8fafc', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
                <div style={{ fontSize: '0.65rem', fontFamily: 'var(--font-mono)', color: '#94a3b8', fontWeight: 700 }}>ALTITUDE AGL</div>
                <div style={{ fontSize: '1.15rem', fontWeight: 800, color: '#c2410c', fontFamily: 'var(--font-sans)' }}>{flock.altitudeM} m</div>
              </div>
              <div style={{ padding: '10px', backgroundColor: '#f8fafc', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
                <div style={{ fontSize: '0.65rem', fontFamily: 'var(--font-mono)', color: '#94a3b8', fontWeight: 700 }}>GROUND SPEED</div>
                <div style={{ fontSize: '1.15rem', fontWeight: 800, color: '#0f172a', fontFamily: 'var(--font-sans)' }}>{flock.speedMs.toFixed(1)} m/s</div>
              </div>
              <div style={{ padding: '10px', backgroundColor: '#f8fafc', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
                <div style={{ fontSize: '0.65rem', fontFamily: 'var(--font-mono)', color: '#94a3b8', fontWeight: 700 }}>EST. RF ENTRY</div>
                <div style={{ fontSize: '1.15rem', fontWeight: 800, color: '#c2410c', fontFamily: 'var(--font-sans)' }}>T-{flock.estEntrySec} sec</div>
              </div>
            </div>

            {/* Proximity Threshold Buffer Selector (draws the ring on the radar) */}
            <div style={{ marginBottom: '14px' }}>
              <div className="flex-row-between" style={{ fontSize: '0.68rem', fontFamily: 'var(--font-mono)', color: '#64748b', fontWeight: 700, marginBottom: '6px' }}>
                <span>RADAR RANGE BUFFER</span>
                <span style={{ color: '#c2410c' }}>{activeBuffer.toUpperCase()} MARKER</span>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '6px', fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>
                {(['500m', '300m', '100m'] as const).map((buffer) => {
                  const selected = activeBuffer === buffer;
                  return (
                    <button
                      key={buffer}
                      onClick={() => setActiveBuffer(buffer)}
                      style={{
                        padding: '6px 0',
                        borderRadius: '4px',
                        textAlign: 'center',
                        border: selected ? '1px solid #fdba74' : '1px solid #e2e8f0',
                        backgroundColor: selected ? '#fed7aa' : '#f8fafc',
                        color: selected ? '#9a3412' : '#64748b',
                        fontWeight: selected ? 800 : 600,
                      }}
                    >
                      {buffer} {buffer === '500m' ? 'PASS' : buffer === '300m' ? 'ALERT' : 'ABORT'}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Live exposure meter (replaces the invented dispersion cone) */}
            <div style={{ marginBottom: '14px' }}>
              <div className="flex-row-between" style={{ fontSize: '0.68rem', fontFamily: 'var(--font-mono)', color: '#64748b', fontWeight: 700, marginBottom: '4px' }}>
                <span>PEAK EXPOSURE / LIMIT</span>
                <span>{limit > 0 ? `${((peak / limit) * 100).toFixed(2)}%` : '—'}</span>
              </div>
              <div style={{ height: '64px', width: '100%', backgroundColor: '#f8fafc', borderRadius: '6px', border: '1px solid #e2e8f0', padding: '8px 10px', position: 'relative' }}>
                <div style={{ height: '8px', width: '100%', backgroundColor: '#e2e8f0', borderRadius: '4px', marginTop: '6px', overflow: 'hidden' }}>
                  <div
                    style={{
                      height: '100%',
                      width: `${Math.min(100, exposureRatio * 100)}%`,
                      backgroundColor: overLimit ? '#dc2626' : '#0d9488',
                      borderRadius: '4px',
                      transition: 'width 0.15s linear',
                    }}
                  />
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.62rem', fontFamily: 'var(--font-mono)', color: '#64748b', marginTop: '6px' }}>
                  <span>0 W/m²</span>
                  <span style={{ fontWeight: 800, color: overLimit ? '#dc2626' : '#005951' }}>{peak.toFixed(5)} W/m²</span>
                  <span>{limit > 0 ? `${limit.toFixed(3)} W/m²` : '—'}</span>
                </div>
              </div>
            </div>

            {/* Served links, all from the frame (BLER derived from SINR) */}
            <div style={{ borderTop: '1px solid #f1f5f9', paddingTop: '12px', display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '0.75rem', fontFamily: 'var(--font-mono)' }}>
              <div className="flex-row-between" style={{ fontSize: '0.68rem', fontWeight: 700, color: '#64748b' }}>
                <span>SERVED LINKS</span>
                <span style={{ color: '#005951' }}>{simState.users.length} UEs</span>
              </div>
              <div className="flex-row-between" style={{ fontSize: '0.72rem', color: '#475569' }}>
                <span>Mean SINR</span>
                <span style={{ fontWeight: 800, color: '#0f172a' }}>{simState.meanSinrDb.toFixed(1)} dB</span>
              </div>
              <div className="flex-row-between" style={{ fontSize: '0.72rem', color: '#475569' }}>
                <span>Mean BLER (derived)</span>
                <span style={{ fontWeight: 800, color: '#0f172a' }}>{simState.meanBlerPercent.toFixed(2)}%</span>
              </div>
              <div className="flex-row-between" style={{ fontSize: '0.72rem', color: '#475569' }}>
                <span>Cell throughput</span>
                <span style={{ fontWeight: 800, color: '#0f172a' }}>{simState.cellThroughputGbps.toFixed(2)} Gbps</span>
              </div>
            </div>
          </div>

          {/* Null vector status (replaces the button that transmitted nothing) */}
          <div style={{ marginTop: '16px', padding: '10px', borderRadius: '6px', backgroundColor: simState.nullActive ? '#c7f9ea' : '#f8fafc', border: `1px solid ${simState.nullActive ? '#99f6e4' : '#e2e8f0'}` }}>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.68rem', fontWeight: 700, color: simState.nullActive ? '#005951' : '#64748b', letterSpacing: '0.04em' }}>
              NULL VECTOR {simState.nullActive ? 'ENGAGED' : 'NOT ENGAGED'}
            </div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.72rem', color: '#334155', marginTop: '2px' }}>
              {simState.nullActive
                ? `${simState.nullDepthDb.toFixed(2)} dB toward ${flock.azimuthDeg.toFixed(1)}° · controller ${simState.engine}`
                : `Controller ${simState.engine || '—'} is serving without a notch this step`}
            </div>
          </div>
        </div>
      </div>

      {/* 4. Bottom Row: Live telemetry (Left) & Classifier Event Log (Right) */}
      <div className="surveillance-bottom-grid">
        {/* Live telemetry replaces the invented environmental panel */}
        <div className="surveillance-card">
          <div>
            <div className="flex-row-between" style={{ marginBottom: '10px' }}>
              <span style={{ fontFamily: 'var(--font-sans)', fontWeight: 800, fontSize: '0.78rem', color: '#0f172a', letterSpacing: '0.04em' }}>
                LIVE RF &amp; TRACK TELEMETRY
              </span>
              <span style={{ fontSize: '0.68rem', color: '#005951', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>● FRAME {simState.step}</span>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', fontSize: '0.75rem', fontFamily: 'var(--font-mono)' }}>
              <div>
                <div style={{ color: '#94a3b8', fontSize: '0.62rem', fontWeight: 700 }}>SERVING BEAM</div>
                <div style={{ fontWeight: 800, color: '#0f172a', fontSize: '0.9rem' }}>#{simState.beamIndex} @ {simState.beamAzimuthDeg.toFixed(1)}°</div>
                <div style={{ fontSize: '0.65rem', color: '#64748b' }}>{simState.beamGainDbi.toFixed(1)} dBi · {simState.beamZenithDeg.toFixed(0)}° elevation</div>
              </div>

              <div>
                <div style={{ color: '#94a3b8', fontSize: '0.62rem', fontWeight: 700 }}>NULL DEPTH</div>
                <div style={{ fontWeight: 800, color: simState.nullActive ? '#005951' : '#0f172a', fontSize: '0.9rem' }}>{simState.nullActive ? `${simState.nullDepthDb.toFixed(2)} dB` : 'OFF'}</div>
                <div style={{ fontSize: '0.65rem', color: '#64748b' }}>{simState.nullActive ? 'peak-cell projection' : 'no notch on air'}</div>
              </div>

              <div>
                <div style={{ color: '#94a3b8', fontSize: '0.62rem', fontWeight: 700 }}>PEAK DENSITY</div>
                <div style={{ fontWeight: 800, color: overLimit ? '#dc2626' : '#0f172a', fontSize: '0.9rem' }}>{peak.toFixed(5)} W/m²</div>
                <div style={{ fontSize: '0.65rem', color: '#64748b' }}>{overLimit ? 'over the limit' : 'under the limit'}</div>
              </div>

              <div>
                <div style={{ color: '#94a3b8', fontSize: '0.62rem', fontWeight: 700 }}>CELL THROUGHPUT</div>
                <div style={{ fontWeight: 800, color: '#0f172a', fontSize: '0.9rem' }}>{simState.cellThroughputGbps.toFixed(2)} Gbps</div>
                <div style={{ fontSize: '0.65rem', color: '#64748b' }}>{simState.users.length} served UEs</div>
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.68rem', fontFamily: 'var(--font-mono)', color: '#64748b', borderTop: '1px solid #f1f5f9', paddingTop: '8px', marginTop: '8px' }}>
            <span>CONTROLLER: <strong style={{ color: '#0f172a' }}>{simState.engine || '—'}</strong></span>
            <span>LATENCY: <strong style={{ color: '#0f172a' }}>{simState.inferenceLatencyMs.toFixed(2)} ms</strong></span>
            <span>W: <strong style={{ color: '#0f172a' }}>{simState.tradeoffWeight.toFixed(2)}</strong></span>
          </div>
        </div>

        {/* Radar & Classifier Event Log */}
        <div className="surveillance-card">
          <div>
            <div className="flex-row-between" style={{ marginBottom: '8px' }}>
              <span style={{ fontFamily: 'var(--font-sans)', fontWeight: 800, fontSize: '0.78rem', color: '#0f172a', letterSpacing: '0.04em' }}>
                RADAR &amp; CLASSIFIER EVENT LOG
              </span>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', fontFamily: 'var(--font-mono)', fontSize: '0.72rem' }}>
                  <button
                    onClick={() => setFilterType('ALL')}
                    style={{ fontWeight: filterType === 'ALL' ? 800 : 500, color: filterType === 'ALL' ? '#0f172a' : '#94a3b8', borderBottom: filterType === 'ALL' ? '2px solid #0f172a' : 'none', paddingBottom: '2px' }}
                  >
                    All Events
                  </button>
                  <button
                    onClick={() => setFilterType('NULL')}
                    style={{ fontWeight: filterType === 'NULL' ? 800 : 500, color: filterType === 'NULL' ? '#0f172a' : '#94a3b8', borderBottom: filterType === 'NULL' ? '2px solid #0f172a' : 'none', paddingBottom: '2px' }}
                  >
                    Null Steers
                  </button>
                  <button
                    onClick={() => setFilterType('ENTRY')}
                    style={{ fontWeight: filterType === 'ENTRY' ? 800 : 500, color: filterType === 'ENTRY' ? '#0f172a' : '#94a3b8', borderBottom: filterType === 'ENTRY' ? '2px solid #0f172a' : 'none', paddingBottom: '2px' }}
                  >
                    Target Enters
                  </button>
                </div>
                <span style={{ fontSize: '0.68rem', fontFamily: 'var(--font-mono)', color: '#94a3b8' }}>BUFFER: {simState.radarEvents.length} RECS</span>
              </div>
            </div>

            <div style={{ overflowX: 'auto' }}>
              <table className="event-table">
                <thead>
                  <tr>
                    <th>TIMESTAMP</th>
                    <th>SENSOR SOURCE</th>
                    <th>EVENT TYPE</th>
                    <th>DETAILS / TARGET</th>
                    <th>ACTION TRIGGERED</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredEvents.length === 0 ? (
                    <tr>
                      <td colSpan={5} style={{ color: '#94a3b8' }}>No events yet — they are appended as the live frames change state.</td>
                    </tr>
                  ) : (
                    filteredEvents.map((ev: RadarEventEntry) => (
                      <tr key={ev.id}>
                        <td style={{ color: '#64748b' }}>{ev.timestamp}</td>
                        <td style={{ fontWeight: 600, color: '#334155' }}>{ev.sensorSource}</td>
                        <td>
                          <span className={`pill-badge ${ev.eventType === 'CORRIDOR_ENTRY' ? 'orange' : ev.eventType === 'SPECIES_CONFIRM' ? 'mint' : 'purple'}`}>
                            {ev.eventType}
                          </span>
                        </td>
                        <td style={{ color: '#334155' }}>{ev.details}</td>
                        <td style={{ fontWeight: 700, color: '#005951' }}>{ev.actionTriggered}</td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
