import React, { useState } from 'react';
import { Lock, Radio, Compass, Cloud, Activity, Layers, CheckCircle2 } from 'lucide-react';
import { SimState, FlockTarget, RadarEventEntry } from '../../types';

interface FlywaySurveillanceProps {
  simState: SimState;
  onSelectFlock: (id: string) => void;
}

export const FlywaySurveillance: React.FC<FlywaySurveillanceProps> = ({
  simState,
  onSelectFlock,
}) => {
  const [activeLayers, setActiveLayers] = useState({
    lidar: false,
    thermal: false,
    rfLobes: true,
  });

  const [activeBuffer, setActiveBuffer] = useState<'500m' | '300m' | '100m'>('300m');
  const [filterType, setFilterType] = useState<'ALL' | 'NULL' | 'ENTRY'>('ALL');
  const [transmittedNotice, setTransmittedNotice] = useState(false);

  const handleTransmit = () => {
    setTransmittedNotice(true);
    setTimeout(() => setTransmittedNotice(false), 3500);
  };

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
            <span>KU-BAND PULSE-DOPPLER ACTIVE</span>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <button
            onClick={() => setActiveLayers(p => ({ ...p, lidar: !p.lidar }))}
            style={{
              padding: '4px 14px',
              borderRadius: '9999px',
              border: activeLayers.lidar ? '1px solid #005951' : '1px solid #cbd5e1',
              backgroundColor: activeLayers.lidar ? '#005951' : '#ffffff',
              color: activeLayers.lidar ? '#ffffff' : '#334155',
              fontSize: '0.75rem',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
            }}
          >
            <Cloud size={13} />
            <span>LiDAR Cloud</span>
          </button>

          <button
            onClick={() => setActiveLayers(p => ({ ...p, thermal: !p.thermal }))}
            style={{
              padding: '4px 14px',
              borderRadius: '9999px',
              border: activeLayers.thermal ? '1px solid #005951' : '1px solid #cbd5e1',
              backgroundColor: activeLayers.thermal ? '#005951' : '#ffffff',
              color: activeLayers.thermal ? '#ffffff' : '#334155',
              fontSize: '0.75rem',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
            }}
          >
            <Activity size={13} />
            <span>Thermal FLIR</span>
          </button>

          <button
            onClick={() => setActiveLayers(p => ({ ...p, rfLobes: !p.rfLobes }))}
            style={{
              padding: '4px 16px',
              borderRadius: '9999px',
              border: '1px solid #005951',
              backgroundColor: activeLayers.rfLobes ? '#005951' : '#ffffff',
              color: activeLayers.rfLobes ? '#ffffff' : '#005951',
              fontSize: '0.75rem',
              fontWeight: 700,
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              boxShadow: '0 1px 2px rgba(0,0,0,0.1)',
            }}
          >
            <Layers size={13} />
            <span>5G RF Lobes</span>
          </button>
        </div>
      </div>

      {/* 2. Top 4 Stat Cards Row */}
      <div className="surveillance-top-grid">
        {/* Card 1: Active Tracked Flocks */}
        <div className="surveillance-card">
          <div className="flex-row-between">
            <span style={{ fontSize: '0.68rem', fontFamily: 'var(--font-mono)', color: '#94a3b8', fontWeight: 700, letterSpacing: '0.06em' }}>
              ACTIVE TRACKED FLOCKS
            </span>
            <span className="pill-badge mint">RADAR 3D</span>
          </div>
          <div style={{ margin: '6px 0', display: 'flex', alignItems: 'baseline', gap: '8px' }}>
            <span style={{ fontSize: '1.9rem', fontWeight: 800, fontFamily: 'var(--font-sans)', color: '#0f172a', lineHeight: 1 }}>14</span>
            <span style={{ fontSize: '1.05rem', fontWeight: 700, color: '#1e293b' }}>Flocks</span>
            <span style={{ fontSize: '0.75rem', color: '#94a3b8', fontFamily: 'var(--font-mono)' }}>1,420 Birds</span>
          </div>
          <div style={{ fontSize: '0.75rem', color: '#005951', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
            ↗ +2 flocks entry (Sector A)
          </div>
        </div>

        {/* Card 2: Airspace Safety Index */}
        <div className="surveillance-card">
          <div className="flex-row-between">
            <span style={{ fontSize: '0.68rem', fontFamily: 'var(--font-mono)', color: '#94a3b8', fontWeight: 700, letterSpacing: '0.06em' }}>
              AIRSPACE SAFETY INDEX
            </span>
            <span className="pill-badge mint">
              <span style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: '#0d9488' }} />
              <span>NOMINAL</span>
            </span>
          </div>
          <div style={{ margin: '6px 0', display: 'flex', alignItems: 'baseline', gap: '12px' }}>
            <span style={{ fontSize: '1.9rem', fontWeight: 800, fontFamily: 'var(--font-sans)', color: '#005951', lineHeight: 1 }}>99.4%</span>
            <span style={{ fontSize: '0.82rem', fontFamily: 'var(--font-mono)', color: '#1e293b', fontWeight: 600 }}>RF Buffer Clear</span>
          </div>
          <div style={{ height: '3px', width: '100%', backgroundColor: '#0d9488', borderRadius: '2px' }} />
        </div>

        {/* Card 3: Mean Migration Vector */}
        <div className="surveillance-card">
          <div className="flex-row-between">
            <span style={{ fontSize: '0.68rem', fontFamily: 'var(--font-mono)', color: '#94a3b8', fontWeight: 700, letterSpacing: '0.06em' }}>
              MEAN MIGRATION VECTOR
            </span>
            <Compass size={14} style={{ color: '#94a3b8' }} />
          </div>
          <div style={{ margin: '6px 0', display: 'flex', alignItems: 'baseline', gap: '6px' }}>
            <span style={{ fontSize: '1.9rem', fontWeight: 800, fontFamily: 'var(--font-sans)', color: '#0f172a', lineHeight: 1 }}>048°</span>
            <span style={{ fontSize: '1rem', fontWeight: 700, color: '#1e293b' }}>NNE</span>
            <span style={{ fontSize: '0.78rem', color: '#64748b', fontFamily: 'var(--font-mono)' }}>@ 14.2 m/s</span>
          </div>
          <div style={{ fontSize: '0.72rem', color: '#64748b', fontFamily: 'var(--font-mono)', display: 'flex', justifyContent: 'space-between' }}>
            <span>ALT AGL: 110-340m</span>
            <span style={{ color: '#005951', fontWeight: 600 }}>Steady Flow</span>
          </div>
        </div>

        {/* Card 4: Sensor Fusion Health */}
        <div className="surveillance-card">
          <div className="flex-row-between">
            <span style={{ fontSize: '0.68rem', fontFamily: 'var(--font-mono)', color: '#94a3b8', fontWeight: 700, letterSpacing: '0.06em' }}>
              SENSOR FUSION HEALTH
            </span>
            <span style={{ fontSize: '0.72rem', fontFamily: 'var(--font-mono)', color: '#005951', fontWeight: 700 }}>ALL ONLINE</span>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '6px', margin: '4px 0', textAlign: 'center', fontFamily: 'var(--font-mono)' }}>
            <div style={{ backgroundColor: '#f1f5f9', borderRadius: '4px', padding: '4px 2px' }}>
              <div style={{ fontSize: '0.62rem', color: '#64748b' }}>LiDAR</div>
              <div style={{ fontSize: '0.78rem', fontWeight: 700, color: '#0f172a' }}>100%</div>
            </div>
            <div style={{ backgroundColor: '#f1f5f9', borderRadius: '4px', padding: '4px 2px' }}>
              <div style={{ fontSize: '0.62rem', color: '#64748b' }}>Ku-Band</div>
              <div style={{ fontSize: '0.78rem', fontWeight: 700, color: '#0f172a' }}>99.8%</div>
            </div>
            <div style={{ backgroundColor: '#f1f5f9', borderRadius: '4px', padding: '4px 2px' }}>
              <div style={{ fontSize: '0.62rem', color: '#64748b' }}>Thermal</div>
              <div style={{ fontSize: '0.78rem', fontWeight: 700, color: '#0f172a' }}>100%</div>
            </div>
          </div>
          <div style={{ fontSize: '0.68rem', color: '#94a3b8', fontFamily: 'var(--font-mono)', display: 'flex', justifyContent: 'space-between' }}>
            <span>Latency: 4.2ms</span>
            <span>Kalman Filter v4.1</span>
          </div>
        </div>
      </div>

      {/* 3. Main Row: Radar Scope (Left) & Flock Target Detail (Right) */}
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
                <span style={{ display: 'flex', alignItems: 'center', gap: '4px', color: '#dc2626', fontWeight: 700, fontSize: '0.68rem' }}>
                  <span style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: '#dc2626' }} />
                  <span>Critical Alert &lt;180m</span>
                </span>
                <span className="pill-badge blue">Range: 6.0 km</span>
              </div>
            </div>

            {/* In-Radar Legend */}
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '20px', fontSize: '0.72rem', fontFamily: 'var(--font-mono)', color: '#475569', padding: '4px 0 8px' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#ef4444' }} />
                <span>&lt; 150m AGL (RF Risk Zone)</span>
              </span>
              <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#005951' }} />
                <span>&gt; 150m AGL (Clear Corridor)</span>
              </span>
            </div>

            {/* Radar Circular Scope Container */}
            <div className="surveillance-radar-wrapper">
              <svg viewBox="0 0 620 400" style={{ width: '100%', height: '100%' }}>
                <defs>
                  <linearGradient id="survRadialSweep" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#0d9488" stopOpacity="0.45" />
                    <stop offset="65%" stopColor="#2dd4bf" stopOpacity="0.12" />
                    <stop offset="100%" stopColor="#2dd4bf" stopOpacity="0.0" />
                  </linearGradient>
                </defs>

                {/* Scope Center at (310, 200) */}
                <g transform="translate(310, 200)">
                  {/* Concentric Distance Rings */}
                  <circle cx="0" cy="0" r="55" fill="none" stroke="#e2e8f0" strokeWidth="1" strokeDasharray="3 3" />
                  <circle cx="0" cy="0" r="110" fill="none" stroke="#e2e8f0" strokeWidth="1" strokeDasharray="3 3" />
                  <circle cx="0" cy="0" r="165" fill="none" stroke="#cbd5e1" strokeWidth="1.2" />
                  <circle cx="0" cy="0" r="225" fill="none" stroke="#e2e8f0" strokeWidth="1" />

                  {/* Cardinal Axis Lines */}
                  <line x1="0" y1="0" x2="0" y2="-185" stroke="#008075" strokeWidth="2" />
                  <text x="6" y="-165" fill="#008075" fontSize="10" fontFamily="JetBrains Mono" fontWeight="700">000° N</text>

                  <line x1="0" y1="0" x2="0" y2="185" stroke="#cbd5e1" strokeWidth="1" />
                  <text x="6" y="175" fill="#94a3b8" fontSize="9" fontFamily="JetBrains Mono">180° S</text>

                  <line x1="-260" y1="0" x2="260" y2="0" stroke="#cbd5e1" strokeWidth="1" />
                  <text x="-250" y="-6" fill="#94a3b8" fontSize="9" fontFamily="JetBrains Mono">270° W</text>
                  <text x="220" y="-6" fill="#94a3b8" fontSize="9" fontFamily="JetBrains Mono">090° E</text>

                  {/* 360-Degree Sweeping Beam */}
                  <g className="radar-sweep-line">
                    <path
                      d="M 0,0 L 225,0 A 225 225 0 0 0 205,-85 Z"
                      fill="url(#survRadialSweep)"
                    />
                    <line x1="0" y1="0" x2="225" y2="0" stroke="#0d9488" strokeWidth="2" opacity="0.8" />
                  </g>

                  {/* User Terminals A & B */}
                  <g transform="translate(-50, 40)">
                    <circle cx="0" cy="0" r="4.5" fill="#e0f2fe" stroke="#0284c7" strokeWidth="1.5" />
                    <text x="0" y="3" textAnchor="middle" fill="#0369a1" fontSize="7.5" fontFamily="JetBrains Mono" fontWeight="bold">A</text>
                    <circle cx="15" cy="-12" r="1.5" fill="#0284c7" opacity="0.7" />
                    <circle cx="30" cy="-24" r="1.5" fill="#0284c7" opacity="0.5" />
                  </g>

                  <g transform="translate(-95, 75)">
                    <circle cx="0" cy="0" r="4.5" fill="#e0f2fe" stroke="#0284c7" strokeWidth="1.5" />
                    <text x="0" y="3" textAnchor="middle" fill="#0369a1" fontSize="7.5" fontFamily="JetBrains Mono" fontWeight="bold">B</text>
                    <circle cx="15" cy="-12" r="1.5" fill="#0284c7" opacity="0.7" />
                    <circle cx="30" cy="-24" r="1.5" fill="#0284c7" opacity="0.5" />
                  </g>

                  {/* Target Flock #409 (Crane) */}
                  <g className="surveillance-flock-animated">
                    <circle cx="10" cy="-35" r="32" fill="none" stroke="#ef4444" strokeWidth="1.2" strokeDasharray="3 3" />
                    <text x="45" y="-35" fill="#ef4444" fontSize="8" fontFamily="JetBrains Mono" fontWeight="bold">
                      RNG (&lt;240m)
                    </text>

                    {/* Orange Vector Arrow */}
                    <line x1="10" y1="-35" x2="70" y2="-80" stroke="#c2410c" strokeWidth="2" />
                    <polygon points="70,-80 58,-77 62,-68" fill="#c2410c" />

                    {/* Crosshair ⨁ */}
                    <circle cx="10" cy="-35" r="9" fill="#ffedd5" stroke="#ea580c" strokeWidth="2" />
                    <line x1="1" y1="-35" x2="19" y2="-35" stroke="#ea580c" strokeWidth="1.5" />
                    <line x1="10" y1="-44" x2="10" y2="-26" stroke="#ea580c" strokeWidth="1.5" />
                    <circle cx="10" cy="-35" r="3" fill="#9a3412" />

                    {/* White Target Label Tag */}
                    <rect x="-65" y="-62" width="140" height="18" rx="3" fill="#ffffff" stroke="#ea580c" strokeWidth="1.2" />
                    <text x="5" y="-50" textAnchor="middle" fill="#9a3412" fontSize="9" fontFamily="JetBrains Mono" fontWeight="700">
                      #409 Crane 120m AGL
                    </text>
                  </g>

                  {/* gNodeB-042 Base Station Shield Tower Icon */}
                  <circle cx="0" cy="0" r="16" fill="none" stroke="#0d9488" strokeWidth="1.2" className="target-ping-1" />
                  <rect x="-10" y="-12" width="20" height="22" rx="4" fill="#005951" stroke="#ffffff" strokeWidth="2" />
                  <path d="M -5,4 L 0,-6 L 5,4 Z" fill="#ffffff" />
                  <line x1="0" y1="-6" x2="0" y2="4" stroke="#005951" strokeWidth="1.5" />
                  <circle cx="0" cy="-7" r="1.5" fill="#34d399" />
                  <text x="0" y="24" textAnchor="middle" fill="#005951" fontSize="10" fontFamily="JetBrains Mono" fontWeight="700">
                    gNodeB-042
                  </text>
                </g>
              </svg>
            </div>
          </div>

          {/* Bottom Telemetry & Target Selection Bar */}
          <div style={{ borderTop: '1px solid #f1f5f9', paddingTop: '10px', marginTop: '10px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', fontFamily: 'var(--font-mono)', color: '#64748b' }}>
              <span>RADAR FREQ: <strong style={{ color: '#0f172a' }}>13.4 GHz</strong>  PRF: <strong style={{ color: '#0f172a' }}>2,400 Hz</strong></span>
              <span>LAT 37°46'29.2"N | LON 122°25'10.4"W</span>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
              <span style={{ fontSize: '0.68rem', fontFamily: 'var(--font-mono)', color: '#94a3b8', fontWeight: 700, letterSpacing: '0.06em' }}>LIVE TARGETS:</span>
              {simState.allFlocks.map((flock: FlockTarget) => {
                const isActive = simState.activeFlock.id === flock.id;
                return (
                  <button
                    key={flock.id}
                    onClick={() => onSelectFlock(flock.id)}
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
                    <span>{flock.id} {flock.commonName} {flock.birdCount} birds</span>
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
                  FLOCK {simState.activeFlock.id}
                </h3>
                <span className="pill-badge orange">RF CRITICAL</span>
              </div>
              <div style={{ color: '#94a3b8', cursor: 'pointer' }}>
                <Lock size={15} />
              </div>
            </div>
            <div style={{ fontSize: '0.78rem', color: '#94a3b8', fontFamily: 'var(--font-sans)', fontStyle: 'italic', marginBottom: '14px' }}>
              {simState.activeFlock.species} / {simState.activeFlock.commonName}
            </div>

            {/* 2x2 Metric Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', marginBottom: '14px' }}>
              <div style={{ padding: '10px', backgroundColor: '#f8fafc', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
                <div style={{ fontSize: '0.65rem', fontFamily: 'var(--font-mono)', color: '#94a3b8', fontWeight: 700 }}>FLOCK COUNT</div>
                <div style={{ fontSize: '1.15rem', fontWeight: 800, color: '#0f172a', fontFamily: 'var(--font-sans)' }}>{simState.activeFlock.birdCount} Birds</div>
              </div>
              <div style={{ padding: '10px', backgroundColor: '#f8fafc', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
                <div style={{ fontSize: '0.65rem', fontFamily: 'var(--font-mono)', color: '#94a3b8', fontWeight: 700 }}>ALTITUDE AGL</div>
                <div style={{ fontSize: '1.15rem', fontWeight: 800, color: '#c2410c', fontFamily: 'var(--font-sans)' }}>{simState.activeFlock.altitudeM} m</div>
              </div>
              <div style={{ padding: '10px', backgroundColor: '#f8fafc', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
                <div style={{ fontSize: '0.65rem', fontFamily: 'var(--font-mono)', color: '#94a3b8', fontWeight: 700 }}>GROUND SPEED</div>
                <div style={{ fontSize: '1.15rem', fontWeight: 800, color: '#0f172a', fontFamily: 'var(--font-sans)' }}>{simState.activeFlock.speedMs} m/s</div>
              </div>
              <div style={{ padding: '10px', backgroundColor: '#f8fafc', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
                <div style={{ fontSize: '0.65rem', fontFamily: 'var(--font-mono)', color: '#94a3b8', fontWeight: 700 }}>EST. RF ENTRY</div>
                <div style={{ fontSize: '1.15rem', fontWeight: 800, color: '#c2410c', fontFamily: 'var(--font-sans)' }}>T-{simState.activeFlock.estEntrySec} sec</div>
              </div>
            </div>

            {/* Proximity Threshold Buffer Selector */}
            <div style={{ marginBottom: '14px' }}>
              <div className="flex-row-between" style={{ fontSize: '0.68rem', fontFamily: 'var(--font-mono)', color: '#64748b', fontWeight: 700, marginBottom: '6px' }}>
                <span>PROXIMITY THRESHOLD BUFFER</span>
                <span style={{ color: '#c2410c' }}>300m WARNING</span>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '6px', fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>
                <button
                  onClick={() => setActiveBuffer('500m')}
                  style={{
                    padding: '6px 0',
                    borderRadius: '4px',
                    textAlign: 'center',
                    border: activeBuffer === '500m' ? '1px solid #a7f3d0' : '1px solid #e2e8f0',
                    backgroundColor: activeBuffer === '500m' ? '#c7f9ea' : '#f8fafc',
                    color: activeBuffer === '500m' ? '#005951' : '#64748b',
                    fontWeight: activeBuffer === '500m' ? 800 : 600,
                  }}
                >
                  500m PASS
                </button>
                <button
                  onClick={() => setActiveBuffer('300m')}
                  style={{
                    padding: '6px 0',
                    borderRadius: '4px',
                    textAlign: 'center',
                    border: activeBuffer === '300m' ? '1px solid #fdba74' : '1px solid #e2e8f0',
                    backgroundColor: activeBuffer === '300m' ? '#fed7aa' : '#f8fafc',
                    color: activeBuffer === '300m' ? '#9a3412' : '#64748b',
                    fontWeight: activeBuffer === '300m' ? 800 : 600,
                  }}
                >
                  300m ALERT
                </button>
                <button
                  onClick={() => setActiveBuffer('100m')}
                  style={{
                    padding: '6px 0',
                    borderRadius: '4px',
                    textAlign: 'center',
                    border: activeBuffer === '100m' ? '1px solid #cbd5e1' : '1px solid #e2e8f0',
                    backgroundColor: activeBuffer === '100m' ? '#cbd5e1' : '#f8fafc',
                    color: activeBuffer === '100m' ? '#0f172a' : '#94a3b8',
                    fontWeight: activeBuffer === '100m' ? 800 : 600,
                  }}
                >
                  100m ABORT
                </button>
              </div>
            </div>

            {/* Trajectory Dispersion Cone Chart */}
            <div style={{ marginBottom: '14px' }}>
              <div className="flex-row-between" style={{ fontSize: '0.68rem', fontFamily: 'var(--font-mono)', color: '#64748b', fontWeight: 700, marginBottom: '4px' }}>
                <span>TRAJECTORY DISPERSION CONE (P=95%)</span>
                <span>σ = ±4.2°</span>
              </div>
              <div style={{ height: '64px', width: '100%', backgroundColor: '#f8fafc', borderRadius: '6px', border: '1px solid #e2e8f0', padding: '4px', position: 'relative' }}>
                <svg viewBox="0 0 200 48" style={{ width: '100%', height: '100%' }}>
                  <path
                    d="M 10,42 Q 60,42 80,24 Q 100,6 120,24 Q 140,42 190,42 Z"
                    fill="rgba(13, 148, 136, 0.2)"
                  />
                  <path
                    d="M 10,42 Q 60,42 80,24 Q 100,6 120,24 Q 140,42 190,42"
                    fill="none"
                    stroke="#008075"
                    strokeWidth="2"
                  />
                  <line x1="100" y1="6" x2="100" y2="44" stroke="#005951" strokeWidth="1.5" strokeDasharray="2 2" />
                  <circle cx="100" cy="6" r="2.5" fill="#005951" />
                </svg>
                <div style={{ position: 'absolute', top: '4px', left: '50%', transform: 'translateX(-50%)', fontSize: '0.58rem', fontFamily: 'var(--font-mono)', fontWeight: 800, color: '#005951' }}>
                  048° Vector
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.55rem', fontFamily: 'var(--font-mono)', color: '#94a3b8', padding: '0 8px', marginTop: '-4px' }}>
                  <span>-15° Drift</span>
                  <span>+15° Drift</span>
                </div>
              </div>
            </div>

            {/* Acoustic / Thermal Biometrics */}
            <div style={{ borderTop: '1px solid #f1f5f9', paddingTop: '12px', display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '0.75rem', fontFamily: 'var(--font-mono)' }}>
              <div className="flex-row-between" style={{ fontSize: '0.68rem', fontWeight: 700, color: '#64748b' }}>
                <span>ACOUSTIC / THERMAL BIOMETRICS</span>
                <span style={{ color: '#005951' }}>95.2% MATCH</span>
              </div>

              <div>
                <div className="flex-row-between" style={{ fontSize: '0.72rem', marginBottom: '2px' }}>
                  <span style={{ color: '#475569' }}>Wingbeat Frequency</span>
                  <span style={{ fontWeight: 800, color: '#0f172a' }}>{simState.activeFlock.wingbeatHz} Hz</span>
                </div>
                <div style={{ height: '4px', width: '100%', backgroundColor: '#f1f5f9', borderRadius: '2px', overflow: 'hidden' }}>
                  <div style={{ height: '100%', width: '85%', backgroundColor: '#005951', borderRadius: '2px' }} />
                </div>
              </div>

              <div>
                <div className="flex-row-between" style={{ fontSize: '0.72rem', marginBottom: '2px' }}>
                  <span style={{ color: '#475569' }}>V-Formation Cohesion Score</span>
                  <span style={{ fontWeight: 800, color: '#0f172a' }}>{simState.activeFlock.cohesionScore} / 1.0</span>
                </div>
                <div style={{ height: '4px', width: '100%', backgroundColor: '#f1f5f9', borderRadius: '2px', overflow: 'hidden' }}>
                  <div style={{ height: '100%', width: '94%', backgroundColor: '#005951', borderRadius: '2px' }} />
                </div>
              </div>

              <div>
                <div className="flex-row-between" style={{ fontSize: '0.72rem', marginBottom: '2px' }}>
                  <span style={{ color: '#475569' }}>Thermal Signature (Core)</span>
                  <span style={{ fontWeight: 800, color: '#0f172a' }}>{simState.activeFlock.thermalC}°C (Nominal)</span>
                </div>
                <div style={{ height: '4px', width: '100%', backgroundColor: '#f1f5f9', borderRadius: '2px', overflow: 'hidden' }}>
                  <div style={{ height: '100%', width: '75%', backgroundColor: '#005951', borderRadius: '2px' }} />
                </div>
              </div>
            </div>
          </div>

          {/* Action Transmit Button */}
          <div style={{ marginTop: '16px' }}>
            <button
              onClick={handleTransmit}
              style={{
                width: '100%',
                padding: '10px',
                borderRadius: '6px',
                fontFamily: 'var(--font-mono)',
                fontSize: '0.78rem',
                fontWeight: 700,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
                boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                backgroundColor: transmittedNotice ? '#10b981' : '#005951',
                color: '#ffffff',
                transition: 'all 0.15s ease',
              }}
            >
              {transmittedNotice ? <CheckCircle2 size={15} /> : <Radio size={15} />}
              <span>{transmittedNotice ? 'DYNAMIC NULL VECTOR SENT TO DQN' : 'Transmit Dynamic Null Vector to DQN'}</span>
            </button>
          </div>
        </div>
      </div>

      {/* 4. Bottom Row: Environmental Sensors (Left) & Classifier Event Log (Right) */}
      <div className="surveillance-bottom-grid">
        {/* Environmental Sensors */}
        <div className="surveillance-card">
          <div>
            <div className="flex-row-between" style={{ marginBottom: '10px' }}>
              <span style={{ fontFamily: 'var(--font-sans)', fontWeight: 800, fontSize: '0.78rem', color: '#0f172a', letterSpacing: '0.04em' }}>
                ENVIRONMENTAL SENSORS
              </span>
              <span style={{ fontSize: '0.68rem', color: '#005951', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>● NOAA FEED</span>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', fontSize: '0.75rem', fontFamily: 'var(--font-mono)' }}>
              <div>
                <div style={{ color: '#94a3b8', fontSize: '0.62rem', fontWeight: 700 }}>BAROMETRIC PRESSURE</div>
                <div style={{ fontWeight: 800, color: '#0f172a', fontSize: '0.9rem' }}>1014.8 hPa</div>
                <div style={{ fontSize: '0.65rem', color: '#64748b' }}>+0.2hPa/hr [Stable]</div>
              </div>

              <div>
                <div style={{ color: '#94a3b8', fontSize: '0.62rem', fontWeight: 700 }}>THERMAL UPDRAFT</div>
                <div style={{ fontWeight: 800, color: '#005951', fontSize: '0.9rem' }}>+1.8 m/s</div>
                <div style={{ fontSize: '0.65rem', color: '#64748b' }}>Optimal soaring lift</div>
              </div>

              <div>
                <div style={{ color: '#94a3b8', fontSize: '0.62rem', fontWeight: 700 }}>WIND SHEAR (0-300M)</div>
                <div style={{ fontWeight: 800, color: '#0f172a', fontSize: '0.9rem' }}>1.2 kt/100ft</div>
                <div style={{ fontSize: '0.65rem', color: '#64748b' }}>Light / Negligible</div>
              </div>

              <div>
                <div style={{ color: '#94a3b8', fontSize: '0.62rem', fontWeight: 700 }}>VISIBILITY INDEX</div>
                <div style={{ fontWeight: 800, color: '#0f172a', fontSize: '0.9rem' }}>18.4 km</div>
                <div style={{ fontSize: '0.65rem', color: '#64748b' }}>Ceiling: 2,800m</div>
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.68rem', fontFamily: 'var(--font-mono)', color: '#64748b', borderTop: '1px solid #f1f5f9', paddingTop: '8px', marginTop: '8px' }}>
            <span>RF REFRACTIVITY: <strong style={{ color: '#0f172a' }}>310 N-units</strong></span>
            <span>HUMIDITY: <strong style={{ color: '#0f172a' }}>63%</strong></span>
            <span>TEMP: <strong style={{ color: '#0f172a' }}>14.6°C</strong></span>
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
                <span style={{ fontSize: '0.68rem', fontFamily: 'var(--font-mono)', color: '#94a3b8' }}>BUFFER: 256 RECS</span>
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
                  {filteredEvents.map((ev: RadarEventEntry) => (
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
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
