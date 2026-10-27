import React, { useState } from 'react';
import { Grid3X3, Activity, Radio, Cpu } from 'lucide-react';
import { SimState } from '../../types';

interface PhasedArrayGridProps {
  simState: SimState;
}

type GridViewMode = 'phase' | 'power';

const N_ROWS = 8;
const N_COLS = 8;
/** Half-wavelength spacing, matching `rf.array.DEFAULT_D_LAMBDA`. */
const D_LAMBDA = 0.5;
const ELEMENTS = N_ROWS * N_COLS;

function wrap180(deg: number): number {
  const wrapped = (((deg + 180) % 360) + 360) % 360 - 180;
  return wrapped === -180 ? 180 : wrapped;
}

/**
 * Transmit phase of one element of the serving beam.
 *
 * Mirrors the model's steering vector (``rf/array.py``): with rows on the
 * elevation axis (m) and columns on the azimuth axis (n), the propagation
 * phase is ``2*pi*d*(n*cos(theta)*sin(phi) + m*sin(theta))``. In degrees with
 * ``d = 0.5`` that is ``180*(n*cos(theta)*sin(phi) + m*sin(theta))``. A global
 * phase offset is physically irrelevant, so it is dropped.
 */
function elementPhaseDeg(
  row: number,
  col: number,
  elevationDeg: number,
  azimuthDeg: number,
): number {
  const theta = (elevationDeg * Math.PI) / 180;
  const phi = (azimuthDeg * Math.PI) / 180;
  const vAzimuth = Math.cos(theta) * Math.sin(phi);
  const vElevation = Math.sin(theta);
  return wrap180(360 * D_LAMBDA * (col * vAzimuth + row * vElevation));
}

function compassPoint(azimuthDeg: number): string {
  const names = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE'];
  const index = Math.round((((azimuthDeg % 360) + 360) % 360) / 45) % 8;
  return names[index];
}

export const PhasedArrayGrid: React.FC<PhasedArrayGridProps> = ({ simState }) => {
  const [viewMode, setViewMode] = useState<GridViewMode>('phase');
  const [selectedCell, setSelectedCell] = useState<{ r: number; c: number }>({ r: 3, c: 4 });

  const azimuthDeg = simState.beam?.azimuthDeg ?? simState.beamAzimuthDeg;
  const elevationDeg = simState.beam?.zenithDeg ?? simState.beamZenithDeg;
  const beamGainDbi = simState.beam?.gainDbi ?? simState.beamGainDbi;
  const powerMw = simState.txRfPowerW * 1000;
  const eirpDbm =
    powerMw > 0 && beamGainDbi > 0 ? 10 * Math.log10(powerMw) + beamGainDbi : Number.NaN;
  const elementPowerDbm = powerMw > 0 ? 10 * Math.log10(powerMw / ELEMENTS) : Number.NaN;
  const flockAzimuthDeg = simState.activeFlock.azimuthDeg;

  // 8x8 element matrix, derived from the live beam angles.
  const gridData = Array.from({ length: N_ROWS }, (_, r) =>
    Array.from({ length: N_COLS }, (_, c) => ({
      r,
      c,
      phase: elementPhaseDeg(r, c, elevationDeg, azimuthDeg),
    })),
  );

  const activeElement = gridData[selectedCell.r][selectedCell.c];
  const activeElementPowerDbm = elementPowerDbm;

  const phaseIntensity = Math.abs(activeElement.phase) / 180;
  const polarX = 150 + 140 * Math.cos((azimuthDeg * Math.PI) / 180);
  const polarY = 160 - 140 * Math.sin((azimuthDeg * Math.PI) / 180);
  const nullX = 150 + 130 * Math.cos((flockAzimuthDeg * Math.PI) / 180);
  const nullY = 160 - 130 * Math.sin((flockAzimuthDeg * Math.PI) / 180);

  return (
    <div className="content-viewport">
      {/* Subheader Banner */}
      <div className="sub-header-strip">
        <div className="sub-header-left">
          <div className="sub-header-title">
            <div className="w-5 h-5 bg-teal-700 text-white rounded flex items-center justify-center text-xs">
              <Grid3X3 size={13} />
            </div>
            <span>64T64R 3.5 GHz Planar Patch Array</span>
            <span className={`badge-pill ${simState.step > 0 ? 'teal' : 'purple'}`}>
              {simState.step > 0 ? 'LIVE' : 'AWAITING FRAME'}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-4 font-mono text-xs">
          <div className="p-1 px-2 bg-white border border-slate-200 rounded">
            <span className="text-gray-400">BAND:</span>{' '}
            <strong className="text-slate-800">3.5 GHz (n78)</strong>
          </div>
          <div className="p-1 px-2 bg-white border border-slate-200 rounded">
            <span className="text-gray-400">LIVE EIRP:</span>{' '}
            <strong className="text-slate-800">
              {Number.isFinite(eirpDbm) ? `${eirpDbm.toFixed(1)} dBm` : '—'}
            </strong>
          </div>
          <div className="p-1 px-2 bg-white border border-slate-200 rounded">
            <span className="text-gray-400">BEAM:</span>{' '}
            <strong className="text-slate-800">
              #{simState.beamIndex} @ {azimuthDeg.toFixed(1)}°
            </strong>
          </div>
          <div
            className={`p-1 px-2 rounded font-bold border ${
              simState.nullActive
                ? 'bg-teal-50 border-teal-300 text-teal-800'
                : 'bg-white border-slate-200 text-slate-500'
            }`}
          >
            NULL: {simState.nullActive ? `${simState.nullDepthDb.toFixed(2)} dB` : 'OFF'}
          </div>
        </div>
      </div>

      {/* Main Grid: 8x8 Array on Left, Polar Projection on Right */}
      <div className="phased-array-layout">
        {/* Left: 8x8 Patch Grid */}
        <div className="tech-card">
          <div className="card-title-row">
            <div>
              <div className="text-[10px] font-mono text-gray-400 uppercase">
                MOD-E04 // LIVE_STEERING_MATRIX
              </div>
              <div className="font-bold font-sans text-sm text-slate-800">
                Antenna Matrix Elements [0,0] to [7,7]
                <span className="ml-2 text-xs font-mono text-teal-700 font-semibold">
                  phase from the live frame
                </span>
              </div>
            </div>

            {/* View Mode Toggle */}
            <div className="flex bg-slate-100 p-1 rounded-md border border-slate-200 font-mono text-xs">
              <button
                onClick={() => setViewMode('phase')}
                className={`px-3 py-1 rounded ${viewMode === 'phase' ? 'bg-white shadow font-bold text-teal-800' : 'text-slate-600'}`}
              >
                Phase Shift (°)
              </button>
              <button
                onClick={() => setViewMode('power')}
                className={`px-3 py-1 rounded ${viewMode === 'power' ? 'bg-white shadow font-bold text-teal-800' : 'text-slate-600'}`}
              >
                RF Power (dBm)
              </button>
            </div>
          </div>

          {/* 8x8 Grid Canvas */}
          <div className="array-grid-container">
            {gridData.flat().map((elem) => {
              const isSelected = selectedCell.r === elem.r && selectedCell.c === elem.c;
              const intensity = Math.min(1, Math.abs(elem.phase) / 180);

              let bg = '#f1f5f9';
              let textColor = '#334155';
              let valText = `${Math.round(elem.phase)}°`;

              if (viewMode === 'phase') {
                bg =
                  elem.phase >= 0
                    ? `hsl(174, 72%, ${92 - 42 * intensity}%)`
                    : `hsl(202, 58%, ${92 - 38 * intensity}%)`;
                textColor = intensity > 0.55 ? '#ffffff' : '#0f172a';
                valText = `${Math.round(elem.phase)}°`;
              } else {
                bg = '#e0f2fe';
                textColor = '#0369a1';
                valText = Number.isFinite(activeElementPowerDbm)
                  ? `${activeElementPowerDbm.toFixed(1)}`
                  : '—';
              }

              return (
                <div
                  key={`${elem.r}-${elem.c}`}
                  onClick={() => setSelectedCell({ r: elem.r, c: elem.c })}
                  className={`array-element-cell ${isSelected ? 'selected' : ''}`}
                  style={{ backgroundColor: bg, color: textColor }}
                  title={`Element [${elem.r},${elem.c}] · phase ${elem.phase.toFixed(1)}°`}
                >
                  <span className="cell-coord">[{elem.r},{elem.c}]</span>
                  <span className="cell-val">{valText}</span>
                </div>
              );
            })}
          </div>

          {/* Bottom Element Inspector */}
          <div className="flex items-center justify-between border-t border-slate-100 pt-3 mt-2 font-mono text-xs">
            <div className="flex items-center gap-4">
              <span className="text-gray-400">-180°</span>
              <div className="w-32 h-2 bg-gradient-to-r from-slate-400 via-teal-400 to-teal-800 rounded" />
              <span className="text-teal-800 font-bold">+180°</span>
              <span className="text-gray-400 text-[10px]">6-Bit DAC (5.625°)</span>
            </div>

            <div className="bg-slate-50 border border-slate-200 px-3 py-1.5 rounded flex items-center gap-4">
              <span className="text-slate-700 font-bold">
                INSPECTING ELEMENT <strong>EL [{activeElement.r}, {activeElement.c}]</strong>
              </span>
              <span>
                Phase: <strong>{activeElement.phase.toFixed(1)}°</strong>
              </span>
              <span>
                Power:{' '}
                <strong>
                  {Number.isFinite(activeElementPowerDbm)
                    ? `${activeElementPowerDbm.toFixed(1)} dBm`
                    : '—'}
                </strong>
              </span>
              {phaseIntensity > 0.66 && (
                <span className="badge-pill teal text-[10px]">Deep Steering Phase</span>
              )}
            </div>
          </div>
        </div>

        {/* Right: Polar Radiation Diagram */}
        <div className="tech-card flex flex-col justify-between gap-4">
          <div>
            <div className="card-title-row">
              <div>
                <div className="text-[10px] font-mono text-gray-400">DIAG-3D // LIVE_SERVING_BEAM</div>
                <div className="font-bold font-sans text-sm text-slate-800">
                  Far-Field Radiation Polar Projection
                </div>
              </div>
              <span className="badge-pill teal font-mono">
                AZ: {azimuthDeg.toFixed(1)}° {compassPoint(azimuthDeg)}
              </span>
            </div>

            {/* Polar Diagram Canvas */}
            <div className="h-56 w-full bg-slate-50 border border-slate-200 rounded-lg p-2 flex items-center justify-center relative">
              <svg viewBox="0 0 300 200" className="w-full h-full">
                <g transform="translate(150, 160)">
                  {/* Grid Rings */}
                  <circle cx="0" cy="0" r="40" fill="none" stroke="#e2e8f0" strokeDasharray="2 2" />
                  <circle cx="0" cy="0" r="80" fill="none" stroke="#e2e8f0" strokeDasharray="2 2" />
                  <circle cx="0" cy="0" r="120" fill="none" stroke="#cbd5e1" strokeWidth="1" />

                  <text x="125" y="-5" fill="#94a3b8" fontSize="8" fontFamily="JetBrains Mono">0 dB</text>
                  <text x="85" y="-5" fill="#94a3b8" fontSize="8" fontFamily="JetBrains Mono">-10 dB</text>
                  <text x="45" y="-5" fill="#94a3b8" fontSize="8" fontFamily="JetBrains Mono">-20 dB</text>

                  {/* Serving lobe, rotated to the live frame's azimuth. SVG rotation
                      is clockwise, while the array convention is counter-clockwise. */}
                  <g transform={`rotate(${-azimuthDeg})`}>
                    <path
                      d="M 0,0 C 40,-34 100,-46 138,-12 C 146,-3 142,6 126,13 C 92,27 40,22 0,0 Z"
                      fill="rgba(13, 148, 136, 0.2)"
                      stroke="#0d9488"
                      strokeWidth="1.5"
                    />
                    <line x1="0" y1="0" x2="138" y2="0" stroke="#0d9488" strokeWidth="1" strokeDasharray="2 3" />
                  </g>

                  {/* Ecological null direction, from the live flock bearing */}
                  {simState.nullActive && (
                    <line
                      x1="0"
                      y1="0"
                      x2={nullX - 150}
                      y2={nullY - 160}
                      stroke="#0f766e"
                      strokeWidth="1.5"
                      strokeDasharray="4 3"
                    />
                  )}

                  {/* Sector span markers */}
                  <line x1="0" y1="0" x2={120 * Math.cos(-Math.PI / 3)} y2={-120 * Math.sin(-Math.PI / 3)} stroke="#cbd5e1" strokeWidth="1" />
                  <line x1="0" y1="0" x2={120 * Math.cos(Math.PI / 3)} y2={-120 * Math.sin(Math.PI / 3)} stroke="#cbd5e1" strokeWidth="1" />

                  <circle cx="0" cy="0" r="3" fill="#005951" />
                  <text x="-18" y="18" fill="#005951" fontSize="8" fontFamily="JetBrains Mono">gNB</text>
                </g>

                <text x={Math.min(280, Math.max(10, polarX - 14))} y={Math.max(12, Math.min(196, polarY - 8))} fill="#005951" fontSize="9" fontFamily="JetBrains Mono" fontWeight="700">
                  BEAM #{simState.beamIndex}
                </text>
                {simState.nullActive && (
                  <text x={Math.min(270, Math.max(6, nullX - 18))} y={Math.max(12, Math.min(196, nullY + 14))} fill="#0f766e" fontSize="8.5" fontFamily="JetBrains Mono" fontWeight="700">
                    NULL {simState.nullDepthDb.toFixed(2)} dB
                  </text>
                )}
              </svg>
            </div>

            {/* Live Polar Diagnostics */}
            <div className="grid grid-cols-2 gap-3 mt-3 font-mono text-xs">
              <div className="p-2 bg-slate-50 border border-slate-200 rounded">
                <div className="text-[10px] text-gray-400">SERVING BEAM GAIN</div>
                <div className="font-bold text-slate-800">{beamGainDbi.toFixed(1)} dBi</div>
                <div className="text-[10px] text-emerald-600">from frame.beam.gainDbi</div>
              </div>
              <div className="p-2 bg-slate-50 border border-slate-200 rounded">
                <div className="text-[10px] text-gray-400">ELEVATION</div>
                <div className="font-bold text-slate-800">{elevationDeg.toFixed(0)}° above horizon</div>
                <div className="text-[10px] text-gray-500">codebook tilt</div>
              </div>
            </div>
          </div>

          {/* Null / calibration box (live, no invented hardware telemetry) */}
          <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <div>
                <div className="text-[10px] font-mono text-gray-400">NULL_PROJECTION // LIVE_STATE</div>
                <div className="text-xs font-bold font-sans text-slate-800">
                  {simState.nullActive ? 'Projection null on air' : 'Serving without a null'}
                </div>
              </div>
              <span className={`badge-pill ${simState.nullActive ? 'teal' : 'blue'} font-mono`}>
                {simState.nullActive ? `${simState.nullDepthDb.toFixed(2)} dB` : 'OFF'}
              </span>
            </div>
            <div className="flex justify-between font-mono text-[11px] text-gray-600 flex-wrap gap-2">
              <span>
                Flock bearing: <strong>{flockAzimuthDeg.toFixed(1)}°</strong>
              </span>
              <span>
                TX power: <strong>{simState.txRfPowerW.toFixed(1)} W</strong>
              </span>
              <span className="text-teal-800 font-bold">{simState.engine || '—'}</span>
            </div>
            <div className="mt-2 text-[10px] font-mono text-gray-500 leading-relaxed">
              Element phases are computed from the live beam angles with the model's own
              steering law (8×8 UPA, 0.5λ). A projection null modifies the transmitted
              weights, which the schema-3 frame does not carry, so the matrix shows the
              serving-beam steering.
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Section: Live RF Diagnostics & Decision Log */}
      <div className="tech-card">
        <div className="card-title-row">
          <span className="card-title">
            <Activity size={14} className="text-teal-600" />
            <span>LIVE RF FRONTEND TELEMETRY</span>
          </span>
          <div className="flex items-center gap-3 font-mono text-xs text-gray-500">
            <span>
              Frame <strong>{simState.step}</strong>
            </span>
            <span className="badge-pill teal">{simState.timeString}</span>
          </div>
        </div>

        {/* 4 Stat Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3 mb-3">
          <div className="p-3 bg-slate-50 border border-slate-200 rounded flex items-center gap-3">
            <div className="w-8 h-8 rounded bg-teal-100 text-teal-800 flex items-center justify-center">
              <Cpu size={16} />
            </div>
            <div>
              <div className="text-[10px] font-mono text-gray-400">BEAM GAIN</div>
              <div className="text-base font-bold text-slate-800 font-sans">{beamGainDbi.toFixed(1)} dBi</div>
              <div className="text-[10px] text-gray-500 font-mono">beam #{simState.beamIndex}</div>
            </div>
          </div>

          <div className="p-3 bg-slate-50 border border-slate-200 rounded flex items-center gap-3">
            <div className="w-8 h-8 rounded bg-blue-100 text-blue-800 flex items-center justify-center">
              <Radio size={16} />
            </div>
            <div>
              <div className="text-[10px] font-mono text-gray-400">TX POWER</div>
              <div className="text-base font-bold text-slate-800 font-sans">{simState.txRfPowerW.toFixed(1)} W</div>
              <div className="text-[10px] text-emerald-600 font-mono">
                EIRP {Number.isFinite(eirpDbm) ? `${eirpDbm.toFixed(1)} dBm` : '—'}
              </div>
            </div>
          </div>

          <div className="p-3 bg-slate-50 border border-slate-200 rounded flex items-center gap-3">
            <div className="w-8 h-8 rounded bg-slate-100 text-slate-800 flex items-center justify-center">
              <Activity size={16} />
            </div>
            <div>
              <div className="text-[10px] font-mono text-gray-400">DECISION LATENCY</div>
              <div className="text-base font-bold text-slate-800 font-sans">
                {simState.inferenceLatencyMs.toFixed(2)} ms
              </div>
              <div className="text-[10px] text-gray-500 font-mono">server-measured</div>
            </div>
          </div>

          <div className="p-3 bg-emerald-50 border border-emerald-200 rounded flex items-center gap-3">
            <div className="w-8 h-8 rounded bg-emerald-100 text-emerald-800 flex items-center justify-center">
              <Radio size={16} />
            </div>
            <div>
              <div className="text-[10px] font-mono text-emerald-700">SUM THROUGHPUT</div>
              <div className="text-base font-bold text-emerald-900 font-sans">
                {simState.cellThroughputGbps.toFixed(2)} Gbps
              </div>
              <div className="text-[10px] text-emerald-700 font-mono">{simState.users.length} UEs</div>
            </div>
          </div>
        </div>

        {/* Live decision log */}
        <div className="overflow-x-auto">
          <table className="event-table">
            <thead>
              <tr>
                <th>TIMESTAMP</th>
                <th>ACTION</th>
                <th>WHAT THE CONTROLLER DID</th>
                <th>DETAILS</th>
              </tr>
            </thead>
            <tbody>
              {simState.decisionLogs.length === 0 ? (
                <tr>
                  <td colSpan={4} style={{ color: '#94a3b8' }}>
                    No decisions yet — entries appear as the live frames change beam, null or power state.
                  </td>
                </tr>
              ) : (
                simState.decisionLogs.slice(0, 4).map((entry) => (
                  <tr key={entry.id}>
                    <td style={{ color: '#64748b' }}>{entry.timestamp}</td>
                    <td className="font-semibold text-teal-800">{entry.actionType}</td>
                    <td>{entry.description}</td>
                    <td style={{ color: '#475569' }}>{entry.details}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
