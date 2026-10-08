import React, { useState } from 'react';
import { Grid3X3, Zap, Shield, RefreshCw, Activity, Cpu } from 'lucide-react';
import { SimState } from '../../types';

interface PhasedArrayGridProps {
  simState: SimState;
}

type GridViewMode = 'phase' | 'power' | 'thermal';

export const PhasedArrayGrid: React.FC<PhasedArrayGridProps> = () => {
  const [viewMode, setViewMode] = useState<GridViewMode>('phase');
  const [selectedCell, setSelectedCell] = useState<{ r: number; c: number }>({ r: 3, c: 4 });
  const [isSynthesizing, setIsSynthesizing] = useState(false);

  // Generate 8x8 matrix data
  const gridData = Array.from({ length: 8 }, (_, r) =>
    Array.from({ length: 8 }, (_, c) => {
      // Core null corridor elements around center [2..5, 3..6]
      const isNull = r >= 2 && r <= 5 && c >= 3 && c <= 6;
      
      let phase = ((r * 35 + c * 25 - 180) % 360);
      if (phase > 180) phase -= 360;
      if (phase < -180) phase += 360;
      
      if (isNull) {
        phase = -((c * 30 + r * 15 + 45) % 180);
      }

      const power = isNull ? 14.5 + (r + c) * 0.4 : 22.0 + (r + c) * 0.3;
      const thermal = 38.5 + (r * 0.6 + c * 0.4);

      return {
        r,
        c,
        isNull,
        phase: Math.round(phase),
        power: +power.toFixed(1),
        thermal: +thermal.toFixed(1),
      };
    })
  );

  const handleReSynthesize = () => {
    setIsSynthesizing(true);
    setTimeout(() => setIsSynthesizing(false), 1200);
  };

  const activeElement = gridData[selectedCell.r][selectedCell.c];

  return (
    <div className="content-viewport">
      {/* Subheader Banner */}
      <div className="sub-header-strip">
        <div className="sub-header-left">
          <div className="sub-header-title">
            <div className="w-5 h-5 bg-teal-700 text-white rounded flex items-center justify-center text-xs">
              <Grid3X3 size={13} />
            </div>
            <span>64T64R mmWave Planar Microstrip Patch Array</span>
            <span className="badge-pill teal">SYNCHRONIZED</span>
          </div>
        </div>

        <div className="flex items-center gap-4 font-mono text-xs">
          <div className="p-1 px-2 bg-white border border-slate-200 rounded">
            <span className="text-gray-400">OPERATING BAND:</span> <strong className="text-slate-800">27.5–28.35 GHz</strong>
          </div>
          <div className="p-1 px-2 bg-white border border-slate-200 rounded">
            <span className="text-gray-400">TOTAL EIRP:</span> <strong className="text-slate-800">58.0 dBm</strong>
          </div>
          <div className="p-1 px-2 bg-white border border-slate-200 rounded">
            <span className="text-gray-400">ARRAY TEMP:</span> <strong className="text-slate-800">41.2°C</strong>
          </div>
          <div className="p-1 px-2 bg-white border border-slate-200 rounded">
            <span className="text-gray-400">VSWR (AVG):</span> <strong className="text-slate-800">1.14 : 1</strong>
          </div>
          <div className="p-1 px-2 bg-teal-50 border border-teal-300 text-teal-800 rounded font-bold">
            CORRIDOR NULL: -28.4 dB @ 047°
          </div>
        </div>
      </div>

      {/* Main Grid: 8x8 Array on Left, Polar Projection on Right */}
      <div className="phased-array-layout">
        {/* Left: 8x8 Patch Grid */}
        <div className="tech-card">
          <div className="card-title-row">
            <div>
              <div className="text-[10px] font-mono text-gray-400 uppercase">MOD-E04 // BEAM_NULL_MATRIX</div>
              <div className="font-bold font-sans text-sm text-slate-800">
                Antenna Matrix Elements [0,0] to [7,7]
                <span className="ml-2 text-xs font-mono text-teal-700 font-semibold">● Active Null Corridor (047°)</span>
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
              <button
                onClick={() => setViewMode('thermal')}
                className={`px-3 py-1 rounded ${viewMode === 'thermal' ? 'bg-white shadow font-bold text-teal-800' : 'text-slate-600'}`}
              >
                Thermal (°C)
              </button>
            </div>
          </div>

          {/* 8x8 Grid Canvas */}
          <div className="array-grid-container">
            {gridData.flat().map((elem) => {
              const isSelected = selectedCell.r === elem.r && selectedCell.c === elem.c;
              
              // Color calculation based on view mode
              let bg = '#f1f5f9';
              let textColor = '#334155';
              let valText = `${elem.phase}°`;

              if (viewMode === 'phase') {
                if (elem.isNull) {
                  bg = '#0f766e';
                  textColor = '#ffffff';
                  valText = `${elem.phase}° NULL`;
                } else if (elem.phase > 0) {
                  bg = '#ccfbf1';
                  textColor = '#0f766e';
                  valText = `+${elem.phase}°`;
                } else {
                  bg = '#e2e8f0';
                  textColor = '#475569';
                  valText = `${elem.phase}°`;
                }
              } else if (viewMode === 'power') {
                bg = elem.isNull ? '#fed7aa' : '#e0f2fe';
                textColor = elem.isNull ? '#9a3412' : '#0369a1';
                valText = `${elem.power} dBm`;
              } else {
                bg = elem.thermal > 41 ? '#fee2e2' : '#fef3c7';
                textColor = elem.thermal > 41 ? '#991b1b' : '#92400e';
                valText = `${elem.thermal}°C`;
              }

              return (
                <div
                  key={`${elem.r}-${elem.c}`}
                  onClick={() => setSelectedCell({ r: elem.r, c: elem.c })}
                  className={`array-element-cell ${elem.isNull ? 'is-null-element' : ''} ${isSelected ? 'selected' : ''}`}
                  style={{ backgroundColor: bg, color: textColor }}
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
              <span className="text-gray-400 text-[10px]">Null Quantization: 6-Bit DAC (5.625°)</span>
            </div>

            <div className="bg-slate-50 border border-slate-200 px-3 py-1.5 rounded flex items-center gap-4">
              <span className="text-slate-700 font-bold">
                INSPECTING ELEMENT <strong>EL [{activeElement.r}, {activeElement.c}]</strong>
              </span>
              <span>Phase: <strong>{activeElement.phase}°</strong></span>
              <span>Power: <strong>{activeElement.power} dBm</strong></span>
              <span>Temp: <strong>{activeElement.thermal}°C</strong></span>
              {activeElement.isNull && (
                <span className="badge-pill teal text-[10px]">Phase Inverted for Nulling</span>
              )}
            </div>
          </div>
        </div>

        {/* Right: Polar Radiation Diagram & AutoCal */}
        <div className="tech-card flex flex-col justify-between gap-4">
          <div>
            <div className="card-title-row">
              <div>
                <div className="text-[10px] font-mono text-gray-400">DIAG-3D // SYNTHESIZED_BEAM</div>
                <div className="font-bold font-sans text-sm text-slate-800">Far-Field Radiation Polar Projection</div>
              </div>
              <span className="badge-pill teal font-mono">AZ: 047° NULL</span>
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

                  {/* Main beam lobe at +35° */}
                  <path
                    d="M 0,0 C 40,-60 90,-120 110,-100 C 130,-80 80,-30 0,0 Z"
                    fill="rgba(13, 148, 136, 0.2)"
                    stroke="#0d9488"
                    strokeWidth="1.5"
                  />

                  {/* Left Side lobe */}
                  <path
                    d="M 0,0 C -30,-40 -70,-80 -90,-60 C -100,-40 -50,-20 0,0 Z"
                    fill="rgba(13, 148, 136, 0.1)"
                    stroke="#2dd4bf"
                    strokeWidth="1"
                  />

                  {/* Deep Avian Null Notch at 047° */}
                  <line x1="0" y1="0" x2="105" y2="-115" stroke="#ef4444" strokeWidth="1.5" strokeDasharray="3 2" />
                  <circle cx="105" cy="-115" r="4" fill="#0f766e" />
                  <text x="115" y="-120" fill="#0f766e" fontSize="9" fontFamily="JetBrains Mono" fontWeight="700">
                    Avian Null: -28.4 dB
                  </text>
                </g>
              </svg>
            </div>

            {/* Polar Diagnostics */}
            <div className="grid grid-cols-2 gap-3 mt-3 font-mono text-xs">
              <div className="p-2 bg-slate-50 border border-slate-200 rounded">
                <div className="text-[10px] text-gray-400">PHASE JITTER RMS</div>
                <div className="font-bold text-slate-800">0.82° rms</div>
                <div className="text-[10px] text-emerald-600">Within ±1.2° Spec</div>
              </div>
              <div className="p-2 bg-slate-50 border border-slate-200 rounded">
                <div className="text-[10px] text-gray-400">SIDELOBE RATIO (SLR)</div>
                <div className="font-bold text-slate-800">-19.6 dB</div>
                <div className="text-[10px] text-gray-500">Taper: Taylor -30dB</div>
              </div>
            </div>
          </div>

          {/* AutoCal / Mutual Coupling Box */}
          <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <div>
                <div className="text-[10px] font-mono text-gray-400">AUTOCAL-S11 // MUTUAL_COUPLING</div>
                <div className="text-xs font-bold font-sans text-slate-800">Element Phase Shift Optimization</div>
              </div>
              <button
                onClick={handleReSynthesize}
                disabled={isSynthesizing}
                className="px-3 py-1.5 bg-teal-800 hover:bg-teal-700 text-white rounded font-mono text-xs font-bold flex items-center gap-1.5 shadow"
              >
                <RefreshCw size={12} className={isSynthesizing ? 'animate-spin' : ''} />
                <span>{isSynthesizing ? 'Calibrating...' : 'Re-Synthesize'}</span>
              </button>
            </div>
            <div className="flex justify-between font-mono text-[11px] text-gray-600">
              <span>Mutual Isolation: <strong>S[i,j] &lt; -24 dB</strong></span>
              <span className="text-teal-800 font-bold">64/64 CALIBRATED</span>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Section: RF Frontend Diagnostics & Log */}
      <div className="tech-card">
        <div className="card-title-row">
          <span className="card-title">
            <Activity size={14} className="text-teal-600" />
            <span>RF FRONTEND TELEMETRY &amp; DIAGNOSTICS STREAM</span>
          </span>
          <div className="flex items-center gap-3 font-mono text-xs text-gray-500">
            <span>Loop Rate: <strong>2.4 kHz</strong></span>
            <span className="badge-pill teal">0 ACTIVE FAULTS</span>
          </div>
        </div>

        {/* 4 Stat Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3 mb-3">
          <div className="p-3 bg-slate-50 border border-slate-200 rounded flex items-center gap-3">
            <div className="w-8 h-8 rounded bg-teal-100 text-teal-800 flex items-center justify-center">
              <Zap size={16} />
            </div>
            <div>
              <div className="text-[10px] font-mono text-gray-400">PA POWER EFFICIENCY</div>
              <div className="text-base font-bold text-slate-800 font-sans">38.4%</div>
              <div className="text-[10px] text-gray-500 font-mono">Doherty Class-AB nominal</div>
            </div>
          </div>

          <div className="p-3 bg-slate-50 border border-slate-200 rounded flex items-center gap-3">
            <div className="w-8 h-8 rounded bg-blue-100 text-blue-800 flex items-center justify-center">
              <Cpu size={16} />
            </div>
            <div>
              <div className="text-[10px] font-mono text-gray-400">PHASE SHIFTER DACS</div>
              <div className="text-base font-bold text-slate-800 font-sans">64/64 Locked</div>
              <div className="text-[10px] text-emerald-600 font-mono">Drift: +0.14°</div>
            </div>
          </div>

          <div className="p-3 bg-slate-50 border border-slate-200 rounded flex items-center gap-3">
            <div className="w-8 h-8 rounded bg-slate-100 text-slate-800 flex items-center justify-center">
              <Activity size={16} />
            </div>
            <div>
              <div className="text-[10px] font-mono text-gray-400">ARRAY MICRO-COOLERS</div>
              <div className="text-base font-bold text-slate-800 font-sans">3,420 RPM</div>
              <div className="text-[10px] text-gray-500 font-mono">Bearing stable</div>
            </div>
          </div>

          <div className="p-3 bg-emerald-50 border border-emerald-200 rounded flex items-center gap-3">
            <div className="w-8 h-8 rounded bg-emerald-100 text-emerald-800 flex items-center justify-center">
              <Shield size={16} />
            </div>
            <div>
              <div className="text-[10px] font-mono text-emerald-700">SAR AVIAN SAFETY</div>
              <div className="text-base font-bold text-emerald-900 font-sans">0.024 W/kg</div>
              <div className="text-[10px] text-emerald-700 font-mono">86% below threshold</div>
            </div>
          </div>
        </div>

        {/* Real-time Subsystem Log Table */}
        <div className="overflow-x-auto">
          <table className="event-table">
            <thead>
              <tr>
                <th>TIMESTAMP</th>
                <th>CHANNEL</th>
                <th>SUBSYSTEM</th>
                <th>TELEMETRY MESSAGE</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>14:28:02.109</td>
                <td className="font-semibold text-teal-800">CH-32..47</td>
                <td>PHASE_ENGINE</td>
                <td>Applied 047° spatial null weight vector (-28.4 dB extinction confirmed by receiver feedback)</td>
              </tr>
              <tr>
                <td>14:27:58.840</td>
                <td>GLOBAL</td>
                <td>COOL_THERM</td>
                <td>Heatsink planar gradient stabilized at Delta-T = 1.8°C across corner elements [0,0] to [7,7]</td>
              </tr>
              <tr>
                <td>14:27:44.201</td>
                <td>S11_SENSE</td>
                <td>RF_IMPEDANCE</td>
                <td>Return loss sweep verified = 18.2 dB on all 64 microstrip feedports. No reflection alerts.</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
