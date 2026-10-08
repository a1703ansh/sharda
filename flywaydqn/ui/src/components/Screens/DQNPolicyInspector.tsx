import React, { useState } from 'react';
import { BrainCircuit, Play, Sparkles, Network, BarChart3, TrendingUp, Sliders } from 'lucide-react';
import { SimState } from '../../types';

interface DQNPolicyInspectorProps {
  simState: SimState;
}

export const DQNPolicyInspector: React.FC<DQNPolicyInspectorProps> = () => {
  const [learningRate, setLearningRate] = useState(0.0001);
  const [gamma, setGamma] = useState(0.99);
  const [avianPenalty, setAvianPenalty] = useState(2.50);
  const [selectedActionIndex, setSelectedActionIndex] = useState(29);
  const [isSimulatingStep, setIsSimulatingStep] = useState(false);

  // Generate 64 discrete action Q-values (bell-shaped centered around action 29)
  const qValues = Array.from({ length: 64 }, (_, i) => {
    const dist = Math.abs(i - 29);
    const val = 58 - dist * 1.6 + Math.sin(i * 0.8) * 3.5;
    return Math.max(8, Math.min(60, val));
  });

  const handleSimulateStep = () => {
    setIsSimulatingStep(true);
    setTimeout(() => {
      setIsSimulatingStep(false);
      setSelectedActionIndex(Math.floor(25 + Math.random() * 8));
    }, 600);
  };

  const handleResetHyperparams = () => {
    setLearningRate(0.0001);
    setGamma(0.99);
    setAvianPenalty(2.50);
  };

  return (
    <div className="content-viewport">
      {/* Top Banner */}
      <div className="sub-header-strip">
        <div className="flex items-center gap-3">
          <div className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
          <div className="font-mono text-xs font-bold text-slate-800">
            AGENT STATUS: <span className="text-teal-700">INFERENCE MODE (v4.2.1-prod)</span>
          </div>
        </div>

        <div className="flex items-center gap-4 font-mono text-xs">
          <div className="p-1 px-2 bg-white border border-slate-200 rounded">
            <span className="text-gray-400">EXPLORATION (ε):</span> <strong className="text-slate-800">0.020 DECAYED</strong>
          </div>
          <div className="p-1 px-2 bg-white border border-slate-200 rounded">
            <span className="text-gray-400">REPLAY BUFFER:</span> <strong className="text-slate-800">1,000,000 FULL</strong>
          </div>
          <div className="p-1 px-2 bg-white border border-slate-200 rounded">
            <span className="text-gray-400">TARGET NET SYNC:</span> <strong className="text-slate-800">15.2s ago</strong>
          </div>
          <div className="p-1 px-2 bg-white border border-slate-200 rounded">
            <span className="text-gray-400">BELLMAN LOSS (MSE):</span> <strong className="text-slate-800">0.00318</strong>
          </div>
          <div className="p-1 px-2 bg-teal-50 border border-teal-300 text-teal-800 rounded font-bold">
            STEP LATENCY: 1.84 ms
          </div>
        </div>
      </div>

      {/* Action Trigger Bar */}
      <div className="flex items-center justify-between p-2 bg-white border border-slate-200 rounded-lg shadow-sm">
        <div className="flex items-center gap-3">
          <button
            onClick={handleSimulateStep}
            disabled={isSimulatingStep}
            className="px-4 py-1.5 bg-teal-800 hover:bg-teal-700 text-white rounded font-mono text-xs font-bold flex items-center gap-2 shadow transition-all active:scale-[0.98]"
          >
            <Play size={12} className={isSimulatingStep ? 'animate-spin' : ''} />
            <span>{isSimulatingStep ? 'Evaluating Q-Network...' : 'Simulate Inference Step'}</span>
          </button>
          <div className="font-mono text-xs text-gray-500">
            POLICY: <span className="badge-pill teal">Greedy (ArgMax)</span>
          </div>
        </div>
        <div className="font-mono text-xs text-gray-400">
          Architecture: <strong>Dueling Double-DQN (DDQN) + Prioritized Experience Replay</strong>
        </div>
      </div>

      {/* Main Grid: Computational Graph on Left, Q-Landscape on Right */}
      <div className="dqn-screen-grid">
        {/* Left: Dueling DQN Computational Graph */}
        <div className="tech-card">
          <div className="card-title-row">
            <div>
              <div className="text-[10px] font-mono text-gray-400">MOD-DQN // FORWARD_PASS</div>
              <div className="font-bold font-sans text-sm text-slate-800 flex items-center gap-2">
                <Network size={14} className="text-teal-600" />
                <span>Dueling DQN Computational Graph</span>
              </div>
            </div>
            <div className="font-mono text-xs text-teal-800 bg-teal-50 px-2 py-1 rounded border border-teal-200 font-bold">
              V(s) + (A(s,a) - mean[A])
            </div>
          </div>

          {/* Node Diagram Container */}
          <div className="comp-graph-box">
            {/* Col 1: State Inputs */}
            <div className="graph-col">
              <div className="text-[10px] font-mono text-gray-400 uppercase font-bold">STATE VECTOR S ∈ ℝ¹⁴</div>
              <div className="graph-node">
                <div className="text-gray-400 text-[10px]">flock_az</div>
                <div className="font-bold text-teal-800">+58.4°</div>
              </div>
              <div className="graph-node">
                <div className="text-gray-400 text-[10px]">flock_dist</div>
                <div className="font-bold text-slate-800">182 m</div>
              </div>
              <div className="graph-node">
                <div className="text-gray-400 text-[10px]">flock_vel</div>
                <div className="font-bold text-slate-800">14.2 m/s</div>
              </div>
              <div className="graph-node">
                <div className="text-gray-400 text-[10px]">ue_pos [1..8]</div>
                <div className="font-bold text-slate-800">Dense</div>
              </div>
              <div className="graph-node">
                <div className="text-gray-400 text-[10px]">rf_demand</div>
                <div className="font-bold text-slate-800">920 Mbps</div>
              </div>
            </div>

            {/* Col 2: Dense Backbone Layers */}
            <div className="graph-col">
              <div className="text-[10px] font-mono text-gray-400 uppercase font-bold text-center">DENSE BACKBONE</div>
              <div className="graph-node text-center">
                <div className="font-bold text-slate-800">FC 256</div>
                <div className="text-[10px] text-gray-400">Layer 1 (ReLU)</div>
              </div>
              <div className="w-0.5 h-6 bg-slate-300 mx-auto" />
              <div className="graph-node text-center">
                <div className="font-bold text-slate-800">FC 128</div>
                <div className="text-[10px] text-gray-400">Layer 2 (Mish)</div>
              </div>
            </div>

            {/* Col 3: Value vs Advantage Streams */}
            <div className="graph-col">
              {/* Value Stream */}
              <div className="graph-node">
                <div className="flex justify-between items-center text-[10px] font-mono text-gray-400">
                  <span>VALUE STREAM V(S)</span>
                  <span className="badge-pill teal text-[9px]">Scalar</span>
                </div>
                <div className="text-xl font-bold font-sans text-teal-800 mt-1">+48.26</div>
                <div className="text-[10px] text-gray-500">State environment potential</div>
              </div>

              {/* Advantage Stream */}
              <div className="graph-node active-highlight">
                <div className="flex justify-between items-center text-[10px] font-mono text-gray-400">
                  <span>ADVANTAGE A(S, A)</span>
                  <span className="badge-pill coral text-[9px]">64-dim</span>
                </div>
                <div className="text-xl font-bold font-sans text-orange-600 mt-1">max: +9.41</div>
                <div className="text-[10px] text-gray-500">Relative beam tilt delta</div>
              </div>
            </div>

            {/* Col 4: Q Aggregation */}
            <div className="graph-col">
              <div className="graph-node active-highlight border-2 border-teal-600 bg-white p-3 shadow-md">
                <div className="text-[10px] font-mono text-gray-400 uppercase font-bold">Q-AGGREGATION</div>
                <div className="text-sm font-bold text-teal-900 font-mono mt-1">Q*(s, a)</div>
                <div className="text-base font-extrabold text-teal-700 font-sans mt-0.5">ArgMax → #{selectedActionIndex}</div>
                <div className="mt-2 p-1.5 bg-teal-50 border border-teal-200 rounded text-[11px] font-mono text-teal-800">
                  <strong>AZ NULL SHIFT: +1.75°</strong>
                </div>
              </div>
            </div>
          </div>

          {/* Sub Diagnostics Row */}
          <div className="grid grid-cols-3 gap-3 mt-3 font-mono text-xs">
            <div className="p-2 bg-slate-50 border border-slate-200 rounded">
              <div className="text-[10px] text-gray-400">L2 WEIGHT REGULARIZER</div>
              <div className="font-bold text-slate-800">1.42e-05</div>
              <div className="text-[10px] text-emerald-600">Normal</div>
            </div>
            <div className="p-2 bg-slate-50 border border-slate-200 rounded">
              <div className="text-[10px] text-gray-400">GRADIENT CLIP NORM</div>
              <div className="font-bold text-slate-800">0.82 / 10.0</div>
              <div className="text-[10px] text-emerald-600">Stable</div>
            </div>
            <div className="p-2 bg-slate-50 border border-slate-200 rounded">
              <div className="text-[10px] text-gray-400">SOFT POLYAK TAU (τ)</div>
              <div className="font-bold text-slate-800">0.005</div>
              <div className="text-[10px] text-gray-500">Every 4 steps</div>
            </div>
          </div>
        </div>

        {/* Right: Discrete Q-Value Landscape */}
        <div className="tech-card flex flex-col justify-between gap-4">
          <div>
            <div className="card-title-row">
              <div className="flex items-center gap-2">
                <BarChart3 size={14} className="text-teal-600" />
                <span className="card-title">Discrete Q-Value Landscape</span>
                <span className="text-xs font-mono text-gray-400">[|A| = 64]</span>
              </div>
              <span className="badge-pill teal font-mono">OPT: A = {selectedActionIndex}</span>
            </div>

            {/* Q-Value Histogram Bars */}
            <div className="q-dist-bars">
              {qValues.map((val, idx) => {
                const isSelected = idx === selectedActionIndex;
                const heightPct = (val / 60) * 100;

                return (
                  <div
                    key={idx}
                    onClick={() => setSelectedActionIndex(idx)}
                    title={`Action #${idx}: Q = ${val.toFixed(2)}`}
                    className={`q-bar ${isSelected ? 'selected-q' : ''}`}
                    style={{
                      height: `${heightPct}%`,
                      backgroundColor: isSelected ? '#0d9488' : undefined,
                    }}
                  />
                );
              })}
            </div>

            <div className="flex justify-between font-mono text-[10px] text-gray-400 mt-1">
              <span>-5.0° Az (A₀)</span>
              <span className="text-teal-800 font-bold">A{selectedActionIndex}: +1.75° Az | -28.5 dB Null</span>
              <span>+5.0° Az (A₆₃)</span>
            </div>

            {/* Selected Action Decomposition */}
            <div className="mt-4 p-2.5 bg-slate-50 border border-slate-200 rounded-lg">
              <div className="text-[10px] font-mono text-gray-400 uppercase font-bold mb-2">
                SELECTED ACTION DECOMPOSITION (A = {selectedActionIndex})
              </div>
              <div className="grid grid-cols-3 gap-2 font-mono text-xs">
                <div>
                  <div className="text-[10px] text-gray-400">Azimuth Tilt</div>
                  <div className="font-bold text-slate-800">+1.75°</div>
                </div>
                <div>
                  <div className="text-[10px] text-gray-400">Null Target</div>
                  <div className="font-bold text-teal-700">-28.5 dB</div>
                </div>
                <div>
                  <div className="text-[10px] text-gray-400">Sidelobe Max</div>
                  <div className="font-bold text-slate-800">-16.2 dB</div>
                </div>
              </div>
            </div>

            {/* Reward Weights Breakdown */}
            <div className="mt-3 space-y-2 font-mono text-xs">
              <div className="flex justify-between text-[11px]">
                <span className="text-gray-500">REWARD WEIGHTS: R = α·R_TP - β·R_EID - γ·R_PWR</span>
                <span className="font-bold text-teal-800">Net: +3.40</span>
              </div>

              <div>
                <div className="flex justify-between text-[10px] mb-0.5">
                  <span className="text-slate-700 font-medium">UE Throughput (α = 1.0)</span>
                  <span className="text-emerald-600 font-bold">+4.80 pts</span>
                </div>
                <div className="h-1.5 w-full bg-slate-100 rounded overflow-hidden">
                  <div className="h-full bg-emerald-500 rounded" style={{ width: '85%' }} />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-[10px] mb-0.5">
                  <span className="text-slate-700 font-medium">Avian Radar Exposure (β = 2.5)</span>
                  <span className="text-red-500 font-bold">-0.95 pts</span>
                </div>
                <div className="h-1.5 w-full bg-slate-100 rounded overflow-hidden">
                  <div className="h-full bg-red-400 rounded" style={{ width: '25%' }} />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-[10px] mb-0.5">
                  <span className="text-slate-700 font-medium">RF Power Penalty (γ = 0.4)</span>
                  <span className="text-orange-500 font-bold">-0.45 pts</span>
                </div>
                <div className="h-1.5 w-full bg-slate-100 rounded overflow-hidden">
                  <div className="h-full bg-orange-400 rounded" style={{ width: '15%' }} />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Section: Training Convergence & Hyperparameter Tuning */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Left 2 Cols: Convergence Chart */}
        <div className="tech-card lg:col-span-2">
          <div className="card-title-row">
            <span className="card-title">
              <TrendingUp size={14} className="text-teal-600" />
              <span>Episode Reward &amp; Nulling Fidelity Convergence</span>
            </span>
            <span className="text-xs font-mono text-gray-400">LAST 2,400 EPISODES</span>
          </div>

          <div className="flex items-center gap-6 font-mono text-xs mb-2">
            <span className="flex items-center gap-2">
              <span className="w-3 h-0.5 bg-teal-600" />
              <span className="text-teal-800 font-semibold">Mean Reward</span>
            </span>
            <span className="flex items-center gap-2">
              <span className="w-3 h-0.5 bg-orange-500" />
              <span className="text-orange-800 font-semibold">Null Depth (dB)</span>
            </span>
          </div>

          {/* Convergence Chart Canvas */}
          <div className="h-28 w-full bg-slate-50 border border-slate-200 rounded p-2">
            <svg viewBox="0 0 600 80" className="w-full h-full">
              {/* Convergence curves */}
              <path
                d="M 10,75 Q 150,70 250,45 Q 380,25 590,20 L 590,78 L 10,78 Z"
                fill="rgba(13, 148, 136, 0.12)"
              />
              <path
                d="M 10,75 Q 150,70 250,45 Q 380,25 590,20"
                fill="none"
                stroke="#0d9488"
                strokeWidth="2"
              />
              <path
                d="M 10,78 Q 150,75 250,55 Q 380,35 590,26"
                fill="none"
                stroke="#ea580c"
                strokeWidth="1.5"
                strokeDasharray="3 2"
              />
            </svg>
          </div>

          <div className="flex justify-between items-center text-[10px] font-mono text-gray-500 mt-2">
            <span>Ep #1 (Random Matrix)</span>
            <span>Ep #1200 (ε-decay 0.1)</span>
            <span className="text-teal-800 font-bold">Ep #2400 (Converged Optimal Policy)</span>
          </div>
        </div>

        {/* Right 1 Col: Live Hyperparameter Tuning */}
        <div className="tech-card flex flex-col justify-between">
          <div>
            <div className="card-title-row">
              <span className="card-title">
                <Sliders size={14} className="text-teal-600" />
                <span>Hyperparameter Tuning</span>
              </span>
              <span className="badge-pill teal text-[10px]">LIVE TUNE</span>
            </div>

            <div className="space-y-3 font-mono text-xs">
              <div>
                <div className="flex justify-between text-gray-500 mb-1">
                  <span>Learning Rate (α)</span>
                  <span className="font-bold text-slate-800">{learningRate}</span>
                </div>
                <input
                  type="range"
                  min="0.00001"
                  max="0.001"
                  step="0.00005"
                  value={learningRate}
                  onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                  className="dial-slider"
                />
              </div>

              <div>
                <div className="flex justify-between text-gray-500 mb-1">
                  <span>Discount Factor (γ)</span>
                  <span className="font-bold text-slate-800">{gamma}</span>
                </div>
                <input
                  type="range"
                  min="0.80"
                  max="0.999"
                  step="0.01"
                  value={gamma}
                  onChange={(e) => setGamma(parseFloat(e.target.value))}
                  className="dial-slider"
                />
              </div>

              <div>
                <div className="flex justify-between text-gray-500 mb-1">
                  <span>Avian Safety Penalty (β)</span>
                  <span className="font-bold text-orange-600">{avianPenalty.toFixed(2)}x</span>
                </div>
                <input
                  type="range"
                  min="0.5"
                  max="5.0"
                  step="0.1"
                  value={avianPenalty}
                  onChange={(e) => setAvianPenalty(parseFloat(e.target.value))}
                  className="dial-slider"
                />
              </div>
            </div>
          </div>

          <div className="flex items-center justify-between border-t border-slate-100 pt-3 mt-3 font-mono text-xs">
            <div>
              <div className="text-[9px] text-gray-400 uppercase">TARGET SOFT SYNC (τ)</div>
              <div className="font-bold text-slate-800">Continuous EMA: 0.005</div>
            </div>
            <button
              onClick={handleResetHyperparams}
              className="px-2.5 py-1 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded text-xs"
            >
              Reset Defaults
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
