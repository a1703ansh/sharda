import React from 'react';
import { Play, Pause, RotateCcw, User, Activity, Radio } from 'lucide-react';
import { SimState } from '../../types';

interface HeaderProps {
  simState: SimState;
  onTogglePlay: () => void;
  onSetSpeed: (speed: 1 | 2 | 4) => void;
  onResetSim: () => void;
  /** Scenario list from `GET /scenarios`; falls back to the current label. */
  scenarios?: ScenarioOption[];
  /** Key of the live session's scenario (selects the dropdown by key). */
  scenarioKey?: string;
  /** True while the WS stream is up; false shows the offline hint. */
  connected?: boolean;
  /** Last fetch/stream error, shown inline instead of silently listing one. */
  error?: string | null;
  onScenarioChange?: (key: string) => void;
}

export interface ScenarioOption {
  key: string;
  label: string;
}

export const Header: React.FC<HeaderProps> = ({
  simState,
  onTogglePlay,
  onSetSpeed,
  onResetSim,
  scenarios = [],
  scenarioKey,
  connected,
  error,
  onScenarioChange,
}) => {
  const scenarioOptions: ScenarioOption[] =
    scenarios.length > 0 ? scenarios : [{ key: simState.scenario, label: simState.scenario }];
  const currentScenarioKey =
    scenarioKey ??
    scenarioOptions.find((option) => option.label === simState.scenario)?.key ??
    scenarioOptions[0]?.key ??
    'scenario_001';
  const listState =
    scenarios.length > 1
      ? `${scenarios.length} scenarios`
      : scenarios.length === 1
        ? '1 scenario'
        : (connected === false || error) ? 'offline — 1 cached' : 'loading…';

  return (
    <header className="app-header">
      <div className="header-left">
        <div className="node-badge">
          <Radio size={13} className="text-teal-600" />
          <span>NODE: <strong>gNodeB-042</strong></span>
        </div>

        <div className="scenario-select-box">
          <span className="text-xs text-gray-500 font-mono">Scenario:</span>
          <select
            className="scenario-select"
            value={currentScenarioKey}
            onChange={(e) => onScenarioChange?.(e.target.value)}
            aria-label="Simulation scenario"
            title={error ?? `${scenarioOptions.length} scenario(s) from GET /scenarios`}
          >
            {scenarioOptions.map((option) => (
              <option key={option.key} value={option.key}>
                {option.label}
              </option>
            ))}
          </select>
          <span
            className="text-[10px] font-mono text-gray-400"
            aria-live="polite"
            title={error ?? 'Scenario list state'}
          >
            ({listState})
          </span>
        </div>

        <div className="status-badge-live" role="status" aria-live="polite">
          <span className="live-pulse-dot" />
          <span>{simState.isPlaying ? 'LIVE / RUNNING' : 'PAUSED'}</span>
        </div>
      </div>

      <div className="header-right">
        <div className="playback-controls">
          <button
            type="button"
            className="control-btn"
            onClick={onTogglePlay}
            title={simState.isPlaying ? 'Pause simulation' : 'Resume simulation'}
            aria-label={simState.isPlaying ? 'Pause simulation' : 'Resume simulation'}
          >
            {simState.isPlaying ? <Pause size={13} /> : <Play size={13} />}
          </button>
          <button type="button" className="control-btn" onClick={onResetSim} title="Reset simulation" aria-label="Reset simulation">
            <RotateCcw size={13} />
          </button>
          <button
            type="button"
            className={`control-btn ${simState.speed === 1 ? 'active' : ''}`}
            onClick={() => onSetSpeed(1)}
            aria-pressed={simState.speed === 1}
          >
            1x
          </button>
          <button
            type="button"
            className={`control-btn ${simState.speed === 2 ? 'active' : ''}`}
            onClick={() => onSetSpeed(2)}
            aria-pressed={simState.speed === 2}
          >
            2x
          </button>
          <button
            type="button"
            className={`control-btn ${simState.speed === 4 ? 'active' : ''}`}
            onClick={() => onSetSpeed(4)}
            aria-pressed={simState.speed === 4}
          >
            4x
          </button>
        </div>

        <div className="timer-readout">
          <Activity size={13} className="text-emerald-600" />
          <span>{simState.timeString}</span>
        </div>

        <div className="avatar-btn" title="Operator Session: RF Mission Specialist" aria-label="Operator session">
          <User size={15} />
        </div>
      </div>
    </header>
  );
};
