import React from 'react';
import { Play, Pause, RotateCcw, User, Activity, Radio } from 'lucide-react';
import { SimState } from '../../types';

interface HeaderProps {
  simState: SimState;
  onTogglePlay: () => void;
  onSetSpeed: (speed: 1 | 2 | 4) => void;
  onResetSim: () => void;
  onScenarioChange?: (scenario: string) => void;
}

export const Header: React.FC<HeaderProps> = ({
  simState,
  onTogglePlay,
  onSetSpeed,
  onResetSim,
}) => {
  return (
    <header className="app-header">
      <div className="header-left">
        <div className="node-badge">
          <Radio size={13} className="text-teal-600" />
          <span>NODE: <strong>gNodeB-042</strong></span>
        </div>

        <div className="scenario-select-box">
          <span className="text-xs text-gray-500 font-mono">Scenario:</span>
          <select className="scenario-select" value={simState.scenario} onChange={() => {}}>
            <option value={simState.scenario}>{simState.scenario}</option>
            <option value="Coastal Flyway Peak Corridor [gNodeB-042]">Coastal Flyway Peak Corridor [gNodeB-042]</option>
            <option value="Storm Divergence Fall Transit [gNodeB-042]">Storm Divergence Fall Transit [gNodeB-042]</option>
          </select>
        </div>

        <div className="status-badge-live">
          <span className="live-pulse-dot" />
          <span>{simState.isPlaying ? 'LIVE / RUNNING' : 'PAUSED'}</span>
        </div>
      </div>

      <div className="header-right">
        <div className="playback-controls">
          <button
            className="control-btn"
            onClick={onTogglePlay}
            title={simState.isPlaying ? 'Pause simulation' : 'Resume simulation'}
          >
            {simState.isPlaying ? <Pause size={13} /> : <Play size={13} />}
          </button>
          <button className="control-btn" onClick={onResetSim} title="Reset simulation">
            <RotateCcw size={13} />
          </button>
          <button
            className={`control-btn ${simState.speed === 1 ? 'active' : ''}`}
            onClick={() => onSetSpeed(1)}
          >
            1x
          </button>
          <button
            className={`control-btn ${simState.speed === 2 ? 'active' : ''}`}
            onClick={() => onSetSpeed(2)}
          >
            2x
          </button>
          <button
            className={`control-btn ${simState.speed === 4 ? 'active' : ''}`}
            onClick={() => onSetSpeed(4)}
          >
            4x
          </button>
        </div>

        <div className="timer-readout">
          <Activity size={13} className="text-emerald-600" />
          <span>{simState.timeString}</span>
        </div>

        <div className="avatar-btn" title="Operator Session: RF Mission Specialist">
          <User size={15} />
        </div>
      </div>
    </header>
  );
};
