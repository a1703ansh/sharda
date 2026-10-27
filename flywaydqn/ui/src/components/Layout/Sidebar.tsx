import React from 'react';
import { Compass, Radar, BrainCircuit, Grid3X3, Feather } from 'lucide-react';
import { ScreenId } from '../../types';

interface SidebarProps {
  currentScreen: ScreenId;
  onSelectScreen: (screen: ScreenId) => void;
  /** Live null depth for the current session; 0 means the null is not applied. */
  nullDepthDb?: number;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentScreen,
  onSelectScreen,
  nullDepthDb,
}) => {
  return (
    <aside className="app-sidebar">
      <div>
        <div className="brand-section">
          <div className="brand-icon-box">
            <Feather size={20} />
          </div>
          <div>
            <h1 className="brand-title">FlywayDQN</h1>
            <div className="brand-version">v2.4 // STIR-RAN</div>
          </div>
        </div>

        <div className="nav-label">MISSION CONSOLE</div>

        <nav className="nav-links" aria-label="Mission console">
          <button
            className={`nav-btn ${currentScreen === 'beam-steering' ? 'active' : ''}`}
            onClick={() => onSelectScreen('beam-steering')}
            aria-current={currentScreen === 'beam-steering' ? 'page' : undefined}
          >
            <Compass size={17} className="nav-icon" />
            <span>Beam Steering Console</span>
          </button>

          <button
            className={`nav-btn ${currentScreen === 'surveillance' ? 'active' : ''}`}
            onClick={() => onSelectScreen('surveillance')}
            aria-current={currentScreen === 'surveillance' ? 'page' : undefined}
          >
            <Radar size={17} className="nav-icon" />
            <span>Flyway Surveillance</span>
          </button>

          <button
            className={`nav-btn ${currentScreen === 'dqn-inspector' ? 'active' : ''}`}
            onClick={() => onSelectScreen('dqn-inspector')}
            aria-current={currentScreen === 'dqn-inspector' ? 'page' : undefined}
          >
            <BrainCircuit size={17} className="nav-icon" />
            <span>DQN Policy Inspector</span>
          </button>

          <button
            className={`nav-btn ${currentScreen === 'phased-array' ? 'active' : ''}`}
            onClick={() => onSelectScreen('phased-array')}
            aria-current={currentScreen === 'phased-array' ? 'page' : undefined}
          >
            <Grid3X3 size={17} className="nav-icon" />
            <span>Phased Array Grid</span>
          </button>
        </nav>
      </div>

      <div className="sidebar-bottom">
        <div className="sidebar-spec-row">
          <span>RF CARRIER</span>
          <span className="sidebar-spec-val">3.5 GHz [n78] mmWave</span>
        </div>
        <div className="sidebar-spec-row">
          <span>NULL DEPTH</span>
          <span className="sidebar-spec-val" style={{ color: '#0d9488' }}>
            {nullDepthDb !== undefined && nullDepthDb !== 0 ? `${nullDepthDb} dB` : 'null off'}
          </span>
        </div>
        <div className="sidebar-spec-row">
          <span>ARRAY FORM</span>
          <span className="sidebar-spec-val">64T64R MASSIVE</span>
        </div>
      </div>
    </aside>
  );
};
