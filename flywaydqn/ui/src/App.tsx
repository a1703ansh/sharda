import React from 'react';
import { useSimulation } from './hooks/useSimulation';
import { Header } from './components/Layout/Header';
import { Sidebar } from './components/Layout/Sidebar';
import { BeamSteeringConsole } from './components/Screens/BeamSteeringConsole';
import { FlywaySurveillance } from './components/Screens/FlywaySurveillance';
import { PhasedArrayGrid } from './components/Screens/PhasedArrayGrid';
import { DQNPolicyInspector } from './components/Screens/DQNPolicyInspector';

export const App: React.FC = () => {
  const {
    currentScreen,
    setCurrentScreen,
    simState,
    togglePlay,
    setSpeed,
    resetSim,
    setEngine,
    setTradeoffWeight,
    selectFlock,
  } = useSimulation();

  return (
    <div className="app-container">
      {/* Fixed Sidebar */}
      <Sidebar
        currentScreen={currentScreen}
        onSelectScreen={setCurrentScreen}
      />

      {/* Main Viewport */}
      <div className="app-main">
        {/* Sticky Header */}
        <Header
          simState={simState}
          onTogglePlay={togglePlay}
          onSetSpeed={setSpeed}
          onResetSim={resetSim}
        />

        {/* Dynamic Screen Component */}
        <main>
          {currentScreen === 'beam-steering' && (
            <BeamSteeringConsole
              simState={simState}
              onSetEngine={setEngine}
              onSetTradeoffWeight={setTradeoffWeight}
            />
          )}

          {currentScreen === 'surveillance' && (
            <FlywaySurveillance
              simState={simState}
              onSelectFlock={selectFlock}
            />
          )}

          {currentScreen === 'phased-array' && (
            <PhasedArrayGrid
              simState={simState}
            />
          )}

          {currentScreen === 'dqn-inspector' && (
            <DQNPolicyInspector
              simState={simState}
            />
          )}
        </main>
      </div>
    </div>
  );
};

export default App;
