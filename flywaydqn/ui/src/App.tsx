import React from 'react';
import { useDemoSession } from './hooks/useDemoSession';
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
    connected,
    error,
    togglePlay,
    setSpeed,
    resetSim,
    setEngine,
    setTradeoffWeight,
    selectFlock,
    selectScenario,
    scenarios,
    scenarioKey,
    policies,
    history,
    mitigationEnabled,
    setMitigationEnabled,
    sessionId,
  } = useDemoSession({ autoConnect: true });


  return (
    <div className="app-container">
      <Sidebar
        currentScreen={currentScreen}
        onSelectScreen={setCurrentScreen}
        nullDepthDb={simState.nullDepthDb}
      />

      <div className="app-main">
        <Header
          simState={simState}
          onTogglePlay={togglePlay}
          onSetSpeed={setSpeed}
          onResetSim={resetSim}
          scenarios={scenarios}
          scenarioKey={scenarioKey}
          connected={connected}
          error={error}
          onScenarioChange={(key) => {
            void selectScenario(key);
          }}
        />

        <main>
          {currentScreen === 'beam-steering' && (
            <BeamSteeringConsole
              simState={simState}
              policies={policies}
              onSetEngine={setEngine}
              onSetTradeoffWeight={setTradeoffWeight}
              scenarios={scenarios}
              scenarioKey={scenarioKey}
              history={history}
              mitigationEnabled={mitigationEnabled}
              onSetMitigation={setMitigationEnabled}
              onSelectScenario={(key) => {
                void selectScenario(key);
              }}
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
            sessionId != null ? (
              <DQNPolicyInspector
                sessionId={sessionId}
                frame={simState}
                policyOptions={policies}
                onBack={() => setCurrentScreen('beam-steering')}
              />
            ) : (
              <div className="min-h-[60vh] flex items-center justify-center">
                <div className="text-center text-slate-600">
                  <div className="text-4xl mb-3">🧠</div>
                  <p className="text-sm font-medium">DQN Policy Inspector</p>
                  <p className="text-xs mt-1 text-slate-400">
                    Start a beam steering session first. The inspector will show the
                    real Q-values and decision trace for the live session once one is
                    active.
                  </p>
                </div>
              </div>
            )
          )}
        </main>
      </div>
    </div>
  );
};

export default App;
