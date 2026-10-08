import { useState, useEffect, useCallback } from 'react';
import { SimState, ScreenId, SteeringEngine, FlockTarget, UEData } from '../types';

const INITIAL_FLOCKS: FlockTarget[] = [
  {
    id: '#409',
    species: 'Grus grus',
    commonName: 'Common Crane',
    birdCount: 180,
    azimuthDeg: 47.2,
    distanceM: 184,
    altitudeM: 120,
    speedMs: 14.2,
    headingDeg: 48,
    riskLevel: 'CRITICAL',
    wingbeatHz: 3.6,
    cohesionScore: 0.94,
    thermalC: 39.8,
    estEntrySec: 62,
  },
  {
    id: '#412',
    species: 'Anser anser',
    commonName: 'Greylag Geese',
    birdCount: 94,
    azimuthDeg: 62.0,
    distanceM: 320,
    altitudeM: 190,
    speedMs: 16.5,
    headingDeg: 52,
    riskLevel: 'ALERT',
    wingbeatHz: 4.2,
    cohesionScore: 0.88,
    thermalC: 40.1,
    estEntrySec: 145,
  },
  {
    id: '#388',
    species: 'Antigone canadensis',
    commonName: 'Sandhill Crane',
    birdCount: 65,
    azimuthDeg: 32.5,
    distanceM: 410,
    altitudeM: 260,
    speedMs: 13.8,
    headingDeg: 45,
    riskLevel: 'NOMINAL',
    wingbeatHz: 3.4,
    cohesionScore: 0.91,
    thermalC: 39.5,
    estEntrySec: 280,
  }
];

const INITIAL_UES: UEData[] = [
  { id: 'UE-101', name: 'UE-101 (4K Mobile)', azimuthDeg: 24.0, distanceM: 95, sinrDb: 28.4, throughputMbps: 820, blerPercent: 0.01, cqi: 15 },
  { id: 'UE-102', name: 'UE-102 (Fixed CPE)', azimuthDeg: 38.5, distanceM: 160, sinrDb: 23.2, throughputMbps: 450, blerPercent: 0.02, cqi: 13, isNulledRisk: true },
  { id: 'UE-103', name: 'UE-103 (Vehicular)', azimuthDeg: 58.0, distanceM: 210, sinrDb: 25.1, throughputMbps: 540, blerPercent: 0.01, cqi: 14 },
  { id: 'UE-104', name: 'UE-104 (IoT Gateway)', azimuthDeg: 72.0, distanceM: 275, sinrDb: 21.0, throughputMbps: 310, blerPercent: 0.03, cqi: 12 },
  { id: 'UE-105', name: 'UE-105 (Smart Grid)', azimuthDeg: 12.0, distanceM: 80, sinrDb: 31.0, throughputMbps: 910, blerPercent: 0.00, cqi: 15 },
  { id: 'UE-106', name: 'UE-106 (AR Headset)', azimuthDeg: 84.0, distanceM: 140, sinrDb: 26.5, throughputMbps: 620, blerPercent: 0.01, cqi: 14 },
  { id: 'UE-107', name: 'UE-107 (Telemetry Hub)', azimuthDeg: 66.0, distanceM: 320, sinrDb: 19.8, throughputMbps: 260, blerPercent: 0.04, cqi: 11 },
  { id: 'UE-108', name: 'UE-108 (Mobile Node)', azimuthDeg: 45.0, distanceM: 130, sinrDb: 22.0, throughputMbps: 390, blerPercent: 0.02, cqi: 12, isNulledRisk: true },
];

export function useSimulation() {
  const [currentScreen, setCurrentScreen] = useState<ScreenId>('beam-steering');
  
  const [simState, setSimState] = useState<SimState>({
    step: 41920,
    timeSec: 15502.09,
    timeString: 'T+04:18:22.09',
    isPlaying: true,
    speed: 1,
    scenario: 'Spring migration, Sector A [gNodeB-042]',
    engine: 'FlywayDQN (Active Null)',
    tradeoffWeight: 0.75, // w = beta / alpha
    
    peakExposureWm2: 0.42,
    exposureLimitWm2: 5.00,
    timeOverLimitSec: 0.0,
    nullDepthDb: -26.4,
    cellThroughputGbps: 2.84,
    baselineThroughputGbps: 2.93,
    meanSinrDb: 24.2,
    meanBlerPercent: 0.02,
    txRfPowerW: 420,
    inferenceLatencyMs: 3.4,
    
    rewardThroughput: 4.80,
    rewardExposure: 0.95,
    rewardPower: 0.45,
    rewardNet: 3.40,
    
    activeFlock: INITIAL_FLOCKS[0],
    allFlocks: INITIAL_FLOCKS,
    users: INITIAL_UES,
    
    decisionLogs: [
      {
        id: 'log-1',
        timestamp: 'T+04:18:22.09',
        step: 41920,
        actionType: 'NULL_DEEPEN',
        description: 'Azimuth 047.2° phase shifted by -4.2°. Null -26.4dB locked.',
        details: 'Q*(s,a) = +48.26 + 9.41'
      },
      {
        id: 'log-2',
        timestamp: 'T+04:18:20.10',
        step: 41919,
        actionType: 'LOBE_SPLIT',
        description: 'Sub-Array J27 reassigned to UE-104 cluster (Azimuth 062.5°).',
        details: 'SINR recovered +3.1 dB'
      },
      {
        id: 'log-3',
        timestamp: 'T+04:18:18.02',
        step: 41918,
        actionType: 'STATE_OBSERVE',
        description: 'Flock trajectory projected at 14.2 m/s. Range: 184m.',
        details: '180 Birds in corridor'
      }
    ],
    
    radarEvents: [
      {
        id: 'ev-1',
        timestamp: '04:18:21.84',
        sensorSource: 'Ku-Radar',
        eventType: 'CORRIDOR_ENTRY',
        eventBadgeClass: 'coral',
        details: 'Flock #409 entered Sector A Outer Fringe (Alt 120m)',
        actionTriggered: 'Null-DQN-Prep'
      },
      {
        id: 'ev-2',
        timestamp: '04:18:19.32',
        sensorSource: 'Acoustic FLIR',
        eventType: 'SPECIES_CONFIRM',
        eventBadgeClass: 'teal',
        details: 'Grus grus (Common Crane), 180 units identified',
        actionTriggered: 'Weight Logged'
      },
      {
        id: 'ev-3',
        timestamp: '04:18:12.48',
        sensorSource: 'LiDAR-3D',
        eventType: 'TRAJECTORY_UPD',
        eventBadgeClass: 'blue',
        details: 'Flock #412 trajectory steady 048° @ 13.8 m/s',
        actionTriggered: 'No-Intervention'
      }
    ]
  });

  // Toggle playback
  const togglePlay = useCallback(() => {
    setSimState((prev: SimState) => ({ ...prev, isPlaying: !prev.isPlaying }));
  }, []);

  const setSpeed = useCallback((speed: 1 | 2 | 4) => {
    setSimState((prev: SimState) => ({ ...prev, speed }));
  }, []);

  const resetSim = useCallback(() => {
    setSimState((prev: SimState) => ({
      ...prev,
      step: 41900,
      timeSec: 15480,
      timeString: 'T+04:18:00.00',
      peakExposureWm2: 0.38,
      cellThroughputGbps: 2.86
    }));
  }, []);

  const setEngine = useCallback((engine: SteeringEngine) => {
    setSimState((prev: SimState) => {
      let exposure = prev.peakExposureWm2;
      let throughput = prev.cellThroughputGbps;
      let nullDepth = prev.nullDepthDb;
      
      if (engine === 'Baseline SON') {
        exposure = 4.85; // high exposure without nulling
        throughput = 2.94;
        nullDepth = -3.2;
      } else if (engine === 'Sector Shutdown') {
        exposure = 0.01;
        throughput = 0.00;
        nullDepth = -60.0;
      } else if (engine === 'Static Back-off') {
        exposure = 1.95;
        throughput = 2.10;
        nullDepth = -12.0;
      } else {
        // FlywayDQN
        exposure = 0.42;
        throughput = 2.84;
        nullDepth = -26.4;
      }
      
      return {
        ...prev,
        engine,
        peakExposureWm2: exposure,
        cellThroughputGbps: throughput,
        nullDepthDb: nullDepth,
      };
    });
  }, []);

  const setTradeoffWeight = useCallback((w: number) => {
    setSimState((prev: SimState) => {
      const exposure = Math.max(0.08, +(0.85 - w * 0.45).toFixed(2));
      const throughput = Math.min(2.93, +(2.93 - w * 0.12).toFixed(2));
      const nullDepth = -20 - w * 7.5;
      
      return {
        ...prev,
        tradeoffWeight: w,
        peakExposureWm2: exposure,
        cellThroughputGbps: throughput,
        nullDepthDb: +nullDepth.toFixed(1),
        rewardThroughput: +(throughput * 1.64).toFixed(2),
        rewardExposure: +(w * exposure * 3.0).toFixed(2),
      };
    });
  }, []);

  const selectFlock = useCallback((flockId: string) => {
    setSimState((prev: SimState) => {
      const found = prev.allFlocks.find((f: FlockTarget) => f.id === flockId);
      if (!found) return prev;
      return { ...prev, activeFlock: found };
    });
  }, []);

  // Real-time animation loop
  useEffect(() => {
    if (!simState.isPlaying) return;

    const interval = setInterval(() => {
      setSimState((prev: SimState) => {
        const nextStep = prev.step + 1;
        const nextTime = prev.timeSec + 0.1 * prev.speed;
        
        const hours = Math.floor(nextTime / 3600);
        const mins = Math.floor((nextTime % 3600) / 60);
        const secs = (nextTime % 60).toFixed(2);
        const timeString = `T+${String(hours).padStart(2, '0')}:${String(mins).padStart(2, '0')}:${secs.padStart(5, '0')}`;
        
        const jitter = (Math.random() - 0.5) * 0.02;
        const latencyJitter = +(3.2 + Math.random() * 0.6).toFixed(1);
        
        return {
          ...prev,
          step: nextStep,
          timeSec: nextTime,
          timeString,
          inferenceLatencyMs: latencyJitter,
          peakExposureWm2: +(prev.peakExposureWm2 + jitter * 0.2).toFixed(2),
        };
      });
    }, 200 / simState.speed);

    return () => clearInterval(interval);
  }, [simState.isPlaying, simState.speed]);

  return {
    currentScreen,
    setCurrentScreen,
    simState,
    togglePlay,
    setSpeed,
    resetSim,
    setEngine,
    setTradeoffWeight,
    selectFlock,
  };
}
