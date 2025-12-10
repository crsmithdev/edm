import { create } from "zustand";

interface WaveformState {
  // State
  waveformBass: number[];
  waveformMids: number[];
  waveformHighs: number[];
  waveformTimes: number[];
  duration: number;
  zoomLevel: number;
  viewportStart: number;
  viewportEnd: number;

  // Actions
  setWaveformData: (data: {
    waveform_bass: number[];
    waveform_mids: number[];
    waveform_highs: number[];
    waveform_times: number[];
    duration: number;
  }) => void;
  zoom: (direction: number, centerTime?: number) => void;
  zoomToFit: () => void;
  pan: (deltaTime: number) => void;
  setViewport: (start: number, end: number) => void;
  reset: () => void;
}

export const useWaveformStore = create<WaveformState>((set, get) => ({
  // Initial state
  waveformBass: [],
  waveformMids: [],
  waveformHighs: [],
  waveformTimes: [],
  duration: 0,
  zoomLevel: 1.0,
  viewportStart: 0,
  viewportEnd: 0,

  // Actions
  setWaveformData: (data) =>
    set({
      waveformBass: data.waveform_bass,
      waveformMids: data.waveform_mids,
      waveformHighs: data.waveform_highs,
      waveformTimes: data.waveform_times,
      duration: data.duration,
      viewportStart: 0,
      viewportEnd: data.duration,
      zoomLevel: 1.0,
    }),

  zoom: (direction, centerTime) => {
    const { duration, viewportStart, viewportEnd, zoomLevel } = get();
    const newZoomLevel = Math.max(0.1, Math.min(10, zoomLevel + direction * 0.2));

    if (centerTime !== undefined) {
      // Zoom centered on specific time
      const viewportDuration = duration / newZoomLevel;
      const center = centerTime;
      let newStart = center - viewportDuration / 2;
      let newEnd = center + viewportDuration / 2;

      // Clamp to track boundaries
      if (newStart < 0) {
        newStart = 0;
        newEnd = viewportDuration;
      }
      if (newEnd > duration) {
        newEnd = duration;
        newStart = Math.max(0, duration - viewportDuration);
      }

      set({
        zoomLevel: newZoomLevel,
        viewportStart: newStart,
        viewportEnd: newEnd,
      });
    } else {
      // Zoom centered on current viewport
      const currentCenter = (viewportStart + viewportEnd) / 2;
      const viewportDuration = duration / newZoomLevel;
      let newStart = currentCenter - viewportDuration / 2;
      let newEnd = currentCenter + viewportDuration / 2;

      if (newStart < 0) {
        newStart = 0;
        newEnd = viewportDuration;
      }
      if (newEnd > duration) {
        newEnd = duration;
        newStart = Math.max(0, duration - viewportDuration);
      }

      set({
        zoomLevel: newZoomLevel,
        viewportStart: newStart,
        viewportEnd: newEnd,
      });
    }
  },

  zoomToFit: () => {
    const { duration } = get();
    set({
      zoomLevel: 1.0,
      viewportStart: 0,
      viewportEnd: duration,
    });
  },

  pan: (deltaTime) => {
    const { duration, viewportStart, viewportEnd } = get();
    let newStart = viewportStart + deltaTime;
    let newEnd = viewportEnd + deltaTime;

    // Clamp to track boundaries
    if (newStart < 0) {
      newStart = 0;
      newEnd = viewportEnd; // Keep the same endpoint, just shift start to 0
    } else if (newEnd > duration) {
      newEnd = duration;
      newStart = viewportStart + deltaTime; // Keep the original shift for start
    }

    set({ viewportStart: newStart, viewportEnd: newEnd });
  },

  setViewport: (start, end) =>
    set({ viewportStart: start, viewportEnd: end }),

  reset: () =>
    set({
      waveformBass: [],
      waveformMids: [],
      waveformHighs: [],
      waveformTimes: [],
      duration: 0,
      zoomLevel: 1.0,
      viewportStart: 0,
      viewportEnd: 0,
    }),
}));
