import { create } from "zustand";
import { getBeatDuration, quantizeToBeat, timeToBar, barToTime } from "@/utils/tempo";

interface TempoState {
  // State
  trackBPM: number;
  trackDownbeat: number;
  tapTimes: number[];

  // Actions
  setBPM: (bpm: number) => void;
  setDownbeat: (time: number) => void;
  tapTempo: () => void;
  resetTapTempo: () => void;
  reset: () => void;

  // Selectors (computed values)
  timeToBar: (time: number) => number;
  barToTime: (bar: number) => number;
  quantizeToBeat: (time: number) => number;
  getBeatDuration: () => number;
}

export const useTempoStore = create<TempoState>((set, get) => ({
  // Initial state
  trackBPM: 0,
  trackDownbeat: 0,
  tapTimes: [],

  // Actions
  setBPM: (bpm) => set({ trackBPM: bpm }),

  setDownbeat: (time) => set({ trackDownbeat: time }),

  tapTempo: () => {
    const now = Date.now() / 1000; // Current time in seconds
    const { tapTimes } = get();
    const newTapTimes = [...tapTimes, now];

    // Keep only last 4 taps
    if (newTapTimes.length > 4) {
      newTapTimes.shift();
    }

    // Calculate BPM from intervals
    if (newTapTimes.length >= 2) {
      const intervals = [];
      for (let i = 1; i < newTapTimes.length; i++) {
        intervals.push(newTapTimes[i] - newTapTimes[i - 1]);
      }
      const avgInterval = intervals.reduce((a, b) => a + b, 0) / intervals.length;
      const bpm = 60 / avgInterval;
      // Check if close to whole number before rounding
      const wholeNumber = Math.round(bpm);
      const diff = Math.abs(bpm - wholeNumber);
      // Use whole number if within 0.25 BPM, otherwise round to 1 decimal
      const finalBPM = diff < 0.25
        ? wholeNumber
        : Math.round(bpm * 10) / 10;
      set({ trackBPM: finalBPM, tapTimes: newTapTimes });
    } else {
      set({ tapTimes: newTapTimes });
    }
  },

  resetTapTempo: () => set({ tapTimes: [] }),

  reset: () =>
    set({
      trackBPM: 0,
      trackDownbeat: 0,
      tapTimes: [],
    }),

  // Selectors
  timeToBar: (time) => {
    const { trackBPM, trackDownbeat } = get();
    return timeToBar(time, trackBPM, trackDownbeat);
  },

  barToTime: (bar) => {
    const { trackBPM, trackDownbeat } = get();
    return barToTime(bar, trackBPM, trackDownbeat);
  },

  quantizeToBeat: (time) => {
    const { trackBPM, trackDownbeat } = get();
    return quantizeToBeat(time, trackBPM, trackDownbeat);
  },

  getBeatDuration: () => {
    const { trackBPM } = get();
    return getBeatDuration(trackBPM);
  },
}));
