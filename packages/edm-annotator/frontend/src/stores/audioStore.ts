import { create } from "zustand";

interface AudioState {
  // State
  player: HTMLAudioElement | null;
  isPlaying: boolean;
  currentTime: number;
  cuePoint: number;

  // Actions
  setPlayer: (player: HTMLAudioElement) => void;
  play: () => void;
  pause: () => void;
  seek: (time: number) => void;
  setCuePoint: (time: number) => void;
  returnToCue: () => void;
  updateCurrentTime: (time: number) => void;
  reset: () => void;
}

export const useAudioStore = create<AudioState>((set, get) => ({
  // Initial state
  player: null,
  isPlaying: false,
  currentTime: 0,
  cuePoint: 0,

  // Actions
  setPlayer: (player) => set({ player }),

  play: () => {
    const { player } = get();
    if (player) {
      player.play();
      set({ isPlaying: true });
    }
  },

  pause: () => {
    const { player } = get();
    if (player) {
      player.pause();
      set({ isPlaying: false });
    }
  },

  seek: (time) => {
    const { player } = get();
    if (player) {
      player.currentTime = time;
      set({ currentTime: time });
    }
  },

  setCuePoint: (time) => set({ cuePoint: time }),

  returnToCue: () => {
    const { cuePoint, player } = get();
    if (player) {
      player.currentTime = cuePoint;
      player.pause();
      set({ currentTime: cuePoint, isPlaying: false });
    }
  },

  updateCurrentTime: (time) => set({ currentTime: time }),

  reset: () =>
    set({
      isPlaying: false,
      currentTime: 0,
      cuePoint: 0,
    }),
}));
