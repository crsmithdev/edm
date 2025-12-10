import { create } from "zustand";
import type { Track } from "@/types/track";
import { trackService } from "@/services/api";

interface TrackState {
  // State
  tracks: Track[];
  selectedTrack: string | null;
  currentTrack: string | null;
  isLoading: boolean;
  error: string | null;

  // Actions
  fetchTracks: () => Promise<void>;
  selectTrack: (filename: string) => void;
  setCurrentTrack: (filename: string) => void;
  previousTrack: () => void;
  nextTrack: () => void;
  reset: () => void;
}

export const useTrackStore = create<TrackState>((set, get) => ({
  // Initial state
  tracks: [],
  selectedTrack: null,
  currentTrack: null,
  isLoading: false,
  error: null,

  // Actions
  fetchTracks: async () => {
    set({ isLoading: true, error: null });
    try {
      const tracks = await trackService.getTracks();
      set({ tracks, isLoading: false });
    } catch (error) {
      set({
        tracks: [],
        error: error instanceof Error ? error.message : "Failed to fetch tracks",
        isLoading: false,
      });
    }
  },

  selectTrack: (filename) => set({ selectedTrack: filename }),

  setCurrentTrack: (filename) => set({ currentTrack: filename }),

  previousTrack: () => {
    const { tracks, selectedTrack } = get();
    if (!selectedTrack || tracks.length === 0) return;

    const currentIndex = tracks.findIndex((t) => t.filename === selectedTrack);
    if (currentIndex > 0) {
      const prevTrack = tracks[currentIndex - 1];
      set({ selectedTrack: prevTrack.filename });
    }
  },

  nextTrack: () => {
    const { tracks, selectedTrack } = get();
    if (!selectedTrack || tracks.length === 0) return;

    const currentIndex = tracks.findIndex((t) => t.filename === selectedTrack);
    if (currentIndex >= 0 && currentIndex < tracks.length - 1) {
      const nextTrack = tracks[currentIndex + 1];
      set({ selectedTrack: nextTrack.filename });
    }
  },

  reset: () =>
    set({
      selectedTrack: null,
      currentTrack: null,
      error: null,
    }),
}));
