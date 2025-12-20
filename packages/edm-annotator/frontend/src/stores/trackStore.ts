import { create } from "zustand";
import type { Track } from "@/types/track";
import { trackService } from "@/services/api";

type SortField = "artist" | "title" | "status";
type SortDirection = "asc" | "desc";

// Helper function to parse filename into artist and title
function parseFilename(filename: string): { artist: string; title: string } {
  // Remove file extension
  const nameWithoutExt = filename.replace(/\.[^.]+$/, "");

  // Try to split by " - " to separate artist and title
  const parts = nameWithoutExt.split(" - ");

  if (parts.length >= 2) {
    return {
      artist: parts[0].trim(),
      title: parts.slice(1).join(" - ").trim(),
    };
  }

  // If no separator, treat entire name as title
  return {
    artist: "",
    title: nameWithoutExt,
  };
}

// Helper function to get annotation status priority for sorting
function getStatusPriority(track: Track): number {
  if (track.has_reference) return 0; // Reference first
  if (track.has_generated) return 1; // Generated second
  return 2; // No annotation last
}

// Helper function to sort tracks
function sortTracks(tracks: Track[], sortBy: SortField, sortDirection: SortDirection): Track[] {
  const sorted = [...tracks].sort((a, b) => {
    let comparison = 0;

    switch (sortBy) {
      case "artist": {
        const artistA = parseFilename(a.filename).artist.toLowerCase();
        const artistB = parseFilename(b.filename).artist.toLowerCase();
        comparison = artistA.localeCompare(artistB);
        break;
      }
      case "title": {
        const titleA = parseFilename(a.filename).title.toLowerCase();
        const titleB = parseFilename(b.filename).title.toLowerCase();
        comparison = titleA.localeCompare(titleB);
        break;
      }
      case "status": {
        const statusA = getStatusPriority(a);
        const statusB = getStatusPriority(b);
        comparison = statusA - statusB;
        break;
      }
    }

    return sortDirection === "asc" ? comparison : -comparison;
  });

  return sorted;
}

interface TrackState {
  // State
  tracks: Track[];
  selectedTrack: string | null;
  currentTrack: string | null;
  isLoading: boolean;
  error: string | null;
  sortBy: SortField;
  sortDirection: SortDirection;

  // Actions
  fetchTracks: () => Promise<void>;
  selectTrack: (filename: string) => void;
  setCurrentTrack: (filename: string) => void;
  updateTrackStatus: (filename: string, has_reference: boolean, has_generated: boolean) => void;
  setSorting: (sortBy: SortField, sortDirection?: SortDirection) => void;
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
  sortBy: "title",
  sortDirection: "asc",

  // Actions
  fetchTracks: async () => {
    set({ isLoading: true, error: null });
    try {
      const tracks = await trackService.getTracks();
      const { sortBy, sortDirection } = get();
      const sortedTracks = sortTracks(tracks, sortBy, sortDirection);
      set({ tracks: sortedTracks, isLoading: false });
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

  updateTrackStatus: (filename, has_reference, has_generated) => {
    const { tracks, sortBy, sortDirection } = get();
    const updatedTracks = tracks.map((track) =>
      track.filename === filename
        ? { ...track, has_reference, has_generated }
        : track
    );
    const sortedTracks = sortTracks(updatedTracks, sortBy, sortDirection);
    set({ tracks: sortedTracks });
  },

  setSorting: (newSortBy, newSortDirection) => {
    const { tracks, sortBy, sortDirection } = get();

    // If clicking the same field, toggle direction
    const finalSortDirection =
      newSortDirection ?? (newSortBy === sortBy && sortDirection === "asc" ? "desc" : "asc");

    const sortedTracks = sortTracks(tracks, newSortBy, finalSortDirection);
    set({ sortBy: newSortBy, sortDirection: finalSortDirection, tracks: sortedTracks });
  },

  previousTrack: () => {
    const { tracks, selectedTrack } = get();
    if (!selectedTrack || tracks.length === 0) return;

    const currentIndex = tracks.findIndex((t) => t.filename === selectedTrack);
    // Wrap around: if at first track (index 0), go to last track
    const prevIndex = currentIndex > 0 ? currentIndex - 1 : tracks.length - 1;
    const prevTrack = tracks[prevIndex];
    set({ selectedTrack: prevTrack.filename });
  },

  nextTrack: () => {
    const { tracks, selectedTrack } = get();
    if (!selectedTrack || tracks.length === 0) return;

    const currentIndex = tracks.findIndex((t) => t.filename === selectedTrack);
    if (currentIndex < 0) return;

    // Wrap around: if at last track, go to first track (index 0)
    const nextIndex = currentIndex < tracks.length - 1 ? currentIndex + 1 : 0;
    const nextTrack = tracks[nextIndex];
    set({ selectedTrack: nextTrack.filename });
  },

  reset: () =>
    set({
      selectedTrack: null,
      currentTrack: null,
      error: null,
    }),
}));
