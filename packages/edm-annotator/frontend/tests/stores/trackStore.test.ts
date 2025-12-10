import { describe, it, expect, beforeEach, vi } from "vitest";
import { useTrackStore } from "@/stores/trackStore";
import { trackService } from "@/services/api";
import type { Track } from "@/types/track";

// Mock the trackService
vi.mock("@/services/api", () => ({
  trackService: {
    getTracks: vi.fn(),
  },
}));

describe("trackStore", () => {
  const mockTracks: Track[] = [
    {
      filename: "track1.mp3",
      path: "/audio/track1.mp3",
      has_reference: true,
      has_generated: false,
    },
    {
      filename: "track2.mp3",
      path: "/audio/track2.mp3",
      has_reference: false,
      has_generated: true,
    },
    {
      filename: "track3.mp3",
      path: "/audio/track3.mp3",
      has_reference: true,
      has_generated: true,
    },
  ];

  beforeEach(() => {
    // Reset store before each test
    useTrackStore.getState().reset();
    // Clear all mocks
    vi.clearAllMocks();
  });

  describe("fetchTracks", () => {
    it("should fetch tracks successfully", async () => {
      vi.mocked(trackService.getTracks).mockResolvedValue(mockTracks);

      const { fetchTracks } = useTrackStore.getState();
      await fetchTracks();

      const { tracks, isLoading, error } = useTrackStore.getState();
      expect(tracks).toEqual(mockTracks);
      expect(isLoading).toBe(false);
      expect(error).toBe(null);
    });

    it("should set loading state during fetch", async () => {
      let resolvePromise: (value: Track[]) => void;
      const promise = new Promise<Track[]>((resolve) => {
        resolvePromise = resolve;
      });
      vi.mocked(trackService.getTracks).mockReturnValue(promise);

      const { fetchTracks } = useTrackStore.getState();
      const fetchPromise = fetchTracks();

      // Check loading state before promise resolves
      const { isLoading: isLoadingDuring } = useTrackStore.getState();
      expect(isLoadingDuring).toBe(true);

      // Resolve and wait
      resolvePromise!(mockTracks);
      await fetchPromise;

      // Check loading state after promise resolves
      const { isLoading: isLoadingAfter } = useTrackStore.getState();
      expect(isLoadingAfter).toBe(false);
    });

    it("should handle fetch error with Error object", async () => {
      const errorMessage = "Network error";
      vi.mocked(trackService.getTracks).mockRejectedValue(
        new Error(errorMessage)
      );

      const { fetchTracks } = useTrackStore.getState();
      await fetchTracks();

      const { tracks, isLoading, error } = useTrackStore.getState();
      expect(tracks).toEqual([]);
      expect(isLoading).toBe(false);
      expect(error).toBe(errorMessage);
    });

    it("should handle fetch error with non-Error object", async () => {
      vi.mocked(trackService.getTracks).mockRejectedValue(
        "Unknown error string"
      );

      const { fetchTracks } = useTrackStore.getState();
      await fetchTracks();

      const { error } = useTrackStore.getState();
      expect(error).toBe("Failed to fetch tracks");
    });

    it("should clear previous error on successful fetch", async () => {
      // First fetch fails
      vi.mocked(trackService.getTracks).mockRejectedValue(
        new Error("First error")
      );
      await useTrackStore.getState().fetchTracks();

      let { error } = useTrackStore.getState();
      expect(error).toBe("First error");

      // Second fetch succeeds
      vi.mocked(trackService.getTracks).mockResolvedValue(mockTracks);
      await useTrackStore.getState().fetchTracks();

      error = useTrackStore.getState().error;
      expect(error).toBe(null);
    });
  });

  describe("selectTrack", () => {
    it("should update selectedTrack", () => {
      const { selectTrack } = useTrackStore.getState();

      selectTrack("track1.mp3");

      const { selectedTrack } = useTrackStore.getState();
      expect(selectedTrack).toBe("track1.mp3");
    });

    it("should allow selecting different tracks", () => {
      const { selectTrack } = useTrackStore.getState();

      selectTrack("track1.mp3");
      selectTrack("track2.mp3");

      const { selectedTrack } = useTrackStore.getState();
      expect(selectedTrack).toBe("track2.mp3");
    });
  });

  describe("setCurrentTrack", () => {
    it("should update currentTrack", () => {
      const { setCurrentTrack } = useTrackStore.getState();

      setCurrentTrack("track1.mp3");

      const { currentTrack } = useTrackStore.getState();
      expect(currentTrack).toBe("track1.mp3");
    });
  });

  describe("previousTrack", () => {
    beforeEach(async () => {
      vi.mocked(trackService.getTracks).mockResolvedValue(mockTracks);
      await useTrackStore.getState().fetchTracks();
    });

    it("should navigate to previous track", () => {
      const { selectTrack, previousTrack } = useTrackStore.getState();

      selectTrack("track2.mp3");
      previousTrack();

      const { selectedTrack } = useTrackStore.getState();
      expect(selectedTrack).toBe("track1.mp3");
    });

    it("should not navigate before first track", () => {
      const { selectTrack, previousTrack } = useTrackStore.getState();

      selectTrack("track1.mp3");
      previousTrack();

      const { selectedTrack } = useTrackStore.getState();
      expect(selectedTrack).toBe("track1.mp3");
    });

    it("should do nothing if no track is selected", () => {
      const { previousTrack } = useTrackStore.getState();

      previousTrack();

      const { selectedTrack } = useTrackStore.getState();
      expect(selectedTrack).toBe(null);
    });

    it("should do nothing if tracks list is empty", () => {
      useTrackStore.setState({ tracks: [], selectedTrack: "track1.mp3" });

      const { previousTrack } = useTrackStore.getState();
      previousTrack();

      const { selectedTrack } = useTrackStore.getState();
      expect(selectedTrack).toBe("track1.mp3");
    });

    it("should handle selecting previous from last track", () => {
      const { selectTrack, previousTrack } = useTrackStore.getState();

      selectTrack("track3.mp3");
      previousTrack();

      const { selectedTrack } = useTrackStore.getState();
      expect(selectedTrack).toBe("track2.mp3");
    });
  });

  describe("nextTrack", () => {
    beforeEach(async () => {
      vi.mocked(trackService.getTracks).mockResolvedValue(mockTracks);
      await useTrackStore.getState().fetchTracks();
    });

    it("should navigate to next track", () => {
      const { selectTrack, nextTrack } = useTrackStore.getState();

      selectTrack("track1.mp3");
      nextTrack();

      const { selectedTrack } = useTrackStore.getState();
      expect(selectedTrack).toBe("track2.mp3");
    });

    it("should not navigate past last track", () => {
      const { selectTrack, nextTrack } = useTrackStore.getState();

      selectTrack("track3.mp3");
      nextTrack();

      const { selectedTrack } = useTrackStore.getState();
      expect(selectedTrack).toBe("track3.mp3");
    });

    it("should do nothing if no track is selected", () => {
      const { nextTrack } = useTrackStore.getState();

      nextTrack();

      const { selectedTrack } = useTrackStore.getState();
      expect(selectedTrack).toBe(null);
    });

    it("should do nothing if tracks list is empty", () => {
      useTrackStore.setState({ tracks: [], selectedTrack: "track1.mp3" });

      const { nextTrack } = useTrackStore.getState();
      nextTrack();

      const { selectedTrack } = useTrackStore.getState();
      expect(selectedTrack).toBe("track1.mp3");
    });

    it("should handle navigating through all tracks", () => {
      const { selectTrack, nextTrack } = useTrackStore.getState();

      selectTrack("track1.mp3");
      nextTrack();

      let { selectedTrack } = useTrackStore.getState();
      expect(selectedTrack).toBe("track2.mp3");

      nextTrack();
      selectedTrack = useTrackStore.getState().selectedTrack;
      expect(selectedTrack).toBe("track3.mp3");
    });
  });

  describe("reset", () => {
    it("should clear selected and current tracks", async () => {
      vi.mocked(trackService.getTracks).mockResolvedValue(mockTracks);
      await useTrackStore.getState().fetchTracks();

      const { selectTrack, setCurrentTrack, reset } =
        useTrackStore.getState();

      selectTrack("track1.mp3");
      setCurrentTrack("track2.mp3");
      reset();

      const { selectedTrack, currentTrack, error } =
        useTrackStore.getState();
      expect(selectedTrack).toBe(null);
      expect(currentTrack).toBe(null);
      expect(error).toBe(null);
    });

    it("should not clear tracks list", async () => {
      vi.mocked(trackService.getTracks).mockResolvedValue(mockTracks);
      await useTrackStore.getState().fetchTracks();

      useTrackStore.getState().reset();

      const { tracks } = useTrackStore.getState();
      expect(tracks).toEqual(mockTracks);
    });

    it("should clear error state", async () => {
      vi.mocked(trackService.getTracks).mockRejectedValue(
        new Error("Test error")
      );
      await useTrackStore.getState().fetchTracks();

      useTrackStore.getState().reset();

      const { error } = useTrackStore.getState();
      expect(error).toBe(null);
    });
  });

  describe("integration scenarios", () => {
    it("should handle full track selection workflow", async () => {
      vi.mocked(trackService.getTracks).mockResolvedValue(mockTracks);

      const { fetchTracks, selectTrack, setCurrentTrack } =
        useTrackStore.getState();

      await fetchTracks();
      selectTrack("track1.mp3");
      setCurrentTrack("track1.mp3");

      const { tracks, selectedTrack, currentTrack } =
        useTrackStore.getState();
      expect(tracks).toHaveLength(3);
      expect(selectedTrack).toBe("track1.mp3");
      expect(currentTrack).toBe("track1.mp3");
    });

    it("should handle navigation after fetch", async () => {
      vi.mocked(trackService.getTracks).mockResolvedValue(mockTracks);

      const { fetchTracks, selectTrack, nextTrack, previousTrack } =
        useTrackStore.getState();

      await fetchTracks();
      selectTrack("track2.mp3");

      previousTrack();
      let { selectedTrack } = useTrackStore.getState();
      expect(selectedTrack).toBe("track1.mp3");

      nextTrack();
      nextTrack();
      selectedTrack = useTrackStore.getState().selectedTrack;
      expect(selectedTrack).toBe("track3.mp3");
    });

    it("should handle error and recovery workflow", async () => {
      // Initial error
      vi.mocked(trackService.getTracks).mockRejectedValue(
        new Error("Network error")
      );
      await useTrackStore.getState().fetchTracks();

      let { error, tracks } = useTrackStore.getState();
      expect(error).toBe("Network error");
      expect(tracks).toEqual([]);

      // Retry and succeed
      vi.mocked(trackService.getTracks).mockResolvedValue(mockTracks);
      await useTrackStore.getState().fetchTracks();

      error = useTrackStore.getState().error;
      tracks = useTrackStore.getState().tracks;
      expect(error).toBe(null);
      expect(tracks).toEqual(mockTracks);
    });
  });
});
