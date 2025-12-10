import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { TrackSelector } from "@/components/TrackList/TrackSelector";
import { trackService } from "@/services/api";
import { useTrackStore, useWaveformStore, useTempoStore, useAudioStore, useStructureStore, useUIStore } from "@/stores";
import type { Track } from "@/types/track";

// Mock the API service
vi.mock("@/services/api", () => ({
  trackService: {
    getTracks: vi.fn(),
    loadTrack: vi.fn(),
    getAudioUrl: vi.fn(),
  },
}));

describe("TrackSelector", () => {
  beforeEach(() => {
    // Reset all stores
    useTrackStore.setState({
      tracks: [],
      selectedTrack: null,
      currentTrack: null,
      isLoading: false,
      error: null,
    });
    useWaveformStore.getState().reset();
    useTempoStore.getState().reset();
    useStructureStore.getState().reset();
    useAudioStore.getState().reset();
    useUIStore.getState().reset();

    // Clear all mocks
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("Loading State", () => {
    it("renders loading state when isLoading is true", async () => {
      // Mock fetchTracks to keep isLoading true
      const mockFetchTracks = vi.fn().mockImplementation(() => {
        useTrackStore.setState({ isLoading: true });
      });
      useTrackStore.setState({ fetchTracks: mockFetchTracks });

      render(<TrackSelector />);

      await waitFor(() => {
        expect(screen.getByText("Loading tracks...")).toBeInTheDocument();
      });
    });
  });

  describe("Empty State", () => {
    it("renders empty state when no audio files available", () => {
      useTrackStore.setState({ tracks: [], isLoading: false });

      render(<TrackSelector />);

      expect(
        screen.getByText(/No audio files found. Check EDM_AUDIO_DIR environment variable./)
      ).toBeInTheDocument();
    });
  });

  describe("Track List Rendering", () => {
    it("renders track list with correct data", async () => {
      const mockTracks: Track[] = [
        {
          filename: "track1.mp3",
          path: "/path/to/track1.mp3",
          has_reference: true,
          has_generated: false,
        },
        {
          filename: "track2.mp3",
          path: "/path/to/track2.mp3",
          has_reference: false,
          has_generated: true,
        },
        {
          filename: "track3.mp3",
          path: "/path/to/track3.mp3",
          has_reference: false,
          has_generated: false,
        },
      ];

      // Mock fetchTracks to directly set tracks in store
      const mockFetchTracks = vi.fn(async () => {
        useTrackStore.setState({ tracks: mockTracks, isLoading: false });
      });
      useTrackStore.setState({ fetchTracks: mockFetchTracks });

      render(<TrackSelector />);

      expect(await screen.findByText("Tracks (3)")).toBeInTheDocument();
      expect(screen.getByText("track1.mp3")).toBeInTheDocument();
      expect(screen.getByText("track2.mp3")).toBeInTheDocument();
      expect(screen.getByText("track3.mp3")).toBeInTheDocument();
    });

    it("displays has_reference indicator correctly", async () => {
      const mockTracks: Track[] = [
        {
          filename: "track1.mp3",
          path: "/path/to/track1.mp3",
          has_reference: true,
          has_generated: false,
        },
      ];

      const mockFetchTracks = vi.fn(async () => {
        useTrackStore.setState({ tracks: mockTracks, isLoading: false });
      });
      useTrackStore.setState({ fetchTracks: mockFetchTracks });

      render(<TrackSelector />);

      expect(await screen.findByText("track1.mp3")).toBeInTheDocument();
      expect(screen.getByText("Reference")).toBeInTheDocument();
      // Check for checkmark
      expect(screen.getByText("✓")).toBeInTheDocument();
    });

    it("displays has_generated indicator correctly", async () => {
      const mockTracks: Track[] = [
        {
          filename: "track2.mp3",
          path: "/path/to/track2.mp3",
          has_reference: false,
          has_generated: true,
        },
      ];

      const mockFetchTracks = vi.fn(async () => {
        useTrackStore.setState({ tracks: mockTracks, isLoading: false });
      });
      useTrackStore.setState({ fetchTracks: mockFetchTracks });

      render(<TrackSelector />);

      expect(await screen.findByText("track2.mp3")).toBeInTheDocument();
      expect(screen.getByText("Generated")).toBeInTheDocument();
    });

    it("displays no annotation indicator when neither has_reference nor has_generated", async () => {
      const mockTracks: Track[] = [
        {
          filename: "track3.mp3",
          path: "/path/to/track3.mp3",
          has_reference: false,
          has_generated: false,
        },
      ];

      const mockFetchTracks = vi.fn(async () => {
        useTrackStore.setState({ tracks: mockTracks, isLoading: false });
      });
      useTrackStore.setState({ fetchTracks: mockFetchTracks });

      render(<TrackSelector />);

      expect(await screen.findByText("track3.mp3")).toBeInTheDocument();
      expect(screen.getByText("No annotation")).toBeInTheDocument();
    });
  });

  describe("Track Click Handler", () => {
    it("triggers handleTrackClick when track is clicked", async () => {
      const user = userEvent.setup();
      const mockTracks: Track[] = [
        {
          filename: "track1.mp3",
          path: "/path/to/track1.mp3",
          has_reference: true,
          has_generated: false,
        },
      ];

      const mockLoadTrackData = {
        filename: "track1.mp3",
        bpm: 128,
        downbeat: 0,
        waveform_bass: [0.1, 0.2, 0.3],
        waveform_mids: [0.2, 0.3, 0.4],
        waveform_highs: [0.3, 0.4, 0.5],
        waveform_times: [0, 1, 2],
        duration: 180,
      };

      const mockFetchTracks = vi.fn(async () => {
        useTrackStore.setState({ tracks: mockTracks, isLoading: false });
      });
      useTrackStore.setState({ fetchTracks: mockFetchTracks });
      vi.mocked(trackService.loadTrack).mockResolvedValue(mockLoadTrackData);
      vi.mocked(trackService.getAudioUrl).mockReturnValue("/api/audio/track1.mp3");

      render(<TrackSelector />);

      const trackElement = await screen.findByText("track1.mp3");
      await user.click(trackElement);

      // Verify that selectTrack was called
      expect(useTrackStore.getState().selectedTrack).toBe("track1.mp3");
    });

    it("calls API and updates stores on track click", async () => {
      const user = userEvent.setup();
      const mockTracks: Track[] = [
        {
          filename: "track1.mp3",
          path: "/path/to/track1.mp3",
          has_reference: true,
          has_generated: false,
        },
      ];

      const mockLoadTrackData = {
        filename: "track1.mp3",
        bpm: 128,
        downbeat: 0.5,
        waveform_bass: [0.1, 0.2, 0.3],
        waveform_mids: [0.2, 0.3, 0.4],
        waveform_highs: [0.3, 0.4, 0.5],
        waveform_times: [0, 1, 2],
        duration: 180,
      };

      const mockFetchTracks = vi.fn(async () => {
        useTrackStore.setState({ tracks: mockTracks, isLoading: false });
      });
      useTrackStore.setState({ fetchTracks: mockFetchTracks });
      vi.mocked(trackService.loadTrack).mockResolvedValue(mockLoadTrackData);
      vi.mocked(trackService.getAudioUrl).mockReturnValue("/api/audio/track1.mp3");

      // Create a mock audio element
      const mockAudioElement = {
        src: "",
        load: vi.fn(),
        play: vi.fn(),
        pause: vi.fn(),
        currentTime: 0,
      } as unknown as HTMLAudioElement;

      useAudioStore.setState({ player: mockAudioElement });

      render(<TrackSelector />);

      const trackElement = await screen.findByText("track1.mp3");
      await user.click(trackElement);

      await waitFor(() => {
        expect(trackService.loadTrack).toHaveBeenCalledWith("track1.mp3");
      });

      // Verify waveform store updated
      const waveformState = useWaveformStore.getState();
      expect(waveformState.waveformBass).toEqual([0.1, 0.2, 0.3]);
      expect(waveformState.waveformMids).toEqual([0.2, 0.3, 0.4]);
      expect(waveformState.waveformHighs).toEqual([0.3, 0.4, 0.5]);
      expect(waveformState.duration).toBe(180);

      // Verify tempo store updated
      const tempoState = useTempoStore.getState();
      expect(tempoState.trackBPM).toBe(128);
      expect(tempoState.trackDownbeat).toBe(0.5);

      // Verify audio URL was set
      expect(trackService.getAudioUrl).toHaveBeenCalledWith("track1.mp3");
      expect(mockAudioElement.load).toHaveBeenCalled();

      // Verify currentTrack was set
      expect(useTrackStore.getState().currentTrack).toBe("track1.mp3");
    });

    it("displays error message on API error", async () => {
      const user = userEvent.setup();
      const mockTracks: Track[] = [
        {
          filename: "track1.mp3",
          path: "/path/to/track1.mp3",
          has_reference: true,
          has_generated: false,
        },
      ];

      const mockFetchTracks = vi.fn(async () => {
        useTrackStore.setState({ tracks: mockTracks, isLoading: false });
      });
      useTrackStore.setState({ fetchTracks: mockFetchTracks });
      vi.mocked(trackService.loadTrack).mockRejectedValue(new Error("Failed to load track"));

      render(<TrackSelector />);

      const trackElement = await screen.findByText("track1.mp3");
      await user.click(trackElement);

      await waitFor(() => {
        expect(useUIStore.getState().statusMessage).toContain("Error loading track");
      });
    });

    it("resets stores before loading new track", async () => {
      const user = userEvent.setup();
      const mockTracks: Track[] = [
        {
          filename: "track1.mp3",
          path: "/path/to/track1.mp3",
          has_reference: true,
          has_generated: false,
        },
      ];

      const mockLoadTrackData = {
        filename: "track1.mp3",
        bpm: 128,
        downbeat: 0.5,
        waveform_bass: [0.1, 0.2, 0.3],
        waveform_mids: [0.2, 0.3, 0.4],
        waveform_highs: [0.3, 0.4, 0.5],
        waveform_times: [0, 1, 2],
        duration: 180,
      };

      const mockFetchTracks = vi.fn(async () => {
        useTrackStore.setState({ tracks: mockTracks, isLoading: false });
      });
      useTrackStore.setState({ fetchTracks: mockFetchTracks });
      vi.mocked(trackService.loadTrack).mockResolvedValue(mockLoadTrackData);
      vi.mocked(trackService.getAudioUrl).mockReturnValue("/api/audio/track1.mp3");

      // Set some initial state
      useWaveformStore.setState({ duration: 200 });
      useTempoStore.setState({ trackBPM: 140 });
      useStructureStore.setState({ boundaries: [10, 20] });

      render(<TrackSelector />);

      const trackElement = await screen.findByText("track1.mp3");
      await user.click(trackElement);

      await waitFor(() => {
        // Verify stores were reset (new values should be from mockLoadTrackData)
        const tempoState = useTempoStore.getState();
        expect(tempoState.trackBPM).toBe(128);

        const waveformState = useWaveformStore.getState();
        expect(waveformState.duration).toBe(180);

        const structureState = useStructureStore.getState();
        expect(structureState.boundaries).toEqual([]);
      });
    });
  });

  describe("Track Selection Visual Feedback", () => {
    it("highlights selected track", async () => {
      const mockTracks: Track[] = [
        {
          filename: "track1.mp3",
          path: "/path/to/track1.mp3",
          has_reference: true,
          has_generated: false,
        },
        {
          filename: "track2.mp3",
          path: "/path/to/track2.mp3",
          has_reference: false,
          has_generated: true,
        },
      ];

      const mockFetchTracks = vi.fn(async () => {
        useTrackStore.setState({ tracks: mockTracks, isLoading: false });
      });
      useTrackStore.setState({ fetchTracks: mockFetchTracks });

      render(<TrackSelector />);

      await screen.findByText("track1.mp3");

      // Set selected track after render
      useTrackStore.setState({ selectedTrack: "track1.mp3" });

      const track1 = screen.getByText("track1.mp3").parentElement;
      const track2 = screen.getByText("track2.mp3").parentElement;

      // Check that selected track has highlighting style
      await waitFor(() => {
        expect(track1).toHaveStyle({ background: "rgba(91, 124, 255, 0.1)" });
        expect(track2).toHaveStyle({ background: "transparent" });
      });
    });
  });

  describe("Effect Hooks", () => {
    it("fetches tracks on mount", () => {
      const fetchTracksSpy = vi.spyOn(useTrackStore.getState(), "fetchTracks");

      render(<TrackSelector />);

      expect(fetchTracksSpy).toHaveBeenCalled();
    });
  });
});
