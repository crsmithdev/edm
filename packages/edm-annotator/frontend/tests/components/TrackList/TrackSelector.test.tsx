import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { TrackSelector } from "@/components/TrackList/TrackSelector";
import { useTrackStore, useWaveformStore, useTempoStore, useAudioStore, useStructureStore, useUIStore } from "@/stores";
import { trackService } from "@/services/api";

vi.mock("@/services/api", () => ({
  trackService: {
    getTracks: vi.fn(),
    loadTrack: vi.fn(),
    getAudioUrl: vi.fn((filename) => `/audio/${filename}`),
  },
}));

describe("TrackSelector", () => {
  beforeEach(() => {
    useTrackStore.getState().reset();
    useWaveformStore.getState().reset();
    useTempoStore.getState().reset();
    useAudioStore.getState().reset();
    useStructureStore.getState().reset();
    useUIStore.getState().reset();

    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("Track List Loading", () => {
    it("fetches tracks on mount", async () => {
      vi.mocked(trackService.getTracks).mockResolvedValue([
        { filename: "track1.mp3", has_reference: true, has_generated: false },
        { filename: "track2.mp3", has_reference: false, has_generated: true },
      ]);

      render(<TrackSelector />);

      await waitFor(() => {
        expect(trackService.getTracks).toHaveBeenCalledTimes(1);
      });
    });

    it("displays list of tracks", async () => {
      vi.mocked(trackService.getTracks).mockResolvedValue([
        { filename: "track1.mp3", has_reference: true, has_generated: false },
        { filename: "track2.mp3", has_reference: false, has_generated: true },
      ]);

      render(<TrackSelector />);

      await waitFor(() => {
        expect(screen.getByText("track1")).toBeInTheDocument();
        expect(screen.getByText("track2")).toBeInTheDocument();
      });
    });

    it("shows reference annotation indicator", async () => {
      vi.mocked(trackService.getTracks).mockResolvedValue([
        { filename: "track1.mp3", has_reference: true, has_generated: false },
      ]);

      render(<TrackSelector />);

      await waitFor(() => {
        expect(screen.getByText("track1")).toBeInTheDocument();
        // Verify CheckCircle icon is present (reference annotation)
        const svg = document.querySelector('svg.lucide-circle-check-big');
        expect(svg).toBeInTheDocument();
      });
    });

    it("shows generated annotation indicator", async () => {
      vi.mocked(trackService.getTracks).mockResolvedValue([
        { filename: "track1.mp3", has_reference: false, has_generated: true },
      ]);

      render(<TrackSelector />);

      await waitFor(() => {
        expect(screen.getByText("track1")).toBeInTheDocument();
        // Verify Sparkles icon is present (generated annotation)
        const svg = document.querySelector('svg.lucide-sparkles');
        expect(svg).toBeInTheDocument();
      });
    });

    it("shows no annotation indicator", async () => {
      vi.mocked(trackService.getTracks).mockResolvedValue([
        { filename: "track1.mp3", has_reference: false, has_generated: false },
      ]);

      render(<TrackSelector />);

      await waitFor(() => {
        expect(screen.getByText("track1")).toBeInTheDocument();
        // Verify Circle icon is present (no annotation)
        const svg = document.querySelector('svg.lucide-circle');
        expect(svg).toBeInTheDocument();
      });
    });

    it("shows checkmark for tracks with reference annotations", async () => {
      vi.mocked(trackService.getTracks).mockResolvedValue([
        { filename: "track1.mp3", has_reference: true, has_generated: false },
      ]);

      render(<TrackSelector />);

      await waitFor(() => {
        const icon = document.querySelector('svg');
        expect(icon).toBeInTheDocument();
      });
    });
  });

  describe("Track Selection", () => {
    it("selects track on click", async () => {
      const user = userEvent.setup();

      vi.mocked(trackService.getTracks).mockResolvedValue([
        { filename: "track1.mp3", has_reference: true, has_generated: false },
        { filename: "track2.mp3", has_reference: false, has_generated: true },
      ]);

      render(<TrackSelector />);

      await waitFor(() => {
        expect(screen.getByText("track1")).toBeInTheDocument();
      });

      await user.click(screen.getByText("track1"));

      const { selectedTrack } = useTrackStore.getState();
      expect(selectedTrack).toBe("track1.mp3");
    });
  });

  describe("Track Loading", () => {
    it("load button is disabled when no track selected", () => {
      render(<TrackSelector />);

      const button = screen.getByText("Load").closest("button");
      expect(button).toBeDisabled();
    });

    it("load button is enabled when track selected", async () => {
      const user = userEvent.setup();

      vi.mocked(trackService.getTracks).mockResolvedValue([
        { filename: "track1.mp3", has_reference: true, has_generated: false },
      ]);

      render(<TrackSelector />);

      await waitFor(() => {
        expect(screen.getByText("track1")).toBeInTheDocument();
      });

      await user.click(screen.getByText("track1"));

      const button = screen.getByText("Load").closest("button");
      expect(button).not.toBeDisabled();
    });

    it("loads track with waveform data", async () => {
      const user = userEvent.setup();

      vi.mocked(trackService.getTracks).mockResolvedValue([
        { filename: "track1.mp3", has_reference: false, has_generated: false },
      ]);

      vi.mocked(trackService.loadTrack).mockResolvedValue({
        filename: "track1.mp3",
        waveform_bass: [0.5, 0.7],
        waveform_mids: [0.4, 0.6],
        waveform_highs: [0.2, 0.3],
        waveform_times: [0, 1],
        duration: 180,
        bpm: null,
        downbeat: 0,
      });

      render(<TrackSelector />);

      await waitFor(() => {
        expect(screen.getByText("track1")).toBeInTheDocument();
      });

      await user.click(screen.getByText("track1"));
      await user.click(screen.getByText("Load"));

      await waitFor(() => {
        const { waveformBass } = useWaveformStore.getState();
        expect(waveformBass).toEqual([0.5, 0.7]);
      });
    });

    it("loads track with reference annotations and sets tier to 1", async () => {
      const user = userEvent.setup();

      vi.mocked(trackService.getTracks).mockResolvedValue([
        { filename: "track1.mp3", has_reference: true, has_generated: false },
      ]);

      vi.mocked(trackService.loadTrack).mockResolvedValue({
        filename: "track1.mp3",
        waveform_bass: [0.5],
        waveform_mids: [0.4],
        waveform_highs: [0.2],
        waveform_times: [0],
        duration: 180,
        bpm: 128,
        downbeat: 0,
        boundaries: [
          { time: 0, label: "intro" },
          { time: 10, label: "buildup" },
        ],
        annotation_tier: 1,
      });

      render(<TrackSelector />);

      await waitFor(() => {
        expect(screen.getByText("track1")).toBeInTheDocument();
      });

      await user.click(screen.getByText("track1"));
      await user.click(screen.getByText("Load"));

      await waitFor(() => {
        const { annotationTier } = useStructureStore.getState();
        expect(annotationTier).toBe(1);
      });
    });

    it("loads track with generated annotations and sets tier to 2", async () => {
      const user = userEvent.setup();

      vi.mocked(trackService.getTracks).mockResolvedValue([
        { filename: "track1.mp3", has_reference: false, has_generated: true },
      ]);

      vi.mocked(trackService.loadTrack).mockResolvedValue({
        filename: "track1.mp3",
        waveform_bass: [0.5],
        waveform_mids: [0.4],
        waveform_highs: [0.2],
        waveform_times: [0],
        duration: 180,
        bpm: 128,
        downbeat: 0,
        boundaries: [
          { time: 0, label: "intro" },
          { time: 10, label: "buildup" },
        ],
        annotation_tier: 2,
      });

      render(<TrackSelector />);

      await waitFor(() => {
        expect(screen.getByText("track1")).toBeInTheDocument();
      });

      await user.click(screen.getByText("track1"));
      await user.click(screen.getByText("Load"));

      await waitFor(() => {
        const { annotationTier } = useStructureStore.getState();
        expect(annotationTier).toBe(2);
      });
    });

    it("adds track duration as final boundary when loading annotations", async () => {
      const user = userEvent.setup();

      vi.mocked(trackService.getTracks).mockResolvedValue([
        { filename: "track1.mp3", has_reference: true, has_generated: false },
      ]);

      vi.mocked(trackService.loadTrack).mockResolvedValue({
        filename: "track1.mp3",
        waveform_bass: [0.5],
        waveform_mids: [0.4],
        waveform_highs: [0.2],
        waveform_times: [0],
        duration: 180,
        bpm: 128,
        downbeat: 0,
        boundaries: [
          { time: 0, label: "intro" },
          { time: 10, label: "buildup" },
          { time: 20, label: "breakdown" },
        ],
        annotation_tier: 1,
      });

      render(<TrackSelector />);

      await waitFor(() => {
        expect(screen.getByText("track1")).toBeInTheDocument();
      });

      await user.click(screen.getByText("track1"));
      await user.click(screen.getByText("Load"));

      await waitFor(() => {
        const { boundaries } = useStructureStore.getState();
        expect(boundaries).toEqual([0, 10, 20, 180]);
      });
    });

    it("sets region labels from loaded annotations", async () => {
      const user = userEvent.setup();

      vi.mocked(trackService.getTracks).mockResolvedValue([
        { filename: "track1.mp3", has_reference: true, has_generated: false },
      ]);

      vi.mocked(trackService.loadTrack).mockResolvedValue({
        filename: "track1.mp3",
        waveform_bass: [0.5],
        waveform_mids: [0.4],
        waveform_highs: [0.2],
        waveform_times: [0],
        duration: 180,
        bpm: 128,
        downbeat: 0,
        boundaries: [
          { time: 0, label: "intro" },
          { time: 10, label: "buildup" },
        ],
        annotation_tier: 1,
      });

      render(<TrackSelector />);

      await waitFor(() => {
        expect(screen.getByText("track1")).toBeInTheDocument();
      });

      await user.click(screen.getByText("track1"));
      await user.click(screen.getByText("Load"));

      await waitFor(() => {
        const { regions } = useStructureStore.getState();
        expect(regions[0].label).toBe("intro");
        expect(regions[1].label).toBe("buildup");
      });
    });

    it("initializes with single region when no annotations", async () => {
      const user = userEvent.setup();

      vi.mocked(trackService.getTracks).mockResolvedValue([
        { filename: "track1.mp3", has_reference: false, has_generated: false },
      ]);

      vi.mocked(trackService.loadTrack).mockResolvedValue({
        filename: "track1.mp3",
        waveform_bass: [0.5],
        waveform_mids: [0.4],
        waveform_highs: [0.2],
        waveform_times: [0],
        duration: 180,
        bpm: null,
        downbeat: 0,
      });

      render(<TrackSelector />);

      await waitFor(() => {
        expect(screen.getByText("track1")).toBeInTheDocument();
      });

      await user.click(screen.getByText("track1"));
      await user.click(screen.getByText("Load"));

      await waitFor(() => {
        const { boundaries, annotationTier } = useStructureStore.getState();
        expect(boundaries).toEqual([0, 180]);
        expect(annotationTier).toBeNull();
      });
    });

    it("sets BPM to 0 when no annotation exists", async () => {
      const user = userEvent.setup();

      vi.mocked(trackService.getTracks).mockResolvedValue([
        { filename: "track1.mp3", has_reference: false, has_generated: false },
      ]);

      vi.mocked(trackService.loadTrack).mockResolvedValue({
        filename: "track1.mp3",
        waveform_bass: [0.5],
        waveform_mids: [0.4],
        waveform_highs: [0.2],
        waveform_times: [0],
        duration: 180,
        bpm: null,
        downbeat: 0,
      });

      render(<TrackSelector />);

      await waitFor(() => {
        expect(screen.getByText("track1")).toBeInTheDocument();
      });

      await user.click(screen.getByText("track1"));
      await user.click(screen.getByText("Load"));

      await waitFor(() => {
        const { trackBPM } = useTempoStore.getState();
        expect(trackBPM).toBe(0);
      });
    });

    it("marks state as saved after loading", async () => {
      const user = userEvent.setup();

      vi.mocked(trackService.getTracks).mockResolvedValue([
        { filename: "track1.mp3", has_reference: false, has_generated: false },
      ]);

      vi.mocked(trackService.loadTrack).mockResolvedValue({
        filename: "track1.mp3",
        waveform_bass: [0.5],
        waveform_mids: [0.4],
        waveform_highs: [0.2],
        waveform_times: [0],
        duration: 180,
        bpm: null,
        downbeat: 0,
      });

      render(<TrackSelector />);

      await waitFor(() => {
        expect(screen.getByText("track1")).toBeInTheDocument();
      });

      await user.click(screen.getByText("track1"));
      await user.click(screen.getByText("Load"));

      await waitFor(() => {
        const { isDirty } = useStructureStore.getState();
        expect(isDirty()).toBe(false);
      });
    });
  });

  describe("Empty State", () => {
    it("shows empty state when no tracks", async () => {
      vi.mocked(trackService.getTracks).mockResolvedValue([]);

      render(<TrackSelector />);

      await waitFor(() => {
        expect(screen.getByText("No tracks loaded")).toBeInTheDocument();
      });
    });
  });
});
