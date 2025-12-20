import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useTrackStore, useStructureStore, useTempoStore, useWaveformStore, useAudioStore } from "@/stores";
import { trackService } from "@/services/api";

vi.mock("@/services/api", () => ({
  trackService: {
    loadTrack: vi.fn(),
    saveAnnotation: vi.fn(),
  },
}));

describe("Track Switching Integration", () => {
  beforeEach(() => {
    useTrackStore.getState().reset();
    useStructureStore.getState().reset();
    useTempoStore.getState().reset();
    useWaveformStore.getState().reset();
    useAudioStore.getState().reset();

    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("Switching Tracks", () => {
    it("resets all stores when switching tracks", async () => {
      // Set up Track 1
      useTrackStore.setState({ currentTrack: "track1.mp3" });
      useStructureStore.setState({
        boundaries: [0, 10, 20, 180],
        annotationTier: 1,
      });
      useTempoStore.setState({ trackBPM: 120, trackDownbeat: 0 });
      useWaveformStore.setState({
        waveformBass: [0.5, 0.7],
        duration: 180,
      });

      // Switch to Track 2
      act(() => {
        useStructureStore.getState().reset();
        useTempoStore.getState().reset();
        useWaveformStore.getState().reset();
        useAudioStore.getState().reset();
      });

      // Load Track 2
      vi.mocked(trackService.loadTrack).mockResolvedValue({
        filename: "track2.mp3",
        waveform_bass: [0.3, 0.4],
        waveform_mids: [0.2, 0.3],
        waveform_highs: [0.1, 0.2],
        waveform_times: [0, 1],
        duration: 200,
        bpm: 140,
        downbeat: 0.5,
        boundaries: [{ time: 0, label: "intro" }],
        annotation_tier: 2,
      });

      const track2Data = await trackService.loadTrack("track2.mp3");

      const { setWaveformData } = useWaveformStore.getState();
      const { setBPM, setDownbeat } = useTempoStore.getState();
      const { setBoundaries, setRegionLabel, setAnnotationTier } = useStructureStore.getState();

      act(() => {
        setWaveformData({
          waveform_bass: track2Data.waveform_bass,
          waveform_mids: track2Data.waveform_mids,
          waveform_highs: track2Data.waveform_highs,
          waveform_times: track2Data.waveform_times,
          duration: track2Data.duration,
        });

        const boundaryTimes = [0, track2Data.duration];
        setBoundaries(boundaryTimes);
        setRegionLabel(0, "intro");

        if (track2Data.bpm) setBPM(track2Data.bpm);
        setDownbeat(track2Data.downbeat);
        setAnnotationTier(track2Data.annotation_tier || null);

        useTrackStore.setState({ currentTrack: "track2.mp3" });
      });

      // Verify Track 2 state
      const structureState = useStructureStore.getState();
      const tempoState = useTempoStore.getState();
      const waveformState = useWaveformStore.getState();
      const trackState = useTrackStore.getState();

      expect(trackState.currentTrack).toBe("track2.mp3");
      expect(waveformState.duration).toBe(200);
      expect(tempoState.trackBPM).toBe(140);
      expect(structureState.annotationTier).toBe(2);
      expect(structureState.regions[0].label).toBe("intro");
    });

    it("warns when switching with unsaved changes", () => {
      // Set up track with unsaved changes
      useTrackStore.setState({ currentTrack: "track1.mp3" });
      useStructureStore.setState({
        boundaries: [0, 10, 180],
        savedState: {
          regions: [{ start: 0, end: 10, label: "intro" }, { start: 10, end: 180, label: "buildup" }],
          boundaries: [0, 10, 180],
        },
      });

      // Make changes
      const { addBoundary, isDirty } = useStructureStore.getState();
      act(() => {
        addBoundary(15);
      });

      const hasDirtyChanges = isDirty();
      expect(hasDirtyChanges).toBe(true);

      // In real app, would show confirmation dialog before switching
    });

    it("preserves saved state when switching between tracks", async () => {
      // Track 1 - Create and save annotation
      useTrackStore.setState({ currentTrack: "track1.mp3" });
      useWaveformStore.setState({ duration: 180 });

      const { setBoundaries, setRegionLabel, markAsSaved, setAnnotationTier } =
        useStructureStore.getState();

      act(() => {
        setBoundaries([0, 10, 20, 180]);
        setRegionLabel(0, "intro");
        setRegionLabel(1, "buildup");
      });

      vi.mocked(trackService.saveAnnotation).mockResolvedValue(undefined);

      const track1Regions = useStructureStore.getState().regions;
      await trackService.saveAnnotation("track1.mp3", {
        boundaries: track1Regions.map((r) => ({ time: r.start, label: r.label })),
      });

      const track1SavedState = {
        regions: [...track1Regions],
        boundaries: [...useStructureStore.getState().boundaries],
      };

      act(() => {
        markAsSaved();
        setAnnotationTier(1);
      });

      // Switch to Track 2
      act(() => {
        useStructureStore.getState().reset();
        useTrackStore.setState({ currentTrack: "track2.mp3" });
      });

      // Load Track 2 with different annotation
      vi.mocked(trackService.loadTrack).mockResolvedValue({
        filename: "track2.mp3",
        waveform_bass: [0.5],
        waveform_mids: [0.4],
        waveform_highs: [0.2],
        waveform_times: [0],
        duration: 200,
        bpm: 140,
        downbeat: 0,
        boundaries: [{ time: 0, label: "breakdown" }],
        annotation_tier: 2,
      });

      const track2Data = await trackService.loadTrack("track2.mp3");

      act(() => {
        const boundaryTimes = [0, track2Data.duration];
        setBoundaries(boundaryTimes);
        setRegionLabel(0, "breakdown");
        setAnnotationTier(2);
        markAsSaved();
      });

      const track2State = {
        regions: [...useStructureStore.getState().regions],
        boundaries: [...useStructureStore.getState().boundaries],
        tier: useStructureStore.getState().annotationTier,
      };

      // Switch back to Track 1
      act(() => {
        useStructureStore.getState().reset();
        useTrackStore.setState({ currentTrack: "track1.mp3" });
      });

      // Reload Track 1
      vi.mocked(trackService.loadTrack).mockResolvedValue({
        filename: "track1.mp3",
        waveform_bass: [0.5],
        waveform_mids: [0.4],
        waveform_highs: [0.2],
        waveform_times: [0],
        duration: 180,
        bpm: 120,
        downbeat: 0,
        boundaries: track1Regions.map((r) => ({ time: r.start, label: r.label })),
        annotation_tier: 1,
      });

      const track1ReloadedData = await trackService.loadTrack("track1.mp3");

      act(() => {
        const boundaryTimes = track1ReloadedData.boundaries!.map((b) => b.time);
        if (boundaryTimes[boundaryTimes.length - 1] !== track1ReloadedData.duration) {
          boundaryTimes.push(track1ReloadedData.duration);
        }
        setBoundaries(boundaryTimes);
        track1ReloadedData.boundaries?.forEach((boundary, idx) => {
          setRegionLabel(idx, boundary.label);
        });
        setAnnotationTier(1);
        markAsSaved();
      });

      const track1ReloadedState = {
        regions: useStructureStore.getState().regions,
        boundaries: useStructureStore.getState().boundaries,
        tier: useStructureStore.getState().annotationTier,
      };

      // Verify Track 1 state was preserved
      expect(track1ReloadedState.regions).toEqual(track1SavedState.regions);
      expect(track1ReloadedState.boundaries).toEqual(track1SavedState.boundaries);
      expect(track1ReloadedState.tier).toBe(1);

      // Verify Track 2 was different
      expect(track2State.tier).toBe(2);
      expect(track2State.regions[0].label).toBe("breakdown");
    });
  });

  describe("Navigation Between Tracks", () => {
    it("implements previousTrack navigation", () => {
      const tracks = ["track1.mp3", "track2.mp3", "track3.mp3"];

      useTrackStore.setState({
        tracks: tracks.map((filename) => ({
          filename,
          has_reference: false,
          has_generated: false,
        })),
        selectedTrack: "track2.mp3",
      });

      const { previousTrack } = useTrackStore.getState();

      act(() => {
        previousTrack();
      });

      const { selectedTrack } = useTrackStore.getState();
      expect(selectedTrack).toBe("track1.mp3");
    });

    it("implements nextTrack navigation", () => {
      const tracks = ["track1.mp3", "track2.mp3", "track3.mp3"];

      useTrackStore.setState({
        tracks: tracks.map((filename) => ({
          filename,
          has_reference: false,
          has_generated: false,
        })),
        selectedTrack: "track2.mp3",
      });

      const { nextTrack } = useTrackStore.getState();

      act(() => {
        nextTrack();
      });

      const { selectedTrack } = useTrackStore.getState();
      expect(selectedTrack).toBe("track3.mp3");
    });

    it("wraps around at start when going to previous track", () => {
      const tracks = ["track1.mp3", "track2.mp3", "track3.mp3"];

      useTrackStore.setState({
        tracks: tracks.map((filename) => ({
          filename,
          has_reference: false,
          has_generated: false,
        })),
        selectedTrack: "track1.mp3",
      });

      const { previousTrack } = useTrackStore.getState();

      act(() => {
        previousTrack();
      });

      const { selectedTrack } = useTrackStore.getState();
      expect(selectedTrack).toBe("track3.mp3");
    });

    it("wraps around at end when going to next track", () => {
      const tracks = ["track1.mp3", "track2.mp3", "track3.mp3"];

      useTrackStore.setState({
        tracks: tracks.map((filename) => ({
          filename,
          has_reference: false,
          has_generated: false,
        })),
        selectedTrack: "track3.mp3",
      });

      const { nextTrack } = useTrackStore.getState();

      act(() => {
        nextTrack();
      });

      const { selectedTrack } = useTrackStore.getState();
      expect(selectedTrack).toBe("track1.mp3");
    });
  });
});
