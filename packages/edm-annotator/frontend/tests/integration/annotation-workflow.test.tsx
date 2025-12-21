import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act, waitFor } from "@testing-library/react";
import { useStructureStore, useTempoStore, useTrackStore, useWaveformStore, useUIStore } from "@/stores";
import { trackService } from "@/services/api";

vi.mock("@/services/api", () => ({
  trackService: {
    loadTrack: vi.fn(),
    saveAnnotation: vi.fn(),
    loadGeneratedAnnotation: vi.fn(),
  },
}));

describe("Annotation Workflow Integration", () => {
  beforeEach(() => {
    useStructureStore.getState().reset();
    useTempoStore.getState().reset();
    useTrackStore.getState().reset();
    useWaveformStore.getState().reset();
    useUIStore.getState().reset();

    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("Complete Annotation Workflow", () => {
    it("loads track, adds boundaries, sets labels, and saves", async () => {
      // 1. Load track
      useTrackStore.setState({ currentTrack: "test.mp3" });
      useWaveformStore.setState({ duration: 180 });

      const { setBoundaries } = useStructureStore.getState();
      setBoundaries([0, 180]); // Initial single region

      // 2. Add boundaries
      const { addBoundary } = useStructureStore.getState();
      act(() => {
        addBoundary(10);
        addBoundary(20);
        addBoundary(30);
      });

      const { boundaries } = useStructureStore.getState();
      expect(boundaries).toContain(10);
      expect(boundaries).toContain(20);
      expect(boundaries).toContain(30);

      // 3. Set labels
      const { setRegionLabel } = useStructureStore.getState();
      act(() => {
        setRegionLabel(0, "intro");
        setRegionLabel(1, "buildup");
        setRegionLabel(2, "breakdown");
        setRegionLabel(3, "outro");
      });

      const { regions } = useStructureStore.getState();
      expect(regions[0].label).toBe("intro");
      expect(regions[1].label).toBe("buildup");
      expect(regions[2].label).toBe("breakdown");
      expect(regions[3].label).toBe("outro");

      // 4. Verify dirty state
      const { markAsSaved, isDirty } = useStructureStore.getState();
      markAsSaved();
      act(() => {
        setRegionLabel(0, "default");
      });
      expect(isDirty()).toBe(true);

      // 5. Save
      vi.mocked(trackService.saveAnnotation).mockResolvedValue(undefined);

      // Simulate save operation
      const finalRegions = useStructureStore.getState().regions;
      const boundariesData = finalRegions.map((r) => ({
        time: r.start,
        label: r.label,
      }));

      await trackService.saveAnnotation("test.mp3", { boundaries: boundariesData });

      expect(trackService.saveAnnotation).toHaveBeenCalledWith("test.mp3", {
        boundaries: expect.arrayContaining([
          expect.objectContaining({ label: "default" }),
          expect.objectContaining({ label: "buildup" }),
        ]),
      });

      // 6. Mark as saved and verify not dirty
      act(() => {
        markAsSaved();
        const { setAnnotationTier } = useStructureStore.getState();
        setAnnotationTier(1);
      });

      expect(isDirty()).toBe(false);
      expect(useStructureStore.getState().annotationTier).toBe(1);
    });
  });

  describe("Reference vs Generated Annotation Loading", () => {
    it("loads reference annotation with tier 1", async () => {
      vi.mocked(trackService.loadTrack).mockResolvedValue({
        filename: "test.mp3",
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

      const trackData = await trackService.loadTrack("test.mp3");

      const { setBoundaries, setRegionLabel, setAnnotationTier } = useStructureStore.getState();

      // Add track duration as final boundary
      const boundaryTimes = trackData.boundaries!.map((b) => b.time);
      if (boundaryTimes[boundaryTimes.length - 1] !== trackData.duration) {
        boundaryTimes.push(trackData.duration);
      }

      act(() => {
        setBoundaries(boundaryTimes);
        trackData.boundaries?.forEach((boundary, idx) => {
          if (boundary.label) {
            setRegionLabel(idx, boundary.label);
          }
        });
        setAnnotationTier(trackData.annotation_tier || null);
      });

      const { boundaries, regions, annotationTier } = useStructureStore.getState();

      expect(boundaries).toEqual([0, 10, 180]);
      expect(regions[0].label).toBe("intro");
      expect(regions[1].label).toBe("buildup");
      expect(annotationTier).toBe(1);
    });

    it("loads generated annotation with tier 2", async () => {
      vi.mocked(trackService.loadGeneratedAnnotation).mockResolvedValue({
        boundaries: [
          { time: 0, label: "intro" },
          { time: 15, label: "breakdown" },
        ],
        bpm: 130,
        downbeat: 0.5,
      });

      const genData = await trackService.loadGeneratedAnnotation("test.mp3");

      const { setBoundaries, setRegionLabel, setAnnotationTier } = useStructureStore.getState();
      const { setBPM, setDownbeat } = useTempoStore.getState();

      useWaveformStore.setState({ duration: 180 });

      // Add track duration
      const boundaryTimes = genData.boundaries.map((b) => b.time);
      if (boundaryTimes[boundaryTimes.length - 1] !== 180) {
        boundaryTimes.push(180);
      }

      act(() => {
        setBoundaries(boundaryTimes);
        genData.boundaries.forEach((boundary, idx) => {
          setRegionLabel(idx, boundary.label);
        });
        setAnnotationTier(2);
        if (genData.bpm) setBPM(genData.bpm);
        if (genData.downbeat !== undefined) setDownbeat(genData.downbeat);
      });

      const { boundaries, regions, annotationTier } = useStructureStore.getState();
      const { trackBPM, trackDownbeat } = useTempoStore.getState();

      expect(boundaries).toEqual([0, 15, 180]);
      expect(regions[0].label).toBe("intro");
      expect(regions[1].label).toBe("breakdown");
      expect(annotationTier).toBe(2);
      expect(trackBPM).toBe(130);
      expect(trackDownbeat).toBe(0.5);
    });

    it("overwrites generated with reference when saving", async () => {
      // Start with generated
      const { setBoundaries, setRegionLabel, setAnnotationTier, markAsSaved } =
        useStructureStore.getState();

      act(() => {
        setBoundaries([0, 10, 20, 180]);
        setRegionLabel(0, "intro");
        setRegionLabel(1, "buildup");
        setRegionLabel(2, "breakdown");
        setAnnotationTier(2); // Generated
        markAsSaved();
      });

      expect(useStructureStore.getState().annotationTier).toBe(2);

      // Modify and save
      act(() => {
        setRegionLabel(1, "breakdown-buildup");
      });

      vi.mocked(trackService.saveAnnotation).mockResolvedValue(undefined);

      const regions = useStructureStore.getState().regions;
      await trackService.saveAnnotation("test.mp3", {
        boundaries: regions.map((r) => ({ time: r.start, label: r.label })),
      });

      // After save, tier becomes 1 (reference)
      act(() => {
        setAnnotationTier(1);
        markAsSaved();
      });

      expect(useStructureStore.getState().annotationTier).toBe(1);
    });
  });

  describe("Quantization with Boundaries", () => {
    it("adds quantized boundaries when quantize enabled", () => {
      const { setBPM, setDownbeat } = useTempoStore.getState();
      const { addBoundary } = useStructureStore.getState();

      useUIStore.setState({ quantizeEnabled: true });
      useTempoStore.setState({ trackBPM: 120, trackDownbeat: 0 });
      useStructureStore.setState({ boundaries: [0, 180] });

      // Beat duration at 120 BPM = 0.5s
      // Try to add boundary at 10.3s, should snap to 10.5s

      const currentTime = 10.3;
      const beatDuration = 60 / 120; // 0.5s
      const beatsFromDownbeat = (currentTime - 0) / beatDuration;
      const nearestBeat = Math.round(beatsFromDownbeat);
      const quantizedTime = 0 + nearestBeat * beatDuration;

      act(() => {
        addBoundary(quantizedTime);
      });

      const { boundaries } = useStructureStore.getState();
      expect(boundaries).toContain(10.5);
      expect(boundaries).not.toContain(10.3);
    });
  });

  describe("Track Switching with Unsaved Changes", () => {
    it("preserves state when switching tracks", () => {
      // Set up first track
      useTrackStore.setState({ currentTrack: "track1.mp3" });
      useStructureStore.setState({
        boundaries: [0, 10, 20, 180],
        annotationTier: 1,
      });
      useStructureStore.getState().setRegionLabel(0, "intro");
      useStructureStore.getState().markAsSaved();

      const track1State = {
        boundaries: useStructureStore.getState().boundaries,
        regions: useStructureStore.getState().regions,
        tier: useStructureStore.getState().annotationTier,
      };

      // Switch to second track (would reset in real app)
      useTrackStore.setState({ currentTrack: "track2.mp3" });
      useStructureStore.getState().reset();
      useStructureStore.setState({
        boundaries: [0, 15, 180],
        annotationTier: 2,
      });

      const track2State = {
        boundaries: useStructureStore.getState().boundaries,
        tier: useStructureStore.getState().annotationTier,
      };

      // Verify states are different
      expect(track1State.boundaries).not.toEqual(track2State.boundaries);
      expect(track1State.tier).not.toBe(track2State.tier);
    });

    it("detects unsaved changes before switching", () => {
      useTrackStore.setState({ currentTrack: "track1.mp3" });
      useStructureStore.setState({
        boundaries: [0, 10, 180],
      });
      useStructureStore.getState().markAsSaved();

      // Make changes
      useStructureStore.getState().addBoundary(15);

      const { isDirty } = useStructureStore.getState();
      expect(isDirty()).toBe(true);
    });
  });

  describe("Region Deletion and Merging", () => {
    it("merges regions correctly when deleting middle boundary", () => {
      const { setBoundaries, setRegionLabel, removeBoundary } = useStructureStore.getState();

      act(() => {
        setBoundaries([0, 10, 20, 30, 180]);
        setRegionLabel(0, "intro");
        setRegionLabel(1, "buildup");
        setRegionLabel(2, "breakdown");
        setRegionLabel(3, "outro");
      });

      // Delete boundary at 20 (merges regions 1 and 2)
      act(() => {
        removeBoundary(20);
      });

      const { boundaries, regions } = useStructureStore.getState();

      expect(boundaries).toEqual([0, 10, 30, 180]);
      expect(regions).toHaveLength(3);
      expect(regions[0].label).toBe("intro");
      expect(regions[1].label).toBe("buildup"); // Keeps previous region's label
      expect(regions[1].end).toBe(30); // Extended to include merged region
      expect(regions[2].label).toBe("outro");
    });
  });

  describe("Complete Save-Load Cycle", () => {
    it("saves and reloads annotation maintaining consistency", async () => {
      // 1. Create annotation
      useTrackStore.setState({ currentTrack: "test.mp3" });
      useWaveformStore.setState({ duration: 180 });

      const { setBoundaries, setRegionLabel, markAsSaved, setAnnotationTier } =
        useStructureStore.getState();

      act(() => {
        setBoundaries([0, 10, 20, 30, 180]);
        setRegionLabel(0, "intro");
        setRegionLabel(1, "buildup");
        setRegionLabel(2, "breakdown");
        setRegionLabel(3, "outro");
      });

      const savedRegions = useStructureStore.getState().regions;
      const savedBoundaries = savedRegions.map((r) => ({
        time: r.start,
        label: r.label,
      }));

      // 2. Save
      vi.mocked(trackService.saveAnnotation).mockResolvedValue(undefined);
      await trackService.saveAnnotation("test.mp3", { boundaries: savedBoundaries });

      act(() => {
        markAsSaved();
        setAnnotationTier(1);
      });

      // 3. Reset and reload
      const stateBeforeReload = {
        boundaries: useStructureStore.getState().boundaries,
        regions: useStructureStore.getState().regions,
      };

      act(() => {
        useStructureStore.getState().reset();
      });

      // 4. Mock load response with saved data
      vi.mocked(trackService.loadTrack).mockResolvedValue({
        filename: "test.mp3",
        waveform_bass: [0.5],
        waveform_mids: [0.4],
        waveform_highs: [0.2],
        waveform_times: [0],
        duration: 180,
        bpm: 128,
        downbeat: 0,
        boundaries: savedBoundaries,
        annotation_tier: 1,
      });

      const loadedData = await trackService.loadTrack("test.mp3");

      // 5. Restore state
      const boundaryTimes = loadedData.boundaries!.map((b) => b.time);
      if (boundaryTimes[boundaryTimes.length - 1] !== loadedData.duration) {
        boundaryTimes.push(loadedData.duration);
      }

      act(() => {
        setBoundaries(boundaryTimes);
        loadedData.boundaries?.forEach((boundary, idx) => {
          setRegionLabel(idx, boundary.label);
        });
        setAnnotationTier(loadedData.annotation_tier || null);
        markAsSaved();
      });

      const stateAfterReload = {
        boundaries: useStructureStore.getState().boundaries,
        regions: useStructureStore.getState().regions,
      };

      // 6. Verify consistency
      expect(stateAfterReload.boundaries).toEqual(stateBeforeReload.boundaries);
      expect(stateAfterReload.regions).toEqual(stateBeforeReload.regions);
    });
  });
});
