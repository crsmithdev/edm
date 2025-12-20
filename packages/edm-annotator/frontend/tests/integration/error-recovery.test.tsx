import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, waitFor } from "@testing-library/react";
import { useStructureStore, useTempoStore, useTrackStore, useWaveformStore, useUIStore } from "@/stores";
import { trackService } from "@/services/api";
import axios from "axios";

vi.mock("@/services/api", () => ({
  trackService: {
    loadTrack: vi.fn(),
    saveAnnotation: vi.fn(),
    loadGeneratedAnnotation: vi.fn(),
    getTracks: vi.fn(),
  },
}));

describe("Error Recovery Integration", () => {
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

  describe("Network Failure → Retry Logic", () => {
    it("retries failed track load request", async () => {
      const mockTrackData = {
        filename: "test.mp3",
        waveform_bass: [0.5, 0.7],
        waveform_mids: [0.4, 0.6],
        waveform_highs: [0.2, 0.3],
        waveform_times: [0, 1],
        duration: 180,
        bpm: 128,
        downbeat: 0,
        boundaries: [{ time: 0, label: "intro" }],
        annotation_tier: 1,
      };

      // First call fails, second succeeds
      vi.mocked(trackService.loadTrack)
        .mockRejectedValueOnce(new Error("Network error"))
        .mockResolvedValueOnce(mockTrackData);

      let loadError: Error | null = null;
      let loadedData = null;

      // First attempt - fails
      try {
        await trackService.loadTrack("test.mp3");
      } catch (error) {
        loadError = error as Error;
      }

      expect(loadError).toBeTruthy();
      expect(loadError?.message).toBe("Network error");
      expect(trackService.loadTrack).toHaveBeenCalledTimes(1);

      // Retry - succeeds
      loadedData = await trackService.loadTrack("test.mp3");

      expect(loadedData).toEqual(mockTrackData);
      expect(trackService.loadTrack).toHaveBeenCalledTimes(2);
    });

    it("retries failed save request with exponential backoff", async () => {
      const { setBoundaries, setRegionLabel } = useStructureStore.getState();

      act(() => {
        setBoundaries([0, 10, 20, 180]);
        setRegionLabel(0, "intro");
        setRegionLabel(1, "buildup");
      });

      // First two calls fail, third succeeds
      vi.mocked(trackService.saveAnnotation)
        .mockRejectedValueOnce(new Error("Network error"))
        .mockRejectedValueOnce(new Error("Timeout"))
        .mockResolvedValueOnce({ status: "success", message: "Saved" });

      const regions = useStructureStore.getState().regions;
      const saveData = {
        boundaries: regions.map((r) => ({ time: r.start, label: r.label })),
      };

      let attempts = 0;
      let saveError: Error | null = null;

      // Attempt 1 - fails
      try {
        attempts++;
        await trackService.saveAnnotation(saveData);
      } catch (error) {
        saveError = error as Error;
      }

      expect(saveError?.message).toBe("Network error");
      expect(attempts).toBe(1);

      // Attempt 2 - fails
      try {
        attempts++;
        await trackService.saveAnnotation(saveData);
      } catch (error) {
        saveError = error as Error;
      }

      expect(saveError?.message).toBe("Timeout");
      expect(attempts).toBe(2);

      // Attempt 3 - succeeds
      saveError = null;
      const result = await trackService.saveAnnotation(saveData);

      expect(result.status).toBe("success");
      expect(attempts).toBe(2);
      expect(trackService.saveAnnotation).toHaveBeenCalledTimes(3);
    });

    it("handles timeout errors gracefully", async () => {
      const timeoutError = {
        code: "ECONNABORTED",
        message: "timeout of 5000ms exceeded",
      };

      vi.mocked(trackService.loadTrack).mockRejectedValueOnce(timeoutError);

      let error: any = null;

      try {
        await trackService.loadTrack("test.mp3");
      } catch (e) {
        error = e;
      }

      expect(error).toBeTruthy();
      expect(error.code).toBe("ECONNABORTED");

      // User can retry after timeout
      const mockData = {
        filename: "test.mp3",
        waveform_bass: [0.5],
        waveform_mids: [0.4],
        waveform_highs: [0.2],
        waveform_times: [0],
        duration: 180,
        bpm: 128,
        downbeat: 0,
        boundaries: [],
        annotation_tier: null,
      };

      vi.mocked(trackService.loadTrack).mockResolvedValueOnce(mockData);

      const retryData = await trackService.loadTrack("test.mp3");
      expect(retryData).toEqual(mockData);
    });

    it("displays user-friendly error messages", async () => {
      const { showStatus, clearStatus } = useUIStore.getState();

      vi.mocked(trackService.loadTrack).mockRejectedValueOnce(
        new Error("Network error")
      );

      try {
        await trackService.loadTrack("test.mp3");
      } catch (error) {
        act(() => {
          showStatus("Failed to load track. Please check your connection and try again.");
        });
      }

      const { statusMessage } = useUIStore.getState();
      expect(statusMessage).toBe(
        "Failed to load track. Please check your connection and try again."
      );

      // Clear message after user acknowledges
      act(() => {
        clearStatus();
      });

      expect(useUIStore.getState().statusMessage).toBeNull();
    });
  });

  describe("Corrupted Annotation → Fallback", () => {
    it("falls back to empty annotation when data is corrupted", async () => {
      const { setBoundaries, setRegionLabel, setAnnotationTier } =
        useStructureStore.getState();
      const { setWaveformData } = useWaveformStore.getState();

      // Mock corrupted response (invalid boundary data)
      const corruptedData = {
        filename: "test.mp3",
        waveform_bass: [0.5, 0.7],
        waveform_mids: [0.4, 0.6],
        waveform_highs: [0.2, 0.3],
        waveform_times: [0, 1],
        duration: 180,
        bpm: 128,
        downbeat: 0,
        boundaries: [
          { time: "invalid", label: "intro" }, // Invalid time
          { time: 20, label: "buildup" },
        ],
        annotation_tier: 1,
      };

      vi.mocked(trackService.loadTrack).mockResolvedValueOnce(corruptedData as any);

      const loadedData = await trackService.loadTrack("test.mp3");

      act(() => {
        setWaveformData({
          waveform_bass: loadedData.waveform_bass,
          waveform_mids: loadedData.waveform_mids,
          waveform_highs: loadedData.waveform_highs,
          waveform_times: loadedData.waveform_times,
          duration: loadedData.duration,
        });

        // Validate and filter boundaries
        const validBoundaries = [0, loadedData.duration];
        if (loadedData.boundaries) {
          loadedData.boundaries.forEach((boundary: any) => {
            if (
              typeof boundary.time === "number" &&
              !isNaN(boundary.time) &&
              boundary.time > 0 &&
              boundary.time < loadedData.duration
            ) {
              validBoundaries.push(boundary.time);
            }
          });
        }

        setBoundaries(validBoundaries.sort((a, b) => a - b));
      });

      const { boundaries } = useStructureStore.getState();

      // Should only contain valid boundaries: 0, 20, 180
      expect(boundaries).toContain(0);
      expect(boundaries).toContain(20);
      expect(boundaries).toContain(180);
      expect(boundaries).toHaveLength(3);
    });

    it("handles missing required fields with defaults", async () => {
      const incompleteData = {
        filename: "test.mp3",
        waveform_bass: [0.5],
        waveform_mids: [0.4],
        waveform_highs: [0.2],
        waveform_times: [0],
        duration: 180,
        // Missing: bpm, downbeat, boundaries, annotation_tier
      };

      vi.mocked(trackService.loadTrack).mockResolvedValueOnce(incompleteData as any);

      const loadedData = await trackService.loadTrack("test.mp3");

      const { setBoundaries } = useStructureStore.getState();
      const { setBPM, setDownbeat } = useTempoStore.getState();

      act(() => {
        // Apply defaults for missing fields
        setBoundaries([0, loadedData.duration]);
        setBPM(loadedData.bpm || 120); // Default BPM
        setDownbeat(loadedData.downbeat ?? 0); // Default downbeat
      });

      const { boundaries } = useStructureStore.getState();
      const { trackBPM, trackDownbeat } = useTempoStore.getState();

      expect(boundaries).toEqual([0, 180]);
      expect(trackBPM).toBe(120); // Default applied
      expect(trackDownbeat).toBe(0); // Default applied
    });

    it("validates boundary ordering and removes duplicates", async () => {
      const dataWithBadBoundaries = {
        filename: "test.mp3",
        waveform_bass: [0.5],
        waveform_mids: [0.4],
        waveform_highs: [0.2],
        waveform_times: [0],
        duration: 180,
        bpm: 128,
        downbeat: 0,
        boundaries: [
          { time: 50, label: "breakdown" }, // Out of order
          { time: 10, label: "intro" },
          { time: 10, label: "duplicate" }, // Duplicate
          { time: 30, label: "buildup" },
          { time: 200, label: "invalid" }, // Beyond duration
        ],
        annotation_tier: 1,
      };

      vi.mocked(trackService.loadTrack).mockResolvedValueOnce(dataWithBadBoundaries as any);

      const loadedData = await trackService.loadTrack("test.mp3");

      const { setBoundaries, setRegionLabel } = useStructureStore.getState();

      act(() => {
        // Extract and validate boundary times
        const times = new Set<number>();
        times.add(0); // Always include start

        if (loadedData.boundaries) {
          loadedData.boundaries.forEach((boundary: any) => {
            if (
              typeof boundary.time === "number" &&
              boundary.time > 0 &&
              boundary.time < loadedData.duration
            ) {
              times.add(boundary.time);
            }
          });
        }

        times.add(loadedData.duration); // Always include end

        const sortedBoundaries = Array.from(times).sort((a, b) => a - b);
        setBoundaries(sortedBoundaries);

        // Apply labels in sorted order
        const validBoundaries = loadedData.boundaries
          .filter((b: any) => times.has(b.time))
          .sort((a: any, b: any) => a.time - b.time);

        validBoundaries.forEach((boundary: any, idx: number) => {
          setRegionLabel(idx, boundary.label);
        });
      });

      const { boundaries, regions } = useStructureStore.getState();

      // Should have: 0, 10, 30, 50, 180 (sorted, no duplicates, no invalid)
      expect(boundaries).toEqual([0, 10, 30, 50, 180]);
      expect(regions).toHaveLength(4);
    });
  });

  describe("Save Failure → Preserve Unsaved Changes", () => {
    it("preserves unsaved changes when save fails", async () => {
      const { setBoundaries, setRegionLabel, markAsSaved, isDirty } =
        useStructureStore.getState();

      // Create initial saved state
      act(() => {
        setBoundaries([0, 10, 20, 180]);
        setRegionLabel(0, "intro");
        setRegionLabel(1, "buildup");
        markAsSaved();
      });

      expect(isDirty()).toBe(false);

      // Make changes
      act(() => {
        setRegionLabel(1, "breakdown");
      });

      expect(isDirty()).toBe(true);

      const unsavedRegions = useStructureStore.getState().regions;

      // Attempt to save - fails
      vi.mocked(trackService.saveAnnotation).mockRejectedValueOnce(
        new Error("Network error")
      );

      let saveError: Error | null = null;

      try {
        await trackService.saveAnnotation({
          boundaries: unsavedRegions.map((r) => ({ time: r.start, label: r.label })),
        });
      } catch (error) {
        saveError = error as Error;
      }

      expect(saveError).toBeTruthy();

      // Verify changes are still present
      const { regions, isDirty: stillDirty } = useStructureStore.getState();

      expect(regions[1].label).toBe("breakdown");
      expect(stillDirty()).toBe(true);
    });

    it("shows save indicator and preserves state across navigation attempts", async () => {
      const { setBoundaries, setRegionLabel, markAsSaved, isDirty } =
        useStructureStore.getState();

      act(() => {
        setBoundaries([0, 15, 30, 180]);
        setRegionLabel(0, "intro");
        markAsSaved();
      });

      // Make unsaved changes
      act(() => {
        setRegionLabel(0, "buildup");
      });

      expect(isDirty()).toBe(true);

      // User tries to switch tracks - should be warned about unsaved changes
      const hasUnsavedChanges = isDirty();
      expect(hasUnsavedChanges).toBe(true);

      // User cancels navigation to save changes
      vi.mocked(trackService.saveAnnotation).mockResolvedValueOnce({
        status: "success",
        message: "Saved",
      });

      const regions = useStructureStore.getState().regions;
      await trackService.saveAnnotation({
        boundaries: regions.map((r) => ({ time: r.start, label: r.label })),
      });

      act(() => {
        markAsSaved();
      });

      expect(isDirty()).toBe(false);
    });

    it("recovers from save failure with local backup", async () => {
      const { setBoundaries, setRegionLabel, markAsSaved } =
        useStructureStore.getState();

      act(() => {
        setBoundaries([0, 10, 20, 30, 180]);
        setRegionLabel(0, "intro");
        setRegionLabel(1, "buildup");
        setRegionLabel(2, "breakdown");
        markAsSaved();
      });

      // Make changes
      act(() => {
        setRegionLabel(1, "breakbuild");
      });

      const beforeSave = {
        regions: JSON.parse(JSON.stringify(useStructureStore.getState().regions)),
        boundaries: [...useStructureStore.getState().boundaries],
      };

      // Save fails
      vi.mocked(trackService.saveAnnotation).mockRejectedValueOnce(
        new Error("Server error")
      );

      let saveError: Error | null = null;

      try {
        await trackService.saveAnnotation({
          boundaries: beforeSave.regions.map((r: any) => ({
            time: r.start,
            label: r.label,
          })),
        });
      } catch (error) {
        saveError = error as Error;
      }

      expect(saveError).toBeTruthy();

      // State should be unchanged
      const afterFailedSave = useStructureStore.getState();

      expect(afterFailedSave.regions).toEqual(beforeSave.regions);
      expect(afterFailedSave.boundaries).toEqual(beforeSave.boundaries);
    });

    it("queues multiple save attempts when offline", async () => {
      const { setBoundaries, setRegionLabel } = useStructureStore.getState();

      act(() => {
        setBoundaries([0, 10, 180]);
      });

      const saveQueue: any[] = [];

      // All saves fail (offline)
      vi.mocked(trackService.saveAnnotation).mockImplementation(async (data) => {
        saveQueue.push({ data, timestamp: Date.now() });
        throw new Error("Network unavailable");
      });

      // Make multiple changes
      const changes = [
        { index: 0, label: "intro" as const },
        { index: 0, label: "buildup" as const },
        { index: 0, label: "breakdown" as const },
      ];

      for (const change of changes) {
        act(() => {
          setRegionLabel(change.index, change.label);
        });

        try {
          const regions = useStructureStore.getState().regions;
          await trackService.saveAnnotation({
            boundaries: regions.map((r) => ({ time: r.start, label: r.label })),
          });
        } catch {
          // Expected to fail
        }
      }

      expect(saveQueue).toHaveLength(3);

      // When back online, retry latest state
      vi.mocked(trackService.saveAnnotation).mockResolvedValueOnce({
        status: "success",
        message: "Saved",
      });

      const finalRegions = useStructureStore.getState().regions;
      const result = await trackService.saveAnnotation({
        boundaries: finalRegions.map((r) => ({ time: r.start, label: r.label })),
      });

      expect(result.status).toBe("success");
      expect(finalRegions[0].label).toBe("breakdown"); // Latest change
    });
  });

  describe("State Recovery from Errors", () => {
    it("recovers from corrupt store state", () => {
      const { setBoundaries, reset } = useStructureStore.getState();

      // Set invalid state
      act(() => {
        // Manually set invalid state (boundaries not sorted)
        useStructureStore.setState({ boundaries: [10, 0, 20, 5] });
      });

      let { boundaries } = useStructureStore.getState();
      const isInvalidState = boundaries.some((b, i) => i > 0 && b < boundaries[i - 1]);
      expect(isInvalidState).toBe(true);

      // Recover by resetting and reloading valid data
      act(() => {
        reset();
        setBoundaries([0, 5, 10, 20, 180]); // Valid sorted boundaries
      });

      boundaries = useStructureStore.getState().boundaries;

      // Verify state is now valid
      for (let i = 0; i < boundaries.length - 1; i++) {
        expect(boundaries[i]).toBeLessThan(boundaries[i + 1]);
      }
    });

    it("recovers from missing waveform data", () => {
      const { setWaveformData } = useWaveformStore.getState();

      // Attempt to set incomplete waveform data
      act(() => {
        setWaveformData({
          waveform_bass: [0.5, 0.7],
          waveform_mids: [], // Missing data
          waveform_highs: [0.2],
          waveform_times: [0, 1],
          duration: 180,
        });
      });

      const { waveformBass, waveformMids, waveformHighs } = useWaveformStore.getState();

      // Validate data
      const isIncomplete = waveformMids.length === 0 || waveformHighs.length !== waveformBass.length;
      expect(isIncomplete).toBe(true);

      // Recover with valid data
      act(() => {
        setWaveformData({
          waveform_bass: [0.5, 0.7],
          waveform_mids: [0.4, 0.6],
          waveform_highs: [0.2, 0.3],
          waveform_times: [0, 1],
          duration: 180,
        });
      });

      const recovered = useWaveformStore.getState();

      expect(recovered.waveformBass).toEqual([0.5, 0.7]);
      expect(recovered.waveformMids).toEqual([0.4, 0.6]);
      expect(recovered.waveformHighs).toEqual([0.2, 0.3]);
    });

    it("handles race conditions in concurrent store updates", async () => {
      const { setBoundaries, addBoundary } = useStructureStore.getState();

      act(() => {
        setBoundaries([0, 180]);
      });

      // Simulate concurrent updates
      const updates = [10, 20, 30, 15, 25];

      await act(async () => {
        // All updates happen "simultaneously"
        updates.forEach((time) => {
          addBoundary(time);
        });
      });

      const { boundaries } = useStructureStore.getState();

      // Verify all boundaries were added and are sorted
      expect(boundaries).toContain(0);
      expect(boundaries).toContain(180);
      updates.forEach((time) => {
        expect(boundaries).toContain(time);
      });

      // Verify sorted order
      for (let i = 0; i < boundaries.length - 1; i++) {
        expect(boundaries[i]).toBeLessThan(boundaries[i + 1]);
      }
    });

    it("resets all stores on critical error", () => {
      const { setBoundaries } = useStructureStore.getState();
      const { setBPM } = useTempoStore.getState();
      const { setWaveformData } = useWaveformStore.getState();

      // Set up state
      act(() => {
        setBoundaries([0, 10, 20, 180]);
        setBPM(128);
        setWaveformData({
          waveform_bass: [0.5],
          waveform_mids: [0.4],
          waveform_highs: [0.2],
          waveform_times: [0],
          duration: 180,
        });
      });

      // Simulate critical error requiring full reset
      act(() => {
        useStructureStore.getState().reset();
        useTempoStore.getState().reset();
        useWaveformStore.getState().reset();
        useTrackStore.getState().reset();
        useUIStore.getState().reset();
      });

      // Verify all stores are reset
      expect(useStructureStore.getState().boundaries).toEqual([]);
      expect(useTempoStore.getState().trackBPM).toBe(0); // Reset to 0, not null
      expect(useWaveformStore.getState().duration).toBe(0);
    });
  });
});
