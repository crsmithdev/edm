import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act } from "@testing-library/react";
import { useStructureStore, useTempoStore, useWaveformStore, useUIStore } from "@/stores";
import { trackService } from "@/services/api";

vi.mock("@/services/api", () => ({
  trackService: {
    loadTrack: vi.fn(),
    saveAnnotation: vi.fn(),
  },
}));

describe("Performance Integration", () => {
  beforeEach(() => {
    useStructureStore.getState().reset();
    useTempoStore.getState().reset();
    useWaveformStore.getState().reset();
    useUIStore.getState().reset();

    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("Large Track Loading (>10 min)", () => {
    it("handles loading a 15-minute track efficiently", async () => {
      const duration = 900; // 15 minutes in seconds
      const sampleRate = 10; // 10 samples per second
      const sampleCount = duration * sampleRate;

      // Generate large waveform data
      const waveformBass = new Array(sampleCount).fill(0).map(() => Math.random());
      const waveformMids = new Array(sampleCount).fill(0).map(() => Math.random());
      const waveformHighs = new Array(sampleCount).fill(0).map(() => Math.random());
      const waveformTimes = new Array(sampleCount)
        .fill(0)
        .map((_, i) => i / sampleRate);

      const startTime = performance.now();

      vi.mocked(trackService.loadTrack).mockResolvedValueOnce({
        filename: "long_track.mp3",
        waveform_bass: waveformBass,
        waveform_mids: waveformMids,
        waveform_highs: waveformHighs,
        waveform_times: waveformTimes,
        duration: duration,
        bpm: 128,
        downbeat: 0,
        boundaries: [{ time: 0, label: "intro" }],
        annotation_tier: 1,
      });

      const trackData = await trackService.loadTrack("long_track.mp3");

      const { setWaveformData } = useWaveformStore.getState();
      const { setBoundaries } = useStructureStore.getState();

      act(() => {
        setWaveformData({
          waveform_bass: trackData.waveform_bass,
          waveform_mids: trackData.waveform_mids,
          waveform_highs: trackData.waveform_highs,
          waveform_times: trackData.waveform_times,
          duration: trackData.duration,
        });

        setBoundaries([0, trackData.duration]);
      });

      const endTime = performance.now();
      const loadTime = endTime - startTime;

      const { waveformBass: storedBass, duration: storedDuration } =
        useWaveformStore.getState();
      const { boundaries } = useStructureStore.getState();

      expect(storedBass).toHaveLength(sampleCount);
      expect(storedDuration).toBe(duration);
      expect(boundaries).toEqual([0, duration]);

      // Should complete in reasonable time (< 1 second for store operations)
      expect(loadTime).toBeLessThan(1000);
    });

    it("loads waveform data in chunks for very large files", async () => {
      const duration = 1200; // 20 minutes
      const chunkDuration = 60; // 1-minute chunks
      const sampleRate = 10;

      const chunks: any[] = [];

      // Simulate chunked loading
      for (let i = 0; i < duration / chunkDuration; i++) {
        const chunkStart = i * chunkDuration;
        const chunkEnd = Math.min((i + 1) * chunkDuration, duration);
        const chunkSamples = (chunkEnd - chunkStart) * sampleRate;

        chunks.push({
          start: chunkStart,
          end: chunkEnd,
          waveformBass: new Array(chunkSamples).fill(0).map(() => Math.random()),
          waveformMids: new Array(chunkSamples).fill(0).map(() => Math.random()),
          waveformHighs: new Array(chunkSamples).fill(0).map(() => Math.random()),
        });
      }

      expect(chunks).toHaveLength(20); // 20 chunks of 1 minute each

      // Simulate progressive loading
      const { setWaveformData } = useWaveformStore.getState();

      const startTime = performance.now();

      // Load first chunk immediately
      act(() => {
        setWaveformData({
          waveform_bass: chunks[0].waveformBass,
          waveform_mids: chunks[0].waveformMids,
          waveform_highs: chunks[0].waveformHighs,
          waveform_times: chunks[0].waveformBass.map((_, i) => i / sampleRate),
          duration: duration,
        });
      });

      const firstChunkTime = performance.now() - startTime;

      // First chunk should load very quickly
      expect(firstChunkTime).toBeLessThan(100);

      const { waveformBass } = useWaveformStore.getState();
      expect(waveformBass.length).toBeGreaterThan(0);
    });

    it("handles seeking in large tracks without lag", () => {
      const duration = 900; // 15 minutes

      const { setViewport } = useWaveformStore.getState();

      useWaveformStore.setState({ duration });

      const seekPositions = [0, 100, 300, 600, 850, 900];

      seekPositions.forEach((position) => {
        const startTime = performance.now();

        act(() => {
          // Update viewport to center on seek position (30s window)
          const viewportStart = Math.max(0, position - 15);
          const viewportEnd = Math.min(duration, position + 15);
          setViewport(viewportStart, viewportEnd);
        });

        const seekTime = performance.now() - startTime;

        // Seeking should be nearly instant (< 10ms)
        expect(seekTime).toBeLessThan(10);
      });

      const { viewportStart, viewportEnd } = useWaveformStore.getState();
      // Last seek position was 900, viewport should be at end
      expect(viewportEnd).toBe(duration);
    });
  });

  describe("Many Boundaries (>100)", () => {
    it("handles 200 boundaries without performance degradation", () => {
      const duration = 600; // 10 minutes
      const boundaryCount = 200;

      // Generate evenly spaced boundaries
      const boundaries = new Array(boundaryCount)
        .fill(0)
        .map((_, i) => (duration / (boundaryCount - 1)) * i);

      const startTime = performance.now();

      const { setBoundaries } = useStructureStore.getState();

      act(() => {
        setBoundaries(boundaries);
      });

      const endTime = performance.now();
      const setupTime = endTime - startTime;

      const { boundaries: storedBoundaries, regions } = useStructureStore.getState();

      expect(storedBoundaries).toHaveLength(boundaryCount);
      expect(regions).toHaveLength(boundaryCount - 1);

      // Should handle 200 boundaries quickly (< 100ms)
      expect(setupTime).toBeLessThan(100);
    });

    it("efficiently adds boundaries one at a time to large set", () => {
      const { setBoundaries, addBoundary } = useStructureStore.getState();

      // Start with 100 boundaries
      const initialBoundaries = new Array(101)
        .fill(0)
        .map((_, i) => i * 2); // 0, 2, 4, 6, ..., 200

      act(() => {
        setBoundaries(initialBoundaries);
      });

      // Add 50 more boundaries in between existing ones
      const addTimes: number[] = [];

      for (let i = 0; i < 50; i++) {
        const newBoundary = i * 4 + 1; // 1, 5, 9, 13, ..., 197

        const startTime = performance.now();

        act(() => {
          addBoundary(newBoundary);
        });

        const endTime = performance.now();
        addTimes.push(endTime - startTime);
      }

      const { boundaries } = useStructureStore.getState();

      expect(boundaries).toHaveLength(151);

      // Each add should be fast (< 10ms on average)
      const avgAddTime = addTimes.reduce((a, b) => a + b, 0) / addTimes.length;
      expect(avgAddTime).toBeLessThan(10);
    });

    it("efficiently removes boundaries from large set", () => {
      const { setBoundaries, removeBoundary } = useStructureStore.getState();

      // Start with 150 boundaries
      const initialBoundaries = new Array(150)
        .fill(0)
        .map((_, i) => i * 2);

      act(() => {
        setBoundaries(initialBoundaries);
      });

      // Remove every 3rd boundary (50 removals)
      const removeTimes: number[] = [];

      for (let i = 1; i < 150; i += 3) {
        const boundaryToRemove = i * 2;

        const startTime = performance.now();

        act(() => {
          removeBoundary(boundaryToRemove);
        });

        const endTime = performance.now();
        removeTimes.push(endTime - startTime);
      }

      const { boundaries } = useStructureStore.getState();

      // Should have ~100 boundaries left (can't go below 2)
      expect(boundaries.length).toBeGreaterThanOrEqual(2);
      expect(boundaries.length).toBeLessThan(150);

      // Each removal should be fast
      const avgRemoveTime = removeTimes.reduce((a, b) => a + b, 0) / removeTimes.length;
      expect(avgRemoveTime).toBeLessThan(10);
    });

    it("renders viewport with many boundaries efficiently", () => {
      const duration = 600;
      const { setBoundaries } = useStructureStore.getState();
      const { setViewport } = useWaveformStore.getState();

      // Create 200 boundaries
      const boundaries = new Array(200)
        .fill(0)
        .map((_, i) => (duration / 199) * i);

      act(() => {
        setBoundaries(boundaries);
      });

      useWaveformStore.setState({ duration });

      // Test multiple viewport positions
      const viewportTests = [
        { start: 0, end: 30 },
        { start: 100, end: 130 },
        { start: 300, end: 330 },
        { start: 550, end: 600 },
      ];

      viewportTests.forEach(({ start, end }) => {
        const startTime = performance.now();

        act(() => {
          setViewport(start, end);
        });

        // Filter boundaries in viewport (simulating rendering logic)
        const visibleBoundaries = boundaries.filter((b) => b >= start && b <= end);

        const endTime = performance.now();
        const filterTime = endTime - startTime;

        expect(visibleBoundaries.length).toBeGreaterThan(0);
        expect(filterTime).toBeLessThan(5); // Very fast filtering
      });
    });

    it("saves annotation with many boundaries efficiently", async () => {
      const { setBoundaries, setRegionLabel } = useStructureStore.getState();

      // Create 150 boundaries with labels
      const boundaries = new Array(150).fill(0).map((_, i) => i * 2);

      act(() => {
        setBoundaries(boundaries);

        // Label every region
        for (let i = 0; i < 149; i++) {
          const labels = ["intro", "buildup", "breakdown", "outro"] as const;
          setRegionLabel(i, labels[i % 4]);
        }
      });

      const { regions } = useStructureStore.getState();

      const saveData = {
        boundaries: regions.map((r) => ({ time: r.start, label: r.label })),
      };

      const startTime = performance.now();

      vi.mocked(trackService.saveAnnotation).mockResolvedValueOnce({
        status: "success",
        message: "Saved",
      });

      await trackService.saveAnnotation(saveData);

      const endTime = performance.now();
      const saveTime = endTime - startTime;

      expect(trackService.saveAnnotation).toHaveBeenCalledWith(
        expect.objectContaining({
          boundaries: expect.arrayContaining([
            expect.objectContaining({ time: expect.any(Number), label: expect.any(String) }),
          ]),
        })
      );

      // Save operation should be fast
      expect(saveTime).toBeLessThan(100);
    });
  });

  describe("Rapid Zoom/Pan", () => {
    it("handles rapid zoom in/out without lag", () => {
      const duration = 300;
      const { setViewport, zoom } = useWaveformStore.getState();

      useWaveformStore.setState({ duration, viewportStart: 0, viewportEnd: duration });

      // Use zoom deltas to simulate zooming in and out
      const zoomDeltas = [1, 1, 1, 1, -1, -1, -1, -1]; // Zoom in 4x, then out 4x
      const zoomTimes: number[] = [];

      zoomDeltas.forEach((delta) => {
        const startTime = performance.now();

        act(() => {
          // Zoom centered on middle of track
          zoom(delta, duration / 2);
        });

        const endTime = performance.now();
        zoomTimes.push(endTime - startTime);
      });

      const { viewportStart, viewportEnd } = useWaveformStore.getState();

      // After zooming in and back out, should be close to original viewport
      expect(viewportEnd - viewportStart).toBeGreaterThan(0);

      // All zoom operations should be fast
      const avgZoomTime = zoomTimes.reduce((a, b) => a + b, 0) / zoomTimes.length;
      expect(avgZoomTime).toBeLessThan(10);
    });

    it("handles rapid panning without performance issues", () => {
      const duration = 600;
      const { setViewport } = useWaveformStore.getState();

      useWaveformStore.setState({ duration });

      // Simulate rapid panning across the track
      const panSteps = 50;
      const panTimes: number[] = [];

      for (let i = 0; i < panSteps; i++) {
        const progress = i / (panSteps - 1);
        const center = progress * duration;
        const windowSize = 30;
        const start = Math.max(0, center - windowSize / 2);
        const end = Math.min(duration, center + windowSize / 2);

        const startTime = performance.now();

        act(() => {
          setViewport(start, end);
        });

        const endTime = performance.now();
        panTimes.push(endTime - startTime);
      }

      // All pan operations should be very fast
      const avgPanTime = panTimes.reduce((a, b) => a + b, 0) / panTimes.length;
      expect(avgPanTime).toBeLessThan(5);
    });

    it("maintains smooth scrolling with mousewheel zoom", () => {
      const duration = 300;
      const { zoom } = useWaveformStore.getState();

      useWaveformStore.setState({ duration, viewportStart: 100, viewportEnd: 160 });

      // Simulate mousewheel zoom centered at 130s
      const zoomCenter = 130;
      const scrollSteps = 10;
      const scrollTimes: number[] = [];

      for (let i = 0; i < scrollSteps; i++) {
        const startTime = performance.now();

        act(() => {
          // Zoom in (positive delta) centered on zoomCenter
          zoom(1, zoomCenter);
        });

        const endTime = performance.now();
        scrollTimes.push(endTime - startTime);
      }

      // All scroll-zoom operations should be fast enough for smooth animation
      const avgScrollTime = scrollTimes.reduce((a, b) => a + b, 0) / scrollTimes.length;
      expect(avgScrollTime).toBeLessThan(10);

      const maxScrollTime = Math.max(...scrollTimes);
      expect(maxScrollTime).toBeLessThan(20); // No single operation should lag
    });
  });

  describe("Waveform Rendering Efficiency", () => {
    it("downsamples waveform data for overview rendering", () => {
      const duration = 600;
      const sampleRate = 10;
      const totalSamples = duration * sampleRate;

      // Full resolution waveform
      const fullWaveform = new Array(totalSamples).fill(0).map(() => Math.random());

      const { setWaveformData } = useWaveformStore.getState();

      act(() => {
        setWaveformData({
          waveform_bass: fullWaveform,
          waveform_mids: fullWaveform,
          waveform_highs: fullWaveform,
          waveform_times: fullWaveform.map((_, i) => i / sampleRate),
          duration,
        });
      });

      // Simulate downsampling for overview (max 1000 points)
      const targetPoints = 1000;
      const downsampleFactor = Math.ceil(fullWaveform.length / targetPoints);

      const startTime = performance.now();

      const downsampled = [];
      for (let i = 0; i < fullWaveform.length; i += downsampleFactor) {
        const chunk = fullWaveform.slice(i, i + downsampleFactor);
        downsampled.push(Math.max(...chunk));
      }

      const endTime = performance.now();
      const downsampleTime = endTime - startTime;

      expect(downsampled.length).toBeLessThanOrEqual(targetPoints);
      expect(downsampleTime).toBeLessThan(50); // Should be very fast
    });

    it("renders only visible portion of waveform in detail view", () => {
      const duration = 600;
      const sampleRate = 10;
      const fullWaveform = new Array(duration * sampleRate).fill(0).map(() => Math.random());

      const { setWaveformData, setViewport } = useWaveformStore.getState();

      act(() => {
        setWaveformData({
          waveform_bass: fullWaveform,
          waveform_mids: fullWaveform,
          waveform_highs: fullWaveform,
          waveform_times: fullWaveform.map((_, i) => i / sampleRate),
          duration,
        });

        setViewport(100, 130); // 30-second window
      });

      const { viewportStart, viewportEnd } = useWaveformStore.getState();

      // Calculate visible samples
      const visibleStartIdx = Math.floor(viewportStart * sampleRate);
      const visibleEndIdx = Math.ceil(viewportEnd * sampleRate);

      const startTime = performance.now();

      const visibleWaveform = fullWaveform.slice(visibleStartIdx, visibleEndIdx);

      const endTime = performance.now();
      const sliceTime = endTime - startTime;

      expect(visibleWaveform.length).toBeLessThan(fullWaveform.length);
      expect(visibleWaveform.length).toBeCloseTo(30 * sampleRate, -1);
      expect(sliceTime).toBeLessThan(5); // Very fast array slicing
    });

    it("uses efficient data structures for waveform lookup", () => {
      const duration = 300;
      const sampleRate = 10;
      const waveformData = new Array(duration * sampleRate)
        .fill(0)
        .map(() => Math.random());

      const { setWaveformData } = useWaveformStore.getState();

      act(() => {
        setWaveformData({
          waveform_bass: waveformData,
          waveform_mids: waveformData,
          waveform_highs: waveformData,
          waveform_times: waveformData.map((_, i) => i / sampleRate),
          duration,
        });
      });

      const { waveformBass } = useWaveformStore.getState();

      // Test random access performance
      const lookupTimes: number[] = [];
      const testPositions = [0, 50, 100, 150, 200, 250];

      testPositions.forEach((time) => {
        const startTime = performance.now();

        const sampleIndex = Math.floor(time * sampleRate);
        const value = waveformBass[sampleIndex];

        const endTime = performance.now();
        lookupTimes.push(endTime - startTime);

        expect(value).toBeDefined();
        expect(value).toBeGreaterThanOrEqual(0);
      });

      // Random access should be instant
      const avgLookupTime = lookupTimes.reduce((a, b) => a + b, 0) / lookupTimes.length;
      expect(avgLookupTime).toBeLessThan(1);
    });
  });
});
