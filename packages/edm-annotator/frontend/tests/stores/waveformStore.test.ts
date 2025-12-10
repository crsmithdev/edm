import { describe, it, expect, beforeEach } from "vitest";
import { useWaveformStore } from "@/stores/waveformStore";

describe("waveformStore", () => {
  const mockWaveformData = {
    waveform_bass: [0.5, 0.7, 0.3, 0.9],
    waveform_mids: [0.4, 0.6, 0.5, 0.8],
    waveform_highs: [0.2, 0.3, 0.4, 0.5],
    waveform_times: [0, 1, 2, 3],
    duration: 180, // 3 minutes
  };

  beforeEach(() => {
    // Reset store before each test
    useWaveformStore.getState().reset();
  });

  describe("setWaveformData", () => {
    it("should initialize waveform data", () => {
      const { setWaveformData } = useWaveformStore.getState();

      setWaveformData(mockWaveformData);

      const state = useWaveformStore.getState();
      expect(state.waveformBass).toEqual(mockWaveformData.waveform_bass);
      expect(state.waveformMids).toEqual(mockWaveformData.waveform_mids);
      expect(state.waveformHighs).toEqual(mockWaveformData.waveform_highs);
      expect(state.waveformTimes).toEqual(mockWaveformData.waveform_times);
      expect(state.duration).toBe(mockWaveformData.duration);
    });

    it("should initialize viewport to full duration", () => {
      const { setWaveformData } = useWaveformStore.getState();

      setWaveformData(mockWaveformData);

      const { viewportStart, viewportEnd, zoomLevel } =
        useWaveformStore.getState();
      expect(viewportStart).toBe(0);
      expect(viewportEnd).toBe(mockWaveformData.duration);
      expect(zoomLevel).toBe(1.0);
    });

    it("should reset zoom level to 1.0", () => {
      const { setWaveformData, zoom } = useWaveformStore.getState();

      setWaveformData(mockWaveformData);
      zoom(2); // Zoom in

      // Set new data
      setWaveformData({
        ...mockWaveformData,
        duration: 240,
      });

      const { zoomLevel } = useWaveformStore.getState();
      expect(zoomLevel).toBe(1.0);
    });
  });

  describe("zoom", () => {
    beforeEach(() => {
      useWaveformStore.getState().setWaveformData(mockWaveformData);
    });

    it("should zoom in centered on viewport", () => {
      const { zoom } = useWaveformStore.getState();

      zoom(1); // Zoom in by 0.2

      const { zoomLevel, viewportStart, viewportEnd } =
        useWaveformStore.getState();
      expect(zoomLevel).toBe(1.2);
      expect(viewportEnd - viewportStart).toBeLessThan(
        mockWaveformData.duration
      );
    });

    it("should zoom out centered on viewport", () => {
      const { zoom } = useWaveformStore.getState();

      zoom(2); // Zoom in first
      zoom(-1); // Zoom out

      const { zoomLevel } = useWaveformStore.getState();
      expect(zoomLevel).toBe(1.2); // 1.0 + 0.4 - 0.2
    });

    it("should zoom centered on specific time", () => {
      const { zoom } = useWaveformStore.getState();

      const centerTime = 90; // Center of track
      zoom(2, centerTime);

      const { viewportStart, viewportEnd } = useWaveformStore.getState();
      const viewportCenter = (viewportStart + viewportEnd) / 2;
      expect(viewportCenter).toBeCloseTo(centerTime, 1);
    });

    it("should clamp zoom level to minimum (0.1)", () => {
      const { zoom } = useWaveformStore.getState();

      // Zoom out far beyond minimum
      for (let i = 0; i < 10; i++) {
        zoom(-1);
      }

      const { zoomLevel } = useWaveformStore.getState();
      expect(zoomLevel).toBe(0.1);
    });

    it("should clamp zoom level to maximum (10)", () => {
      const { zoom } = useWaveformStore.getState();

      // Zoom in far beyond maximum
      for (let i = 0; i < 100; i++) {
        zoom(1);
      }

      const { zoomLevel } = useWaveformStore.getState();
      expect(zoomLevel).toBe(10);
    });

    it("should clamp viewport to track start", () => {
      const { zoom } = useWaveformStore.getState();

      // Zoom in on time near start
      zoom(5, 5);

      const { viewportStart } = useWaveformStore.getState();
      expect(viewportStart).toBeGreaterThanOrEqual(0);
    });

    it("should clamp viewport to track end", () => {
      const { zoom } = useWaveformStore.getState();

      // Zoom in on time near end
      zoom(5, mockWaveformData.duration - 5);

      const { viewportEnd } = useWaveformStore.getState();
      expect(viewportEnd).toBeLessThanOrEqual(mockWaveformData.duration);
    });

    it("should maintain viewport duration based on zoom level", () => {
      const { zoom } = useWaveformStore.getState();

      zoom(1); // zoomLevel = 1.2

      const { viewportStart, viewportEnd, duration, zoomLevel } =
        useWaveformStore.getState();
      const expectedDuration = duration / zoomLevel;
      expect(viewportEnd - viewportStart).toBeCloseTo(expectedDuration, 5);
    });

    it("should zoom progressively", () => {
      const { zoom } = useWaveformStore.getState();

      zoom(1);
      const { zoomLevel: level1 } = useWaveformStore.getState();

      zoom(1);
      const { zoomLevel: level2 } = useWaveformStore.getState();

      zoom(1);
      const { zoomLevel: level3 } = useWaveformStore.getState();

      expect(level2).toBeGreaterThan(level1);
      expect(level3).toBeGreaterThan(level2);
    });
  });

  describe("zoomToFit", () => {
    beforeEach(() => {
      useWaveformStore.getState().setWaveformData(mockWaveformData);
    });

    it("should reset zoom to full track view", () => {
      const { zoom, zoomToFit } = useWaveformStore.getState();

      // Zoom in
      zoom(5);

      // Reset
      zoomToFit();

      const { zoomLevel, viewportStart, viewportEnd } =
        useWaveformStore.getState();
      expect(zoomLevel).toBe(1.0);
      expect(viewportStart).toBe(0);
      expect(viewportEnd).toBe(mockWaveformData.duration);
    });

    it("should work after panning", () => {
      const { pan, zoomToFit } = useWaveformStore.getState();

      pan(50); // Pan right
      zoomToFit();

      const { viewportStart, viewportEnd } = useWaveformStore.getState();
      expect(viewportStart).toBe(0);
      expect(viewportEnd).toBe(mockWaveformData.duration);
    });
  });

  describe("pan", () => {
    beforeEach(() => {
      useWaveformStore.getState().setWaveformData(mockWaveformData);
    });

    it("should pan viewport right", () => {
      const { pan } = useWaveformStore.getState();

      const { viewportStart: startBefore } = useWaveformStore.getState();

      pan(30);

      const { viewportStart: startAfter } = useWaveformStore.getState();
      expect(startAfter).toBe(startBefore + 30);
    });

    it("should pan viewport left", () => {
      const { pan } = useWaveformStore.getState();

      // Pan right first
      pan(50);
      const { viewportStart: startBefore } = useWaveformStore.getState();

      // Pan left
      pan(-30);

      const { viewportStart: startAfter } = useWaveformStore.getState();
      expect(startAfter).toBe(startBefore - 30);
    });

    it("should clamp panning at track start", () => {
      const { pan } = useWaveformStore.getState();

      pan(-100); // Try to pan before start

      const { viewportStart } = useWaveformStore.getState();
      expect(viewportStart).toBe(0);
    });

    it("should clamp panning at track end", () => {
      const { pan } = useWaveformStore.getState();

      pan(1000); // Try to pan past end

      const { viewportEnd } = useWaveformStore.getState();
      expect(viewportEnd).toBe(mockWaveformData.duration);
    });

    it("should maintain viewport duration during pan", () => {
      const { zoom, pan } = useWaveformStore.getState();

      zoom(2); // Zoom in first
      const {
        viewportStart: startBefore,
        viewportEnd: endBefore,
      } = useWaveformStore.getState();
      const durationBefore = endBefore - startBefore;

      pan(20);

      const {
        viewportStart: startAfter,
        viewportEnd: endAfter,
      } = useWaveformStore.getState();
      const durationAfter = endAfter - startAfter;

      expect(durationAfter).toBeCloseTo(durationBefore, 5);
    });

    it("should pan while zoomed in", () => {
      const { zoom, pan } = useWaveformStore.getState();

      zoom(3); // Zoom in
      pan(10);

      const { viewportStart } = useWaveformStore.getState();
      expect(viewportStart).toBeGreaterThan(0);
    });
  });

  describe("setViewport", () => {
    beforeEach(() => {
      useWaveformStore.getState().setWaveformData(mockWaveformData);
    });

    it("should set viewport start and end", () => {
      const { setViewport } = useWaveformStore.getState();

      setViewport(30, 90);

      const { viewportStart, viewportEnd } = useWaveformStore.getState();
      expect(viewportStart).toBe(30);
      expect(viewportEnd).toBe(90);
    });

    it("should allow arbitrary viewport ranges", () => {
      const { setViewport } = useWaveformStore.getState();

      setViewport(45, 135);

      const { viewportStart, viewportEnd } = useWaveformStore.getState();
      expect(viewportStart).toBe(45);
      expect(viewportEnd).toBe(135);
    });
  });

  describe("reset", () => {
    it("should clear all waveform data", () => {
      const { setWaveformData, zoom, pan, reset } =
        useWaveformStore.getState();

      setWaveformData(mockWaveformData);
      zoom(3);
      pan(30);

      reset();

      const state = useWaveformStore.getState();
      expect(state.waveformBass).toEqual([]);
      expect(state.waveformMids).toEqual([]);
      expect(state.waveformHighs).toEqual([]);
      expect(state.waveformTimes).toEqual([]);
      expect(state.duration).toBe(0);
      expect(state.zoomLevel).toBe(1.0);
      expect(state.viewportStart).toBe(0);
      expect(state.viewportEnd).toBe(0);
    });
  });

  describe("complex zoom and pan scenarios", () => {
    beforeEach(() => {
      useWaveformStore.getState().setWaveformData(mockWaveformData);
    });

    it("should handle zoom in, pan, zoom out sequence", () => {
      const { zoom, pan } = useWaveformStore.getState();

      zoom(3); // Zoom in
      pan(20); // Pan right
      zoom(-1); // Zoom out slightly

      const { zoomLevel, viewportStart } = useWaveformStore.getState();
      expect(zoomLevel).toBeGreaterThan(1.0);
      expect(viewportStart).toBeGreaterThan(0);
    });

    it("should handle extreme zoom and viewport boundaries", () => {
      const { zoom, pan } = useWaveformStore.getState();

      zoom(10); // Max zoom
      pan(1000); // Try to pan far right

      const { viewportEnd, duration } = useWaveformStore.getState();
      expect(viewportEnd).toBe(duration);
    });

    it("should maintain consistency across zoom operations at different centers", () => {
      const store1 = useWaveformStore.getState();
      store1.setWaveformData(mockWaveformData);
      store1.zoom(2, 60);

      const { zoomLevel: level1 } = useWaveformStore.getState();

      store1.reset();
      store1.setWaveformData(mockWaveformData);
      store1.zoom(2, 120);

      const { zoomLevel: level2 } = useWaveformStore.getState();

      // Zoom level should be same regardless of center
      expect(level1).toBe(level2);
    });

    it("should handle rapid zoom in and out", () => {
      const { zoom } = useWaveformStore.getState();

      zoom(5);
      zoom(-2);
      zoom(3);
      zoom(-4);

      const { zoomLevel } = useWaveformStore.getState();
      // Net change: +5 -2 +3 -4 = +2 → 1.0 + 2*0.2 = 1.4
      expect(zoomLevel).toBeCloseTo(1.4, 5);
    });

    it("should handle edge case: zoom at track start", () => {
      const { zoom } = useWaveformStore.getState();

      zoom(5, 0);

      const { viewportStart } = useWaveformStore.getState();
      expect(viewportStart).toBe(0);
    });

    it("should handle edge case: zoom at track end", () => {
      const { zoom } = useWaveformStore.getState();

      zoom(5, mockWaveformData.duration);

      const { viewportEnd } = useWaveformStore.getState();
      expect(viewportEnd).toBe(mockWaveformData.duration);
    });
  });

  describe("viewport calculations", () => {
    beforeEach(() => {
      useWaveformStore.getState().setWaveformData(mockWaveformData);
    });

    it("should calculate viewport duration correctly at different zoom levels", () => {
      const { zoom } = useWaveformStore.getState();

      const testZoomLevels = [1, 2, 3, -1, -2];
      const results = [];

      for (const delta of testZoomLevels) {
        zoom(delta);
        const { viewportStart, viewportEnd, duration, zoomLevel } =
          useWaveformStore.getState();
        results.push({
          zoomLevel,
          viewportDuration: viewportEnd - viewportStart,
          expectedDuration: duration / zoomLevel,
        });
      }

      for (const result of results) {
        expect(result.viewportDuration).toBeCloseTo(
          result.expectedDuration,
          5
        );
      }
    });

    it("should handle fractional zoom levels", () => {
      const { zoom } = useWaveformStore.getState();

      // Create fractional zoom level
      zoom(0.5); // Not exact step

      const { zoomLevel } = useWaveformStore.getState();
      expect(zoomLevel).toBeCloseTo(1.1, 5); // 1.0 + 0.5*0.2
    });
  });
});
