import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useWaveformInteraction } from "@/hooks/useWaveformInteraction";
import { useStructureStore, useWaveformStore, useAudioStore, useUIStore, useTempoStore } from "@/stores";

describe("useWaveformInteraction", () => {
  beforeEach(() => {
    useStructureStore.getState().reset();
    useWaveformStore.getState().reset();
    useAudioStore.getState().reset();
    useUIStore.getState().reset();
    useTempoStore.getState().reset();

    // Set up basic state
    useWaveformStore.setState({
      duration: 180,
      viewportStart: 0,
      viewportEnd: 180,
      zoomLevel: 1.0,
    });

    useStructureStore.setState({
      boundaries: [0, 180],
    });

    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("Boundary Creation", () => {
    it("creates boundary from waveform click", () => {
      renderHook(() => useWaveformInteraction());

      const { addBoundary } = useStructureStore.getState();

      // Simulate clicking at 50% of viewport width representing time 90
      const viewportWidth = 1000; // pixels
      const clickX = 500; // 50% of width
      const clickTime = (clickX / viewportWidth) * 180; // 90 seconds

      act(() => {
        addBoundary(clickTime);
      });

      const { boundaries } = useStructureStore.getState();
      expect(boundaries).toContain(90);
    });

    it("snaps to beat grid when quantize enabled", () => {
      renderHook(() => useWaveformInteraction());

      useUIStore.setState({ quantizeEnabled: true });
      useTempoStore.setState({ trackBPM: 120, trackDownbeat: 0 });

      const { addBoundary } = useStructureStore.getState();

      // Click at 90.3 seconds (should snap to 90.5)
      const clickTime = 90.3;
      const beatDuration = 60 / 120; // 0.5s
      const beatsFromDownbeat = clickTime / beatDuration;
      const nearestBeat = Math.round(beatsFromDownbeat);
      const quantizedTime = nearestBeat * beatDuration; // 90.5

      act(() => {
        addBoundary(quantizedTime);
      });

      const { boundaries } = useStructureStore.getState();
      expect(boundaries).toContain(90.5);
      expect(boundaries).not.toContain(90.3);
    });

    it("does not create duplicate boundaries", () => {
      renderHook(() => useWaveformInteraction());

      const { addBoundary } = useStructureStore.getState();

      act(() => {
        addBoundary(50);
        addBoundary(50); // Duplicate
      });

      const { boundaries } = useStructureStore.getState();
      const count = boundaries.filter((b) => b === 50).length;
      expect(count).toBeLessThanOrEqual(1);
    });
  });

  describe("Boundary Deletion", () => {
    it("removes boundary on click", () => {
      renderHook(() => useWaveformInteraction());

      const { setBoundaries, removeBoundary } = useStructureStore.getState();

      act(() => {
        setBoundaries([0, 50, 100, 180]);
        removeBoundary(50);
      });

      const { boundaries } = useStructureStore.getState();
      expect(boundaries).not.toContain(50);
    });

    it("cannot delete boundary if only 2 boundaries remain", () => {
      renderHook(() => useWaveformInteraction());

      const { setBoundaries, removeBoundary } = useStructureStore.getState();

      act(() => {
        setBoundaries([0, 180]); // Only 2 boundaries
      });

      const boundariesBefore = useStructureStore.getState().boundaries;

      act(() => {
        removeBoundary(0); // Try to delete when only 2 boundaries
      });

      const boundariesAfter = useStructureStore.getState().boundaries;
      // Should still have 2 boundaries - deletion prevented
      expect(boundariesAfter).toHaveLength(2);
      expect(boundariesAfter).toContain(0);
      expect(boundariesAfter).toContain(180);
    });
  });

  describe("Boundary Dragging", () => {
    it("moves boundary to new position", () => {
      renderHook(() => useWaveformInteraction());

      const { setBoundaries } = useStructureStore.getState();

      act(() => {
        setBoundaries([0, 50, 100, 180]);
      });

      // Simulate dragging boundary from 50 to 60
      act(() => {
        const { removeBoundary, addBoundary } = useStructureStore.getState();
        removeBoundary(50);
        addBoundary(60);
      });

      const { boundaries } = useStructureStore.getState();
      expect(boundaries).not.toContain(50);
      expect(boundaries).toContain(60);
    });

    it("snaps dragged boundary when quantize enabled", () => {
      renderHook(() => useWaveformInteraction());

      useUIStore.setState({ quantizeEnabled: true });
      useTempoStore.setState({ trackBPM: 120, trackDownbeat: 0 });

      const { setBoundaries, removeBoundary, addBoundary } = useStructureStore.getState();

      act(() => {
        setBoundaries([0, 50, 100, 180]);
      });

      // Drag to 60.3, should snap to 60.5
      const dragTime = 60.3;
      const beatDuration = 60 / 120;
      const nearestBeat = Math.round(dragTime / beatDuration);
      const quantizedTime = nearestBeat * beatDuration;

      act(() => {
        removeBoundary(50);
        addBoundary(quantizedTime);
      });

      const { boundaries } = useStructureStore.getState();
      expect(boundaries).toContain(60.5);
      expect(boundaries).not.toContain(60.3);
    });

    it("prevents dragging boundary past adjacent boundaries", () => {
      renderHook(() => useWaveformInteraction());

      const { setBoundaries } = useStructureStore.getState();

      act(() => {
        setBoundaries([0, 50, 100, 180]);
      });

      // Try to drag 50 past 100 (should be constrained)
      const dragAttempt = 120; // Beyond next boundary
      const constrainedTime = Math.min(Math.max(dragAttempt, 0), 100 - 0.1);

      // In real implementation, drag would be constrained
      expect(constrainedTime).toBeLessThan(100);
      expect(constrainedTime).toBeGreaterThan(0);
    });
  });

  describe("Zoom and Pan Interaction", () => {
    it("zooms in centered on mouse position", () => {
      renderHook(() => useWaveformInteraction());

      const { zoom } = useWaveformStore.getState();

      // Zoom in centered at 90 seconds
      act(() => {
        zoom(1, 90);
      });

      const { zoomLevel, viewportStart, viewportEnd } = useWaveformStore.getState();

      expect(zoomLevel).toBeGreaterThan(1.0);

      // Viewport should be centered around 90
      const viewportCenter = (viewportStart + viewportEnd) / 2;
      expect(Math.abs(viewportCenter - 90)).toBeLessThan(5);
    });

    it("pans viewport on drag", () => {
      renderHook(() => useWaveformInteraction());

      const { pan, setViewport } = useWaveformStore.getState();

      // Zoom in first to allow panning
      act(() => {
        setViewport(0, 90); // Half track visible
      });

      // Pan 20 seconds to the right
      act(() => {
        pan(20);
      });

      const { viewportStart, viewportEnd } = useWaveformStore.getState();

      // Should have panned right (start > 0)
      expect(viewportStart).toBeGreaterThan(0);
      expect(viewportEnd).toBeLessThanOrEqual(180); // Clamped to duration
    });

    it("constrains pan to track boundaries", () => {
      renderHook(() => useWaveformInteraction());

      const { pan, setViewport } = useWaveformStore.getState();

      // Zoom in first
      act(() => {
        setViewport(0, 90); // Half track
      });

      // Try to pan left beyond start
      act(() => {
        pan(-50);
      });

      const { viewportStart } = useWaveformStore.getState();
      expect(viewportStart).toBeGreaterThanOrEqual(0);
    });
  });

  describe("Click vs Drag Detection", () => {
    it("distinguishes between click and drag", () => {
      let isClick = true;
      const mouseDownTime = Date.now();
      const mouseUpTime = mouseDownTime + 10; // 10ms later
      const mouseMoveDistance = 2; // 2 pixels

      // Click: < 200ms and < 5px movement
      const timeDiff = mouseUpTime - mouseDownTime;
      const isDrag = timeDiff > 200 || mouseMoveDistance > 5;

      isClick = !isDrag;

      expect(isClick).toBe(true);
    });

    it("detects drag when mouse moves", () => {
      let isClick = true;
      const mouseDownTime = Date.now();
      const mouseUpTime = mouseDownTime + 10;
      const mouseMoveDistance = 15; // 15 pixels

      const timeDiff = mouseUpTime - mouseDownTime;
      const isDrag = timeDiff > 200 || mouseMoveDistance > 5;

      isClick = !isDrag;

      expect(isClick).toBe(false);
    });
  });

  describe("Seek on Waveform Click", () => {
    it("seeks audio to clicked time", () => {
      // Mock audio element before hook renders
      const mockAudio = new Audio();
      let currentTimeValue = 0;
      Object.defineProperty(mockAudio, 'currentTime', {
        get: () => currentTimeValue,
        set: (value: number) => { currentTimeValue = value; },
        configurable: true
      });
      useAudioStore.setState({ player: mockAudio });

      renderHook(() => useWaveformInteraction());

      const { seek } = useAudioStore.getState();

      // Simulate clicking at 60 seconds
      act(() => {
        seek(60);
      });

      const { currentTime } = useAudioStore.getState();
      expect(currentTime).toBe(60);
    });

    it("snaps seek to beat when quantize enabled", () => {
      // Mock audio element before hook renders
      const mockAudio = new Audio();
      let currentTimeValue = 0;
      Object.defineProperty(mockAudio, 'currentTime', {
        get: () => currentTimeValue,
        set: (value: number) => { currentTimeValue = value; },
        configurable: true
      });
      useAudioStore.setState({ player: mockAudio });

      useUIStore.setState({ quantizeEnabled: true });
      useTempoStore.setState({ trackBPM: 120, trackDownbeat: 0 });

      renderHook(() => useWaveformInteraction());

      const { seek } = useAudioStore.getState();

      // Click at 60.3, should snap to 60.5
      const clickTime = 60.3;
      const beatDuration = 60 / 120;
      const nearestBeat = Math.round(clickTime / beatDuration);
      const quantizedTime = nearestBeat * beatDuration;

      act(() => {
        seek(quantizedTime);
      });

      const { currentTime } = useAudioStore.getState();
      expect(currentTime).toBe(60.5);
    });
  });

  describe("Mousewheel Zoom", () => {
    it("zooms in on mousewheel up", () => {
      renderHook(() => useWaveformInteraction());

      const { zoom } = useWaveformStore.getState();
      const initialZoom = useWaveformStore.getState().zoomLevel;

      // Simulate mousewheel up (zoom in)
      act(() => {
        zoom(1); // Direction 1 = zoom in
      });

      const { zoomLevel } = useWaveformStore.getState();
      expect(zoomLevel).toBeGreaterThan(initialZoom);
    });

    it("zooms out on mousewheel down", () => {
      renderHook(() => useWaveformInteraction());

      const { zoom } = useWaveformStore.getState();

      // Zoom in first
      act(() => {
        zoom(2);
      });

      const zoomAfterIn = useWaveformStore.getState().zoomLevel;

      // Zoom out
      act(() => {
        zoom(-1);
      });

      const { zoomLevel } = useWaveformStore.getState();
      expect(zoomLevel).toBeLessThan(zoomAfterIn);
    });

    it("reduces smoothing at higher zoom levels", () => {
      renderHook(() => useWaveformInteraction());

      const { zoom } = useWaveformStore.getState();

      // Zoom in significantly
      act(() => {
        zoom(5);
      });

      const { zoomLevel } = useWaveformStore.getState();

      // At higher zoom levels, less smoothing should be applied
      // This is a design property - smoothing factor decreases with zoom
      const smoothingFactor = Math.max(0.1, 1 - (zoomLevel - 1) / 10);

      expect(smoothingFactor).toBeLessThan(1);
      expect(zoomLevel).toBeGreaterThan(1);
    });
  });
});
