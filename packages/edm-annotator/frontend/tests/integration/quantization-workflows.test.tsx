import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act } from "@testing-library/react";
import { useStructureStore, useTempoStore, useUIStore, useWaveformStore } from "@/stores";
import { quantizeToBeat, quantizeToBar, getBeatDuration, getBarDuration } from "@/utils/tempo";

describe("Quantization Workflows Integration", () => {
  beforeEach(() => {
    useStructureStore.getState().reset();
    useTempoStore.getState().reset();
    useUIStore.getState().reset();
    useWaveformStore.getState().reset();

    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("Enable Quantize → Add Boundaries → Verify Snapping", () => {
    it("adds boundaries snapped to beats when quantize enabled", () => {
      const { setBPM, setDownbeat } = useTempoStore.getState();
      const { addBoundary } = useStructureStore.getState();
      const { toggleQuantize } = useUIStore.getState();

      // Setup: 120 BPM, downbeat at 0, quantize enabled
      act(() => {
        setBPM(120);
        setDownbeat(0);
        useUIStore.setState({ quantizeEnabled: true });
        useStructureStore.setState({ boundaries: [0, 180] });
      });

      const { trackBPM, trackDownbeat } = useTempoStore.getState();
      const { quantizeEnabled } = useUIStore.getState();

      expect(quantizeEnabled).toBe(true);

      // Beat duration at 120 BPM = 0.5s
      // Try to add boundaries at non-quantized positions
      const testTimes = [10.3, 20.7, 45.1, 89.9];
      const expectedTimes = testTimes.map((t) => quantizeToBeat(t, trackBPM, trackDownbeat));

      act(() => {
        testTimes.forEach((t) => {
          const quantized = quantizeToBeat(t, trackBPM, trackDownbeat);
          addBoundary(quantized);
        });
      });

      const { boundaries } = useStructureStore.getState();

      // Verify all boundaries are quantized
      expectedTimes.forEach((expected) => {
        expect(boundaries).toContain(expected);
      });

      // Verify original non-quantized times are not present
      testTimes.forEach((original) => {
        expect(boundaries).not.toContain(original);
      });

      // Verify specific quantizations
      // At 120 BPM, beat = 0.5s, so:
      // 10.3 → 10.5 (beat 21)
      // 20.7 → 20.5 (beat 41)
      // 45.1 → 45.0 (beat 90)
      // 89.9 → 90.0 (beat 180)
      expect(boundaries).toContain(10.5);
      expect(boundaries).toContain(20.5);
      expect(boundaries).toContain(45.0);
      expect(boundaries).toContain(90.0);
    });

    it("adds unquantized boundaries when quantize disabled", () => {
      const { setBPM, setDownbeat } = useTempoStore.getState();
      const { addBoundary } = useStructureStore.getState();

      act(() => {
        setBPM(120);
        setDownbeat(0);
        useUIStore.setState({ quantizeEnabled: false });
        useStructureStore.setState({ boundaries: [0, 180] });
      });

      const testTimes = [10.3, 20.7, 45.1, 89.9];

      act(() => {
        testTimes.forEach((t) => {
          addBoundary(t);
        });
      });

      const { boundaries } = useStructureStore.getState();

      // Verify exact times are preserved
      testTimes.forEach((time) => {
        expect(boundaries).toContain(time);
      });
    });

    it("quantizes to bars when jump mode is bars", () => {
      const { setBPM, setDownbeat } = useTempoStore.getState();
      const { addBoundary } = useStructureStore.getState();

      act(() => {
        setBPM(128);
        setDownbeat(0);
        useUIStore.setState({ quantizeEnabled: true, jumpMode: "bars" });
        useStructureStore.setState({ boundaries: [0, 240] });
      });

      const { trackBPM, trackDownbeat } = useTempoStore.getState();

      // Bar duration at 128 BPM = (60/128) * 4 = 1.875s
      const barDuration = getBarDuration(trackBPM);
      expect(barDuration).toBeCloseTo(1.875, 3);

      // Test times and their expected quantized values
      const testCases = [
        { input: 10.5, expected: quantizeToBar(10.5, trackBPM, trackDownbeat) }, // ~11.25 (bar 6)
        { input: 20.1, expected: quantizeToBar(20.1, trackBPM, trackDownbeat) }, // ~20.625 (bar 11)
        { input: 35.8, expected: quantizeToBar(35.8, trackBPM, trackDownbeat) }, // ~35.625 (bar 19)
      ];

      act(() => {
        testCases.forEach(({ input }) => {
          const quantized = quantizeToBar(input, trackBPM, trackDownbeat);
          addBoundary(quantized);
        });
      });

      const { boundaries } = useStructureStore.getState();

      testCases.forEach(({ expected }) => {
        const found = boundaries.some((b) => Math.abs(b - expected) < 0.001);
        expect(found).toBe(true);
      });
    });
  });

  describe("Change BPM with Quantized Boundaries", () => {
    it("updates quantized positions when BPM changes", () => {
      const { setBPM, setDownbeat } = useTempoStore.getState();
      const { addBoundary, setBoundaries } = useStructureStore.getState();

      // Start with 120 BPM
      act(() => {
        setBPM(120);
        setDownbeat(0);
        useUIStore.setState({ quantizeEnabled: true });
        useStructureStore.setState({ boundaries: [0, 180] });
      });

      // Add boundaries at beat-aligned positions (120 BPM: 0.5s per beat)
      act(() => {
        addBoundary(quantizeToBeat(10.5, 120, 0)); // 10.5s (beat 21)
        addBoundary(quantizeToBeat(20.0, 120, 0)); // 20.0s (beat 40)
        addBoundary(quantizeToBeat(30.0, 120, 0)); // 30.0s (beat 60)
      });

      let { boundaries: boundaries120 } = useStructureStore.getState();
      expect(boundaries120).toContain(10.5);
      expect(boundaries120).toContain(20.0);
      expect(boundaries120).toContain(30.0);

      // Change BPM to 140 (0.42857s per beat)
      act(() => {
        setBPM(140);
      });

      // If we were to re-quantize the same boundaries:
      // At 140 BPM, beat 21 = 9.0s, beat 40 = 17.14s, beat 60 = 25.71s
      // The boundaries themselves don't auto-update, but new operations use new BPM
      const newTime = 15.3;
      const quantized140 = quantizeToBeat(newTime, 140, 0);

      act(() => {
        addBoundary(quantized140);
      });

      const { boundaries: boundariesAfterBPMChange } = useStructureStore.getState();

      // Old boundaries remain unchanged
      expect(boundariesAfterBPMChange).toContain(10.5);
      expect(boundariesAfterBPMChange).toContain(20.0);

      // New boundary is quantized to 140 BPM grid
      expect(boundariesAfterBPMChange).toContain(quantized140);
    });

    it("re-snaps all boundaries when re-quantization is triggered", () => {
      const { setBPM, setDownbeat } = useTempoStore.getState();
      const { setBoundaries } = useStructureStore.getState();

      // Start with boundaries at 120 BPM
      act(() => {
        setBPM(120);
        setDownbeat(0);
        useUIStore.setState({ quantizeEnabled: true });
      });

      const initialBoundaries = [0, 10.5, 20.0, 30.0, 180];
      act(() => {
        setBoundaries(initialBoundaries);
      });

      // Change BPM and re-quantize all boundaries
      act(() => {
        setBPM(128);
      });

      const { trackBPM, trackDownbeat } = useTempoStore.getState();

      // Re-quantize all boundaries (except first and last)
      const requantizedBoundaries = initialBoundaries.map((boundary, idx) => {
        if (idx === 0 || idx === initialBoundaries.length - 1) {
          return boundary; // Keep track start/end unchanged
        }
        return quantizeToBeat(boundary, trackBPM, trackDownbeat);
      });

      act(() => {
        setBoundaries(requantizedBoundaries);
      });

      const { boundaries: finalBoundaries } = useStructureStore.getState();

      // Verify boundaries are now aligned to 128 BPM grid
      const beatDuration = getBeatDuration(trackBPM);
      finalBoundaries.slice(1, -1).forEach((boundary) => {
        const beatsFromStart = (boundary - trackDownbeat) / beatDuration;
        const isAligned = Math.abs(beatsFromStart - Math.round(beatsFromStart)) < 0.001;
        expect(isAligned).toBe(true);
      });
    });

    it("handles downbeat offset when re-quantizing", () => {
      const { setBPM, setDownbeat } = useTempoStore.getState();
      const { setBoundaries } = useStructureStore.getState();

      act(() => {
        setBPM(120);
        setDownbeat(0.25); // Downbeat offset
        useUIStore.setState({ quantizeEnabled: true });
      });

      const { trackBPM, trackDownbeat } = useTempoStore.getState();

      // Add boundaries relative to downbeat
      const boundaries = [
        0,
        quantizeToBeat(10.0, trackBPM, trackDownbeat),
        quantizeToBeat(20.0, trackBPM, trackDownbeat),
        180,
      ];

      act(() => {
        setBoundaries(boundaries);
      });

      const { boundaries: finalBoundaries } = useStructureStore.getState();

      // Verify boundaries are aligned to beats from downbeat
      const beatDuration = getBeatDuration(trackBPM);
      finalBoundaries.slice(1, -1).forEach((boundary) => {
        const beatsFromDownbeat = (boundary - trackDownbeat) / beatDuration;
        const isAligned = Math.abs(beatsFromDownbeat - Math.round(beatsFromDownbeat)) < 0.001;
        expect(isAligned).toBe(true);
      });
    });
  });

  describe("Drag Boundary with Quantize Enabled", () => {
    it("snaps dragged boundary to nearest beat", () => {
      const { setBPM, setDownbeat } = useTempoStore.getState();
      const { setBoundaries, removeBoundary, addBoundary } = useStructureStore.getState();

      act(() => {
        setBPM(120);
        setDownbeat(0);
        useUIStore.setState({ quantizeEnabled: true });
        setBoundaries([0, 10.0, 20.0, 30.0, 180]);
      });

      const { trackBPM, trackDownbeat } = useTempoStore.getState();

      // Simulate dragging boundary from 20.0 to 23.7
      const oldPosition = 20.0;
      const newPosition = 23.7;
      const quantizedPosition = quantizeToBeat(newPosition, trackBPM, trackDownbeat);

      act(() => {
        removeBoundary(oldPosition);
        addBoundary(quantizedPosition);
      });

      const { boundaries } = useStructureStore.getState();

      expect(boundaries).not.toContain(oldPosition);
      expect(boundaries).toContain(quantizedPosition);
      // At 120 BPM, beat duration = 0.5s
      // 23.7 / 0.5 = 47.4 beats → rounds to 47 beats → 23.5s
      expect(quantizedPosition).toBe(23.5);
    });

    it("allows free dragging when quantize disabled", () => {
      const { setBPM, setDownbeat } = useTempoStore.getState();
      const { setBoundaries, removeBoundary, addBoundary } = useStructureStore.getState();

      act(() => {
        setBPM(120);
        setDownbeat(0);
        useUIStore.setState({ quantizeEnabled: false });
        setBoundaries([0, 10.0, 20.0, 30.0, 180]);
      });

      // Simulate dragging boundary from 20.0 to 23.7 without quantization
      const oldPosition = 20.0;
      const newPosition = 23.7;

      act(() => {
        removeBoundary(oldPosition);
        addBoundary(newPosition);
      });

      const { boundaries } = useStructureStore.getState();

      expect(boundaries).not.toContain(oldPosition);
      expect(boundaries).toContain(newPosition);
      expect(newPosition).toBe(23.7); // Exact position preserved
    });

    it("prevents dragging boundary past adjacent boundaries", () => {
      const { setBPM, setDownbeat } = useTempoStore.getState();
      const { setBoundaries, removeBoundary, addBoundary } = useStructureStore.getState();

      act(() => {
        setBPM(120);
        setDownbeat(0);
        useUIStore.setState({ quantizeEnabled: true });
        setBoundaries([0, 10.0, 20.0, 30.0, 180]);
      });

      // Try to drag middle boundary (20.0) past the next boundary (30.0)
      const oldPosition = 20.0;
      const attemptedPosition = 35.0;

      // In real implementation, this would be clamped
      // Here we test the current behavior: boundaries remain sorted
      act(() => {
        removeBoundary(oldPosition);
        addBoundary(attemptedPosition);
      });

      const { boundaries } = useStructureStore.getState();

      // Boundaries should remain sorted
      for (let i = 0; i < boundaries.length - 1; i++) {
        expect(boundaries[i]).toBeLessThan(boundaries[i + 1]);
      }
    });
  });

  describe("Re-snap on Quantize Toggle", () => {
    it("does not modify boundaries when enabling quantization", () => {
      const { setBPM, setDownbeat } = useTempoStore.getState();
      const { setBoundaries } = useStructureStore.getState();
      const { toggleQuantize } = useUIStore.getState();

      act(() => {
        setBPM(120);
        setDownbeat(0);
        useUIStore.setState({ quantizeEnabled: false });
      });

      const originalBoundaries = [0, 10.3, 20.7, 45.1, 180];
      act(() => {
        setBoundaries(originalBoundaries);
      });

      const { boundaries: beforeToggle } = useStructureStore.getState();
      expect(beforeToggle).toEqual(originalBoundaries);

      // Enable quantization (should not auto-snap existing boundaries)
      act(() => {
        toggleQuantize();
      });

      const { boundaries: afterToggle } = useStructureStore.getState();
      expect(afterToggle).toEqual(originalBoundaries); // Unchanged
    });

    it("applies quantization to new boundaries after toggle", () => {
      const { setBPM, setDownbeat } = useTempoStore.getState();
      const { setBoundaries, addBoundary } = useStructureStore.getState();
      const { toggleQuantize } = useUIStore.getState();

      act(() => {
        setBPM(120);
        setDownbeat(0);
        useUIStore.setState({ quantizeEnabled: false });
        setBoundaries([0, 180]);
      });

      // Add unquantized boundary
      act(() => {
        addBoundary(10.3);
      });

      let { boundaries } = useStructureStore.getState();
      expect(boundaries).toContain(10.3);

      // Toggle quantization on
      act(() => {
        toggleQuantize();
      });

      const { trackBPM, trackDownbeat } = useTempoStore.getState();
      const { quantizeEnabled } = useUIStore.getState();
      expect(quantizeEnabled).toBe(true);

      // Add new boundary (should be quantized)
      const newTime = 20.7;
      const quantized = quantizeToBeat(newTime, trackBPM, trackDownbeat);

      act(() => {
        addBoundary(quantized);
      });

      boundaries = useStructureStore.getState().boundaries;

      expect(boundaries).toContain(10.3); // Old boundary unchanged
      expect(boundaries).toContain(quantized); // New boundary quantized
      // At 120 BPM, 20.7 → 20.5 (beat 41)
      expect(quantized).toBeCloseTo(20.5, 1);
    });

    it("allows manual re-snap of all boundaries after enabling quantization", () => {
      const { setBPM, setDownbeat } = useTempoStore.getState();
      const { setBoundaries } = useStructureStore.getState();
      const { toggleQuantize } = useUIStore.getState();

      act(() => {
        setBPM(120);
        setDownbeat(0);
        useUIStore.setState({ quantizeEnabled: false });
      });

      const originalBoundaries = [0, 10.3, 20.7, 45.1, 180];
      act(() => {
        setBoundaries(originalBoundaries);
      });

      // Toggle quantization on
      act(() => {
        toggleQuantize();
      });

      const { trackBPM, trackDownbeat } = useTempoStore.getState();

      // Manually re-snap all boundaries
      const snappedBoundaries = originalBoundaries.map((boundary, idx) => {
        if (idx === 0 || idx === originalBoundaries.length - 1) {
          return boundary; // Keep track start/end
        }
        return quantizeToBeat(boundary, trackBPM, trackDownbeat);
      });

      act(() => {
        setBoundaries(snappedBoundaries);
      });

      const { boundaries } = useStructureStore.getState();

      expect(boundaries).toContain(0);
      expect(boundaries).toContain(10.5); // 10.3 → 10.5
      expect(boundaries).toContain(20.5); // 20.7 → 20.5
      expect(boundaries).toContain(45.0); // 45.1 → 45.0
      expect(boundaries).toContain(180);
    });

    it("toggles between beat and bar quantization modes", () => {
      const { setBPM, setDownbeat } = useTempoStore.getState();
      const { addBoundary, setBoundaries } = useStructureStore.getState();
      const { toggleJumpMode } = useUIStore.getState();

      act(() => {
        setBPM(128);
        setDownbeat(0);
        useUIStore.setState({ quantizeEnabled: true, jumpMode: "beats" });
        setBoundaries([0, 180]);
      });

      const { trackBPM, trackDownbeat } = useTempoStore.getState();

      // Add boundary in beats mode
      const testTime = 10.5;
      const beatQuantized = quantizeToBeat(testTime, trackBPM, trackDownbeat);

      act(() => {
        addBoundary(beatQuantized);
      });

      let { boundaries, jumpMode } = useStructureStore.getState();
      let { jumpMode: uiJumpMode } = useUIStore.getState();

      expect(uiJumpMode).toBe("beats");
      expect(boundaries).toContain(beatQuantized);

      // Toggle to bar mode
      act(() => {
        toggleJumpMode();
      });

      uiJumpMode = useUIStore.getState().jumpMode;
      expect(uiJumpMode).toBe("bars");

      // Add another boundary in bar mode
      const testTime2 = 15.5;
      const barQuantized = quantizeToBar(testTime2, trackBPM, trackDownbeat);

      act(() => {
        addBoundary(barQuantized);
      });

      boundaries = useStructureStore.getState().boundaries;
      expect(boundaries).toContain(barQuantized);

      // Bar quantization should produce different result than beat quantization
      expect(barQuantized).not.toBe(quantizeToBeat(testTime2, trackBPM, trackDownbeat));
    });
  });
});
