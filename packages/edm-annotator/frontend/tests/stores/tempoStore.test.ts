import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
import { useTempoStore } from "@/stores/tempoStore";

describe("tempoStore", () => {
  beforeEach(() => {
    // Reset store before each test
    useTempoStore.getState().reset();
    // Mock Date.now for tap tempo tests
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  describe("setBPM", () => {
    it("should update BPM", () => {
      const { setBPM } = useTempoStore.getState();

      setBPM(140);

      const { trackBPM } = useTempoStore.getState();
      expect(trackBPM).toBe(140);
    });

    it("should accept fractional BPM", () => {
      const { setBPM } = useTempoStore.getState();

      setBPM(127.5);

      const { trackBPM } = useTempoStore.getState();
      expect(trackBPM).toBe(127.5);
    });
  });

  describe("setDownbeat", () => {
    it("should update downbeat time", () => {
      const { setDownbeat } = useTempoStore.getState();

      setDownbeat(2.5);

      const { trackDownbeat } = useTempoStore.getState();
      expect(trackDownbeat).toBe(2.5);
    });

    it("should accept zero as downbeat", () => {
      const { setDownbeat } = useTempoStore.getState();

      setDownbeat(0);

      const { trackDownbeat } = useTempoStore.getState();
      expect(trackDownbeat).toBe(0);
    });
  });

  describe("tapTempo", () => {
    it("should calculate BPM from 2 taps", () => {
      const { tapTempo } = useTempoStore.getState();

      // First tap at 0s
      vi.setSystemTime(0);
      tapTempo();

      // Second tap at 0.5s (120 BPM: 60/0.5 = 120)
      vi.setSystemTime(500);
      tapTempo();

      const { trackBPM } = useTempoStore.getState();
      expect(trackBPM).toBe(120);
    });

    it("should average intervals from multiple taps", () => {
      const { tapTempo } = useTempoStore.getState();

      // Tap at 0s, 0.5s, 1.0s, 1.5s
      // Intervals: 0.5, 0.5, 0.5
      // Average: 0.5s → 120 BPM
      vi.setSystemTime(0);
      tapTempo();

      vi.setSystemTime(500);
      tapTempo();

      vi.setSystemTime(1000);
      tapTempo();

      vi.setSystemTime(1500);
      tapTempo();

      const { trackBPM } = useTempoStore.getState();
      expect(trackBPM).toBe(120);
    });

    it("should keep only last 4 taps", () => {
      const { tapTempo } = useTempoStore.getState();

      // 5 taps, should drop first one
      for (let i = 0; i < 5; i++) {
        vi.setSystemTime(i * 500);
        tapTempo();
      }

      const { tapTimes } = useTempoStore.getState();
      expect(tapTimes).toHaveLength(4);
      expect(tapTimes[0]).toBe(0.5); // First tap dropped
    });

    it("should handle varying tap intervals", () => {
      const { tapTempo } = useTempoStore.getState();

      // Intervals: 0.4s, 0.5s, 0.6s
      // Average: 0.5s → 120 BPM
      vi.setSystemTime(0);
      tapTempo();

      vi.setSystemTime(400);
      tapTempo();

      vi.setSystemTime(900);
      tapTempo();

      vi.setSystemTime(1500);
      tapTempo();

      const { trackBPM } = useTempoStore.getState();
      expect(trackBPM).toBe(120);
    });

    it("should round BPM to 1 decimal place", () => {
      const { tapTempo } = useTempoStore.getState();

      // Interval: 0.467s → 128.477... BPM → rounds to 128.5
      vi.setSystemTime(0);
      tapTempo();

      vi.setSystemTime(467);
      tapTempo();

      const { trackBPM } = useTempoStore.getState();
      expect(trackBPM).toBe(128.5);
    });

    it("should calculate 140 BPM correctly", () => {
      const { tapTempo } = useTempoStore.getState();

      // 140 BPM: beat duration = 60/140 = 0.42857s
      const beatDuration = (60 / 140) * 1000; // in ms

      vi.setSystemTime(0);
      tapTempo();

      vi.setSystemTime(beatDuration);
      tapTempo();

      vi.setSystemTime(beatDuration * 2);
      tapTempo();

      const { trackBPM } = useTempoStore.getState();
      expect(trackBPM).toBe(140);
    });

    it("should calculate 170 BPM (drum & bass tempo)", () => {
      const { tapTempo } = useTempoStore.getState();

      // 170 BPM: beat duration = 60/170 ≈ 0.353s
      const beatDuration = (60 / 170) * 1000;

      vi.setSystemTime(0);
      tapTempo();

      vi.setSystemTime(beatDuration);
      tapTempo();

      vi.setSystemTime(beatDuration * 2);
      tapTempo();

      const { trackBPM } = useTempoStore.getState();
      expect(trackBPM).toBe(170);
    });

    it("should not update BPM with single tap", () => {
      const { tapTempo } = useTempoStore.getState();

      vi.setSystemTime(0);
      tapTempo();

      const { trackBPM, tapTimes } = useTempoStore.getState();
      expect(trackBPM).toBe(0); // Default value (undefined BPM)
      expect(tapTimes).toHaveLength(1);
    });
  });

  describe("resetTapTempo", () => {
    it("should clear tap history", () => {
      const { tapTempo, resetTapTempo } = useTempoStore.getState();

      vi.setSystemTime(0);
      tapTempo();
      vi.setSystemTime(500);
      tapTempo();

      resetTapTempo();

      const { tapTimes } = useTempoStore.getState();
      expect(tapTimes).toEqual([]);
    });

    it("should not affect current BPM", () => {
      const { tapTempo, resetTapTempo } = useTempoStore.getState();

      vi.setSystemTime(0);
      tapTempo();
      vi.setSystemTime(500);
      tapTempo();

      const { trackBPM: bpmBefore } = useTempoStore.getState();
      resetTapTempo();
      const { trackBPM: bpmAfter } = useTempoStore.getState();

      expect(bpmAfter).toBe(bpmBefore);
    });
  });

  describe("reset", () => {
    it("should reset to default values", () => {
      const { setBPM, setDownbeat, tapTempo, reset } =
        useTempoStore.getState();

      setBPM(140);
      setDownbeat(5);
      vi.setSystemTime(0);
      tapTempo();

      reset();

      const { trackBPM, trackDownbeat, tapTimes } = useTempoStore.getState();
      expect(trackBPM).toBe(0); // Default value (undefined BPM displays as "--")
      expect(trackDownbeat).toBe(0);
      expect(tapTimes).toEqual([]);
    });
  });

  describe("timeToBar selector", () => {
    it("should convert time to bar number", () => {
      const { setBPM, setDownbeat, timeToBar } = useTempoStore.getState();

      setBPM(120);
      setDownbeat(0);

      // Bar duration at 120 BPM = (60/120) * 4 = 2 seconds
      expect(timeToBar(0)).toBe(1); // First bar
      expect(timeToBar(2)).toBe(2); // Second bar
      expect(timeToBar(4)).toBe(3); // Third bar
    });

    it("should handle non-zero downbeat", () => {
      const { setBPM, setDownbeat, timeToBar } = useTempoStore.getState();

      setBPM(128);
      setDownbeat(1);

      // Bar duration at 128 BPM = (60/128) * 4 = 1.875 seconds
      expect(timeToBar(1)).toBe(1); // Downbeat is bar 1
      expect(timeToBar(2.875)).toBe(2); // Next bar
    });

    it("should return 1 for times before downbeat", () => {
      const { setBPM, setDownbeat, timeToBar } = useTempoStore.getState();

      setBPM(120);
      setDownbeat(2);

      expect(timeToBar(0)).toBe(1);
      expect(timeToBar(1)).toBe(1);
    });
  });

  describe("barToTime selector", () => {
    it("should convert bar number to time", () => {
      const { setBPM, setDownbeat, barToTime } = useTempoStore.getState();

      setBPM(120);
      setDownbeat(0);

      // Bar duration at 120 BPM = 2 seconds
      expect(barToTime(1)).toBe(0);
      expect(barToTime(2)).toBe(2);
      expect(barToTime(5)).toBe(8);
    });

    it("should handle non-zero downbeat", () => {
      const { setBPM, setDownbeat, barToTime } = useTempoStore.getState();

      setBPM(128);
      setDownbeat(1);

      // Bar duration at 128 BPM = 1.875 seconds
      expect(barToTime(1)).toBe(1);
      expect(barToTime(2)).toBeCloseTo(2.875, 5);
    });
  });

  describe("quantizeToBeat selector", () => {
    it("should quantize to nearest beat", () => {
      const { setBPM, setDownbeat, quantizeToBeat } =
        useTempoStore.getState();

      setBPM(120);
      setDownbeat(0);

      // Beat duration at 120 BPM = 0.5 seconds
      expect(quantizeToBeat(0.2)).toBe(0); // Closer to 0
      expect(quantizeToBeat(0.3)).toBe(0.5); // Closer to 0.5
      expect(quantizeToBeat(0.7)).toBe(0.5); // Closer to 0.5
      expect(quantizeToBeat(0.8)).toBe(1.0); // Closer to 1.0
    });

    it("should quantize with non-zero downbeat", () => {
      const { setBPM, setDownbeat, quantizeToBeat } =
        useTempoStore.getState();

      setBPM(128);
      setDownbeat(1);

      // Beat duration at 128 BPM = 60/128 = 0.46875 seconds
      expect(quantizeToBeat(1.2)).toBeCloseTo(1, 5); // Closer to downbeat
      expect(quantizeToBeat(1.4)).toBeCloseTo(1.46875, 5); // Closer to next beat
    });

    it("should handle exact beat positions", () => {
      const { setBPM, setDownbeat, quantizeToBeat } =
        useTempoStore.getState();

      setBPM(120);
      setDownbeat(0);

      expect(quantizeToBeat(0)).toBe(0);
      expect(quantizeToBeat(0.5)).toBe(0.5);
      expect(quantizeToBeat(1.0)).toBe(1.0);
    });
  });

  describe("getBeatDuration selector", () => {
    it("should calculate beat duration correctly", () => {
      const { setBPM, getBeatDuration } = useTempoStore.getState();

      setBPM(120);
      expect(getBeatDuration()).toBe(0.5); // 60/120 = 0.5s

      setBPM(128);
      expect(getBeatDuration()).toBeCloseTo(0.46875, 5); // 60/128

      setBPM(140);
      expect(getBeatDuration()).toBeCloseTo(0.42857, 5); // 60/140
    });

    it("should handle fractional BPM", () => {
      const { setBPM, getBeatDuration } = useTempoStore.getState();

      setBPM(127.5);
      expect(getBeatDuration()).toBeCloseTo(0.47058823, 5); // 60/127.5
    });
  });

  describe("edge cases", () => {
    it("should handle very fast tempo (200 BPM)", () => {
      const { setBPM, getBeatDuration } = useTempoStore.getState();

      setBPM(200);
      expect(getBeatDuration()).toBe(0.3); // 60/200
    });

    it("should handle very slow tempo (60 BPM)", () => {
      const { setBPM, getBeatDuration } = useTempoStore.getState();

      setBPM(60);
      expect(getBeatDuration()).toBe(1.0); // 60/60
    });

    it("should handle rapid tap tempo changes", () => {
      const { tapTempo, resetTapTempo } = useTempoStore.getState();

      // First set of taps
      vi.setSystemTime(0);
      tapTempo();
      vi.setSystemTime(500);
      tapTempo();

      const { trackBPM: bpm1 } = useTempoStore.getState();

      // Reset and tap again
      resetTapTempo();
      vi.setSystemTime(1000);
      tapTempo();
      vi.setSystemTime(1400);
      tapTempo();

      const { trackBPM: bpm2 } = useTempoStore.getState();

      expect(bpm1).toBe(120);
      expect(bpm2).toBe(150);
    });
  });
});
