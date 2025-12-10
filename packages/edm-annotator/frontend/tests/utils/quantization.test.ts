import { describe, it, expect } from "vitest";
import { quantizeToBeat, quantizeToBar } from "@/utils/quantization";

describe("quantizeToBeat", () => {
  describe("basic functionality", () => {
    it("should snap to nearest beat", () => {
      const bpm = 120;
      const downbeat = 0;
      const beatDuration = 60 / 120; // 0.5 seconds

      // Snap forward
      expect(quantizeToBeat(0.3, bpm, downbeat)).toBe(0.5);

      // Snap backward
      expect(quantizeToBeat(0.6, bpm, downbeat)).toBe(0.5);

      // Exact beat
      expect(quantizeToBeat(1.0, bpm, downbeat)).toBe(1.0);
    });

    it("should handle different BPMs", () => {
      const downbeat = 0;

      // BPM 60 -> beat duration = 1 second
      expect(quantizeToBeat(1.4, 60, downbeat)).toBe(1.0);
      expect(quantizeToBeat(1.6, 60, downbeat)).toBe(2.0);

      // BPM 90 -> beat duration = 0.666... seconds
      const beatDuration90 = 60 / 90;
      expect(quantizeToBeat(1.0, 90, downbeat)).toBeCloseTo(
        beatDuration90,
        10
      );
    });

    it("should handle non-zero downbeat", () => {
      const bpm = 120;
      const downbeat = 2.0;
      const beatDuration = 60 / 120; // 0.5 seconds

      // First beat after downbeat
      expect(quantizeToBeat(2.3, bpm, downbeat)).toBe(2.5);

      // Second beat after downbeat
      expect(quantizeToBeat(2.8, bpm, downbeat)).toBe(3.0);
    });
  });

  describe("edge cases", () => {
    it("should handle time exactly at downbeat", () => {
      const bpm = 120;
      const downbeat = 1.5;

      expect(quantizeToBeat(1.5, bpm, downbeat)).toBe(1.5);
    });

    it("should handle time before downbeat", () => {
      const bpm = 120;
      const downbeat = 2.0;
      const beatDuration = 60 / 120; // 0.5 seconds

      // One beat before downbeat
      expect(quantizeToBeat(1.5, bpm, downbeat)).toBe(1.5);

      // Just before downbeat
      expect(quantizeToBeat(1.8, bpm, downbeat)).toBe(2.0);
    });

    it("should handle negative times with positive downbeat", () => {
      const bpm = 120;
      const downbeat = 2.0;
      const beatDuration = 60 / 120; // 0.5 seconds

      // Negative time should quantize relative to downbeat
      expect(quantizeToBeat(-0.5, bpm, downbeat)).toBe(-0.5);
    });

    it("should handle time far from downbeat", () => {
      const bpm = 120;
      const downbeat = 1.0;
      const beatDuration = 60 / 120; // 0.5 seconds

      // Many beats after downbeat
      expect(quantizeToBeat(100.3, bpm, downbeat)).toBe(100.5);
    });

    it("should handle high BPMs", () => {
      const bpm = 180;
      const downbeat = 0;
      const beatDuration = 60 / 180; // 0.333... seconds

      expect(quantizeToBeat(0.5, bpm, downbeat)).toBeCloseTo(
        beatDuration,
        10
      );
    });

    it("should handle low BPMs", () => {
      const bpm = 60;
      const downbeat = 0;
      const beatDuration = 60 / 60; // 1 second

      expect(quantizeToBeat(1.4, bpm, downbeat)).toBe(1.0);
      expect(quantizeToBeat(1.6, bpm, downbeat)).toBe(2.0);
    });
  });

  describe("parametrized tests", () => {
    const testCases = [
      { time: 0.0, bpm: 120, downbeat: 0, expected: 0.0 },
      { time: 0.25, bpm: 120, downbeat: 0, expected: 0.5 },
      { time: 0.5, bpm: 120, downbeat: 0, expected: 0.5 },
      { time: 1.0, bpm: 120, downbeat: 0, expected: 1.0 },
      { time: 2.3, bpm: 120, downbeat: 2.0, expected: 2.5 },
      { time: 5.7, bpm: 90, downbeat: 0, expected: 60 / 90 * 9 },
    ];

    testCases.forEach(({ time, bpm, downbeat, expected }) => {
      it(`quantizeToBeat(${time}, ${bpm}, ${downbeat}) should be ${expected}`, () => {
        expect(quantizeToBeat(time, bpm, downbeat)).toBeCloseTo(expected, 10);
      });
    });
  });
});

describe("quantizeToBar", () => {
  describe("basic functionality", () => {
    it("should snap to nearest bar boundary", () => {
      const bpm = 120;
      const downbeat = 0;
      const barDuration = (60 / 120) * 4; // 2 seconds

      // Snap to first bar
      expect(quantizeToBar(0.8, bpm, downbeat)).toBe(0.0);

      // Snap to second bar
      expect(quantizeToBar(1.2, bpm, downbeat)).toBe(2.0);

      // Exact bar
      expect(quantizeToBar(2.0, bpm, downbeat)).toBe(2.0);
    });

    it("should handle different BPMs", () => {
      const downbeat = 0;

      // BPM 60 -> bar duration = 4 seconds
      expect(quantizeToBar(1.5, 60, downbeat)).toBe(0.0);
      expect(quantizeToBar(3.0, 60, downbeat)).toBe(4.0);

      // BPM 90 -> bar duration = 2.666... seconds
      const barDuration90 = (60 / 90) * 4;
      expect(quantizeToBar(3.0, 90, downbeat)).toBeCloseTo(
        barDuration90,
        10
      );
    });

    it("should handle non-zero downbeat", () => {
      const bpm = 120;
      const downbeat = 1.0;
      const barDuration = (60 / 120) * 4; // 2 seconds

      // First bar after downbeat
      expect(quantizeToBar(2.0, bpm, downbeat)).toBe(1.0);

      // Second bar after downbeat
      expect(quantizeToBar(3.5, bpm, downbeat)).toBe(3.0);
    });
  });

  describe("edge cases", () => {
    it("should handle time exactly at downbeat", () => {
      const bpm = 120;
      const downbeat = 2.0;

      expect(quantizeToBar(2.0, bpm, downbeat)).toBe(2.0);
    });

    it("should handle time before downbeat", () => {
      const bpm = 120;
      const downbeat = 4.0;
      const barDuration = (60 / 120) * 4; // 2 seconds

      // One bar before downbeat
      expect(quantizeToBar(2.0, bpm, downbeat)).toBe(2.0);

      // Just before downbeat
      expect(quantizeToBar(3.5, bpm, downbeat)).toBe(4.0);
    });

    it("should handle negative times with positive downbeat", () => {
      const bpm = 120;
      const downbeat = 4.0;
      const barDuration = (60 / 120) * 4; // 2 seconds

      // Negative time should quantize relative to downbeat
      expect(quantizeToBar(-1.0, bpm, downbeat)).toBe(0.0);
    });

    it("should handle time far from downbeat", () => {
      const bpm = 120;
      const downbeat = 1.0;
      const barDuration = (60 / 120) * 4; // 2 seconds

      // Many bars after downbeat
      expect(quantizeToBar(100.5, bpm, downbeat)).toBe(101.0);
    });

    it("should handle high BPMs", () => {
      const bpm = 180;
      const downbeat = 0;
      const barDuration = (60 / 180) * 4; // 1.333... seconds

      expect(quantizeToBar(2.0, bpm, downbeat)).toBeCloseTo(
        barDuration,
        10
      );
    });

    it("should handle low BPMs", () => {
      const bpm = 60;
      const downbeat = 0;
      const barDuration = (60 / 60) * 4; // 4 seconds

      expect(quantizeToBar(3.5, bpm, downbeat)).toBe(4.0);
      expect(quantizeToBar(5.5, bpm, downbeat)).toBe(4.0);
    });
  });

  describe("parametrized tests", () => {
    const testCases = [
      { time: 0.0, bpm: 120, downbeat: 0, expected: 0.0 },
      { time: 0.5, bpm: 120, downbeat: 0, expected: 0.0 },
      { time: 1.5, bpm: 120, downbeat: 0, expected: 2.0 },
      { time: 2.0, bpm: 120, downbeat: 0, expected: 2.0 },
      { time: 3.5, bpm: 120, downbeat: 1.0, expected: 3.0 },
      { time: 10.0, bpm: 90, downbeat: 0, expected: (60 / 90) * 4 * 4 },
    ];

    testCases.forEach(({ time, bpm, downbeat, expected }) => {
      it(`quantizeToBar(${time}, ${bpm}, ${downbeat}) should be ${expected}`, () => {
        expect(quantizeToBar(time, bpm, downbeat)).toBeCloseTo(expected, 10);
      });
    });
  });
});
