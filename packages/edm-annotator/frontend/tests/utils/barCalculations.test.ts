import { describe, it, expect } from "vitest";
import {
  timeToBar,
  barToTime,
  getBeatDuration,
  getBarDuration,
} from "@/utils/barCalculations";

describe("timeToBar", () => {
  describe("basic functionality", () => {
    it("should convert time to bar number (1-indexed)", () => {
      const bpm = 120;
      const downbeat = 0;
      const barDuration = (60 / 120) * 4; // 2 seconds

      // First bar (0-2s)
      expect(timeToBar(0.0, bpm, downbeat)).toBe(1);
      expect(timeToBar(1.0, bpm, downbeat)).toBe(1);
      expect(timeToBar(1.99, bpm, downbeat)).toBe(1);

      // Second bar (2-4s)
      expect(timeToBar(2.0, bpm, downbeat)).toBe(2);
      expect(timeToBar(3.0, bpm, downbeat)).toBe(2);

      // Third bar (4-6s)
      expect(timeToBar(4.0, bpm, downbeat)).toBe(3);
    });

    it("should handle different BPMs", () => {
      const downbeat = 0;

      // BPM 60 -> bar duration = 4 seconds
      expect(timeToBar(0.0, 60, downbeat)).toBe(1);
      expect(timeToBar(3.9, 60, downbeat)).toBe(1);
      expect(timeToBar(4.0, 60, downbeat)).toBe(2);
      expect(timeToBar(8.0, 60, downbeat)).toBe(3);

      // BPM 180 -> bar duration = 1.333... seconds
      const barDuration180 = (60 / 180) * 4;
      expect(timeToBar(0.0, 180, downbeat)).toBe(1);
      expect(timeToBar(barDuration180, 180, downbeat)).toBe(2);
      expect(timeToBar(barDuration180 * 2, 180, downbeat)).toBe(3);
    });

    it("should handle non-zero downbeat", () => {
      const bpm = 120;
      const downbeat = 2.0;
      const barDuration = (60 / 120) * 4; // 2 seconds

      // At downbeat
      expect(timeToBar(2.0, bpm, downbeat)).toBe(1);

      // First bar after downbeat (2-4s)
      expect(timeToBar(3.0, bpm, downbeat)).toBe(1);

      // Second bar after downbeat (4-6s)
      expect(timeToBar(4.0, bpm, downbeat)).toBe(2);
      expect(timeToBar(5.0, bpm, downbeat)).toBe(2);

      // Third bar after downbeat (6-8s)
      expect(timeToBar(6.0, bpm, downbeat)).toBe(3);
    });
  });

  describe("edge cases", () => {
    it("should return 1 for time before downbeat", () => {
      const bpm = 120;
      const downbeat = 2.0;

      expect(timeToBar(0.0, bpm, downbeat)).toBe(1);
      expect(timeToBar(1.0, bpm, downbeat)).toBe(1);
      expect(timeToBar(1.99, bpm, downbeat)).toBe(1);
    });

    it("should handle time exactly at downbeat", () => {
      const bpm = 120;
      const downbeat = 1.5;

      expect(timeToBar(1.5, bpm, downbeat)).toBe(1);
    });

    it("should handle negative times", () => {
      const bpm = 120;
      const downbeat = 2.0;

      // Negative times should return 1 (clamped)
      expect(timeToBar(-1.0, bpm, downbeat)).toBe(1);
      expect(timeToBar(-10.0, bpm, downbeat)).toBe(1);
    });

    it("should handle very large bar numbers", () => {
      const bpm = 120;
      const downbeat = 0;
      const barDuration = (60 / 120) * 4; // 2 seconds

      // Bar 100 starts at 198s
      expect(timeToBar(198.0, bpm, downbeat)).toBe(100);

      // Bar 1000 starts at 1998s
      expect(timeToBar(1998.0, bpm, downbeat)).toBe(1000);
    });

    it("should handle boundary between bars", () => {
      const bpm = 120;
      const downbeat = 0;
      const barDuration = (60 / 120) * 4; // 2 seconds

      // Just before bar boundary
      expect(timeToBar(1.999999, bpm, downbeat)).toBe(1);

      // Exactly at bar boundary
      expect(timeToBar(2.0, bpm, downbeat)).toBe(2);

      // Just after bar boundary
      expect(timeToBar(2.000001, bpm, downbeat)).toBe(2);
    });
  });

  describe("parametrized tests", () => {
    const testCases = [
      { time: 0.0, bpm: 120, downbeat: 0, expected: 1 },
      { time: 1.99, bpm: 120, downbeat: 0, expected: 1 },
      { time: 2.0, bpm: 120, downbeat: 0, expected: 2 },
      { time: 4.0, bpm: 120, downbeat: 0, expected: 3 },
      { time: 3.0, bpm: 120, downbeat: 2.0, expected: 1 },
      { time: 4.0, bpm: 120, downbeat: 2.0, expected: 2 },
      { time: 0.0, bpm: 60, downbeat: 0, expected: 1 },
      { time: 4.0, bpm: 60, downbeat: 0, expected: 2 },
      { time: -1.0, bpm: 120, downbeat: 0, expected: 1 },
    ];

    testCases.forEach(({ time, bpm, downbeat, expected }) => {
      it(`timeToBar(${time}, ${bpm}, ${downbeat}) should be ${expected}`, () => {
        expect(timeToBar(time, bpm, downbeat)).toBe(expected);
      });
    });
  });
});

describe("barToTime", () => {
  describe("basic functionality", () => {
    it("should convert bar number to time", () => {
      const bpm = 120;
      const downbeat = 0;
      const barDuration = (60 / 120) * 4; // 2 seconds

      // Bar 1 starts at downbeat
      expect(barToTime(1, bpm, downbeat)).toBe(0.0);

      // Bar 2 starts at 2s
      expect(barToTime(2, bpm, downbeat)).toBe(2.0);

      // Bar 3 starts at 4s
      expect(barToTime(3, bpm, downbeat)).toBe(4.0);

      // Bar 10 starts at 18s
      expect(barToTime(10, bpm, downbeat)).toBe(18.0);
    });

    it("should handle different BPMs", () => {
      const downbeat = 0;

      // BPM 60 -> bar duration = 4 seconds
      expect(barToTime(1, 60, downbeat)).toBe(0.0);
      expect(barToTime(2, 60, downbeat)).toBe(4.0);
      expect(barToTime(3, 60, downbeat)).toBe(8.0);

      // BPM 180 -> bar duration = 1.333... seconds
      const barDuration180 = (60 / 180) * 4;
      expect(barToTime(1, 180, downbeat)).toBeCloseTo(0.0, 10);
      expect(barToTime(2, 180, downbeat)).toBeCloseTo(barDuration180, 10);
      expect(barToTime(3, 180, downbeat)).toBeCloseTo(barDuration180 * 2, 10);
    });

    it("should handle non-zero downbeat", () => {
      const bpm = 120;
      const downbeat = 2.0;
      const barDuration = (60 / 120) * 4; // 2 seconds

      // Bar 1 starts at downbeat
      expect(barToTime(1, bpm, downbeat)).toBe(2.0);

      // Bar 2 starts at downbeat + barDuration
      expect(barToTime(2, bpm, downbeat)).toBe(4.0);

      // Bar 3 starts at downbeat + 2*barDuration
      expect(barToTime(3, bpm, downbeat)).toBe(6.0);
    });
  });

  describe("edge cases", () => {
    it("should handle bar 1", () => {
      const bpm = 120;
      const downbeat = 1.5;

      expect(barToTime(1, bpm, downbeat)).toBe(1.5);
    });

    it("should handle large bar numbers", () => {
      const bpm = 120;
      const downbeat = 0;
      const barDuration = (60 / 120) * 4; // 2 seconds

      // Bar 100 starts at 198s
      expect(barToTime(100, bpm, downbeat)).toBe(198.0);

      // Bar 1000 starts at 1998s
      expect(barToTime(1000, bpm, downbeat)).toBe(1998.0);
    });

    it("should handle bar 0 (edge case)", () => {
      const bpm = 120;
      const downbeat = 2.0;
      const barDuration = (60 / 120) * 4; // 2 seconds

      // Bar 0 would be one bar before downbeat
      expect(barToTime(0, bpm, downbeat)).toBe(0.0);
    });

    it("should handle negative bar numbers (edge case)", () => {
      const bpm = 120;
      const downbeat = 4.0;
      const barDuration = (60 / 120) * 4; // 2 seconds

      // Bar -1 would be two bars before downbeat
      expect(barToTime(-1, bpm, downbeat)).toBe(0.0);
    });
  });

  describe("round-trip consistency", () => {
    it("should be consistent with timeToBar", () => {
      const bpm = 120;
      const downbeat = 1.0;

      // Convert bar to time, then time back to bar
      for (let bar = 1; bar <= 10; bar++) {
        const time = barToTime(bar, bpm, downbeat);
        const barBack = timeToBar(time, bpm, downbeat);
        expect(barBack).toBe(bar);
      }
    });

    it("should handle different BPMs in round-trip", () => {
      const bpms = [60, 90, 120, 150, 180];
      const downbeat = 0;

      bpms.forEach((bpm) => {
        for (let bar = 1; bar <= 5; bar++) {
          const time = barToTime(bar, bpm, downbeat);
          const barBack = timeToBar(time, bpm, downbeat);
          expect(barBack).toBe(bar);
        }
      });
    });
  });

  describe("parametrized tests", () => {
    const testCases = [
      { bar: 1, bpm: 120, downbeat: 0, expected: 0.0 },
      { bar: 2, bpm: 120, downbeat: 0, expected: 2.0 },
      { bar: 3, bpm: 120, downbeat: 0, expected: 4.0 },
      { bar: 1, bpm: 120, downbeat: 2.0, expected: 2.0 },
      { bar: 2, bpm: 120, downbeat: 2.0, expected: 4.0 },
      { bar: 1, bpm: 60, downbeat: 0, expected: 0.0 },
      { bar: 2, bpm: 60, downbeat: 0, expected: 4.0 },
      { bar: 10, bpm: 120, downbeat: 0, expected: 18.0 },
    ];

    testCases.forEach(({ bar, bpm, downbeat, expected }) => {
      it(`barToTime(${bar}, ${bpm}, ${downbeat}) should be ${expected}`, () => {
        expect(barToTime(bar, bpm, downbeat)).toBeCloseTo(expected, 10);
      });
    });
  });
});

describe("getBeatDuration", () => {
  describe("basic functionality", () => {
    it("should calculate beat duration from BPM", () => {
      // BPM 60 -> 1 beat per second
      expect(getBeatDuration(60)).toBe(1.0);

      // BPM 120 -> 2 beats per second = 0.5s per beat
      expect(getBeatDuration(120)).toBe(0.5);

      // BPM 90 -> 1.5 beats per second
      expect(getBeatDuration(90)).toBeCloseTo(60 / 90, 10);

      // BPM 180 -> 3 beats per second
      expect(getBeatDuration(180)).toBeCloseTo(60 / 180, 10);
    });

    it("should handle common EDM BPMs", () => {
      // House: ~120-130 BPM
      expect(getBeatDuration(125)).toBeCloseTo(0.48, 2);

      // Techno: ~120-135 BPM
      expect(getBeatDuration(130)).toBeCloseTo(60 / 130, 5);

      // Drum & Bass: ~160-180 BPM
      expect(getBeatDuration(174)).toBeCloseTo(0.345, 3);

      // Dubstep: ~140 BPM
      expect(getBeatDuration(140)).toBeCloseTo(0.429, 3);
    });
  });

  describe("edge cases", () => {
    it("should handle very high BPMs", () => {
      expect(getBeatDuration(300)).toBeCloseTo(0.2, 10);
      expect(getBeatDuration(600)).toBeCloseTo(0.1, 10);
    });

    it("should handle very low BPMs", () => {
      expect(getBeatDuration(30)).toBeCloseTo(2.0, 10);
      expect(getBeatDuration(10)).toBeCloseTo(6.0, 10);
    });

    it("should return Infinity for BPM=0", () => {
      expect(getBeatDuration(0)).toBe(Infinity);
    });

    it("should return negative values for negative BPM", () => {
      expect(getBeatDuration(-60)).toBe(-1.0);
      expect(getBeatDuration(-120)).toBe(-0.5);
    });
  });

  describe("parametrized tests", () => {
    const testCases = [
      { bpm: 60, expected: 1.0 },
      { bpm: 90, expected: 60 / 90 },
      { bpm: 120, expected: 0.5 },
      { bpm: 150, expected: 0.4 },
      { bpm: 180, expected: 60 / 180 },
    ];

    testCases.forEach(({ bpm, expected }) => {
      it(`getBeatDuration(${bpm}) should be ${expected}`, () => {
        expect(getBeatDuration(bpm)).toBeCloseTo(expected, 10);
      });
    });
  });
});

describe("getBarDuration", () => {
  describe("basic functionality", () => {
    it("should calculate bar duration from BPM (4/4 time)", () => {
      // BPM 60 -> 4 seconds per bar
      expect(getBarDuration(60)).toBe(4.0);

      // BPM 120 -> 2 seconds per bar
      expect(getBarDuration(120)).toBe(2.0);

      // BPM 90 -> 2.666... seconds per bar
      expect(getBarDuration(90)).toBeCloseTo((60 / 90) * 4, 10);

      // BPM 180 -> 1.333... seconds per bar
      expect(getBarDuration(180)).toBeCloseTo((60 / 180) * 4, 10);
    });

    it("should handle common EDM BPMs", () => {
      // House: ~120-130 BPM
      expect(getBarDuration(125)).toBeCloseTo(1.92, 2);

      // Techno: ~120-135 BPM
      expect(getBarDuration(130)).toBeCloseTo(1.846, 3);

      // Drum & Bass: ~160-180 BPM
      expect(getBarDuration(174)).toBeCloseTo(1.379, 3);

      // Dubstep: ~140 BPM
      expect(getBarDuration(140)).toBeCloseTo(1.714, 3);
    });
  });

  describe("edge cases", () => {
    it("should handle very high BPMs", () => {
      expect(getBarDuration(300)).toBeCloseTo(0.8, 10);
      expect(getBarDuration(600)).toBeCloseTo(0.4, 10);
    });

    it("should handle very low BPMs", () => {
      expect(getBarDuration(30)).toBeCloseTo(8.0, 10);
      expect(getBarDuration(10)).toBeCloseTo(24.0, 10);
    });

    it("should return Infinity for BPM=0", () => {
      expect(getBarDuration(0)).toBe(Infinity);
    });

    it("should return negative values for negative BPM", () => {
      expect(getBarDuration(-60)).toBe(-4.0);
      expect(getBarDuration(-120)).toBe(-2.0);
    });
  });

  describe("relationship with getBeatDuration", () => {
    it("should be 4x beat duration (4/4 time)", () => {
      const bpms = [60, 90, 120, 150, 180];

      bpms.forEach((bpm) => {
        const beatDuration = getBeatDuration(bpm);
        const barDuration = getBarDuration(bpm);
        expect(barDuration).toBeCloseTo(beatDuration * 4, 10);
      });
    });
  });

  describe("parametrized tests", () => {
    const testCases = [
      { bpm: 60, expected: 4.0 },
      { bpm: 90, expected: (60 / 90) * 4 },
      { bpm: 120, expected: 2.0 },
      { bpm: 150, expected: 1.6 },
      { bpm: 180, expected: (60 / 180) * 4 },
    ];

    testCases.forEach(({ bpm, expected }) => {
      it(`getBarDuration(${bpm}) should be ${expected}`, () => {
        expect(getBarDuration(bpm)).toBeCloseTo(expected, 10);
      });
    });
  });
});
