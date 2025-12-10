import { describe, it, expect } from "vitest";
import { formatTime, formatTimeShort } from "@/utils/timeFormat";

describe("formatTime", () => {
  describe("basic functionality", () => {
    it("should format time as MM:SS.mmm", () => {
      expect(formatTime(0)).toBe("00:00.000");
      expect(formatTime(1)).toBe("00:01.000");
      expect(formatTime(59)).toBe("00:59.000");
      expect(formatTime(60)).toBe("01:00.000");
      expect(formatTime(61)).toBe("01:01.000");
      expect(formatTime(125)).toBe("02:05.000");
    });

    it("should format milliseconds correctly", () => {
      expect(formatTime(0.001)).toBe("00:00.001");
      expect(formatTime(0.01)).toBe("00:00.010");
      expect(formatTime(0.1)).toBe("00:00.100");
      expect(formatTime(0.999)).toBe("00:00.999");
      expect(formatTime(1.5)).toBe("00:01.500");
      expect(formatTime(1.123)).toBe("00:01.123");
    });

    it("should handle multiple minutes", () => {
      expect(formatTime(120)).toBe("02:00.000");
      expect(formatTime(180)).toBe("03:00.000");
      expect(formatTime(300)).toBe("05:00.000");
      expect(formatTime(600)).toBe("10:00.000");
    });

    it("should handle combined minutes, seconds, and milliseconds", () => {
      expect(formatTime(125.456)).toBe("02:05.456");
      expect(formatTime(65.789)).toBe("01:05.789");
      expect(formatTime(3661.234)).toBe("61:01.234");
    });
  });

  describe("edge cases", () => {
    it("should handle zero", () => {
      expect(formatTime(0)).toBe("00:00.000");
    });

    it("should handle very small values", () => {
      expect(formatTime(0.0001)).toBe("00:00.000");
      expect(formatTime(0.0009)).toBe("00:00.000");
    });

    it("should handle large times", () => {
      expect(formatTime(3599)).toBe("59:59.000");
      expect(formatTime(3600)).toBe("60:00.000");
      expect(formatTime(7200)).toBe("120:00.000");
    });

    it("should handle fractional seconds near boundaries", () => {
      expect(formatTime(59.999)).toBe("00:59.999");
      expect(formatTime(60.001)).toBe("01:00.001");
    });

    it("should truncate milliseconds (not round)", () => {
      expect(formatTime(1.9999)).toBe("00:01.999");
      expect(formatTime(59.9999)).toBe("00:59.999");
    });

    it("should handle negative values (edge case - probably invalid input)", () => {
      // Negative values will produce strange output, but testing behavior
      expect(formatTime(-1)).toBe("-1:-1.000");
      expect(formatTime(-0.5)).toBe("-1:-1.500");
    });
  });

  describe("padding", () => {
    it("should pad minutes with leading zero", () => {
      expect(formatTime(0)).toBe("00:00.000");
      expect(formatTime(30)).toBe("00:30.000");
      expect(formatTime(300)).toBe("05:00.000");
    });

    it("should pad seconds with leading zero", () => {
      expect(formatTime(1)).toBe("00:01.000");
      expect(formatTime(60)).toBe("01:00.000");
      expect(formatTime(61)).toBe("01:01.000");
    });

    it("should pad milliseconds with leading zeros", () => {
      expect(formatTime(0.001)).toBe("00:00.001");
      expect(formatTime(0.01)).toBe("00:00.010");
      expect(formatTime(0.1)).toBe("00:00.100");
    });

    it("should not pad minutes beyond 2 digits", () => {
      expect(formatTime(6000)).toBe("100:00.000");
      expect(formatTime(60000)).toBe("1000:00.000");
    });
  });

  describe("parametrized tests", () => {
    const testCases = [
      { seconds: 0, expected: "00:00.000" },
      { seconds: 1, expected: "00:01.000" },
      { seconds: 10, expected: "00:10.000" },
      { seconds: 60, expected: "01:00.000" },
      { seconds: 61.5, expected: "01:01.500" },
      { seconds: 125.456, expected: "02:05.456" },
      { seconds: 3661.789, expected: "61:01.789" },
      { seconds: 0.123, expected: "00:00.123" },
    ];

    testCases.forEach(({ seconds, expected }) => {
      it(`formatTime(${seconds}) should be ${expected}`, () => {
        expect(formatTime(seconds)).toBe(expected);
      });
    });
  });
});

describe("formatTimeShort", () => {
  describe("basic functionality", () => {
    it("should format time as MM:SS", () => {
      expect(formatTimeShort(0)).toBe("00:00");
      expect(formatTimeShort(1)).toBe("00:01");
      expect(formatTimeShort(59)).toBe("00:59");
      expect(formatTimeShort(60)).toBe("01:00");
      expect(formatTimeShort(61)).toBe("01:01");
      expect(formatTimeShort(125)).toBe("02:05");
    });

    it("should ignore milliseconds", () => {
      expect(formatTimeShort(0.001)).toBe("00:00");
      expect(formatTimeShort(0.999)).toBe("00:00");
      expect(formatTimeShort(1.5)).toBe("00:01");
      expect(formatTimeShort(1.999)).toBe("00:01");
    });

    it("should handle multiple minutes", () => {
      expect(formatTimeShort(120)).toBe("02:00");
      expect(formatTimeShort(180)).toBe("03:00");
      expect(formatTimeShort(300)).toBe("05:00");
      expect(formatTimeShort(600)).toBe("10:00");
    });

    it("should handle combined minutes and seconds", () => {
      expect(formatTimeShort(125.456)).toBe("02:05");
      expect(formatTimeShort(65.789)).toBe("01:05");
      expect(formatTimeShort(3661.234)).toBe("61:01");
    });
  });

  describe("edge cases", () => {
    it("should handle zero", () => {
      expect(formatTimeShort(0)).toBe("00:00");
    });

    it("should handle very small values", () => {
      expect(formatTimeShort(0.0001)).toBe("00:00");
      expect(formatTimeShort(0.9999)).toBe("00:00");
    });

    it("should handle large times", () => {
      expect(formatTimeShort(3599)).toBe("59:59");
      expect(formatTimeShort(3600)).toBe("60:00");
      expect(formatTimeShort(7200)).toBe("120:00");
    });

    it("should truncate fractional seconds (not round)", () => {
      expect(formatTimeShort(1.9)).toBe("00:01");
      expect(formatTimeShort(59.9)).toBe("00:59");
      expect(formatTimeShort(60.9)).toBe("01:00");
    });

    it("should handle negative values (edge case - probably invalid input)", () => {
      // Negative values will produce strange output, but testing behavior
      expect(formatTimeShort(-1)).toBe("-1:-1");
      expect(formatTimeShort(-0.5)).toBe("-1:-1");
    });
  });

  describe("padding", () => {
    it("should pad minutes with leading zero", () => {
      expect(formatTimeShort(0)).toBe("00:00");
      expect(formatTimeShort(30)).toBe("00:30");
      expect(formatTimeShort(300)).toBe("05:00");
    });

    it("should pad seconds with leading zero", () => {
      expect(formatTimeShort(1)).toBe("00:01");
      expect(formatTimeShort(60)).toBe("01:00");
      expect(formatTimeShort(61)).toBe("01:01");
    });

    it("should not pad minutes beyond 2 digits", () => {
      expect(formatTimeShort(6000)).toBe("100:00");
      expect(formatTimeShort(60000)).toBe("1000:00");
    });
  });

  describe("comparison with formatTime", () => {
    it("should match formatTime without milliseconds", () => {
      const times = [0, 1, 60, 125, 3661];

      times.forEach((time) => {
        const short = formatTimeShort(time);
        const long = formatTime(time);
        // Short version should match long version without .mmm suffix
        expect(long.startsWith(short)).toBe(true);
        expect(long).toBe(`${short}.000`);
      });
    });

    it("should truncate milliseconds like formatTime", () => {
      const times = [0.5, 1.9, 59.999, 125.456];

      times.forEach((time) => {
        const short = formatTimeShort(time);
        const long = formatTime(time);
        // Short version should match long version up to the decimal point
        const longWithoutMillis = long.substring(0, long.lastIndexOf("."));
        expect(short).toBe(longWithoutMillis);
      });
    });
  });

  describe("parametrized tests", () => {
    const testCases = [
      { seconds: 0, expected: "00:00" },
      { seconds: 1, expected: "00:01" },
      { seconds: 10, expected: "00:10" },
      { seconds: 60, expected: "01:00" },
      { seconds: 61.5, expected: "01:01" },
      { seconds: 125.456, expected: "02:05" },
      { seconds: 3661.789, expected: "61:01" },
      { seconds: 0.999, expected: "00:00" },
    ];

    testCases.forEach(({ seconds, expected }) => {
      it(`formatTimeShort(${seconds}) should be ${expected}`, () => {
        expect(formatTimeShort(seconds)).toBe(expected);
      });
    });
  });
});
