import { describe, it, expect } from "vitest";
import { labelColors, labelBorderColors } from "@/utils/colors";
import type { SectionLabel } from "@/types/structure";

describe("labelColors", () => {
  describe("color definitions", () => {
    it("should have colors for all section labels", () => {
      const expectedLabels: SectionLabel[] = [
        "intro",
        "buildup",
        "breakdown",
        "breakbuild",
        "outro",
        "unlabeled",
      ];

      expectedLabels.forEach((label) => {
        expect(labelColors).toHaveProperty(label);
        expect(labelColors[label]).toBeDefined();
        expect(typeof labelColors[label]).toBe("string");
      });
    });

    it("should have exactly 6 color definitions", () => {
      expect(Object.keys(labelColors)).toHaveLength(6);
    });

    it("should use rgba format with transparency", () => {
      const rgbaPattern = /^rgba\(\d+,\s*\d+,\s*\d+,\s*[\d.]+\)$/;

      Object.entries(labelColors).forEach(([label, color]) => {
        expect(color).toMatch(rgbaPattern);
      });
    });
  });

  describe("individual colors", () => {
    it("should define intro as blue", () => {
      expect(labelColors.intro).toBe("rgba(91, 124, 255, 0.2)");
    });

    it("should define buildup as orange", () => {
      expect(labelColors.buildup).toBe("rgba(255, 184, 0, 0.2)");
    });

    it("should define breakdown as cyan", () => {
      expect(labelColors.breakdown).toBe("rgba(0, 230, 184, 0.2)");
    });

    it("should define breakbuild as red", () => {
      expect(labelColors.breakbuild).toBe("rgba(255, 107, 107, 0.2)");
    });

    it("should define outro as purple", () => {
      expect(labelColors.outro).toBe("rgba(156, 39, 176, 0.2)");
    });

    it("should define unlabeled as gray", () => {
      expect(labelColors.unlabeled).toBe("rgba(128, 128, 128, 0.1)");
    });
  });

  describe("color opacity", () => {
    it("should use 0.2 opacity for most labels", () => {
      const labelsWithOpacity02: SectionLabel[] = [
        "intro",
        "buildup",
        "breakdown",
        "breakbuild",
        "outro",
      ];

      labelsWithOpacity02.forEach((label) => {
        expect(labelColors[label]).toContain("0.2)");
      });
    });

    it("should use 0.1 opacity for unlabeled", () => {
      expect(labelColors.unlabeled).toContain("0.1)");
    });
  });

  describe("color uniqueness", () => {
    it("should have unique colors for each label", () => {
      const colors = Object.values(labelColors);
      const uniqueColors = new Set(colors);
      expect(uniqueColors.size).toBe(colors.length);
    });
  });
});

describe("labelBorderColors", () => {
  describe("border color definitions", () => {
    it("should have border colors for all section labels", () => {
      const expectedLabels: SectionLabel[] = [
        "intro",
        "buildup",
        "breakdown",
        "breakbuild",
        "outro",
        "unlabeled",
      ];

      expectedLabels.forEach((label) => {
        expect(labelBorderColors).toHaveProperty(label);
        expect(labelBorderColors[label]).toBeDefined();
        expect(typeof labelBorderColors[label]).toBe("string");
      });
    });

    it("should have exactly 6 border color definitions", () => {
      expect(Object.keys(labelBorderColors)).toHaveLength(6);
    });

    it("should use rgba format with transparency", () => {
      const rgbaPattern = /^rgba\(\d+,\s*\d+,\s*\d+,\s*[\d.]+\)$/;

      Object.entries(labelBorderColors).forEach(([label, color]) => {
        expect(color).toMatch(rgbaPattern);
      });
    });
  });

  describe("individual border colors", () => {
    it("should define intro border as blue", () => {
      expect(labelBorderColors.intro).toBe("rgba(91, 124, 255, 0.8)");
    });

    it("should define buildup border as orange", () => {
      expect(labelBorderColors.buildup).toBe("rgba(255, 184, 0, 0.8)");
    });

    it("should define breakdown border as cyan", () => {
      expect(labelBorderColors.breakdown).toBe("rgba(0, 230, 184, 0.8)");
    });

    it("should define breakbuild border as red", () => {
      expect(labelBorderColors.breakbuild).toBe("rgba(255, 107, 107, 0.8)");
    });

    it("should define outro border as purple", () => {
      expect(labelBorderColors.outro).toBe("rgba(156, 39, 176, 0.8)");
    });

    it("should define unlabeled border as gray", () => {
      expect(labelBorderColors.unlabeled).toBe("rgba(128, 128, 128, 0.5)");
    });
  });

  describe("border color opacity", () => {
    it("should use 0.8 opacity for most labels", () => {
      const labelsWithOpacity08: SectionLabel[] = [
        "intro",
        "buildup",
        "breakdown",
        "breakbuild",
        "outro",
      ];

      labelsWithOpacity08.forEach((label) => {
        expect(labelBorderColors[label]).toContain("0.8)");
      });
    });

    it("should use 0.5 opacity for unlabeled", () => {
      expect(labelBorderColors.unlabeled).toContain("0.5)");
    });
  });

  describe("border color uniqueness", () => {
    it("should have unique border colors for each label", () => {
      const colors = Object.values(labelBorderColors);
      const uniqueColors = new Set(colors);
      expect(uniqueColors.size).toBe(colors.length);
    });
  });
});

describe("color consistency", () => {
  describe("matching RGB values", () => {
    it("should use same RGB values for fill and border colors", () => {
      const labels: SectionLabel[] = [
        "intro",
        "buildup",
        "breakdown",
        "breakbuild",
        "outro",
        "unlabeled",
      ];

      labels.forEach((label) => {
        const fillColor = labelColors[label];
        const borderColor = labelBorderColors[label];

        // Extract RGB values
        const fillRGB = fillColor.match(/rgba\((\d+),\s*(\d+),\s*(\d+)/);
        const borderRGB = borderColor.match(/rgba\((\d+),\s*(\d+),\s*(\d+)/);

        expect(fillRGB).toBeTruthy();
        expect(borderRGB).toBeTruthy();

        if (fillRGB && borderRGB) {
          expect(fillRGB[1]).toBe(borderRGB[1]); // Red
          expect(fillRGB[2]).toBe(borderRGB[2]); // Green
          expect(fillRGB[3]).toBe(borderRGB[3]); // Blue
        }
      });
    });

    it("should have higher opacity for borders than fills", () => {
      const labels: SectionLabel[] = [
        "intro",
        "buildup",
        "breakdown",
        "breakbuild",
        "outro",
        "unlabeled",
      ];

      labels.forEach((label) => {
        const fillColor = labelColors[label];
        const borderColor = labelBorderColors[label];

        // Extract alpha values
        const fillAlpha = parseFloat(fillColor.match(/,\s*([\d.]+)\)$/)?.[1] || "0");
        const borderAlpha = parseFloat(borderColor.match(/,\s*([\d.]+)\)$/)?.[1] || "0");

        expect(borderAlpha).toBeGreaterThan(fillAlpha);
      });
    });
  });

  describe("both objects have same keys", () => {
    it("should have matching keys in labelColors and labelBorderColors", () => {
      const fillKeys = Object.keys(labelColors).sort();
      const borderKeys = Object.keys(labelBorderColors).sort();

      expect(fillKeys).toEqual(borderKeys);
    });
  });
});

describe("type safety", () => {
  it("should allow access with SectionLabel type", () => {
    const label: SectionLabel = "intro";

    // These should compile without errors
    const fillColor = labelColors[label];
    const borderColor = labelBorderColors[label];

    expect(fillColor).toBeDefined();
    expect(borderColor).toBeDefined();
  });

  it("should be Record<SectionLabel, string>", () => {
    // Type checking - this will fail at compile time if types are wrong
    const fillColors: Record<SectionLabel, string> = labelColors;
    const borderColors: Record<SectionLabel, string> = labelBorderColors;

    expect(fillColors).toBeDefined();
    expect(borderColors).toBeDefined();
  });
});
