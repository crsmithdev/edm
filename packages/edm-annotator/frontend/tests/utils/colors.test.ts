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
        "breakdown-buildup",
        "outro",
        "default",
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

      Object.values(labelColors).forEach((color) => {
        expect(color).toMatch(rgbaPattern);
      });
    });
  });

  describe("individual colors", () => {
    it("should define intro as purple", () => {
      expect(labelColors.intro).toBe("rgba(123, 106, 255, 0.2)");
    });

    it("should define buildup as cyan", () => {
      expect(labelColors.buildup).toBe("rgba(0, 229, 204, 0.2)");
    });

    it("should define breakdown as pink", () => {
      expect(labelColors.breakdown).toBe("rgba(255, 107, 181, 0.2)");
    });

    it("should define breakdown-buildup as light purple", () => {
      expect(labelColors["breakdown-buildup"]).toBe("rgba(167, 139, 250, 0.2)");
    });

    it("should define outro as medium purple", () => {
      expect(labelColors.outro).toBe("rgba(139, 122, 255, 0.2)");
    });

    it("should define default as gray", () => {
      expect(labelColors.default).toBe("rgba(96, 96, 104, 0.1)");
    });
  });

  describe("color opacity", () => {
    it("should use 0.2 opacity for most labels", () => {
      const labelsWithOpacity02: SectionLabel[] = [
        "intro",
        "buildup",
        "breakdown",
        "breakdown-buildup",
        "outro",
      ];

      labelsWithOpacity02.forEach((label) => {
        expect(labelColors[label]).toContain("0.2)");
      });
    });

    it("should use 0.1 opacity for default", () => {
      expect(labelColors.default).toContain("0.1)");
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
        "breakdown-buildup",
        "outro",
        "default",
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

    it("should use hex format", () => {
      const hexPattern = /^#[0-9a-f]{6}$/i;

      Object.values(labelBorderColors).forEach((color) => {
        expect(color).toMatch(hexPattern);
      });
    });
  });

  describe("individual border colors", () => {
    it("should define intro border as purple", () => {
      expect(labelBorderColors.intro).toBe("#7b6aff");
    });

    it("should define buildup border as cyan", () => {
      expect(labelBorderColors.buildup).toBe("#00e5cc");
    });

    it("should define breakdown border as pink", () => {
      expect(labelBorderColors.breakdown).toBe("#ff6bb5");
    });

    it("should define breakdown-buildup border as light purple", () => {
      expect(labelBorderColors["breakdown-buildup"]).toBe("#a78bfa");
    });

    it("should define outro border as medium purple", () => {
      expect(labelBorderColors.outro).toBe("#8b7aff");
    });

    it("should define default border as gray", () => {
      expect(labelBorderColors.default).toBe("#606068");
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
  describe("color format consistency", () => {
    it("should use rgba format for fill colors", () => {
      const labels: SectionLabel[] = [
        "intro",
        "buildup",
        "breakdown",
        "breakdown-buildup",
        "outro",
        "default",
      ];

      labels.forEach((label) => {
        const fillColor = labelColors[label];
        expect(fillColor).toMatch(/^rgba\(\d+,\s*\d+,\s*\d+,\s*[\d.]+\)$/);
      });
    });

    it("should use hex format for border colors", () => {
      const labels: SectionLabel[] = [
        "intro",
        "buildup",
        "breakdown",
        "breakdown-buildup",
        "outro",
        "default",
      ];

      labels.forEach((label) => {
        const borderColor = labelBorderColors[label];
        expect(borderColor).toMatch(/^#[0-9a-f]{6}$/i);
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
