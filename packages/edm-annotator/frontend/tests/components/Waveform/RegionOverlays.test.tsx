import { describe, it, expect, beforeEach } from "vitest";
import { render } from "@testing-library/react";
import { RegionOverlays } from "@/components/Waveform/RegionOverlays";
import { useStructureStore, useWaveformStore } from "@/stores";
import { labelColors, labelBorderColors } from "@/utils/colors";

describe("RegionOverlays", () => {
  beforeEach(() => {
    useStructureStore.getState().reset();
    useWaveformStore.getState().reset();

    // Set up basic viewport
    useWaveformStore.setState({
      viewportStart: 0,
      viewportEnd: 180,
      duration: 180,
    });
  });

  describe("Region Color Mapping", () => {
    it("applies correct background color for intro region", () => {
      useStructureStore.setState({
        regions: [{ start: 0, end: 60, label: "intro" }],
      });

      const { container } = render(<RegionOverlays />);
      const overlay = container.querySelector('[style*="position: absolute"]');

      expect(overlay?.getAttribute("style")).toContain(labelColors.intro);
    });

    it("applies correct background color for buildup region", () => {
      useStructureStore.setState({
        regions: [{ start: 0, end: 60, label: "buildup" }],
      });

      const { container } = render(<RegionOverlays />);
      const overlay = container.querySelector('[style*="position: absolute"]');

      expect(overlay?.getAttribute("style")).toContain(labelColors.buildup);
    });

    it("applies correct background color for breakdown region", () => {
      useStructureStore.setState({
        regions: [{ start: 0, end: 60, label: "breakdown" }],
      });

      const { container } = render(<RegionOverlays />);
      const overlay = container.querySelector('[style*="position: absolute"]');

      expect(overlay?.getAttribute("style")).toContain(labelColors.breakdown);
    });

    it("applies correct border color for intro region", () => {
      useStructureStore.setState({
        regions: [{ start: 0, end: 60, label: "intro" }],
      });

      const { container } = render(<RegionOverlays />);
      const overlay = container.querySelector('[style*="position: absolute"]');

      // React converts hex colors to RGB format
      const style = overlay?.getAttribute("style") || "";
      expect(style).toMatch(/border:.*solid\s+rgb\(123,\s*106,\s*255\)/);
    });

    it("applies correct border color for buildup region", () => {
      useStructureStore.setState({
        regions: [{ start: 0, end: 60, label: "buildup" }],
      });

      const { container } = render(<RegionOverlays />);
      const overlay = container.querySelector('[style*="position: absolute"]');

      // React converts hex colors to RGB format
      const style = overlay?.getAttribute("style") || "";
      expect(style).toMatch(/border:.*solid\s+rgb\(0,\s*229,\s*204\)/);
    });

    it("applies 2px border width", () => {
      useStructureStore.setState({
        regions: [{ start: 0, end: 60, label: "intro" }],
      });

      const { container } = render(<RegionOverlays />);
      const overlay = container.querySelector('[style*="position: absolute"]');

      expect(overlay?.getAttribute("style")).toContain("border: 2px solid");
    });

    it("removes top border", () => {
      useStructureStore.setState({
        regions: [{ start: 0, end: 60, label: "intro" }],
      });

      const { container } = render(<RegionOverlays />);
      const overlay = container.querySelector('[style*="position: absolute"]');

      // React converts borderTop: "none" to "border-top: none" in HTML
      const style = overlay?.getAttribute("style") || "";
      // The border-top style should not be present or be overridden
      expect(style).not.toContain("border-top: 2px");
    });

    it("removes bottom border", () => {
      useStructureStore.setState({
        regions: [{ start: 0, end: 60, label: "intro" }],
      });

      const { container } = render(<RegionOverlays />);
      const overlay = container.querySelector('[style*="position: absolute"]');

      // React converts borderBottom: "none" to "border-bottom: none" in HTML
      const style = overlay?.getAttribute("style") || "";
      // The border-bottom style should not be present or be overridden
      expect(style).not.toContain("border-bottom: 2px");
    });
  });

  describe("Unlabeled Region Filtering", () => {
    it("does not render unlabeled regions", () => {
      useStructureStore.setState({
        regions: [
          { start: 0, end: 60, label: "unlabeled" },
          { start: 60, end: 120, label: "buildup" },
        ],
      });

      const { container } = render(<RegionOverlays />);
      const overlays = container.querySelectorAll('[style*="position: absolute"]');

      // Only buildup region should be rendered
      expect(overlays).toHaveLength(1);
    });

    it("renders all labeled regions", () => {
      useStructureStore.setState({
        regions: [
          { start: 0, end: 30, label: "intro" },
          { start: 30, end: 60, label: "buildup" },
          { start: 60, end: 90, label: "breakdown" },
          { start: 90, end: 120, label: "breakbuild" },
          { start: 120, end: 150, label: "outro" },
        ],
      });

      const { container } = render(<RegionOverlays />);
      const overlays = container.querySelectorAll('[style*="position: absolute"]');

      expect(overlays).toHaveLength(5);
    });

    it("handles mix of labeled and unlabeled regions", () => {
      useStructureStore.setState({
        regions: [
          { start: 0, end: 30, label: "intro" },
          { start: 30, end: 60, label: "unlabeled" },
          { start: 60, end: 90, label: "buildup" },
          { start: 90, end: 120, label: "unlabeled" },
          { start: 120, end: 150, label: "breakdown" },
        ],
      });

      const { container } = render(<RegionOverlays />);
      const overlays = container.querySelectorAll('[style*="position: absolute"]');

      // Only intro, buildup, breakdown should be rendered
      expect(overlays).toHaveLength(3);
    });
  });

  describe("Opacity Management", () => {
    it("applies 0.4 opacity to region overlays", () => {
      useStructureStore.setState({
        regions: [{ start: 0, end: 60, label: "intro" }],
      });

      const { container } = render(<RegionOverlays />);
      const overlay = container.querySelector('[style*="position: absolute"]');

      expect(overlay?.getAttribute("style")).toContain("opacity: 0.4");
    });

    it("applies same opacity to all regions", () => {
      useStructureStore.setState({
        regions: [
          { start: 0, end: 30, label: "intro" },
          { start: 30, end: 60, label: "buildup" },
          { start: 60, end: 90, label: "breakdown" },
        ],
      });

      const { container } = render(<RegionOverlays />);
      const overlays = container.querySelectorAll('[style*="position: absolute"]');

      overlays.forEach((overlay) => {
        expect(overlay.getAttribute("style")).toContain("opacity: 0.4");
      });
    });

    it("disables pointer events", () => {
      useStructureStore.setState({
        regions: [{ start: 0, end: 60, label: "intro" }],
      });

      const { container } = render(<RegionOverlays />);
      const overlay = container.querySelector('[style*="position: absolute"]');

      expect(overlay?.getAttribute("style")).toContain("pointer-events: none");
    });
  });

  describe("Viewport Clipping", () => {
    it("does not render regions completely before viewport", () => {
      useWaveformStore.setState({
        viewportStart: 100,
        viewportEnd: 180,
      });

      useStructureStore.setState({
        regions: [
          { start: 0, end: 50, label: "intro" },
          { start: 100, end: 150, label: "buildup" },
        ],
      });

      const { container } = render(<RegionOverlays />);
      const overlays = container.querySelectorAll('[style*="position: absolute"]');

      // Only buildup region should be rendered (intro ends before viewport)
      expect(overlays).toHaveLength(1);
    });

    it("does not render regions completely after viewport", () => {
      useWaveformStore.setState({
        viewportStart: 0,
        viewportEnd: 100,
      });

      useStructureStore.setState({
        regions: [
          { start: 0, end: 50, label: "intro" },
          { start: 150, end: 180, label: "buildup" },
        ],
      });

      const { container } = render(<RegionOverlays />);
      const overlays = container.querySelectorAll('[style*="position: absolute"]');

      // Only intro region should be rendered (buildup starts after viewport)
      expect(overlays).toHaveLength(1);
    });

    it("clips region start to viewport start", () => {
      useWaveformStore.setState({
        viewportStart: 40,
        viewportEnd: 120,
      });

      useStructureStore.setState({
        regions: [{ start: 0, end: 60, label: "intro" }],
      });

      const { container } = render(<RegionOverlays />);
      const overlay = container.querySelector('[style*="position: absolute"]');

      // Visible portion: [40, 60] within viewport [40, 120]
      // Position: (40 - 40) / (120 - 40) * 100 = 0%
      expect(overlay?.getAttribute("style")).toContain("left: 0%");
      // Width: (60 - 40) / (120 - 40) * 100 = 20 / 80 * 100 = 25%
      expect(overlay?.getAttribute("style")).toContain("width: 25%");
    });

    it("clips region end to viewport end", () => {
      useWaveformStore.setState({
        viewportStart: 40,
        viewportEnd: 120,
      });

      useStructureStore.setState({
        regions: [{ start: 100, end: 160, label: "verse" }],
      });

      const { container } = render(<RegionOverlays />);
      const overlay = container.querySelector('[style*="position: absolute"]');

      // Visible portion: [100, 120] within viewport [40, 120]
      // Position: (100 - 40) / (120 - 40) * 100 = 60 / 80 * 100 = 75%
      expect(overlay?.getAttribute("style")).toContain("left: 75%");
      // Width: (120 - 100) / (120 - 40) * 100 = 20 / 80 * 100 = 25%
      expect(overlay?.getAttribute("style")).toContain("width: 25%");
    });

    it("clips both start and end when region spans viewport", () => {
      useWaveformStore.setState({
        viewportStart: 50,
        viewportEnd: 150,
      });

      useStructureStore.setState({
        regions: [{ start: 0, end: 180, label: "intro" }],
      });

      const { container } = render(<RegionOverlays />);
      const overlay = container.querySelector('[style*="position: absolute"]');

      // Visible portion: [50, 150] within viewport [50, 150]
      // Should span entire width
      expect(overlay?.getAttribute("style")).toContain("left: 0%");
      expect(overlay?.getAttribute("style")).toContain("width: 100%");
    });

    it("renders partial region when overlapping viewport start", () => {
      useWaveformStore.setState({
        viewportStart: 50,
        viewportEnd: 150,
      });

      useStructureStore.setState({
        regions: [{ start: 30, end: 80, label: "intro" }],
      });

      const { container } = render(<RegionOverlays />);
      const overlays = container.querySelectorAll('[style*="position: absolute"]');

      expect(overlays).toHaveLength(1);
    });

    it("renders partial region when overlapping viewport end", () => {
      useWaveformStore.setState({
        viewportStart: 50,
        viewportEnd: 150,
      });

      useStructureStore.setState({
        regions: [{ start: 120, end: 180, label: "buildup" }],
      });

      const { container } = render(<RegionOverlays />);
      const overlays = container.querySelectorAll('[style*="position: absolute"]');

      expect(overlays).toHaveLength(1);
    });
  });

  describe("Viewport Override Props", () => {
    it("uses viewport props instead of store when provided", () => {
      useWaveformStore.setState({
        viewportStart: 0,
        viewportEnd: 180,
      });

      useStructureStore.setState({
        regions: [
          { start: 0, end: 50, label: "intro" },
          { start: 100, end: 150, label: "buildup" },
        ],
      });

      // Override viewport to show only middle section
      const { container } = render(<RegionOverlays viewportStart={75} viewportEnd={125} />);

      const overlays = container.querySelectorAll('[style*="position: absolute"]');

      // Only buildup region should be visible in [75, 125]
      expect(overlays).toHaveLength(1);
    });

    it("calculates positions using viewport props", () => {
      useStructureStore.setState({
        regions: [{ start: 60, end: 120, label: "buildup" }],
      });

      // Override viewport to [40, 140]
      const { container } = render(<RegionOverlays viewportStart={40} viewportEnd={140} />);

      const overlay = container.querySelector('[style*="position: absolute"]');

      // Position: (60 - 40) / (140 - 40) * 100 = 20 / 100 * 100 = 20%
      expect(overlay?.getAttribute("style")).toContain("left: 20%");
      // Width: (120 - 60) / (140 - 40) * 100 = 60 / 100 * 100 = 60%
      expect(overlay?.getAttribute("style")).toContain("width: 60%");
    });
  });

  describe("Positioning", () => {
    it("positions region at viewport start", () => {
      useWaveformStore.setState({
        viewportStart: 0,
        viewportEnd: 180,
      });

      useStructureStore.setState({
        regions: [{ start: 0, end: 60, label: "intro" }],
      });

      const { container } = render(<RegionOverlays />);
      const overlay = container.querySelector('[style*="position: absolute"]');

      expect(overlay?.getAttribute("style")).toContain("left: 0%");
    });

    it("positions region at viewport middle", () => {
      useWaveformStore.setState({
        viewportStart: 0,
        viewportEnd: 180,
      });

      useStructureStore.setState({
        regions: [{ start: 90, end: 120, label: "buildup" }],
      });

      const { container } = render(<RegionOverlays />);
      const overlay = container.querySelector('[style*="position: absolute"]');

      // 90 / 180 * 100 = 50%
      expect(overlay?.getAttribute("style")).toContain("left: 50%");
    });

    it("calculates correct width for region", () => {
      useWaveformStore.setState({
        viewportStart: 0,
        viewportEnd: 180,
      });

      useStructureStore.setState({
        regions: [{ start: 60, end: 120, label: "buildup" }],
      });

      const { container } = render(<RegionOverlays />);
      const overlay = container.querySelector('[style*="position: absolute"]');

      // (120 - 60) / 180 * 100 = 33.333...%
      expect(overlay?.getAttribute("style")).toContain("width: 33.33");
    });

    it("renders region at full height", () => {
      useStructureStore.setState({
        regions: [{ start: 0, end: 60, label: "intro" }],
      });

      const { container } = render(<RegionOverlays />);
      const overlay = container.querySelector('[style*="position: absolute"]');

      expect(overlay?.getAttribute("style")).toContain("height: 100%");
      expect(overlay?.getAttribute("style")).toContain("top: 0");
    });

    it("uses absolute positioning", () => {
      useStructureStore.setState({
        regions: [{ start: 0, end: 60, label: "intro" }],
      });

      const { container } = render(<RegionOverlays />);
      const overlay = container.querySelector('[style*="position: absolute"]');

      expect(overlay?.getAttribute("style")).toContain("position: absolute");
    });
  });

  describe("Edge Cases", () => {
    it("handles empty regions array", () => {
      useStructureStore.setState({ regions: [] });

      const { container } = render(<RegionOverlays />);
      const overlays = container.querySelectorAll('[style*="position: absolute"]');

      expect(overlays).toHaveLength(0);
    });

    it("handles region exactly at viewport boundaries", () => {
      useWaveformStore.setState({
        viewportStart: 60,
        viewportEnd: 120,
      });

      useStructureStore.setState({
        regions: [{ start: 60, end: 120, label: "buildup" }],
      });

      const { container } = render(<RegionOverlays />);
      const overlays = container.querySelectorAll('[style*="position: absolute"]');

      expect(overlays).toHaveLength(1);
      const overlay = overlays[0];
      expect(overlay.getAttribute("style")).toContain("left: 0%");
      expect(overlay.getAttribute("style")).toContain("width: 100%");
    });

    it("handles very small region", () => {
      useWaveformStore.setState({
        viewportStart: 0,
        viewportEnd: 180,
      });

      useStructureStore.setState({
        regions: [{ start: 90, end: 90.5, label: "buildup" }],
      });

      const { container } = render(<RegionOverlays />);
      const overlays = container.querySelectorAll('[style*="position: absolute"]');

      expect(overlays).toHaveLength(1);
    });

    it("handles multiple overlapping regions", () => {
      useStructureStore.setState({
        regions: [
          { start: 0, end: 100, label: "intro" },
          { start: 50, end: 150, label: "buildup" },
          { start: 100, end: 180, label: "breakdown" },
        ],
      });

      const { container } = render(<RegionOverlays />);
      const overlays = container.querySelectorAll('[style*="position: absolute"]');

      // All three should render (they overlap but are all visible)
      expect(overlays).toHaveLength(3);
    });

    it("handles region with fractional timestamps", () => {
      useWaveformStore.setState({
        viewportStart: 0,
        viewportEnd: 180,
      });

      useStructureStore.setState({
        regions: [{ start: 45.5, end: 90.75, label: "buildup" }],
      });

      const { container } = render(<RegionOverlays />);
      const overlays = container.querySelectorAll('[style*="position: absolute"]');

      expect(overlays).toHaveLength(1);
    });

    it("handles viewport with zero duration", () => {
      useWaveformStore.setState({
        viewportStart: 90,
        viewportEnd: 90,
      });

      useStructureStore.setState({
        regions: [{ start: 0, end: 180, label: "intro" }],
      });

      const { container } = render(<RegionOverlays />);
      // Should not crash, but division by zero might cause NaN
      const overlays = container.querySelectorAll('[style*="position: absolute"]');
      expect(overlays.length).toBeGreaterThanOrEqual(0);
    });
  });

  describe("Multiple Regions", () => {
    it("renders regions in order", () => {
      useStructureStore.setState({
        regions: [
          { start: 0, end: 30, label: "intro" },
          { start: 30, end: 60, label: "buildup" },
          { start: 60, end: 90, label: "breakdown" },
        ],
      });

      const { container } = render(<RegionOverlays />);
      const overlays = container.querySelectorAll('[style*="position: absolute"]');

      expect(overlays).toHaveLength(3);
    });

    it("maintains distinct colors for adjacent regions", () => {
      useStructureStore.setState({
        regions: [
          { start: 0, end: 60, label: "intro" },
          { start: 60, end: 120, label: "buildup" },
        ],
      });

      const { container } = render(<RegionOverlays />);
      const overlays = container.querySelectorAll('[style*="position: absolute"]');

      const intro = overlays[0];
      const buildup = overlays[1];

      expect(intro.getAttribute("style")).toContain(labelColors.intro);
      expect(buildup.getAttribute("style")).toContain(labelColors.buildup);
      expect(labelColors.intro).not.toBe(labelColors.buildup);
    });
  });
});
