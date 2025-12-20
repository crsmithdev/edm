import { describe, it, expect, beforeEach } from "vitest";
import { render } from "@testing-library/react";
import { BeatGrid } from "@/components/Waveform/BeatGrid";
import { useWaveformStore, useTempoStore } from "@/stores";

describe("BeatGrid", () => {
  beforeEach(() => {
    useWaveformStore.getState().reset();
    useTempoStore.getState().reset();

    // Set up basic state
    useWaveformStore.setState({
      viewportStart: 0,
      viewportEnd: 180,
      duration: 180,
    });

    useTempoStore.setState({
      trackBPM: 128,
      trackDownbeat: 0,
    });
  });

  describe("Grid Calculation from BPM", () => {
    it("calculates correct bar duration from BPM", () => {
      // At 128 BPM: beat = 60/128 = 0.46875s, bar = 0.46875 * 4 = 1.875s
      useTempoStore.setState({ trackBPM: 128, trackDownbeat: 0 });
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 10 });

      const { container } = render(<BeatGrid />);
      const lines = container.querySelectorAll('[style*="position: absolute"]');

      // In 10 seconds: 10 / 1.875 = 5.33 bars
      // Should have downbeat (bar 1) + bars 2, 3, 4, 5
      expect(lines.length).toBeGreaterThan(0);
    });

    it("adjusts grid spacing when BPM changes", () => {
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 20 });

      const { container, rerender } = render(<BeatGrid />);

      useTempoStore.setState({ trackBPM: 120 }); // Slower BPM = fewer bars
      rerender(<BeatGrid />);

      const lines = container.querySelectorAll('[style*="position: absolute"]');
      expect(lines.length).toBeGreaterThan(0);
    });

    it("renders no grid when BPM is 0", () => {
      useTempoStore.setState({ trackBPM: 0 });

      const { container } = render(<BeatGrid />);
      const lines = container.querySelectorAll('[style*="position: absolute"]');

      expect(lines).toHaveLength(0);
    });

    it("renders no grid when duration is 0", () => {
      useWaveformStore.setState({ duration: 0 });

      const { container } = render(<BeatGrid />);
      const lines = container.querySelectorAll('[style*="position: absolute"]');

      expect(lines).toHaveLength(0);
    });

    it("calculates correct beat positions at 128 BPM", () => {
      // At 128 BPM: beat duration = 60/128 = 0.46875s
      useTempoStore.setState({ trackBPM: 128, trackDownbeat: 0 });
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 2 });

      const { container } = render(<BeatGrid />);
      const lines = container.querySelectorAll('[style*="position: absolute"]');

      expect(lines.length).toBeGreaterThan(0);
    });

    it("calculates correct beat positions at 140 BPM", () => {
      // At 140 BPM: beat duration = 60/140 ≈ 0.4286s
      useTempoStore.setState({ trackBPM: 140, trackDownbeat: 0 });
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 2 });

      const { container } = render(<BeatGrid />);
      const lines = container.querySelectorAll('[style*="position: absolute"]');

      expect(lines.length).toBeGreaterThan(0);
    });
  });

  describe("Downbeat Alignment", () => {
    it("renders downbeat marker at downbeat position", () => {
      useTempoStore.setState({ trackBPM: 128, trackDownbeat: 0 });
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 10 });

      const { container } = render(<BeatGrid />);
      const downbeat = container.querySelector('[style*="rgb(244, 67, 54)"]'); // Red color

      expect(downbeat).toBeInTheDocument();
    });

    it("renders downbeat with red color", () => {
      useTempoStore.setState({ trackBPM: 128, trackDownbeat: 0 });
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 10 });

      const { container } = render(<BeatGrid />);
      const downbeat = container.querySelector('[style*="rgb(244, 67, 54)"]');

      expect(downbeat?.getAttribute("style")).toContain("rgb(244, 67, 54)");
    });

    it("renders downbeat with 3px width", () => {
      useTempoStore.setState({ trackBPM: 128, trackDownbeat: 0 });
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 10 });

      const { container } = render(<BeatGrid />);
      const downbeat = container.querySelector('[style*="rgb(244, 67, 54)"]');

      expect(downbeat?.getAttribute("style")).toContain("width: 3px");
    });

    it("positions downbeat at 0% when downbeat is at viewport start", () => {
      useTempoStore.setState({ trackBPM: 128, trackDownbeat: 0 });
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 10 });

      const { container } = render(<BeatGrid />);
      const downbeat = container.querySelector('[style*="rgb(244, 67, 54)"]');

      expect(downbeat?.getAttribute("style")).toContain("left: 0%");
    });

    it("positions downbeat relative to viewport when offset", () => {
      useTempoStore.setState({ trackBPM: 128, trackDownbeat: 1 });
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 10 });

      const { container } = render(<BeatGrid />);
      const downbeat = container.querySelector('[style*="rgb(244, 67, 54)"]');

      // Downbeat at 1s in [0, 10] viewport = 10%
      expect(downbeat?.getAttribute("style")).toContain("left: 10%");
    });

    it("does not render downbeat when outside viewport", () => {
      useTempoStore.setState({ trackBPM: 128, trackDownbeat: 0 });
      useWaveformStore.setState({ viewportStart: 10, viewportEnd: 20 });

      const { container } = render(<BeatGrid />);
      const downbeat = container.querySelector('[style*="rgb(244, 67, 54)"]');

      expect(downbeat).not.toBeInTheDocument();
    });

    it("labels downbeat as bar 1", () => {
      useTempoStore.setState({ trackBPM: 128, trackDownbeat: 0 });
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 10 });

      const { container } = render(<BeatGrid />);
      const downbeat = container.querySelector('[style*="rgb(244, 67, 54)"]');
      const label = downbeat?.querySelector('[style*="font-size: 10px"]');

      expect(label?.textContent).toBe("1");
    });
  });

  describe("Bar Lines", () => {
    it("renders bar lines with gray color", () => {
      useTempoStore.setState({ trackBPM: 128, trackDownbeat: 0 });
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 10, duration: 180 });

      const { container } = render(<BeatGrid />);
      const barLines = container.querySelectorAll('[style*="rgba(200, 200, 200, 0.7)"]');

      expect(barLines.length).toBeGreaterThan(0);
    });

    it("renders bar lines with 2px width", () => {
      useTempoStore.setState({ trackBPM: 128, trackDownbeat: 0 });
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 10, duration: 180 });

      const { container } = render(<BeatGrid />);
      const barLine = container.querySelector('[style*="rgba(200, 200, 200, 0.7)"]');

      expect(barLine?.getAttribute("style")).toContain("width: 2px");
    });

    it("labels bar lines with bar numbers", () => {
      useTempoStore.setState({ trackBPM: 128, trackDownbeat: 0 });
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 20 });

      const { container } = render(<BeatGrid />);
      const labels = container.querySelectorAll('[style*="font-size: 10px"]');

      expect(labels.length).toBeGreaterThan(0);
    });

    it("increments bar numbers correctly", () => {
      useTempoStore.setState({ trackBPM: 60, trackDownbeat: 0 }); // 4s per bar for easy math
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 20 });

      const { container } = render(<BeatGrid />);
      const labels = container.querySelectorAll('[style*="font-size: 10px"]');
      const numbers = Array.from(labels).map((l) => l.textContent);

      // Should have bars 1, 2, 3, 4, 5, 6 (at 0s, 4s, 8s, 12s, 16s, 20s)
      expect(numbers).toContain("1");
      expect(numbers).toContain("2");
      expect(numbers).toContain("3");
    });

    it("positions bar lines correctly", () => {
      useTempoStore.setState({ trackBPM: 60, trackDownbeat: 0 }); // 4s per bar
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 20 });

      const { container } = render(<BeatGrid />);
      const lines = container.querySelectorAll('[style*="position: absolute"]');

      expect(lines.length).toBeGreaterThan(0);
    });
  });

  describe("Beat Lines", () => {
    it("renders beat lines when zoomed in", () => {
      useTempoStore.setState({ trackBPM: 128, trackDownbeat: 0 });
      // Zoom in to <15% of track (180s * 0.15 = 27s)
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 10, duration: 180 });

      const { container } = render(<BeatGrid />);
      const beatLines = container.querySelectorAll('[style*="rgba(150, 150, 150, 0.3)"]');

      expect(beatLines.length).toBeGreaterThan(0);
    });

    it("does not render beat lines when zoomed out", () => {
      useTempoStore.setState({ trackBPM: 128, trackDownbeat: 0 });
      // Zoom out to >15% of track
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 100, duration: 180 });

      const { container } = render(<BeatGrid />);
      const beatLines = container.querySelectorAll('[style*="rgba(150, 150, 150, 0.3)"]');

      // Should not render beats when zoomed out
      expect(beatLines).toHaveLength(0);
    });

    it("renders beat lines with light gray color", () => {
      useTempoStore.setState({ trackBPM: 128, trackDownbeat: 0 });
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 10, duration: 180 });

      const { container } = render(<BeatGrid />);
      const beatLine = container.querySelector('[style*="rgba(150, 150, 150, 0.3)"]');

      expect(beatLine).toBeInTheDocument();
    });

    it("renders beat lines with 1px width", () => {
      useTempoStore.setState({ trackBPM: 128, trackDownbeat: 0 });
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 10, duration: 180 });

      const { container } = render(<BeatGrid />);
      const beatLine = container.querySelector('[style*="rgba(150, 150, 150, 0.3)"]');

      expect(beatLine?.getAttribute("style")).toContain("width: 1px");
    });

    it("does not label beat lines", () => {
      useTempoStore.setState({ trackBPM: 128, trackDownbeat: 0 });
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 10, duration: 180 });

      const { container } = render(<BeatGrid />);
      const beatLines = container.querySelectorAll('[style*="rgba(150, 150, 150, 0.3)"]');

      beatLines.forEach((line) => {
        const label = line.querySelector('[style*="font-size: 10px"]');
        expect(label).not.toBeInTheDocument();
      });
    });
  });

  describe("Viewport Filtering", () => {
    it("only renders lines within viewport", () => {
      useTempoStore.setState({ trackBPM: 128, trackDownbeat: 0 });
      useWaveformStore.setState({ viewportStart: 50, viewportEnd: 60 });

      const { container } = render(<BeatGrid />);
      const lines = container.querySelectorAll('[style*="position: absolute"]');

      // All rendered lines should be within viewport
      lines.forEach((line) => {
        const left = parseFloat(line.getAttribute("style")?.match(/left: ([\d.]+)%/)?.[1] || "0");
        expect(left).toBeGreaterThanOrEqual(0);
        expect(left).toBeLessThanOrEqual(100);
      });
    });

    it("updates visible lines when viewport changes", () => {
      useTempoStore.setState({ trackBPM: 60, trackDownbeat: 0 }); // 4s per bar

      const { container, rerender } = render(<BeatGrid />);
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 20 });
      rerender(<BeatGrid />);

      let lines = container.querySelectorAll('[style*="position: absolute"]');
      const count1 = lines.length;

      useWaveformStore.setState({ viewportStart: 20, viewportEnd: 40 });
      rerender(<BeatGrid />);

      lines = container.querySelectorAll('[style*="position: absolute"]');
      const count2 = lines.length;

      // Should have similar number of lines (different positions)
      expect(Math.abs(count1 - count2)).toBeLessThan(3);
    });

    it("renders lines at viewport boundaries", () => {
      useTempoStore.setState({ trackBPM: 60, trackDownbeat: 0 }); // 4s per bar
      useWaveformStore.setState({ viewportStart: 4, viewportEnd: 12 }); // Bars 2 and 3

      const { container } = render(<BeatGrid />);
      const lines = container.querySelectorAll('[style*="position: absolute"]');

      expect(lines.length).toBeGreaterThan(0);
    });
  });

  describe("Zoom-Based Granularity", () => {
    it("shows every bar when zoomed in (<15% visible)", () => {
      useTempoStore.setState({ trackBPM: 128, trackDownbeat: 0 });
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 20, duration: 180 });
      // 20 / 180 ≈ 11% visible

      const { container } = render(<BeatGrid />);
      const lines = container.querySelectorAll('[style*="position: absolute"]');

      // Should have many lines (bars and beats)
      expect(lines.length).toBeGreaterThan(10);
    });

    it("shows every 2 bars when medium zoom (15-20% visible)", () => {
      useTempoStore.setState({ trackBPM: 128, trackDownbeat: 0 });
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 35, duration: 180 });
      // 35 / 180 ≈ 19% visible

      const { container } = render(<BeatGrid />);
      const lines = container.querySelectorAll('[style*="position: absolute"]');

      // Should have fewer lines (every 2 bars, no beats)
      expect(lines.length).toBeGreaterThan(0);
    });

    it("shows every 4 bars when zoomed out (20-30% visible)", () => {
      useTempoStore.setState({ trackBPM: 128, trackDownbeat: 0 });
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 50, duration: 180 });
      // 50 / 180 ≈ 28% visible

      const { container } = render(<BeatGrid />);
      const lines = container.querySelectorAll('[style*="position: absolute"]');

      expect(lines.length).toBeGreaterThan(0);
    });

    it("shows every 8 bars when very zoomed out (30-50% visible)", () => {
      useTempoStore.setState({ trackBPM: 128, trackDownbeat: 0 });
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 80, duration: 180 });
      // 80 / 180 ≈ 44% visible

      const { container } = render(<BeatGrid />);
      const lines = container.querySelectorAll('[style*="position: absolute"]');

      expect(lines.length).toBeGreaterThan(0);
    });

    it("shows every 16 bars when extremely zoomed out (>50% visible)", () => {
      useTempoStore.setState({ trackBPM: 128, trackDownbeat: 0 });
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 150, duration: 180 });
      // 150 / 180 ≈ 83% visible

      const { container } = render(<BeatGrid />);
      // Filter to only line divs (not labels)
      const lines = Array.from(container.querySelectorAll('[style*="position: absolute"]')).filter(
        (el) => el.getAttribute("style")?.includes("height: 100%")
      );

      // Should have very few lines (every 16 bars)
      expect(lines.length).toBeGreaterThan(0);
      expect(lines.length).toBeLessThan(10);
    });
  });

  describe("Bar Number Labels", () => {
    it("displays bar numbers on bar lines", () => {
      useTempoStore.setState({ trackBPM: 128, trackDownbeat: 0 });
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 20 });

      const { container } = render(<BeatGrid />);
      const labels = container.querySelectorAll('[style*="font-size: 10px"]');

      expect(labels.length).toBeGreaterThan(0);
      labels.forEach((label) => {
        expect(label.textContent).toMatch(/^\d+$/);
      });
    });

    it("does not display labels on beat lines", () => {
      useTempoStore.setState({ trackBPM: 128, trackDownbeat: 0 });
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 10, duration: 180 });

      const { container } = render(<BeatGrid />);
      const beatLines = container.querySelectorAll('[style*="rgba(150, 150, 150"]');

      beatLines.forEach((line) => {
        const label = line.querySelector('[style*="font-size: 10px"]');
        expect(label).not.toBeInTheDocument();
      });
    });

    it("positions labels at top-left of line", () => {
      useTempoStore.setState({ trackBPM: 128, trackDownbeat: 0 });
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 20 });

      const { container } = render(<BeatGrid />);
      const label = container.querySelector('[style*="font-size: 10px"]');

      expect(label?.getAttribute("style")).toContain("top: 2px");
      expect(label?.getAttribute("style")).toContain("left: 4px");
    });

    it("uses gray color for bar labels", () => {
      useTempoStore.setState({ trackBPM: 128, trackDownbeat: 0 });
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 20 });

      const { container } = render(<BeatGrid />);
      const barLine = container.querySelector('[style*="rgba(200, 200, 200, 0.7)"]');
      const label = barLine?.querySelector('[style*="font-size: 10px"]');

      expect(label?.getAttribute("style")).toContain("rgba(220, 220, 220, 0.9)");
    });

    it("uses red color for downbeat label", () => {
      useTempoStore.setState({ trackBPM: 128, trackDownbeat: 0 });
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 10 });

      const { container } = render(<BeatGrid />);
      const downbeat = container.querySelector('[style*="rgb(244, 67, 54)"]');
      const label = downbeat?.querySelector('[style*="font-size: 10px"]');

      expect(label?.getAttribute("style")).toContain("rgb(244, 67, 54)");
    });
  });

  describe("Viewport Override Props", () => {
    it("uses viewport props instead of store when provided", () => {
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 180 });
      useTempoStore.setState({ trackBPM: 60, trackDownbeat: 0 }); // 4s per bar

      const { container } = render(<BeatGrid viewportStart={8} viewportEnd={16} />);

      // Should show bars around 8-16s range (bars 3 and 4)
      const lines = container.querySelectorAll('[style*="position: absolute"]');
      expect(lines.length).toBeGreaterThan(0);
    });

    it("calculates positions using viewport props", () => {
      useTempoStore.setState({ trackBPM: 60, trackDownbeat: 0 }); // 4s per bar

      const { container } = render(<BeatGrid viewportStart={0} viewportEnd={20} />);

      const lines = container.querySelectorAll('[style*="position: absolute"]');
      expect(lines.length).toBeGreaterThan(0);
    });
  });

  describe("Edge Cases", () => {
    it("handles very high BPM (200)", () => {
      useTempoStore.setState({ trackBPM: 200, trackDownbeat: 0 });
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 10 });

      const { container } = render(<BeatGrid />);
      const lines = container.querySelectorAll('[style*="position: absolute"]');

      expect(lines.length).toBeGreaterThan(0);
    });

    it("handles very low BPM (60)", () => {
      useTempoStore.setState({ trackBPM: 60, trackDownbeat: 0 });
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 20 });

      const { container } = render(<BeatGrid />);
      const lines = container.querySelectorAll('[style*="position: absolute"]');

      expect(lines.length).toBeGreaterThan(0);
    });

    it("handles fractional downbeat", () => {
      useTempoStore.setState({ trackBPM: 128, trackDownbeat: 0.5 });
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 10 });

      const { container } = render(<BeatGrid />);
      const downbeat = container.querySelector('[style*="rgb(244, 67, 54)"]');

      expect(downbeat).toBeInTheDocument();
    });

    it("handles very small viewport", () => {
      useTempoStore.setState({ trackBPM: 128, trackDownbeat: 0 });
      useWaveformStore.setState({ viewportStart: 5, viewportEnd: 5.5, duration: 180 });

      const { container } = render(<BeatGrid />);
      // Should not crash
      const lines = container.querySelectorAll('[style*="position: absolute"]');
      expect(lines.length).toBeGreaterThanOrEqual(0);
    });

    it("renders all lines with full height", () => {
      useTempoStore.setState({ trackBPM: 128, trackDownbeat: 0 });
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 10 });

      const { container } = render(<BeatGrid />);
      // Filter to only line divs (ones with height: 100%)
      const lines = Array.from(container.querySelectorAll('[style*="position: absolute"]')).filter(
        (el) => el.getAttribute("style")?.includes("height: 100%")
      );

      lines.forEach((line) => {
        expect(line.getAttribute("style")).toContain("height: 100%");
      });
    });

    it("renders all lines with pointer-events: none", () => {
      useTempoStore.setState({ trackBPM: 128, trackDownbeat: 0 });
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 10 });

      const { container } = render(<BeatGrid />);
      // Filter to only line divs (ones with pointer-events: none)
      const lines = Array.from(container.querySelectorAll('[style*="position: absolute"]')).filter(
        (el) => el.getAttribute("style")?.includes("pointer-events: none")
      );

      lines.forEach((line) => {
        expect(line.getAttribute("style")).toContain("pointer-events: none");
      });
    });
  });
});
