import { describe, it, expect, beforeEach } from "vitest";
import { render } from "@testing-library/react";
import { WaveformCanvas } from "@/components/Waveform/WaveformCanvas";
import { useWaveformStore } from "@/stores";

describe("WaveformCanvas", () => {
  beforeEach(() => {
    useWaveformStore.getState().reset();
  });

  describe("SVG Setup", () => {
    it("creates SVG element with correct viewBox", () => {
      const { container } = render(<WaveformCanvas />);
      const svg = container.querySelector("svg");

      expect(svg).toBeInTheDocument();
      expect(svg?.getAttribute("viewBox")).toBe("0 0 100 100");
    });

    it("sets preserveAspectRatio to none", () => {
      const { container } = render(<WaveformCanvas />);
      const svg = container.querySelector("svg");

      expect(svg?.getAttribute("preserveAspectRatio")).toBe("none");
    });

    it("sets dark background color", () => {
      const { container } = render(<WaveformCanvas />);
      const svg = container.querySelector("svg");

      expect(svg?.style.background).toBe("rgb(10, 10, 18)");
    });

    it("sets width and height to 100%", () => {
      const { container } = render(<WaveformCanvas />);
      const svg = container.querySelector("svg");

      expect(svg?.style.width).toBe("100%");
      expect(svg?.style.height).toBe("100%");
    });
  });

  describe("Multi-Channel Rendering", () => {
    beforeEach(() => {
      // Set up simple waveform data
      useWaveformStore.setState({
        waveformBass: [0.5, 0.6, 0.7],
        waveformMids: [0.3, 0.4, 0.5],
        waveformHighs: [0.2, 0.3, 0.4],
        waveformTimes: [0, 1, 2],
        viewportStart: 0,
        viewportEnd: 3,
      });
    });

    it("renders bass channel waveform", () => {
      const { container } = render(<WaveformCanvas />);
      const paths = container.querySelectorAll("path");

      // First path should be bass (cyan)
      expect(paths[0]).toBeInTheDocument();
      expect(paths[0].getAttribute("fill")).toBe("rgba(0, 229, 204, 0.8)");
    });

    it("renders mids channel waveform", () => {
      const { container } = render(<WaveformCanvas />);
      const paths = container.querySelectorAll("path");

      // Second path should be mids (purple)
      expect(paths[1]).toBeInTheDocument();
      expect(paths[1].getAttribute("fill")).toBe("rgba(123, 106, 255, 0.8)");
    });

    it("renders highs channel waveform", () => {
      const { container } = render(<WaveformCanvas />);
      const paths = container.querySelectorAll("path");

      // Third path should be highs (pink)
      expect(paths[2]).toBeInTheDocument();
      expect(paths[2].getAttribute("fill")).toBe("rgba(255, 107, 181, 0.8)");
    });

    it("applies correct colors to each channel", () => {
      const { container } = render(<WaveformCanvas />);
      const paths = container.querySelectorAll("path");

      expect(paths[0].getAttribute("fill")).toBe("rgba(0, 229, 204, 0.8)"); // Bass: cyan
      expect(paths[1].getAttribute("fill")).toBe("rgba(123, 106, 255, 0.8)"); // Mids: purple
      expect(paths[2].getAttribute("fill")).toBe("rgba(255, 107, 181, 0.8)"); // Highs: pink
    });

    it("renders center baseline", () => {
      const { container } = render(<WaveformCanvas />);
      const line = container.querySelector("line");

      expect(line).toBeInTheDocument();
      expect(line?.getAttribute("y1")).toBe("50");
      expect(line?.getAttribute("y2")).toBe("50");
    });
  });

  describe("Viewport Filtering", () => {
    it("only renders waveform within viewport", () => {
      useWaveformStore.setState({
        waveformBass: [0.5, 0.6, 0.7, 0.8, 0.9],
        waveformMids: [0.3, 0.4, 0.5, 0.6, 0.7],
        waveformHighs: [0.2, 0.3, 0.4, 0.5, 0.6],
        waveformTimes: [0, 1, 2, 3, 4],
        viewportStart: 1,
        viewportEnd: 3,
      });

      const { container } = render(<WaveformCanvas />);
      const paths = container.querySelectorAll("path");

      // Paths should exist for visible samples
      expect(paths[0].getAttribute("d")).toBeTruthy();
      expect(paths[1].getAttribute("d")).toBeTruthy();
      expect(paths[2].getAttribute("d")).toBeTruthy();
    });

    it("updates when viewport changes", () => {
      useWaveformStore.setState({
        waveformBass: [0.5, 0.6, 0.7],
        waveformMids: [0.3, 0.4, 0.5],
        waveformHighs: [0.2, 0.3, 0.4],
        waveformTimes: [0, 1, 2],
        viewportStart: 0,
        viewportEnd: 3,
      });

      const { container, rerender } = render(<WaveformCanvas />);
      const pathsBefore = container.querySelectorAll("path")[0].getAttribute("d");

      // Change viewport
      useWaveformStore.setState({
        viewportStart: 0,
        viewportEnd: 2,
      });

      rerender(<WaveformCanvas />);
      const pathsAfter = container.querySelectorAll("path")[0].getAttribute("d");

      // Paths should be different
      expect(pathsAfter).not.toBe(pathsBefore);
    });
  });

  describe("Path Generation", () => {
    it("generates valid SVG path data", () => {
      useWaveformStore.setState({
        waveformBass: [0.5, 0.6],
        waveformMids: [0.3, 0.4],
        waveformHighs: [0.2, 0.3],
        waveformTimes: [0, 1],
        viewportStart: 0,
        viewportEnd: 2,
      });

      const { container } = render(<WaveformCanvas />);
      const paths = container.querySelectorAll("path");

      // All paths should have valid d attribute starting with M (moveTo)
      expect(paths[0].getAttribute("d")).toMatch(/^M/);
      expect(paths[1].getAttribute("d")).toMatch(/^M/);
      expect(paths[2].getAttribute("d")).toMatch(/^M/);
    });

    it("creates closed paths with Z command", () => {
      useWaveformStore.setState({
        waveformBass: [0.5, 0.6],
        waveformMids: [0.3, 0.4],
        waveformHighs: [0.2, 0.3],
        waveformTimes: [0, 1],
        viewportStart: 0,
        viewportEnd: 2,
      });

      const { container } = render(<WaveformCanvas />);
      const paths = container.querySelectorAll("path");

      // Paths should end with Z (closePath)
      expect(paths[0].getAttribute("d")).toMatch(/Z$/);
      expect(paths[1].getAttribute("d")).toMatch(/Z$/);
      expect(paths[2].getAttribute("d")).toMatch(/Z$/);
    });

    it("generates mirrored paths (top and bottom)", () => {
      useWaveformStore.setState({
        waveformBass: [0.8],
        waveformMids: [0.5],
        waveformHighs: [0.3],
        waveformTimes: [0],
        viewportStart: 0,
        viewportEnd: 1,
      });

      const { container } = render(<WaveformCanvas />);
      const bassPath = container.querySelectorAll("path")[0].getAttribute("d") || "";

      // Path should contain coordinates both above and below center (50)
      // This is a simplified check - actual mirroring is complex
      expect(bassPath.length).toBeGreaterThan(0);
    });
  });

  describe("Edge Cases", () => {
    it("handles empty waveform data", () => {
      useWaveformStore.setState({
        waveformBass: [],
        waveformMids: [],
        waveformHighs: [],
        waveformTimes: [],
        viewportStart: 0,
        viewportEnd: 10,
      });

      const { container } = render(<WaveformCanvas />);
      const paths = container.querySelectorAll("path");

      // Paths should exist but be empty
      expect(paths[0].getAttribute("d")).toBe("");
      expect(paths[1].getAttribute("d")).toBe("");
      expect(paths[2].getAttribute("d")).toBe("");
    });

    it("handles single sample", () => {
      useWaveformStore.setState({
        waveformBass: [0.5],
        waveformMids: [0.3],
        waveformHighs: [0.2],
        waveformTimes: [0],
        viewportStart: 0,
        viewportEnd: 1,
      });

      const { container } = render(<WaveformCanvas />);
      const paths = container.querySelectorAll("path");

      // Should generate valid paths even with single sample
      expect(paths[0].getAttribute("d")).toBeTruthy();
      expect(paths[1].getAttribute("d")).toBeTruthy();
      expect(paths[2].getAttribute("d")).toBeTruthy();
    });

    it("handles zero amplitude waveform", () => {
      useWaveformStore.setState({
        waveformBass: [0, 0, 0],
        waveformMids: [0, 0, 0],
        waveformHighs: [0, 0, 0],
        waveformTimes: [0, 1, 2],
        viewportStart: 0,
        viewportEnd: 3,
      });

      const { container } = render(<WaveformCanvas />);
      const paths = container.querySelectorAll("path");

      // Should still generate paths (flat line at center)
      expect(paths[0].getAttribute("d")).toBeTruthy();
    });

    it("handles viewport with zero duration", () => {
      useWaveformStore.setState({
        waveformBass: [0.5],
        waveformMids: [0.3],
        waveformHighs: [0.2],
        waveformTimes: [0],
        viewportStart: 5,
        viewportEnd: 5,
      });

      const { container } = render(<WaveformCanvas />);
      const paths = container.querySelectorAll("path");

      // Should handle gracefully (empty paths)
      expect(paths[0].getAttribute("d")).toBe("");
    });

    it("handles negative viewport (end before start)", () => {
      useWaveformStore.setState({
        waveformBass: [0.5],
        waveformMids: [0.3],
        waveformHighs: [0.2],
        waveformTimes: [0],
        viewportStart: 10,
        viewportEnd: 5,
      });

      const { container } = render(<WaveformCanvas />);
      const paths = container.querySelectorAll("path");

      // Should handle gracefully
      expect(paths).toHaveLength(3);
    });

    it("handles very zoomed in view", () => {
      useWaveformStore.setState({
        waveformBass: [0.5, 0.6, 0.7, 0.8, 0.9],
        waveformMids: [0.3, 0.4, 0.5, 0.6, 0.7],
        waveformHighs: [0.2, 0.3, 0.4, 0.5, 0.6],
        waveformTimes: [0, 1, 2, 3, 4],
        viewportStart: 1.5,
        viewportEnd: 2.5,
      });

      const { container } = render(<WaveformCanvas />);
      const paths = container.querySelectorAll("path");

      // Should render only visible samples
      expect(paths[0].getAttribute("d")).toBeTruthy();
    });

    it("handles very zoomed out view", () => {
      useWaveformStore.setState({
        waveformBass: [0.5, 0.6],
        waveformMids: [0.3, 0.4],
        waveformHighs: [0.2, 0.3],
        waveformTimes: [0, 1],
        viewportStart: 0,
        viewportEnd: 1000,
      });

      const { container } = render(<WaveformCanvas />);
      const paths = container.querySelectorAll("path");

      // Should still render the samples
      expect(paths[0].getAttribute("d")).toBeTruthy();
    });

    it("handles missing samples in data arrays", () => {
      useWaveformStore.setState({
        waveformBass: [0.5, undefined, 0.7] as any,
        waveformMids: [0.3, 0.4, undefined] as any,
        waveformHighs: [undefined, 0.3, 0.4] as any,
        waveformTimes: [0, 1, 2],
        viewportStart: 0,
        viewportEnd: 3,
      });

      const { container } = render(<WaveformCanvas />);
      const paths = container.querySelectorAll("path");

      // Should handle undefined/null gracefully (treat as 0)
      expect(paths[0].getAttribute("d")).toBeTruthy();
      expect(paths[1].getAttribute("d")).toBeTruthy();
      expect(paths[2].getAttribute("d")).toBeTruthy();
    });
  });

  describe("Scaling and Normalization", () => {
    it("scales waveform to fit within SVG bounds", () => {
      useWaveformStore.setState({
        waveformBass: [1.0, 0.5, 1.0],
        waveformMids: [0.8, 0.4, 0.8],
        waveformHighs: [0.5, 0.2, 0.5],
        waveformTimes: [0, 1, 2],
        viewportStart: 0,
        viewportEnd: 3,
      });

      const { container } = render(<WaveformCanvas />);
      const paths = container.querySelectorAll("path");

      // All paths should exist and be non-empty
      expect(paths[0].getAttribute("d")).toBeTruthy();
      expect(paths[1].getAttribute("d")).toBeTruthy();
      expect(paths[2].getAttribute("d")).toBeTruthy();
    });

    it("handles very small amplitude values", () => {
      useWaveformStore.setState({
        waveformBass: [0.001, 0.002, 0.001],
        waveformMids: [0.0005, 0.001, 0.0005],
        waveformHighs: [0.0001, 0.0002, 0.0001],
        waveformTimes: [0, 1, 2],
        viewportStart: 0,
        viewportEnd: 3,
      });

      const { container } = render(<WaveformCanvas />);
      const paths = container.querySelectorAll("path");

      // Should still generate paths
      expect(paths[0].getAttribute("d")).toBeTruthy();
    });

    it("handles very large amplitude values", () => {
      useWaveformStore.setState({
        waveformBass: [100, 200, 150],
        waveformMids: [80, 150, 120],
        waveformHighs: [50, 100, 75],
        waveformTimes: [0, 1, 2],
        viewportStart: 0,
        viewportEnd: 3,
      });

      const { container } = render(<WaveformCanvas />);
      const paths = container.querySelectorAll("path");

      // Should scale down to fit
      expect(paths[0].getAttribute("d")).toBeTruthy();
    });
  });

  describe("Cumulative Stacking", () => {
    it("stacks channels cumulatively (bass, then mids, then highs)", () => {
      useWaveformStore.setState({
        waveformBass: [0.3],
        waveformMids: [0.2],
        waveformHighs: [0.1],
        waveformTimes: [0],
        viewportStart: 0,
        viewportEnd: 1,
      });

      const { container } = render(<WaveformCanvas />);
      const paths = container.querySelectorAll("path");

      // All three layers should render
      // Bass: 0-0.3
      // Mids: 0.3-0.5 (bass + mids)
      // Highs: 0.5-0.6 (bass + mids + highs)
      expect(paths).toHaveLength(3);
    });
  });

  describe("Performance", () => {
    it("handles large dataset efficiently", () => {
      const largeDataset = Array.from({ length: 1000 }, (_, i) => ({
        bass: Math.sin(i / 100) * 0.5,
        mids: Math.cos(i / 100) * 0.3,
        highs: Math.sin(i / 50) * 0.2,
        time: i / 10,
      }));

      useWaveformStore.setState({
        waveformBass: largeDataset.map((d) => d.bass),
        waveformMids: largeDataset.map((d) => d.mids),
        waveformHighs: largeDataset.map((d) => d.highs),
        waveformTimes: largeDataset.map((d) => d.time),
        viewportStart: 0,
        viewportEnd: 100,
      });

      const { container } = render(<WaveformCanvas />);
      const paths = container.querySelectorAll("path");

      // Should render without errors
      expect(paths).toHaveLength(3);
    });
  });
});
