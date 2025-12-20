import { describe, it, expect, beforeEach } from "vitest";
import { render } from "@testing-library/react";
import { Playhead } from "@/components/Waveform/Playhead";
import { useAudioStore, useWaveformStore } from "@/stores";

describe("Playhead", () => {
  beforeEach(() => {
    useAudioStore.getState().reset();
    useWaveformStore.getState().reset();

    // Set up basic viewport
    useWaveformStore.setState({
      viewportStart: 0,
      viewportEnd: 180,
    });

    useAudioStore.setState({ currentTime: 90 });
  });

  describe("Position Tracking", () => {
    it("renders playhead at current time position", () => {
      useAudioStore.setState({ currentTime: 90 });
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 180 });

      const { container } = render(<Playhead />);
      const playhead = container.querySelector('[style*="position: absolute"]');

      // 90 / 180 * 100 = 50%
      expect(playhead?.getAttribute("style")).toContain("left: 50%");
    });

    it("updates position when currentTime changes", () => {
      useAudioStore.setState({ currentTime: 45 });
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 180 });

      const { container, rerender } = render(<Playhead />);
      let playhead = container.querySelector('[style*="position: absolute"]');

      // 45 / 180 * 100 = 25%
      expect(playhead?.getAttribute("style")).toContain("left: 25%");

      useAudioStore.setState({ currentTime: 135 });
      rerender(<Playhead />);

      playhead = container.querySelector('[style*="position: absolute"]');
      // 135 / 180 * 100 = 75%
      expect(playhead?.getAttribute("style")).toContain("left: 75%");
    });

    it("positions at 0% when at viewport start", () => {
      useAudioStore.setState({ currentTime: 0 });
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 180 });

      const { container } = render(<Playhead />);
      const playhead = container.querySelector('[style*="position: absolute"]');

      expect(playhead?.getAttribute("style")).toContain("left: 0%");
    });

    it("positions at 100% when at viewport end", () => {
      useAudioStore.setState({ currentTime: 180 });
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 180 });

      const { container } = render(<Playhead />);
      const playhead = container.querySelector('[style*="position: absolute"]');

      expect(playhead?.getAttribute("style")).toContain("left: 100%");
    });

    it("calculates correct position for middle of viewport", () => {
      useAudioStore.setState({ currentTime: 75 });
      useWaveformStore.setState({ viewportStart: 50, viewportEnd: 100 });

      const { container } = render(<Playhead />);
      const playhead = container.querySelector('[style*="position: absolute"]');

      // (75 - 50) / (100 - 50) * 100 = 25 / 50 * 100 = 50%
      expect(playhead?.getAttribute("style")).toContain("left: 50%");
    });

    it("handles fractional currentTime", () => {
      useAudioStore.setState({ currentTime: 45.789 });
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 180 });

      const { container } = render(<Playhead />);
      const playhead = container.querySelector('[style*="position: absolute"]');

      expect(playhead).toBeInTheDocument();
    });

    it("handles fractional viewport boundaries", () => {
      useAudioStore.setState({ currentTime: 50.5 });
      useWaveformStore.setState({ viewportStart: 40.25, viewportEnd: 60.75 });

      const { container } = render(<Playhead />);
      const playhead = container.querySelector('[style*="position: absolute"]');

      expect(playhead).toBeInTheDocument();
    });
  });

  describe("Viewport Conversion", () => {
    it("converts time to viewport percentage correctly", () => {
      useAudioStore.setState({ currentTime: 60 });
      useWaveformStore.setState({ viewportStart: 40, viewportEnd: 80 });

      const { container } = render(<Playhead />);
      const playhead = container.querySelector('[style*="position: absolute"]');

      // (60 - 40) / (80 - 40) * 100 = 20 / 40 * 100 = 50%
      expect(playhead?.getAttribute("style")).toContain("left: 50%");
    });

    it("handles narrow viewport", () => {
      useAudioStore.setState({ currentTime: 90.5 });
      useWaveformStore.setState({ viewportStart: 90, viewportEnd: 91 });

      const { container } = render(<Playhead />);
      const playhead = container.querySelector('[style*="position: absolute"]');

      // (90.5 - 90) / (91 - 90) * 100 = 0.5 / 1 * 100 = 50%
      expect(playhead?.getAttribute("style")).toContain("left: 50%");
    });

    it("handles wide viewport", () => {
      useAudioStore.setState({ currentTime: 300 });
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 600 });

      const { container } = render(<Playhead />);
      const playhead = container.querySelector('[style*="position: absolute"]');

      // 300 / 600 * 100 = 50%
      expect(playhead?.getAttribute("style")).toContain("left: 50%");
    });

    it("updates position when viewport changes", () => {
      useAudioStore.setState({ currentTime: 90 });
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 180 });

      const { container, rerender } = render(<Playhead />);
      let playhead = container.querySelector('[style*="position: absolute"]');

      // 90 / 180 * 100 = 50%
      expect(playhead?.getAttribute("style")).toContain("left: 50%");

      // Change viewport but keep currentTime same
      useWaveformStore.setState({ viewportStart: 80, viewportEnd: 100 });
      rerender(<Playhead />);

      playhead = container.querySelector('[style*="position: absolute"]');
      // (90 - 80) / (100 - 80) * 100 = 10 / 20 * 100 = 50%
      expect(playhead?.getAttribute("style")).toContain("left: 50%");
    });
  });

  describe("Visibility", () => {
    it("renders when currentTime is within viewport", () => {
      useAudioStore.setState({ currentTime: 90 });
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 180 });

      const { container } = render(<Playhead />);
      const playhead = container.querySelector('[style*="position: absolute"]');

      expect(playhead).toBeInTheDocument();
    });

    it("does not render when currentTime is before viewport", () => {
      useAudioStore.setState({ currentTime: 50 });
      useWaveformStore.setState({ viewportStart: 100, viewportEnd: 180 });

      const { container } = render(<Playhead />);
      const playhead = container.querySelector('[style*="position: absolute"]');

      expect(playhead).not.toBeInTheDocument();
    });

    it("does not render when currentTime is after viewport", () => {
      useAudioStore.setState({ currentTime: 200 });
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 180 });

      const { container } = render(<Playhead />);
      const playhead = container.querySelector('[style*="position: absolute"]');

      expect(playhead).not.toBeInTheDocument();
    });

    it("renders when currentTime equals viewport start", () => {
      useAudioStore.setState({ currentTime: 50 });
      useWaveformStore.setState({ viewportStart: 50, viewportEnd: 100 });

      const { container } = render(<Playhead />);
      const playhead = container.querySelector('[style*="position: absolute"]');

      expect(playhead).toBeInTheDocument();
    });

    it("renders when currentTime equals viewport end", () => {
      useAudioStore.setState({ currentTime: 100 });
      useWaveformStore.setState({ viewportStart: 50, viewportEnd: 100 });

      const { container } = render(<Playhead />);
      const playhead = container.querySelector('[style*="position: absolute"]');

      expect(playhead).toBeInTheDocument();
    });

    it("hides when currentTime moves outside viewport", () => {
      useAudioStore.setState({ currentTime: 90 });
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 180 });

      const { container, rerender } = render(<Playhead />);
      let playhead = container.querySelector('[style*="position: absolute"]');
      expect(playhead).toBeInTheDocument();

      // Move currentTime outside viewport
      useAudioStore.setState({ currentTime: 200 });
      rerender(<Playhead />);

      playhead = container.querySelector('[style*="position: absolute"]');
      expect(playhead).not.toBeInTheDocument();
    });

    it("shows when currentTime moves into viewport", () => {
      useAudioStore.setState({ currentTime: 200 });
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 180 });

      const { container, rerender } = render(<Playhead />);
      let playhead = container.querySelector('[style*="position: absolute"]');
      expect(playhead).not.toBeInTheDocument();

      // Move currentTime into viewport
      useAudioStore.setState({ currentTime: 90 });
      rerender(<Playhead />);

      playhead = container.querySelector('[style*="position: absolute"]');
      expect(playhead).toBeInTheDocument();
    });
  });

  describe("Visual Styling", () => {
    it("renders with cyan gradient background", () => {
      const { container } = render(<Playhead />);
      const playhead = container.querySelector('[style*="position: absolute"]');

      expect(playhead?.getAttribute("style")).toContain("linear-gradient");
      expect(playhead?.getAttribute("style")).toContain("#1affef");
      expect(playhead?.getAttribute("style")).toContain("#00e5cc");
    });

    it("renders with glow shadow", () => {
      const { container } = render(<Playhead />);
      const playhead = container.querySelector('[style*="position: absolute"]');

      expect(playhead?.getAttribute("style")).toContain("box-shadow");
      expect(playhead?.getAttribute("style")).toContain("rgba(26, 255, 239");
    });

    it("renders with 2px width", () => {
      const { container } = render(<Playhead />);
      const playhead = container.querySelector('[style*="position: absolute"]');

      expect(playhead?.getAttribute("style")).toContain("width: 2px");
    });

    it("renders at full height", () => {
      const { container } = render(<Playhead />);
      const playhead = container.querySelector('[style*="position: absolute"]');

      expect(playhead?.getAttribute("style")).toContain("height: 100%");
      expect(playhead?.getAttribute("style")).toContain("top: 0");
    });

    it("uses absolute positioning", () => {
      const { container } = render(<Playhead />);
      const playhead = container.querySelector('[style*="position: absolute"]');

      expect(playhead?.getAttribute("style")).toContain("position: absolute");
    });

    it("disables pointer events", () => {
      const { container } = render(<Playhead />);
      const playhead = container.querySelector('[style*="position: absolute"]');

      expect(playhead?.getAttribute("style")).toContain("pointer-events: none");
    });

    it("renders with z-index 10", () => {
      const { container } = render(<Playhead />);
      const playhead = container.querySelector('[style*="position: absolute"]');

      expect(playhead?.getAttribute("style")).toContain("z-index: 10");
    });

    it("applies double shadow for enhanced glow", () => {
      const { container } = render(<Playhead />);
      const playhead = container.querySelector('[style*="position: absolute"]');
      const style = playhead?.getAttribute("style") || "";

      // Should have two shadows (0 0 15px and 0 0 30px)
      expect(style).toContain("0 0 15px");
      expect(style).toContain("0 0 30px");
    });
  });

  describe("Edge Cases", () => {
    it("handles zero viewport duration", () => {
      useAudioStore.setState({ currentTime: 90 });
      useWaveformStore.setState({ viewportStart: 90, viewportEnd: 90 });

      const { container } = render(<Playhead />);
      // Should render (currentTime equals both boundaries)
      const playhead = container.querySelector('[style*="position: absolute"]');
      expect(playhead).toBeInTheDocument();
    });

    it("handles negative viewport start", () => {
      useAudioStore.setState({ currentTime: 0 });
      useWaveformStore.setState({ viewportStart: -10, viewportEnd: 10 });

      const { container } = render(<Playhead />);
      const playhead = container.querySelector('[style*="position: absolute"]');

      expect(playhead).toBeInTheDocument();
    });

    it("handles very large viewport", () => {
      useAudioStore.setState({ currentTime: 5000 });
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 10000 });

      const { container } = render(<Playhead />);
      const playhead = container.querySelector('[style*="position: absolute"]');

      expect(playhead).toBeInTheDocument();
      expect(playhead?.getAttribute("style")).toContain("left: 50%");
    });

    it("handles currentTime at 0", () => {
      useAudioStore.setState({ currentTime: 0 });
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 180 });

      const { container } = render(<Playhead />);
      const playhead = container.querySelector('[style*="position: absolute"]');

      expect(playhead).toBeInTheDocument();
      expect(playhead?.getAttribute("style")).toContain("left: 0%");
    });

    it("handles very small currentTime increment", () => {
      useAudioStore.setState({ currentTime: 90.001 });
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 180 });

      const { container } = render(<Playhead />);
      const playhead = container.querySelector('[style*="position: absolute"]');

      expect(playhead).toBeInTheDocument();
    });

    it("handles viewport with fractional duration", () => {
      useAudioStore.setState({ currentTime: 90.5 });
      useWaveformStore.setState({ viewportStart: 89.5, viewportEnd: 91.5 });

      const { container } = render(<Playhead />);
      const playhead = container.querySelector('[style*="position: absolute"]');

      expect(playhead).toBeInTheDocument();
      expect(playhead?.getAttribute("style")).toContain("left: 50%");
    });
  });

  describe("Rapid Updates", () => {
    it("handles rapid currentTime changes", () => {
      useWaveformStore.setState({ viewportStart: 0, viewportEnd: 180 });

      const { container, rerender } = render(<Playhead />);

      // Simulate rapid playback updates
      for (let t = 0; t <= 180; t += 10) {
        useAudioStore.setState({ currentTime: t });
        rerender(<Playhead />);

        const playhead = container.querySelector('[style*="position: absolute"]');
        expect(playhead).toBeInTheDocument();
      }
    });

    it("handles rapid viewport changes", () => {
      useAudioStore.setState({ currentTime: 90 });

      const { container, rerender } = render(<Playhead />);

      // Simulate rapid viewport scrolling
      for (let start = 0; start <= 100; start += 10) {
        useWaveformStore.setState({ viewportStart: start, viewportEnd: start + 20 });
        rerender(<Playhead />);

        const playhead = container.querySelector('[style*="position: absolute"]');
        // May or may not be visible depending on viewport position
        if (90 >= start && 90 <= start + 20) {
          expect(playhead).toBeInTheDocument();
        } else {
          expect(playhead).not.toBeInTheDocument();
        }
      }
    });

    it("maintains correct position during simultaneous time and viewport changes", () => {
      const { container, rerender } = render(<Playhead />);

      // Change both at once
      useAudioStore.setState({ currentTime: 100 });
      useWaveformStore.setState({ viewportStart: 50, viewportEnd: 150 });
      rerender(<Playhead />);

      const playhead = container.querySelector('[style*="position: absolute"]');
      expect(playhead).toBeInTheDocument();
      // (100 - 50) / (150 - 50) * 100 = 50%
      expect(playhead?.getAttribute("style")).toContain("left: 50%");
    });
  });
});
