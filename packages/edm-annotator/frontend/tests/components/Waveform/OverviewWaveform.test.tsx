import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { userEvent } from "@testing-library/user-event";
import { OverviewWaveform } from "@/components/Waveform/OverviewWaveform";
import {
  useWaveformStore,
  useAudioStore,
  useTempoStore,
  useUIStore,
  useStructureStore,
} from "@/stores";

describe("OverviewWaveform", () => {
  const user = userEvent.setup();

  beforeEach(() => {
    useWaveformStore.getState().reset();
    useAudioStore.getState().reset();
    useTempoStore.getState().reset();
    useUIStore.getState().reset();
    useStructureStore.getState().reset();

    // Set up basic waveform data
    const sampleCount = 100;
    const duration = 180;
    const waveformBass = Array.from({ length: sampleCount }, (_, i) => Math.sin(i * 0.1) * 0.5);
    const waveformMids = Array.from({ length: sampleCount }, (_, i) => Math.sin(i * 0.15) * 0.3);
    const waveformHighs = Array.from({ length: sampleCount }, (_, i) => Math.sin(i * 0.2) * 0.2);
    const waveformTimes = Array.from(
      { length: sampleCount },
      (_, i) => (i / sampleCount) * duration
    );

    useWaveformStore.setState({
      waveformBass,
      waveformMids,
      waveformHighs,
      waveformTimes,
      duration,
    });

    useAudioStore.setState({ currentTime: 90 });
    useTempoStore.setState({ trackBPM: 128, trackDownbeat: 0 });
  });

  describe("Full Track Rendering", () => {
    it("renders SVG waveform", () => {
      const { container } = render(<OverviewWaveform />);
      const svg = container.querySelector("svg");

      expect(svg).toBeInTheDocument();
      expect(svg?.getAttribute("viewBox")).toBe("0 0 100 100");
    });

    it("renders single combined waveform path", () => {
      const { container } = render(<OverviewWaveform />);
      const paths = container.querySelectorAll("path");

      expect(paths).toHaveLength(1);
    });

    it("applies blue color to waveform", () => {
      const { container } = render(<OverviewWaveform />);
      const path = container.querySelector("path");

      expect(path?.getAttribute("fill")).toContain("rgba(100, 140, 180");
    });

    it("generates valid SVG path data", () => {
      const { container } = render(<OverviewWaveform />);
      const path = container.querySelector("path");
      const d = path?.getAttribute("d");

      expect(d).toBeTruthy();
      expect(d).toMatch(/^M/); // Starts with M (moveTo)
      expect(d).toMatch(/Z$/); // Ends with Z (closePath)
    });

    it("extends waveform upward from baseline", () => {
      const { container } = render(<OverviewWaveform />);
      const path = container.querySelector("path");
      const d = path?.getAttribute("d");

      // Should start at baseline (y=100) and have points above
      expect(d).toContain("M0,100");
    });

    it("renders empty path when no waveform data", () => {
      useWaveformStore.setState({
        waveformBass: [],
        waveformMids: [],
        waveformHighs: [],
        waveformTimes: [],
        duration: 0,
      });

      const { container } = render(<OverviewWaveform />);
      const path = container.querySelector("path");

      expect(path?.getAttribute("d")).toBe("");
    });
  });

  describe("Playhead Movement", () => {
    it("renders playhead at current time position", () => {
      useAudioStore.setState({ currentTime: 90 });
      useWaveformStore.setState({ duration: 180 });

      const { container } = render(<OverviewWaveform />);
      const playhead = container.querySelector('[style*="background: rgb(26, 255, 239)"]');

      expect(playhead).toBeInTheDocument();
      // 90 / 180 * 100 = 50%
      expect(playhead?.getAttribute("style")).toContain("left: 50%");
    });

    it("updates playhead position when currentTime changes", () => {
      useAudioStore.setState({ currentTime: 45 });
      useWaveformStore.setState({ duration: 180 });

      const { container, rerender } = render(<OverviewWaveform />);
      let playhead = container.querySelector('[style*="background: rgb(26, 255, 239)"]');

      // 45 / 180 * 100 = 25%
      expect(playhead?.getAttribute("style")).toContain("left: 25%");

      useAudioStore.setState({ currentTime: 135 });
      rerender(<OverviewWaveform />);

      playhead = container.querySelector('[style*="background: rgb(26, 255, 239)"]');
      // 135 / 180 * 100 = 75%
      expect(playhead?.getAttribute("style")).toContain("left: 75%");
    });

    it("positions playhead at 0% when at track start", () => {
      useAudioStore.setState({ currentTime: 0 });
      useWaveformStore.setState({ duration: 180 });

      const { container } = render(<OverviewWaveform />);
      const playhead = container.querySelector('[style*="background: rgb(26, 255, 239)"]');

      expect(playhead?.getAttribute("style")).toContain("left: 0%");
    });

    it("positions playhead at 100% when at track end", () => {
      useAudioStore.setState({ currentTime: 180 });
      useWaveformStore.setState({ duration: 180 });

      const { container } = render(<OverviewWaveform />);
      const playhead = container.querySelector('[style*="background: rgb(26, 255, 239)"]');

      expect(playhead?.getAttribute("style")).toContain("left: 100%");
    });

    it("applies cyan color to playhead", () => {
      const { container } = render(<OverviewWaveform />);
      const playhead = container.querySelector('[style*="background: rgb(26, 255, 239)"]');

      expect(playhead).toBeInTheDocument();
      expect(playhead?.getAttribute("style")).toContain("rgb(26, 255, 239)");
    });

    it("applies glow shadow to playhead", () => {
      const { container } = render(<OverviewWaveform />);
      const playhead = container.querySelector('[style*="background: rgb(26, 255, 239)"]');

      expect(playhead?.getAttribute("style")).toContain("box-shadow");
      expect(playhead?.getAttribute("style")).toContain("rgba(26, 255, 239");
    });

    it("renders playhead with 2px width", () => {
      const { container } = render(<OverviewWaveform />);
      const playhead = container.querySelector('[style*="background: rgb(26, 255, 239)"]');

      expect(playhead?.getAttribute("style")).toContain("width: 2px");
    });

    it("renders playhead at full height", () => {
      const { container } = render(<OverviewWaveform />);
      const playhead = container.querySelector('[style*="background: rgb(26, 255, 239)"]');

      expect(playhead?.getAttribute("style")).toContain("height: 100%");
    });
  });

  describe("Click-to-Seek", () => {
    it("seeks to clicked position", () => {
      const seekSpy = vi.spyOn(useAudioStore.getState(), "seek");
      useWaveformStore.setState({ duration: 180 });

      const { container } = render(<OverviewWaveform />);
      const waveform = container.querySelector('[style*="position: relative"]') as HTMLElement;

      vi.spyOn(waveform, "getBoundingClientRect").mockReturnValue({
        left: 0,
        width: 800,
        top: 0,
        height: 60,
        right: 800,
        bottom: 60,
        x: 0,
        y: 0,
        toJSON: () => {},
      });

      // Click at 50% = 90 seconds
      fireEvent.click(waveform, { clientX: 400 });

      expect(seekSpy).toHaveBeenCalledWith(90);
    });

    it("seeks to beginning when clicking at start", () => {
      const seekSpy = vi.spyOn(useAudioStore.getState(), "seek");
      useWaveformStore.setState({ duration: 180 });

      const { container } = render(<OverviewWaveform />);
      const waveform = container.querySelector('[style*="position: relative"]') as HTMLElement;

      vi.spyOn(waveform, "getBoundingClientRect").mockReturnValue({
        left: 0,
        width: 800,
        top: 0,
        height: 60,
        right: 800,
        bottom: 60,
        x: 0,
        y: 0,
        toJSON: () => {},
      });

      fireEvent.click(waveform, { clientX: 0 });

      expect(seekSpy).toHaveBeenCalledWith(0);
    });

    it("seeks to end when clicking at end", () => {
      const seekSpy = vi.spyOn(useAudioStore.getState(), "seek");
      useWaveformStore.setState({ duration: 180 });

      const { container } = render(<OverviewWaveform />);
      const waveform = container.querySelector('[style*="position: relative"]') as HTMLElement;

      vi.spyOn(waveform, "getBoundingClientRect").mockReturnValue({
        left: 0,
        width: 800,
        top: 0,
        height: 60,
        right: 800,
        bottom: 60,
        x: 0,
        y: 0,
        toJSON: () => {},
      });

      fireEvent.click(waveform, { clientX: 800 });

      expect(seekSpy).toHaveBeenCalledWith(180);
    });

    it("quantizes seek position to nearest bar when enabled", () => {
      const seekSpy = vi.spyOn(useAudioStore.getState(), "seek");
      useUIStore.setState({ quantizeEnabled: true });
      useTempoStore.setState({ trackBPM: 128, trackDownbeat: 0 });
      useWaveformStore.setState({ duration: 180 });

      const { container } = render(<OverviewWaveform />);
      const waveform = container.querySelector('[style*="position: relative"]') as HTMLElement;

      vi.spyOn(waveform, "getBoundingClientRect").mockReturnValue({
        left: 0,
        width: 800,
        top: 0,
        height: 60,
        right: 800,
        bottom: 60,
        x: 0,
        y: 0,
        toJSON: () => {},
      });

      // Click at position that would be 91s, should snap to nearest bar
      // Bar duration at 128 BPM = 1.875s
      fireEvent.click(waveform, { clientX: 405 });

      // Should be called with quantized time
      expect(seekSpy).toHaveBeenCalled();
      const calledTime = seekSpy.mock.calls[0][0];
      // Should be close to a bar boundary (multiple of 1.875)
      const barDuration = (60 / 128) * 4;
      const remainder = calledTime % barDuration;
      expect(remainder).toBeCloseTo(0, 1);
    });

    it("does not quantize when quantize disabled", () => {
      const seekSpy = vi.spyOn(useAudioStore.getState(), "seek");
      useUIStore.setState({ quantizeEnabled: false });
      useWaveformStore.setState({ duration: 180 });

      const { container } = render(<OverviewWaveform />);
      const waveform = container.querySelector('[style*="position: relative"]') as HTMLElement;

      vi.spyOn(waveform, "getBoundingClientRect").mockReturnValue({
        left: 0,
        width: 800,
        top: 0,
        height: 60,
        right: 800,
        bottom: 60,
        x: 0,
        y: 0,
        toJSON: () => {},
      });

      fireEvent.click(waveform, { clientX: 405 });

      // Should seek to exact position (405 / 800 * 180 = 91.125)
      expect(seekSpy).toHaveBeenCalledWith(91.125);
    });

    it("clamps seek position to duration", () => {
      const seekSpy = vi.spyOn(useAudioStore.getState(), "seek");
      useWaveformStore.setState({ duration: 180 });

      const { container } = render(<OverviewWaveform />);
      const waveform = container.querySelector('[style*="position: relative"]') as HTMLElement;

      vi.spyOn(waveform, "getBoundingClientRect").mockReturnValue({
        left: 0,
        width: 800,
        top: 0,
        height: 60,
        right: 800,
        bottom: 60,
        x: 0,
        y: 0,
        toJSON: () => {},
      });

      // Click beyond end
      fireEvent.click(waveform, { clientX: 1000 });

      // Should be clamped to duration
      const calledTime = seekSpy.mock.calls[0][0];
      expect(calledTime).toBeLessThanOrEqual(180);
    });
  });

  describe("Region Overlays", () => {
    it("renders region overlays for labeled regions", () => {
      useStructureStore.setState({
        regions: [
          { start: 0, end: 60, label: "intro" },
          { start: 60, end: 120, label: "verse" },
          { start: 120, end: 180, label: "chorus" },
        ],
      });

      const { container } = render(<OverviewWaveform />);
      const overlays = container.querySelectorAll('[style*="position: absolute"]');

      // Should have 3 region overlays + 1 playhead + 0 boundaries = 4
      expect(overlays.length).toBeGreaterThanOrEqual(3);
    });

    it("does not render default regions", () => {
      useStructureStore.setState({
        regions: [
          { start: 0, end: 60, label: "default" },
          { start: 60, end: 120, label: "verse" },
        ],
      });

      const { container } = render(<OverviewWaveform />);
      // Only verse region should be visible
      const overlays = container.querySelectorAll('[style*="opacity: 0.35"]');

      expect(overlays).toHaveLength(1);
    });

    it("applies correct colors to region overlays", () => {
      useStructureStore.setState({
        regions: [{ start: 0, end: 60, label: "intro" }],
      });

      const { container } = render(<OverviewWaveform />);
      const overlay = container.querySelector('[style*="opacity: 0.35"]');

      // Should have a background color from labelColors
      expect(overlay?.getAttribute("style")).toContain("background");
    });

    it("positions region overlays correctly", () => {
      useWaveformStore.setState({ duration: 180 });
      useStructureStore.setState({
        regions: [{ start: 60, end: 120, label: "verse" }],
      });

      const { container } = render(<OverviewWaveform />);
      const overlay = container.querySelector('[style*="opacity: 0.35"]');

      // 60 / 180 * 100 = 33.333...%
      expect(overlay?.getAttribute("style")).toContain("left: 33.33");
      // (120 - 60) / 180 * 100 = 33.333...%
      expect(overlay?.getAttribute("style")).toContain("width: 33.33");
    });

    it("applies low opacity to region overlays", () => {
      useStructureStore.setState({
        regions: [{ start: 0, end: 60, label: "intro" }],
      });

      const { container } = render(<OverviewWaveform />);
      const overlay = container.querySelector('[style*="opacity: 0.35"]');

      expect(overlay).toBeInTheDocument();
    });

    it("renders region overlays behind waveform", () => {
      useStructureStore.setState({
        regions: [{ start: 0, end: 60, label: "intro" }],
      });

      const { container } = render(<OverviewWaveform />);
      const overlay = container.querySelector('[style*="opacity: 0.35"]');
      const svg = container.querySelector("svg");

      // Overlay should appear before SVG in DOM
      const parent = overlay?.parentElement;
      const children = Array.from(parent?.children || []);
      const overlayIndex = children.indexOf(overlay!);
      const svgIndex = children.indexOf(svg!);

      expect(overlayIndex).toBeLessThan(svgIndex);
    });
  });

  describe("Boundary Markers", () => {
    it("renders boundary markers", () => {
      useStructureStore.setState({
        boundaries: [30, 60, 90, 120, 150],
      });

      const { container } = render(<OverviewWaveform />);
      const boundaries = container.querySelectorAll('[style*="rgba(123, 106, 255, 0.7)"]');

      expect(boundaries).toHaveLength(5);
    });

    it("positions boundary markers correctly", () => {
      useWaveformStore.setState({ duration: 180 });
      useStructureStore.setState({
        boundaries: [90],
      });

      const { container } = render(<OverviewWaveform />);
      const boundary = container.querySelector('[style*="rgba(123, 106, 255, 0.7)"]');

      // 90 / 180 * 100 = 50%
      expect(boundary?.getAttribute("style")).toContain("left: 50%");
    });

    it("applies purple color to boundary markers", () => {
      useStructureStore.setState({
        boundaries: [90],
      });

      const { container } = render(<OverviewWaveform />);
      const boundary = container.querySelector('[style*="rgba(123, 106, 255, 0.7)"]');

      expect(boundary).toBeInTheDocument();
    });

    it("renders boundary markers with 2px width", () => {
      useStructureStore.setState({
        boundaries: [90],
      });

      const { container } = render(<OverviewWaveform />);
      const boundary = container.querySelector('[style*="rgba(123, 106, 255, 0.7)"]');

      expect(boundary?.getAttribute("style")).toContain("width: 2px");
    });

    it("renders boundary markers at full height", () => {
      useStructureStore.setState({
        boundaries: [90],
      });

      const { container } = render(<OverviewWaveform />);
      const boundary = container.querySelector('[style*="rgba(123, 106, 255, 0.7)"]');

      expect(boundary?.getAttribute("style")).toContain("height: 100%");
    });
  });

  describe("Edge Cases", () => {
    it("handles zero duration", () => {
      useWaveformStore.setState({ duration: 0 });
      const { container } = render(<OverviewWaveform />);

      const path = container.querySelector("path");
      expect(path?.getAttribute("d")).toBe("");
    });

    it("handles empty boundaries array", () => {
      useStructureStore.setState({ boundaries: [] });
      const { container } = render(<OverviewWaveform />);

      const boundaries = container.querySelectorAll('[style*="rgba(123, 106, 255"]');
      expect(boundaries).toHaveLength(0);
    });

    it("handles empty regions array", () => {
      useStructureStore.setState({ regions: [] });
      const { container } = render(<OverviewWaveform />);

      const overlays = container.querySelectorAll('[style*="opacity: 0.15"]');
      expect(overlays).toHaveLength(0);
    });

    it("handles very long duration", () => {
      useWaveformStore.setState({ duration: 10000 });
      useAudioStore.setState({ currentTime: 5000 });

      const { container } = render(<OverviewWaveform />);
      const playhead = container.querySelector('[style*="background: rgb(26, 255, 239)"]');

      // 5000 / 10000 * 100 = 50%
      expect(playhead?.getAttribute("style")).toContain("left: 50%");
    });

    it("handles fractional currentTime", () => {
      useWaveformStore.setState({ duration: 180 });
      useAudioStore.setState({ currentTime: 45.789 });

      const { container } = render(<OverviewWaveform />);
      const playhead = container.querySelector('[style*="background: rgb(26, 255, 239)"]');

      expect(playhead).toBeInTheDocument();
    });

    it("does not crash with zero BPM when quantize enabled", () => {
      useUIStore.setState({ quantizeEnabled: true });
      useTempoStore.setState({ trackBPM: 0 });
      useWaveformStore.setState({ duration: 180 });

      const { container } = render(<OverviewWaveform />);
      const waveform = container.querySelector('[style*="position: relative"]') as HTMLElement;

      vi.spyOn(waveform, "getBoundingClientRect").mockReturnValue({
        left: 0,
        width: 800,
        top: 0,
        height: 60,
        right: 800,
        bottom: 60,
        x: 0,
        y: 0,
        toJSON: () => {},
      });

      // Should not crash, should just not quantize
      fireEvent.click(waveform, { clientX: 400 });
    });
  });

  describe("Waveform Downsampling", () => {
    it("downsamples large waveform data for performance", () => {
      // Create large waveform (10000 samples)
      const sampleCount = 10000;
      const duration = 600; // 10 minutes
      const waveformBass = Array.from({ length: sampleCount }, () => Math.random() * 0.5);
      const waveformMids = Array.from({ length: sampleCount }, () => Math.random() * 0.3);
      const waveformHighs = Array.from({ length: sampleCount }, () => Math.random() * 0.2);
      const waveformTimes = Array.from(
        { length: sampleCount },
        (_, i) => (i / sampleCount) * duration
      );

      useWaveformStore.setState({
        waveformBass,
        waveformMids,
        waveformHighs,
        waveformTimes,
        duration,
      });

      const { container } = render(<OverviewWaveform />);
      const path = container.querySelector("path");
      const d = path?.getAttribute("d") || "";

      // Should have significantly fewer points than original 10000
      // Target is ~500 samples, so path should have ~500 L commands
      const pointCount = (d.match(/L/g) || []).length;
      expect(pointCount).toBeLessThan(1000);
    });
  });
});
