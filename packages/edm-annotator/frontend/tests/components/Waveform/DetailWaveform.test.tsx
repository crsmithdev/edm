import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { userEvent } from "@testing-library/user-event";
import { DetailWaveform } from "@/components/Waveform/DetailWaveform";
import {
  useWaveformStore,
  useAudioStore,
  useStructureStore,
  useUIStore,
  useTempoStore,
} from "@/stores";

// Mock child components
vi.mock("@/components/Waveform/BeatGrid", () => ({
  BeatGrid: ({ viewportStart, viewportEnd }: any) => (
    <div data-testid="beat-grid" data-viewport-start={viewportStart} data-viewport-end={viewportEnd}>
      Beat Grid
    </div>
  ),
}));

vi.mock("@/components/Waveform/BoundaryMarkers", () => ({
  BoundaryMarkers: ({ viewportStart, viewportEnd }: any) => (
    <div
      data-testid="boundary-markers"
      data-viewport-start={viewportStart}
      data-viewport-end={viewportEnd}
    >
      Boundary Markers
    </div>
  ),
}));

vi.mock("@/components/Waveform/RegionOverlays", () => ({
  RegionOverlays: ({ viewportStart, viewportEnd }: any) => (
    <div
      data-testid="region-overlays"
      data-viewport-start={viewportStart}
      data-viewport-end={viewportEnd}
    >
      Region Overlays
    </div>
  ),
}));

describe("DetailWaveform", () => {
  const user = userEvent.setup();

  beforeEach(() => {
    useWaveformStore.getState().reset();
    useAudioStore.getState().reset();
    useStructureStore.getState().reset();
    useUIStore.getState().reset();
    useTempoStore.getState().reset();

    // Set up basic waveform data
    useWaveformStore.setState({
      waveformBass: [0.5, 0.7, 0.6, 0.8],
      waveformMids: [0.3, 0.4, 0.5, 0.3],
      waveformHighs: [0.2, 0.3, 0.2, 0.4],
      waveformTimes: [0, 0.5, 1.0, 1.5],
      duration: 180,
    });

    useAudioStore.setState({ currentTime: 90 });
    useTempoStore.setState({ trackBPM: 128, trackDownbeat: 0 });
  });

  describe("Centered Playhead Rendering", () => {
    it("renders centered playhead at 50% position", () => {
      const { container } = render(<DetailWaveform span={16} />);
      const playhead = container.querySelector('[style*="left: 50%"]');

      expect(playhead).toBeInTheDocument();
      expect(playhead?.getAttribute("style")).toContain("width: 2px");
      expect(playhead?.getAttribute("style")).toContain("height: 100%");
    });

    it("applies gradient background to playhead", () => {
      const { container } = render(<DetailWaveform span={16} />);
      const playhead = container.querySelector('[style*="left: 50%"]');

      expect(playhead?.getAttribute("style")).toContain("linear-gradient");
      expect(playhead?.getAttribute("style")).toContain("#1affef");
      expect(playhead?.getAttribute("style")).toContain("#00e5cc");
    });

    it("applies glow shadow to playhead", () => {
      const { container } = render(<DetailWaveform span={16} />);
      const playhead = container.querySelector('[style*="left: 50%"]');

      expect(playhead?.getAttribute("style")).toContain("box-shadow");
      expect(playhead?.getAttribute("style")).toContain("rgba(26, 255, 239");
    });

    it("renders playhead with z-index 10", () => {
      const { container } = render(<DetailWaveform span={16} />);
      const playhead = container.querySelector('[style*="left: 50%"]');

      expect(playhead?.getAttribute("style")).toContain("z-index: 10");
    });

    it("renders playhead with pointer-events: none", () => {
      const { container } = render(<DetailWaveform span={16} />);
      const playhead = container.querySelector('[style*="left: 50%"]');

      expect(playhead?.getAttribute("style")).toContain("pointer-events: none");
    });
  });

  describe("Waveform Scrolling", () => {
    it("calculates viewport centered on currentTime", () => {
      useAudioStore.setState({ currentTime: 90 });
      render(<DetailWaveform span={16} />);

      const beatGrid = screen.getByTestId("beat-grid");
      // With span=16 and currentTime=90, viewport should be [82, 98]
      expect(beatGrid).toHaveAttribute("data-viewport-start", "82");
      expect(beatGrid).toHaveAttribute("data-viewport-end", "98");
    });

    it("updates viewport when currentTime changes", () => {
      useAudioStore.setState({ currentTime: 50 });
      const { rerender } = render(<DetailWaveform span={16} />);

      let beatGrid = screen.getByTestId("beat-grid");
      // [42, 58]
      expect(beatGrid).toHaveAttribute("data-viewport-start", "42");
      expect(beatGrid).toHaveAttribute("data-viewport-end", "58");

      useAudioStore.setState({ currentTime: 100 });
      rerender(<DetailWaveform span={16} />);

      beatGrid = screen.getByTestId("beat-grid");
      // [92, 108]
      expect(beatGrid).toHaveAttribute("data-viewport-start", "92");
      expect(beatGrid).toHaveAttribute("data-viewport-end", "108");
    });

    it("updates viewport when span changes", () => {
      useAudioStore.setState({ currentTime: 90 });
      const { rerender } = render(<DetailWaveform span={16} />);

      let beatGrid = screen.getByTestId("beat-grid");
      expect(beatGrid).toHaveAttribute("data-viewport-start", "82");
      expect(beatGrid).toHaveAttribute("data-viewport-end", "98");

      // Zoom in (smaller span)
      rerender(<DetailWaveform span={8} />);

      beatGrid = screen.getByTestId("beat-grid");
      // [86, 94]
      expect(beatGrid).toHaveAttribute("data-viewport-start", "86");
      expect(beatGrid).toHaveAttribute("data-viewport-end", "94");
    });

    it("allows viewport to extend beyond track start", () => {
      useAudioStore.setState({ currentTime: 5 });
      render(<DetailWaveform span={16} />);

      const beatGrid = screen.getByTestId("beat-grid");
      // Viewport would be [-3, 13], passed unclamped to child components
      const start = parseFloat(beatGrid.getAttribute("data-viewport-start") || "0");
      expect(start).toBe(-3); // Passed unclamped to children
    });

    it("allows viewport to extend beyond track end", () => {
      useWaveformStore.setState({ duration: 180 });
      useAudioStore.setState({ currentTime: 175 });
      render(<DetailWaveform span={16} />);

      const beatGrid = screen.getByTestId("beat-grid");
      // Viewport would be [167, 183], passed unclamped to child components
      const end = parseFloat(beatGrid.getAttribute("data-viewport-end") || "0");
      expect(end).toBe(183); // Passed unclamped to children
    });
  });

  describe("Waveform SVG Rendering", () => {
    it("renders SVG with three frequency band paths", () => {
      const { container } = render(<DetailWaveform span={16} />);
      const svg = container.querySelector("svg");
      const paths = svg?.querySelectorAll("path");

      expect(paths).toHaveLength(3); // bass, mids, highs
    });

    it("renders bass path with cyan color", () => {
      const { container } = render(<DetailWaveform span={16} />);
      const paths = container.querySelectorAll("path");

      expect(paths[0].getAttribute("fill")).toContain("rgba(0, 229, 204");
    });

    it("renders mids path with purple color", () => {
      const { container } = render(<DetailWaveform span={16} />);
      const paths = container.querySelectorAll("path");

      expect(paths[1].getAttribute("fill")).toContain("rgba(123, 106, 255");
    });

    it("renders highs path with pink color", () => {
      const { container } = render(<DetailWaveform span={16} />);
      const paths = container.querySelectorAll("path");

      expect(paths[2].getAttribute("fill")).toContain("rgba(255, 107, 181");
    });

    it("renders center baseline", () => {
      const { container } = render(<DetailWaveform span={16} />);
      const line = container.querySelector("line");

      expect(line).toBeInTheDocument();
      expect(line?.getAttribute("y1")).toBe("50");
      expect(line?.getAttribute("y2")).toBe("50");
    });

    it("renders empty paths when no waveform data", () => {
      useWaveformStore.setState({
        waveformBass: [],
        waveformMids: [],
        waveformHighs: [],
        waveformTimes: [],
      });

      const { container } = render(<DetailWaveform span={16} />);
      const paths = container.querySelectorAll("path");

      // Paths exist but have empty d attribute
      expect(paths[0].getAttribute("d")).toBe("");
      expect(paths[1].getAttribute("d")).toBe("");
      expect(paths[2].getAttribute("d")).toBe("");
    });
  });

  describe("Cue Point Rendering", () => {
    it("renders cue point when within viewport", () => {
      useAudioStore.setState({ currentTime: 90, cuePoint: 92 });
      const { container } = render(<DetailWaveform span={16} />);

      // Viewport is [82, 98], cuePoint at 92 should be visible
      const allDivs = container.querySelectorAll('div');
      const cuePoint = Array.from(allDivs).find(div =>
        div.getAttribute('style')?.includes('linear-gradient') &&
        div.getAttribute('style')?.includes('#ff9500')
      );
      expect(cuePoint).toBeTruthy();
    });

    it("does not render cue point when outside viewport", () => {
      useAudioStore.setState({ currentTime: 90, cuePoint: 50 });
      const { container } = render(<DetailWaveform span={16} />);

      // Viewport is [82, 98], cuePoint at 50 is outside
      const allDivs = container.querySelectorAll('div');
      const cuePoint = Array.from(allDivs).find(div =>
        div.getAttribute('style')?.includes('linear-gradient') &&
        div.getAttribute('style')?.includes('#ff9500')
      );
      expect(cuePoint).toBeFalsy();
    });

    it("positions cue point relative to viewport", () => {
      useAudioStore.setState({ currentTime: 90, cuePoint: 86 });
      const { container } = render(<DetailWaveform span={16} />);

      // Viewport is [82, 98], cuePoint at 86
      // Position: (86 - 82) / (98 - 82) * 100 = 4 / 16 * 100 = 25%
      const allDivs = container.querySelectorAll('div');
      const cuePoint = Array.from(allDivs).find(div =>
        div.getAttribute('style')?.includes('linear-gradient') &&
        div.getAttribute('style')?.includes('#ff9500')
      );
      expect(cuePoint?.getAttribute("style")).toContain("left: 25%");
    });

    it("applies orange gradient to cue point", () => {
      useAudioStore.setState({ currentTime: 90, cuePoint: 92 });
      const { container } = render(<DetailWaveform span={16} />);

      const allDivs = container.querySelectorAll('div');
      const cuePoint = Array.from(allDivs).find(div =>
        div.getAttribute('style')?.includes('linear-gradient') &&
        div.getAttribute('style')?.includes('#ff9500')
      );
      expect(cuePoint?.getAttribute("style")).toContain("#ff6b00");
    });
  });

  describe("Boundary Interaction", () => {
    it("renders BoundaryMarkers component with correct viewport", () => {
      render(<DetailWaveform span={16} />);
      const markers = screen.getByTestId("boundary-markers");

      expect(markers).toBeInTheDocument();
      expect(markers).toHaveAttribute("data-viewport-start", "82");
      expect(markers).toHaveAttribute("data-viewport-end", "98");
    });

    it("adds boundary at click position with Ctrl key", async () => {
      const addBoundarySpy = vi.spyOn(useStructureStore.getState(), "addBoundary");
      const { container } = render(<DetailWaveform span={16} />);

      const waveform = container.querySelector('[style*="position: relative"]');
      // Click at 50% = middle of viewport [82, 98] = 90
      fireEvent.click(waveform!, { clientX: 400, ctrlKey: true });

      expect(addBoundarySpy).toHaveBeenCalled();
    });

    it("quantizes boundary to nearest beat when quantize enabled", async () => {
      const addBoundarySpy = vi.spyOn(useStructureStore.getState(), "addBoundary");
      useUIStore.setState({ quantizeEnabled: true });
      useTempoStore.setState({ trackBPM: 128, trackDownbeat: 0 });

      const { container } = render(<DetailWaveform span={16} />);
      const waveform = container.querySelector('[style*="position: relative"]');

      // Mock getBoundingClientRect
      vi.spyOn(waveform!, "getBoundingClientRect").mockReturnValue({
        left: 0,
        width: 800,
        top: 0,
        height: 200,
        right: 800,
        bottom: 200,
        x: 0,
        y: 0,
        toJSON: () => {},
      });

      fireEvent.click(waveform!, { clientX: 400, ctrlKey: true });

      // Should snap to nearest beat
      expect(addBoundarySpy).toHaveBeenCalled();
    });

    it("does not add boundary on click without Ctrl key", () => {
      const addBoundarySpy = vi.spyOn(useStructureStore.getState(), "addBoundary");
      const { container } = render(<DetailWaveform span={16} />);

      const waveform = container.querySelector('[style*="position: relative"]');
      fireEvent.click(waveform!, { clientX: 400 });

      expect(addBoundarySpy).not.toHaveBeenCalled();
    });
  });

  describe("Scrubbing Interaction", () => {
    it("enters dragging state on mousedown", () => {
      const { container } = render(<DetailWaveform span={16} />);
      const waveform = container.querySelector('[style*="position: relative"]') as HTMLElement;

      fireEvent.mouseDown(waveform, { clientX: 400 });

      // Cursor should change to grabbing
      expect(waveform.style.cursor).toContain("grabbing");
    });

    it("does not start drag on Ctrl+click", () => {
      const { container } = render(<DetailWaveform span={16} />);
      const waveform = container.querySelector('[style*="position: relative"]') as HTMLElement;

      fireEvent.mouseDown(waveform, { clientX: 400, ctrlKey: true });

      // Cursor should not be grabbing
      expect(waveform.style.cursor).not.toContain("grabbing");
    });

    it("seeks to new position when dragging left", () => {
      const seekSpy = vi.spyOn(useAudioStore.getState(), "seek");
      const { container } = render(<DetailWaveform span={16} />);
      const waveform = container.querySelector('[style*="position: relative"]') as HTMLElement;

      vi.spyOn(waveform, "getBoundingClientRect").mockReturnValue({
        left: 0,
        width: 800,
        top: 0,
        height: 200,
        right: 800,
        bottom: 200,
        x: 0,
        y: 0,
        toJSON: () => {},
      });

      fireEvent.mouseDown(waveform, { clientX: 400 });
      fireEvent.mouseMove(window, { clientX: 300 });
      fireEvent.mouseUp(window);

      // Dragging left should move forward in time
      expect(seekSpy).toHaveBeenCalled();
    });

    it("exits dragging state on mouseup", () => {
      const { container } = render(<DetailWaveform span={16} />);
      const waveform = container.querySelector('[style*="position: relative"]') as HTMLElement;

      fireEvent.mouseDown(waveform, { clientX: 400 });
      expect(waveform.style.cursor).toContain("grabbing");

      fireEvent.mouseUp(window);
      expect(waveform.style.cursor).not.toContain("grabbing");
    });
  });

  describe("Region Overlay Rendering", () => {
    it("renders RegionOverlays component with correct viewport", () => {
      render(<DetailWaveform span={16} />);
      const overlays = screen.getByTestId("region-overlays");

      expect(overlays).toBeInTheDocument();
      expect(overlays).toHaveAttribute("data-viewport-start", "82");
      expect(overlays).toHaveAttribute("data-viewport-end", "98");
    });

    it("renders region overlays behind waveform", () => {
      const { container } = render(<DetailWaveform span={16} />);

      const overlays = screen.getByTestId("region-overlays");
      const svg = container.querySelector("svg");

      // RegionOverlays should appear before SVG in DOM
      const parent = overlays.parentElement;
      const children = Array.from(parent?.children || []);
      const overlaysIndex = children.indexOf(overlays);
      const svgIndex = children.indexOf(svg!);

      expect(overlaysIndex).toBeLessThan(svgIndex);
    });
  });

  describe("Beat Grid Rendering", () => {
    it("renders BeatGrid component with correct viewport", () => {
      render(<DetailWaveform span={16} />);
      const beatGrid = screen.getByTestId("beat-grid");

      expect(beatGrid).toBeInTheDocument();
      expect(beatGrid).toHaveAttribute("data-viewport-start", "82");
      expect(beatGrid).toHaveAttribute("data-viewport-end", "98");
    });

    it("updates BeatGrid viewport when currentTime changes", () => {
      useAudioStore.setState({ currentTime: 90 });
      const { rerender } = render(<DetailWaveform span={16} />);

      let beatGrid = screen.getByTestId("beat-grid");
      expect(beatGrid).toHaveAttribute("data-viewport-start", "82");

      useAudioStore.setState({ currentTime: 100 });
      rerender(<DetailWaveform span={16} />);

      beatGrid = screen.getByTestId("beat-grid");
      expect(beatGrid).toHaveAttribute("data-viewport-start", "92");
    });
  });

  describe("Zoom Interaction", () => {
    it("calls onZoomIn when scrolling up", () => {
      const onZoomIn = vi.fn();
      const onZoomOut = vi.fn();
      const { container } = render(
        <DetailWaveform span={16} onZoomIn={onZoomIn} onZoomOut={onZoomOut} />
      );

      const waveform = container.querySelector('[style*="position: relative"]');
      fireEvent.wheel(waveform!, { deltaY: -100 });

      expect(onZoomIn).toHaveBeenCalled();
      expect(onZoomOut).not.toHaveBeenCalled();
    });

    it("calls onZoomOut when scrolling down", () => {
      const onZoomIn = vi.fn();
      const onZoomOut = vi.fn();
      const { container } = render(
        <DetailWaveform span={16} onZoomIn={onZoomIn} onZoomOut={onZoomOut} />
      );

      const waveform = container.querySelector('[style*="position: relative"]');
      fireEvent.wheel(waveform!, { deltaY: 100 });

      expect(onZoomOut).toHaveBeenCalled();
      expect(onZoomIn).not.toHaveBeenCalled();
    });

    it("does not zoom when callbacks not provided", () => {
      const { container } = render(<DetailWaveform span={16} />);

      const waveform = container.querySelector('[style*="position: relative"]');
      // Should not throw
      fireEvent.wheel(waveform!, { deltaY: 100 });
      fireEvent.wheel(waveform!, { deltaY: -100 });
    });
  });

  describe("Cursor States", () => {
    it("shows grab cursor by default", () => {
      const { container } = render(<DetailWaveform span={16} />);
      const waveform = container.querySelector('[style*="position: relative"]') as HTMLElement;

      expect(waveform.style.cursor).toContain("grab");
    });

    it("shows grabbing cursor when dragging", () => {
      const { container } = render(<DetailWaveform span={16} />);
      const waveform = container.querySelector('[style*="position: relative"]') as HTMLElement;

      fireEvent.mouseDown(waveform, { clientX: 400 });
      expect(waveform.style.cursor).toContain("grabbing");
    });

    it("shows crosshair cursor when Ctrl key is down", () => {
      const { container } = render(<DetailWaveform span={16} />);
      const waveform = container.querySelector('[style*="position: relative"]') as HTMLElement;

      fireEvent.keyDown(window, { key: "Control" });
      expect(waveform.style.cursor).toContain("crosshair");

      fireEvent.keyUp(window, { key: "Control" });
      expect(waveform.style.cursor).not.toContain("crosshair");
    });
  });

  describe("Edge Cases", () => {
    it("handles empty waveform data gracefully", () => {
      useWaveformStore.setState({
        waveformBass: [],
        waveformMids: [],
        waveformHighs: [],
        waveformTimes: [],
        duration: 0,
      });

      const { container } = render(<DetailWaveform span={16} />);
      const paths = container.querySelectorAll("path");

      expect(paths).toHaveLength(3);
      expect(paths[0].getAttribute("d")).toBe("");
    });

    it("handles zero duration", () => {
      useWaveformStore.setState({ duration: 0 });
      const { container } = render(<DetailWaveform span={16} />);

      expect(container).toBeInTheDocument();
    });

    it("handles very small span", () => {
      useAudioStore.setState({ currentTime: 90 });
      render(<DetailWaveform span={0.5} />);

      const beatGrid = screen.getByTestId("beat-grid");
      expect(beatGrid).toBeInTheDocument();
      // Viewport should be [89.75, 90.25]
      expect(beatGrid).toHaveAttribute("data-viewport-start", "89.75");
      expect(beatGrid).toHaveAttribute("data-viewport-end", "90.25");
    });

    it("handles very large span", () => {
      useAudioStore.setState({ currentTime: 90 });
      render(<DetailWaveform span={200} />);

      const beatGrid = screen.getByTestId("beat-grid");
      expect(beatGrid).toBeInTheDocument();
      // Viewport should be [-10, 190], passed unclamped to children
      expect(beatGrid).toHaveAttribute("data-viewport-start", "-10");
      expect(beatGrid).toHaveAttribute("data-viewport-end", "190");
    });
  });
});
