import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { userEvent } from "@testing-library/user-event";
import { BoundaryMarkers } from "@/components/Waveform/BoundaryMarkers";
import { useStructureStore, useWaveformStore, useAudioStore } from "@/stores";

describe("BoundaryMarkers", () => {
  const user = userEvent.setup();

  beforeEach(() => {
    useStructureStore.getState().reset();
    useWaveformStore.getState().reset();
    useAudioStore.getState().reset();

    // Set up basic viewport
    useWaveformStore.setState({
      viewportStart: 0,
      viewportEnd: 180,
      duration: 180,
    });

    // Set up some boundaries
    useStructureStore.setState({
      boundaries: [0, 50, 100, 150, 180],
    });
  });

  describe("Rendering", () => {
    it("renders boundary markers for each boundary", () => {
      const { container } = render(<BoundaryMarkers />);
      const markers = container.querySelectorAll('[style*="position: absolute"]');
      // All 5 boundaries are within viewport [0, 180]
      expect(markers).toHaveLength(5);
    });

    it("only renders boundaries within viewport", () => {
      // Set viewport to show only middle section
      useWaveformStore.setState({
        viewportStart: 40,
        viewportEnd: 120,
      });

      const { container } = render(<BoundaryMarkers />);
      const markers = container.querySelectorAll('[style*="position: absolute"]');

      // Only boundaries at 50 and 100 are within [40, 120]
      expect(markers).toHaveLength(2);
    });

    it("calculates correct xPercent position for boundary at start", () => {
      useStructureStore.setState({ boundaries: [0, 180] });

      const { container } = render(<BoundaryMarkers />);
      const marker = container.querySelector('[style*="left: 0%"]');

      expect(marker).toBeInTheDocument();
    });

    it("calculates correct xPercent position for boundary at middle", () => {
      useStructureStore.setState({ boundaries: [90, 180] });
      useWaveformStore.setState({
        viewportStart: 0,
        viewportEnd: 180,
      });

      const { container } = render(<BoundaryMarkers />);
      // 90 is at 50% of [0, 180]
      const marker = container.querySelector('[style*="left: 50%"]');

      expect(marker).toBeInTheDocument();
    });

    it("calculates correct xPercent position for boundary at end", () => {
      useStructureStore.setState({ boundaries: [0, 180] });

      const { container } = render(<BoundaryMarkers />);
      const marker = container.querySelector('[style*="left: 100%"]');

      expect(marker).toBeInTheDocument();
    });

    it("does not render boundaries outside viewport", () => {
      useWaveformStore.setState({
        viewportStart: 100,
        viewportEnd: 150,
      });

      const { container } = render(<BoundaryMarkers />);
      const markers = container.querySelectorAll('[style*="position: absolute"]');

      // Only boundaries at 100 and 150 are within [100, 150]
      expect(markers).toHaveLength(2);
    });
  });

  describe("Interaction", () => {
    it("seeks to boundary time on click", async () => {
      const seekSpy = vi.spyOn(useAudioStore.getState(), "seek");

      const { container } = render(<BoundaryMarkers />);
      const markers = container.querySelectorAll('[style*="position: absolute"]');

      // Click the second marker (boundary at 50)
      await user.click(markers[1] as HTMLElement);

      expect(seekSpy).toHaveBeenCalledWith(50);
    });

    it("removes boundary on Ctrl+Click", () => {
      const removeBoundarySpy = vi.spyOn(useStructureStore.getState(), "removeBoundary");

      const { container } = render(<BoundaryMarkers />);
      const markers = container.querySelectorAll('[style*="position: absolute"]');

      // Ctrl+Click the second marker (boundary at 50)
      fireEvent.click(markers[1] as HTMLElement, { ctrlKey: true });

      expect(removeBoundarySpy).toHaveBeenCalledWith(50);
    });

    it("removes boundary on Cmd+Click (Mac)", () => {
      const removeBoundarySpy = vi.spyOn(useStructureStore.getState(), "removeBoundary");

      const { container } = render(<BoundaryMarkers />);
      const markers = container.querySelectorAll('[style*="position: absolute"]');

      // Cmd+Click the second marker (boundary at 50)
      fireEvent.click(markers[1] as HTMLElement, { metaKey: true });

      expect(removeBoundarySpy).toHaveBeenCalledWith(50);
    });

    it("stops propagation on click", async () => {
      const parentClickHandler = vi.fn();

      const { container } = render(
        <div onClick={parentClickHandler}>
          <BoundaryMarkers />
        </div>
      );

      const markers = container.querySelectorAll('[style*="position: absolute"]');
      await user.click(markers[1] as HTMLElement);

      // Parent handler should not be called due to stopPropagation
      expect(parentClickHandler).not.toHaveBeenCalled();
    });

    it("does not seek when Ctrl+Click is used for deletion", () => {
      const seekSpy = vi.spyOn(useAudioStore.getState(), "seek");

      const { container } = render(<BoundaryMarkers />);
      const markers = container.querySelectorAll('[style*="position: absolute"]');

      // Ctrl+Click should only remove, not seek
      fireEvent.click(markers[1] as HTMLElement, { ctrlKey: true });

      expect(seekSpy).not.toHaveBeenCalled();
    });
  });

  describe("Visual Properties", () => {
    it("renders with purple color (#7b6aff)", () => {
      const { container } = render(<BoundaryMarkers />);
      const marker = container.querySelector('[style*="background: rgb(123, 106, 255)"]');

      expect(marker).toBeInTheDocument();
    });

    it("renders with 3px width", () => {
      const { container } = render(<BoundaryMarkers />);
      const marker = container.querySelector('[style*="width: 3px"]');

      expect(marker).toBeInTheDocument();
    });

    it("renders with glow shadow", () => {
      const { container } = render(<BoundaryMarkers />);
      const marker = container.querySelector('[style*="box-shadow"]');

      expect(marker).toBeInTheDocument();
      expect(marker?.getAttribute("style")).toContain("rgba(123, 106, 255, 0.5)");
    });

    it("has pointer cursor", () => {
      const { container } = render(<BoundaryMarkers />);
      const marker = container.querySelector('[style*="cursor: pointer"]');

      expect(marker).toBeInTheDocument();
    });

    it("renders full height", () => {
      const { container } = render(<BoundaryMarkers />);
      const marker = container.querySelector('[style*="height: 100%"]');

      expect(marker).toBeInTheDocument();
    });
  });

  describe("Tooltip", () => {
    it("shows formatted time in tooltip", () => {
      const { container } = render(<BoundaryMarkers />);
      const markers = container.querySelectorAll('[title]');

      // Second boundary at 50 seconds should show 00:50.000
      const marker50 = Array.from(markers).find((m) =>
        m.getAttribute("title")?.includes("00:50.000")
      );

      expect(marker50).toBeInTheDocument();
    });

    it("shows instructions in tooltip", () => {
      const { container } = render(<BoundaryMarkers />);
      const markers = container.querySelectorAll('[title]');

      expect(markers[0].getAttribute("title")).toContain("Click to seek");
      expect(markers[0].getAttribute("title")).toContain("Ctrl+Click to remove");
    });

    it("formats tooltip time correctly for different boundaries", () => {
      useStructureStore.setState({ boundaries: [0, 65.5, 180] });

      const { container } = render(<BoundaryMarkers />);
      const markers = container.querySelectorAll('[title]');

      // Boundary at 65.5 should show 01:05.500
      const marker65 = Array.from(markers).find((m) =>
        m.getAttribute("title")?.includes("01:05.500")
      );

      expect(marker65).toBeInTheDocument();
    });
  });

  describe("Edge Cases", () => {
    it("handles empty boundaries array", () => {
      useStructureStore.setState({ boundaries: [] });

      const { container } = render(<BoundaryMarkers />);
      const markers = container.querySelectorAll('[style*="position: absolute"]');

      expect(markers).toHaveLength(0);
    });

    it("handles single boundary", () => {
      useStructureStore.setState({ boundaries: [90] });

      const { container } = render(<BoundaryMarkers />);
      const markers = container.querySelectorAll('[style*="position: absolute"]');

      expect(markers).toHaveLength(1);
    });

    it("handles boundaries at viewport edges", () => {
      useStructureStore.setState({ boundaries: [0, 180] });
      useWaveformStore.setState({
        viewportStart: 0,
        viewportEnd: 180,
      });

      const { container } = render(<BoundaryMarkers />);
      const markers = container.querySelectorAll('[style*="position: absolute"]');

      // Both boundaries at edges should be rendered
      expect(markers).toHaveLength(2);
    });

    it("handles viewport override props", () => {
      useStructureStore.setState({ boundaries: [0, 50, 100, 150, 180] });
      useWaveformStore.setState({
        viewportStart: 0,
        viewportEnd: 180,
      });

      // Override viewport to show only middle section
      const { container } = render(
        <BoundaryMarkers viewportStart={40} viewportEnd={120} />
      );

      const markers = container.querySelectorAll('[style*="position: absolute"]');

      // Only boundaries at 50 and 100 are within override [40, 120]
      expect(markers).toHaveLength(2);
    });

    it("calculates position correctly with viewport override", () => {
      useStructureStore.setState({ boundaries: [60] });

      // Viewport is [40, 120], boundary at 60 should be at 25%
      // (60 - 40) / (120 - 40) * 100 = 20 / 80 * 100 = 25%
      const { container } = render(
        <BoundaryMarkers viewportStart={40} viewportEnd={120} />
      );

      const marker = container.querySelector('[style*="left: 25%"]');
      expect(marker).toBeInTheDocument();
    });

    it("handles boundary exactly at viewport start", () => {
      useStructureStore.setState({ boundaries: [50] });
      useWaveformStore.setState({
        viewportStart: 50,
        viewportEnd: 100,
      });

      const { container } = render(<BoundaryMarkers />);
      const markers = container.querySelectorAll('[style*="position: absolute"]');

      // Boundary at exact viewport start should be rendered
      expect(markers).toHaveLength(1);
    });

    it("handles boundary exactly at viewport end", () => {
      useStructureStore.setState({ boundaries: [100] });
      useWaveformStore.setState({
        viewportStart: 50,
        viewportEnd: 100,
      });

      const { container } = render(<BoundaryMarkers />);
      const markers = container.querySelectorAll('[style*="position: absolute"]');

      // Boundary at exact viewport end should be rendered
      expect(markers).toHaveLength(1);
    });

    it("handles very small viewport", () => {
      useStructureStore.setState({ boundaries: [50, 50.5, 51] });
      useWaveformStore.setState({
        viewportStart: 50,
        viewportEnd: 51,
      });

      const { container } = render(<BoundaryMarkers />);
      const markers = container.querySelectorAll('[style*="position: absolute"]');

      // All 3 boundaries within small 1-second viewport
      expect(markers).toHaveLength(3);
    });

    it("handles boundaries with fractional seconds", () => {
      useStructureStore.setState({ boundaries: [45.123, 90.456] });

      const { container } = render(<BoundaryMarkers />);
      const markers = container.querySelectorAll('[title]');

      expect(markers).toHaveLength(2);
      // Should format with 3 decimal places (milliseconds)
      expect(markers[0].getAttribute("title")).toContain("00:45.123");
      expect(markers[1].getAttribute("title")).toContain("01:30.456");
    });
  });

  describe("Multiple Viewport Scenarios", () => {
    it("updates visible markers when viewport changes", () => {
      const { container, rerender } = render(<BoundaryMarkers />);

      let markers = container.querySelectorAll('[style*="position: absolute"]');
      expect(markers).toHaveLength(5); // All boundaries visible in [0, 180]

      // Change viewport
      useWaveformStore.setState({
        viewportStart: 75,
        viewportEnd: 125,
      });

      rerender(<BoundaryMarkers />);

      markers = container.querySelectorAll('[style*="position: absolute"]');
      expect(markers).toHaveLength(1); // Only boundary at 100 visible in [75, 125]
    });

    it("recalculates positions when viewport changes", () => {
      useStructureStore.setState({ boundaries: [100] });
      useWaveformStore.setState({
        viewportStart: 0,
        viewportEnd: 200,
      });

      const { container, rerender } = render(<BoundaryMarkers />);

      // Initially at 50% of [0, 200]
      let marker = container.querySelector('[style*="left: 50%"]');
      expect(marker).toBeInTheDocument();

      // Change viewport
      useWaveformStore.setState({
        viewportStart: 50,
        viewportEnd: 150,
      });

      rerender(<BoundaryMarkers />);

      // Now at 50% of [50, 150]
      marker = container.querySelector('[style*="left: 50%"]');
      expect(marker).toBeInTheDocument();
    });
  });
});
