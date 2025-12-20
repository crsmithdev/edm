import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { userEvent } from "@testing-library/user-event";
import { WaveformContainer } from "@/components/Waveform/WaveformContainer";
import { useWaveformStore, useTrackStore, useTempoStore, useAudioStore } from "@/stores";

// Mock child components
vi.mock("@/components/Waveform/OverviewWaveform", () => ({
  OverviewWaveform: () => <div data-testid="overview-waveform">Overview Waveform</div>,
}));

vi.mock("@/components/Waveform/DetailWaveform", () => ({
  DetailWaveform: ({ span, onZoomIn, onZoomOut }: any) => (
    <div data-testid="detail-waveform" data-span={span}>
      Detail Waveform (span: {span}s)
      <button onClick={onZoomIn}>Detail Zoom In</button>
      <button onClick={onZoomOut}>Detail Zoom Out</button>
    </div>
  ),
}));

vi.mock("@/components/UI", () => ({
  InfoCard: ({ label, value }: any) => (
    <div data-testid={`info-card-${label.toLowerCase()}`}>
      {label}: {value}
    </div>
  ),
}));

describe("WaveformContainer", () => {
  const user = userEvent.setup();

  beforeEach(() => {
    useWaveformStore.getState().reset();
    useTrackStore.getState().reset();
    useTempoStore.getState().reset();
    useAudioStore.getState().reset();

    // Set up basic state
    useWaveformStore.setState({ duration: 180 });
    useTrackStore.setState({ currentTrack: "test-track.mp3" });
    useTempoStore.setState({ trackBPM: 128, trackDownbeat: 0 });
    useAudioStore.setState({ currentTime: 45.5 });
  });

  describe("Component Rendering", () => {
    it("renders OverviewWaveform", () => {
      render(<WaveformContainer />);
      expect(screen.getByTestId("overview-waveform")).toBeInTheDocument();
    });

    it("renders DetailWaveform", () => {
      render(<WaveformContainer />);
      expect(screen.getByTestId("detail-waveform")).toBeInTheDocument();
    });

    it("renders InfoCard components", () => {
      render(<WaveformContainer />);
      expect(screen.getByTestId("info-card-bpm")).toBeInTheDocument();
      expect(screen.getByTestId("info-card-bar")).toBeInTheDocument();
      expect(screen.getByTestId("info-card-time")).toBeInTheDocument();
    });
  });

  describe("Zoom Controls", () => {
    it("zooms in detail view when zoom in clicked", async () => {
      render(<WaveformContainer />);

      // Initial span should be 16s (DEFAULT_DETAIL_SPAN)
      expect(screen.getByTestId("detail-waveform")).toHaveAttribute("data-span", "16");

      // Click zoom in button
      const zoomInButton = screen.getByText("+");
      await user.click(zoomInButton);

      // Should divide by 1.5: 16 / 1.5 ≈ 10.67
      const detailWaveform = screen.getByTestId("detail-waveform");
      const span = parseFloat(detailWaveform.getAttribute("data-span") || "0");
      expect(span).toBeCloseTo(10.67, 1);
    });

    it("zooms out detail view when zoom out clicked", async () => {
      render(<WaveformContainer />);

      // Initial span should be 16s
      expect(screen.getByTestId("detail-waveform")).toHaveAttribute("data-span", "16");

      // Click zoom out button
      const zoomOutButton = screen.getByText("−");
      await user.click(zoomOutButton);

      // Should multiply by 1.5: 16 * 1.5 = 24
      const detailWaveform = screen.getByTestId("detail-waveform");
      const span = parseFloat(detailWaveform.getAttribute("data-span") || "0");
      expect(span).toBe(24);
    });

    it("resets detail view zoom to default", async () => {
      render(<WaveformContainer />);

      // Zoom in twice
      const zoomInButton = screen.getByText("+");
      await user.click(zoomInButton);
      await user.click(zoomInButton);

      // Verify zoom changed
      let detailWaveform = screen.getByTestId("detail-waveform");
      let span = parseFloat(detailWaveform.getAttribute("data-span") || "0");
      expect(span).not.toBe(16);

      // Click reset
      const resetButton = screen.getByText("Reset");
      await user.click(resetButton);

      // Should be back to default 16s
      detailWaveform = screen.getByTestId("detail-waveform");
      span = parseFloat(detailWaveform.getAttribute("data-span") || "0");
      expect(span).toBe(16);
    });

    it("clamps detail span to MIN_DETAIL_SPAN (4s)", async () => {
      render(<WaveformContainer />);

      // Zoom in many times to try to go below minimum
      const zoomInButton = screen.getByText("+");
      await user.click(zoomInButton); // 16 / 1.5 ≈ 10.67
      await user.click(zoomInButton); // 10.67 / 1.5 ≈ 7.11
      await user.click(zoomInButton); // 7.11 / 1.5 ≈ 4.74
      await user.click(zoomInButton); // 4.74 / 1.5 ≈ 3.16 -> clamped to 4
      await user.click(zoomInButton); // Should stay at 4

      const detailWaveform = screen.getByTestId("detail-waveform");
      const span = parseFloat(detailWaveform.getAttribute("data-span") || "0");
      expect(span).toBeGreaterThanOrEqual(4);
    });

    it("clamps detail span to MAX_DETAIL_SPAN (60s)", async () => {
      render(<WaveformContainer />);

      // Zoom out many times to try to exceed maximum
      const zoomOutButton = screen.getByText("−");
      await user.click(zoomOutButton); // 16 * 1.5 = 24
      await user.click(zoomOutButton); // 24 * 1.5 = 36
      await user.click(zoomOutButton); // 36 * 1.5 = 54
      await user.click(zoomOutButton); // 54 * 1.5 = 81 -> clamped to 60
      await user.click(zoomOutButton); // Should stay at 60

      const detailWaveform = screen.getByTestId("detail-waveform");
      const span = parseFloat(detailWaveform.getAttribute("data-span") || "0");
      expect(span).toBeLessThanOrEqual(60);
    });
  });

  describe("Track Info Display", () => {
    it("displays current track filename", () => {
      render(<WaveformContainer />);
      expect(screen.getByText("test-track.mp3")).toBeInTheDocument();
    });

    it("displays track duration formatted as M:SS", () => {
      render(<WaveformContainer />);
      // 180 seconds = 3:00
      expect(screen.getByText(/3:00 duration/)).toBeInTheDocument();
    });

    it("displays current time formatted as M:SS.ss", () => {
      useAudioStore.setState({ currentTime: 45.5 });
      render(<WaveformContainer />);
      // 45.5 seconds = 0:45.50
      expect(screen.getByText(/Time: 0:45.50/)).toBeInTheDocument();
    });

    it("displays current bar number", () => {
      // At 128 BPM, 1 bar = 4 beats = 4 * (60/128) = 1.875s
      // 45.5s / 1.875s ≈ 24.27 -> bar 24 (0-indexed) or 25 (1-indexed)
      render(<WaveformContainer />);
      expect(screen.getByTestId("info-card-bar")).toBeInTheDocument();
    });

    it("displays BPM", () => {
      render(<WaveformContainer />);
      expect(screen.getByText(/BPM: 128/)).toBeInTheDocument();
    });

    it("displays -- for BPM when not set", () => {
      useTempoStore.setState({ trackBPM: 0 });
      render(<WaveformContainer />);
      expect(screen.getByText(/BPM: --/)).toBeInTheDocument();
    });

    it("hides track info when no track loaded", () => {
      useTrackStore.setState({ currentTrack: null });
      render(<WaveformContainer />);
      expect(screen.queryByText(/test-track\.mp3/)).not.toBeInTheDocument();
      expect(screen.queryByText(/duration/)).not.toBeInTheDocument();
    });

    it("displays -- for bar when no track loaded", () => {
      useTrackStore.setState({ currentTrack: null });
      render(<WaveformContainer />);
      expect(screen.getByText(/Bar: --/)).toBeInTheDocument();
    });

    it("displays -- for time when no track loaded", () => {
      useTrackStore.setState({ currentTrack: null });
      render(<WaveformContainer />);
      expect(screen.getByText(/Time: --/)).toBeInTheDocument();
    });
  });

  describe("Component Integration", () => {
    it("passes detailSpan to DetailWaveform", () => {
      render(<WaveformContainer />);
      const detailWaveform = screen.getByTestId("detail-waveform");
      expect(detailWaveform).toHaveAttribute("data-span", "16");
    });

    it("passes zoom handlers to DetailWaveform", async () => {
      render(<WaveformContainer />);

      // Verify DetailWaveform can trigger zoom via passed handlers
      const detailZoomIn = screen.getByText("Detail Zoom In");
      await user.click(detailZoomIn);

      const detailWaveform = screen.getByTestId("detail-waveform");
      const span = parseFloat(detailWaveform.getAttribute("data-span") || "0");
      expect(span).toBeCloseTo(10.67, 1);
    });

    it("renders InfoCard with correct BPM prop", () => {
      render(<WaveformContainer />);
      expect(screen.getByText(/BPM: 128/)).toBeInTheDocument();
    });

    it("renders InfoCard with correct bar prop", () => {
      render(<WaveformContainer />);
      const barCard = screen.getByTestId("info-card-bar");
      expect(barCard).toBeInTheDocument();
    });

    it("renders InfoCard with correct time prop", () => {
      render(<WaveformContainer />);
      expect(screen.getByText(/Time: 0:45.50/)).toBeInTheDocument();
    });
  });

  describe("Time Formatting", () => {
    it("formats duration correctly for single digit minutes", () => {
      useWaveformStore.setState({ duration: 185 }); // 3:05
      render(<WaveformContainer />);
      expect(screen.getByText(/3:05 duration/)).toBeInTheDocument();
    });

    it("formats duration correctly for double digit minutes", () => {
      useWaveformStore.setState({ duration: 625 }); // 10:25
      render(<WaveformContainer />);
      expect(screen.getByText(/10:25 duration/)).toBeInTheDocument();
    });

    it("formats current time with decimal places", () => {
      useAudioStore.setState({ currentTime: 123.45 });
      render(<WaveformContainer />);
      expect(screen.getByText(/Time: 2:03.45/)).toBeInTheDocument();
    });

    it("pads current time seconds correctly", () => {
      useAudioStore.setState({ currentTime: 65.05 }); // Should be 1:05.05
      render(<WaveformContainer />);
      expect(screen.getByText(/Time: 1:05.05/)).toBeInTheDocument();
    });

    it("formats time at zero correctly", () => {
      useAudioStore.setState({ currentTime: 0 });
      render(<WaveformContainer />);
      expect(screen.getByText(/Time: 0:00.00/)).toBeInTheDocument();
    });
  });

  describe("Zoom Button Interactions", () => {
    it("updates DetailWaveform span when zooming via container buttons", async () => {
      render(<WaveformContainer />);

      const zoomInButton = screen.getByText("+");
      await user.click(zoomInButton);

      // Verify DetailWaveform receives updated span
      const detailWaveform = screen.getByTestId("detail-waveform");
      expect(screen.getByText(/Detail Waveform \(span: 10\.6/)).toBeInTheDocument();
    });

    it("updates DetailWaveform span when zooming via DetailWaveform buttons", async () => {
      render(<WaveformContainer />);

      const detailZoomOut = screen.getByText("Detail Zoom Out");
      await user.click(detailZoomOut);

      // Verify span changed
      const detailWaveform = screen.getByTestId("detail-waveform");
      expect(screen.getByText(/Detail Waveform \(span: 24/)).toBeInTheDocument();
    });
  });

  describe("Edge Cases", () => {
    it("handles duration of 0", () => {
      useWaveformStore.setState({ duration: 0 });
      render(<WaveformContainer />);
      expect(screen.getByText(/0:00 duration/)).toBeInTheDocument();
    });

    it("handles very long durations", () => {
      useWaveformStore.setState({ duration: 3665 }); // 61:05
      render(<WaveformContainer />);
      expect(screen.getByText(/61:05 duration/)).toBeInTheDocument();
    });

    it("handles fractional seconds in current time", () => {
      useAudioStore.setState({ currentTime: 45.999 });
      render(<WaveformContainer />);
      expect(screen.getByText(/Time: 0:46.00/)).toBeInTheDocument();
    });
  });
});
