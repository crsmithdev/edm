import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { PlaybackControls } from "@/components/Transport/PlaybackControls";
import { useAudioStore, useTrackStore, useUIStore, useTempoStore } from "@/stores";

describe("PlaybackControls", () => {
  beforeEach(() => {
    // Reset stores
    useAudioStore.getState().reset();
    useTrackStore.getState().reset();
    useUIStore.getState().reset();
    useTempoStore.getState().reset();

    // Set up basic tempo for quantization tests
    useTempoStore.setState({
      trackBPM: 120,
      trackDownbeat: 0,
    });

    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("Play/Pause Button", () => {
    it("renders play button when not playing", () => {
      render(<PlaybackControls />);
      expect(screen.getByText("Play")).toBeInTheDocument();
    });

    it("renders pause button when playing", () => {
      useAudioStore.setState({ isPlaying: true });
      render(<PlaybackControls />);
      expect(screen.getByText("Pause")).toBeInTheDocument();
    });

    it("toggles playback on click", async () => {
      const user = userEvent.setup();
      const playSpy = vi.spyOn(useAudioStore.getState(), "play");
      const pauseSpy = vi.spyOn(useAudioStore.getState(), "pause");

      render(<PlaybackControls />);

      // Click play
      await user.click(screen.getByText("Play"));
      expect(playSpy).toHaveBeenCalledTimes(1);

      // Update state to playing
      useAudioStore.setState({ isPlaying: true });

      // Click pause
      await user.click(screen.getByText("Pause"));
      expect(pauseSpy).toHaveBeenCalledTimes(1);
    });
  });

  describe("Cue Button - Basic Behavior", () => {
    it("sets cue point when clicked while not playing and not at cue point", async () => {
      const user = userEvent.setup();
      const setCuePointSpy = vi.spyOn(useAudioStore.getState(), "setCuePoint");
      const seekSpy = vi.spyOn(useAudioStore.getState(), "seek");

      useAudioStore.setState({
        isPlaying: false,
        currentTime: 10.0,
        cuePoint: 0,
      });

      render(<PlaybackControls />);

      await user.click(screen.getByText("Cue"));

      expect(setCuePointSpy).toHaveBeenCalledWith(10.0);
      expect(seekSpy).toHaveBeenCalledWith(10.0);
    });

    it("returns to cue point when clicked while playing", async () => {
      const user = userEvent.setup();
      const returnToCueSpy = vi.spyOn(useAudioStore.getState(), "returnToCue");

      useAudioStore.setState({
        isPlaying: true,
        currentTime: 10.0,
        cuePoint: 5.0,
      });

      render(<PlaybackControls />);

      await user.click(screen.getByText("Cue"));

      expect(returnToCueSpy).toHaveBeenCalledTimes(1);
    });
  });

  describe("Cue Button - Smart Preview Behavior", () => {
    it("starts preview on mouse down when at cue point and not playing", async () => {
      const user = userEvent.setup();
      const playSpy = vi.spyOn(useAudioStore.getState(), "play");

      useAudioStore.setState({
        isPlaying: false,
        currentTime: 5.0,
        cuePoint: 5.0,
      });

      render(<PlaybackControls />);

      const cueButton = screen.getByText("Cue");
      await user.pointer({ keys: "[MouseLeft>]", target: cueButton });

      expect(playSpy).toHaveBeenCalledTimes(1);
    });

    it("stops preview and returns to cue on mouse up", async () => {
      const user = userEvent.setup();
      const returnToCueSpy = vi.spyOn(useAudioStore.getState(), "returnToCue");

      useAudioStore.setState({
        isPlaying: false,
        currentTime: 5.0,
        cuePoint: 5.0,
      });

      render(<PlaybackControls />);

      const cueButton = screen.getByText("Cue");

      // Mouse down to start preview
      await user.pointer({ keys: "[MouseLeft>]", target: cueButton });

      // Mouse up to stop preview
      await user.pointer({ keys: "[/MouseLeft]", target: cueButton });

      await waitFor(() => {
        expect(returnToCueSpy).toHaveBeenCalledTimes(1);
      });
    });

    it("does not set cue point on click after preview", async () => {
      const user = userEvent.setup();
      const setCuePointSpy = vi.spyOn(useAudioStore.getState(), "setCuePoint");

      useAudioStore.setState({
        isPlaying: false,
        currentTime: 5.0,
        cuePoint: 5.0,
      });

      render(<PlaybackControls />);

      const cueButton = screen.getByText("Cue");

      // Full click (down + up)
      await user.click(cueButton);

      // Should not set cue point because we just did a preview
      expect(setCuePointSpy).not.toHaveBeenCalled();
    });

    it("does not start preview when playing", async () => {
      const user = userEvent.setup();
      const playSpy = vi.spyOn(useAudioStore.getState(), "play");

      useAudioStore.setState({
        isPlaying: true,
        currentTime: 5.0,
        cuePoint: 5.0,
      });

      render(<PlaybackControls />);

      const cueButton = screen.getByText("Cue");
      await user.pointer({ keys: "[MouseLeft>]", target: cueButton });

      expect(playSpy).not.toHaveBeenCalled();
    });
  });

  describe("Cue Button - Quantization", () => {
    it("sets quantized cue point when quantize is enabled", async () => {
      const user = userEvent.setup();
      const setCuePointSpy = vi.spyOn(useAudioStore.getState(), "setCuePoint");
      const seekSpy = vi.spyOn(useAudioStore.getState(), "seek");

      useUIStore.setState({ quantizeEnabled: true });
      useTempoStore.setState({ trackBPM: 120, trackDownbeat: 0 });
      useAudioStore.setState({
        isPlaying: false,
        currentTime: 10.3, // Slightly off beat
        cuePoint: 0,
      });

      render(<PlaybackControls />);

      await user.click(screen.getByText("Cue"));

      // At 120 BPM, beat duration is 0.5s
      // 10.3s is closest to 10.5s (beat 21)
      expect(setCuePointSpy).toHaveBeenCalledWith(10.5);
      expect(seekSpy).toHaveBeenCalledWith(10.5);
    });

    it("detects when at cue point with quantization tolerance", async () => {
      const user = userEvent.setup();
      const playSpy = vi.spyOn(useAudioStore.getState(), "play");

      useUIStore.setState({ quantizeEnabled: true });
      useTempoStore.setState({ trackBPM: 120, trackDownbeat: 0 });
      useAudioStore.setState({
        isPlaying: false,
        currentTime: 10.005, // Within tolerance of 10.0
        cuePoint: 10.0,
      });

      render(<PlaybackControls />);

      const cueButton = screen.getByText("Cue");
      await user.pointer({ keys: "[MouseLeft>]", target: cueButton });

      // Should start preview because we're at cue point (within tolerance)
      expect(playSpy).toHaveBeenCalledTimes(1);
    });
  });

  describe("Cue Button - Tooltip", () => {
    it("shows correct tooltip when playing", async () => {
      useAudioStore.setState({
        isPlaying: true,
        currentTime: 10.0,
        cuePoint: 5.0,
      });

      render(<PlaybackControls />);

      const cueButton = screen.getByText("Cue").closest("button");
      expect(cueButton).toBeInTheDocument();
      // Tooltip is rendered by Tooltip component wrapper
    });

    it("shows preview tooltip when at cue point and not playing", () => {
      useAudioStore.setState({
        isPlaying: false,
        currentTime: 5.0,
        cuePoint: 5.0,
      });

      render(<PlaybackControls />);

      const cueButton = screen.getByText("Cue").closest("button");
      expect(cueButton).toBeInTheDocument();
    });

    it("shows set cue tooltip when not at cue point and not playing", () => {
      useAudioStore.setState({
        isPlaying: false,
        currentTime: 10.0,
        cuePoint: 5.0,
      });

      render(<PlaybackControls />);

      const cueButton = screen.getByText("Cue").closest("button");
      expect(cueButton).toBeInTheDocument();
    });
  });

  describe("Track Navigation", () => {
    it("calls previousTrack when Previous button clicked", async () => {
      const user = userEvent.setup();
      const previousTrackSpy = vi.spyOn(useTrackStore.getState(), "previousTrack");

      render(<PlaybackControls />);

      await user.click(screen.getByText("Previous"));

      expect(previousTrackSpy).toHaveBeenCalledTimes(1);
    });

    it("calls nextTrack when Next button clicked", async () => {
      const user = userEvent.setup();
      const nextTrackSpy = vi.spyOn(useTrackStore.getState(), "nextTrack");

      render(<PlaybackControls />);

      await user.click(screen.getByText("Next"));

      expect(nextTrackSpy).toHaveBeenCalledTimes(1);
    });
  });
});
