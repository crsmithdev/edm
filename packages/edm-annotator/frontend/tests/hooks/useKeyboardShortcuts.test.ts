import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act, waitFor } from "@testing-library/react";
import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcuts";
import {
  useAudioStore,
  useStructureStore,
  useTempoStore,
  useWaveformStore,
  useUIStore,
  useTrackStore,
} from "@/stores";

describe("useKeyboardShortcuts", () => {
  beforeEach(() => {
    // Reset all stores
    useAudioStore.getState().reset();
    useStructureStore.getState().reset();
    useTempoStore.getState().reset();
    useWaveformStore.getState().reset();
    useUIStore.getState().reset();
    useTrackStore.getState().reset();

    // Set up basic tempo for quantization
    useTempoStore.setState({
      trackBPM: 120,
      trackDownbeat: 0,
    });

    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  const fireKeyDown = (key: string, options: Partial<KeyboardEvent> = {}) => {
    const event = new KeyboardEvent("keydown", {
      key,
      bubbles: true,
      cancelable: true,
      ...options,
    });
    window.dispatchEvent(event);
  };

  const fireKeyUp = (key: string, options: Partial<KeyboardEvent> = {}) => {
    const event = new KeyboardEvent("keyup", {
      key,
      bubbles: true,
      cancelable: true,
      ...options,
    });
    window.dispatchEvent(event);
  };

  describe("Space - Play/Pause", () => {
    it("plays when spacebar pressed and not playing", () => {
      useAudioStore.setState({ isPlaying: false });
      const playSpy = vi.spyOn(useAudioStore.getState(), "play");

      renderHook(() => useKeyboardShortcuts());

      fireKeyDown(" ");

      expect(playSpy).toHaveBeenCalledTimes(1);
    });

    it("pauses when spacebar pressed and playing", () => {
      useAudioStore.setState({ isPlaying: true });
      const pauseSpy = vi.spyOn(useAudioStore.getState(), "pause");

      renderHook(() => useKeyboardShortcuts());

      fireKeyDown(" ");

      expect(pauseSpy).toHaveBeenCalledTimes(1);
    });

    it("does not trigger when typing in input field", () => {
      const playSpy = vi.spyOn(useAudioStore.getState(), "play");

      renderHook(() => useKeyboardShortcuts());

      const input = document.createElement("input");
      document.body.appendChild(input);

      const event = new KeyboardEvent("keydown", {
        key: " ",
        bubbles: true,
        cancelable: true,
      });
      Object.defineProperty(event, "target", { value: input, enumerable: true });
      window.dispatchEvent(event);

      expect(playSpy).not.toHaveBeenCalled();

      document.body.removeChild(input);
    });
  });

  describe("B - Add Boundary", () => {
    it("adds boundary at current playhead position", () => {
      useAudioStore.setState({ currentTime: 15.5 });
      const addBoundarySpy = vi.spyOn(useStructureStore.getState(), "addBoundary");

      renderHook(() => useKeyboardShortcuts());

      fireKeyDown("b");

      expect(addBoundarySpy).toHaveBeenCalledWith(15.5);
    });

    it("shows status message when boundary added", () => {
      useAudioStore.setState({ currentTime: 15.5 });
      const showStatusSpy = vi.spyOn(useUIStore.getState(), "showStatus");

      renderHook(() => useKeyboardShortcuts());

      fireKeyDown("b");

      expect(showStatusSpy).toHaveBeenCalledWith(expect.stringContaining("15.50"));
    });
  });

  describe("D - Set Downbeat", () => {
    it("sets downbeat at current playhead position", () => {
      useAudioStore.setState({ currentTime: 2.5 });
      const setDownbeatSpy = vi.spyOn(useTempoStore.getState(), "setDownbeat");

      renderHook(() => useKeyboardShortcuts());

      fireKeyDown("d");

      expect(setDownbeatSpy).toHaveBeenCalledWith(2.5);
    });

    it("shows status message when downbeat set", () => {
      useAudioStore.setState({ currentTime: 2.5 });
      const showStatusSpy = vi.spyOn(useUIStore.getState(), "showStatus");

      renderHook(() => useKeyboardShortcuts());

      fireKeyDown("d");

      expect(showStatusSpy).toHaveBeenCalledWith(expect.stringContaining("2.50"));
    });
  });

  describe("Q - Toggle Quantize", () => {
    it("toggles quantize mode", () => {
      const toggleQuantizeSpy = vi.spyOn(useUIStore.getState(), "toggleQuantize");

      renderHook(() => useKeyboardShortcuts());

      fireKeyDown("q");

      expect(toggleQuantizeSpy).toHaveBeenCalledTimes(1);
    });
  });

  describe("C - Smart Cue Behavior", () => {
    it("returns to cue when playing", () => {
      useAudioStore.setState({
        isPlaying: true,
        currentTime: 10.0,
        cuePoint: 5.0,
      });
      const returnToCueSpy = vi.spyOn(useAudioStore.getState(), "returnToCue");

      renderHook(() => useKeyboardShortcuts());

      fireKeyDown("c");

      expect(returnToCueSpy).toHaveBeenCalledTimes(1);
    });

    it("sets cue point when not playing and not at cue point", () => {
      useAudioStore.setState({
        isPlaying: false,
        currentTime: 10.0,
        cuePoint: 5.0,
      });
      const setCuePointSpy = vi.spyOn(useAudioStore.getState(), "setCuePoint");
      const seekSpy = vi.spyOn(useAudioStore.getState(), "seek");

      renderHook(() => useKeyboardShortcuts());

      fireKeyDown("c");

      expect(setCuePointSpy).toHaveBeenCalledWith(10.0);
      expect(seekSpy).toHaveBeenCalledWith(10.0);
    });

    it("starts preview when not playing and at cue point", () => {
      const playSpy = vi.spyOn(useAudioStore.getState(), "play");

      renderHook(() => useKeyboardShortcuts());

      useAudioStore.setState({
        isPlaying: false,
        currentTime: 5.0,
        cuePoint: 5.0,
      });

      fireKeyDown("c");

      expect(playSpy).toHaveBeenCalledTimes(1);
    });

    it("stops preview and returns to cue on key up", async () => {
      const playSpy = vi.spyOn(useAudioStore.getState(), "play");
      const returnToCueSpy = vi.spyOn(useAudioStore.getState(), "returnToCue");

      renderHook(() => useKeyboardShortcuts());

      useAudioStore.setState({
        isPlaying: false,
        currentTime: 5.0,
        cuePoint: 5.0,
      });

      // Key down starts preview
      fireKeyDown("c");
      expect(playSpy).toHaveBeenCalledTimes(1);

      // Key up stops preview
      fireKeyUp("c");

      await waitFor(() => {
        expect(returnToCueSpy).toHaveBeenCalledTimes(1);
      });
    });

    it("ignores repeated keydown events while key is held", () => {
      const playSpy = vi.spyOn(useAudioStore.getState(), "play");

      renderHook(() => useKeyboardShortcuts());

      useAudioStore.setState({
        isPlaying: false,
        currentTime: 5.0,
        cuePoint: 5.0,
      });

      // First keydown
      fireKeyDown("c");
      expect(playSpy).toHaveBeenCalledTimes(1);

      // Repeated keydown (key held)
      fireKeyDown("c", { repeat: true });
      expect(playSpy).toHaveBeenCalledTimes(1); // Still 1
    });

    it("allows Ctrl+C for copy", () => {
      const setCuePointSpy = vi.spyOn(useAudioStore.getState(), "setCuePoint");

      renderHook(() => useKeyboardShortcuts());

      useAudioStore.setState({
        isPlaying: false,
        currentTime: 10.0,
        cuePoint: 5.0,
      });

      fireKeyDown("c", { ctrlKey: true });

      expect(setCuePointSpy).not.toHaveBeenCalled();
    });

    it("uses quantized position when quantize enabled", () => {
      useUIStore.setState({ quantizeEnabled: true });
      useTempoStore.setState({ trackBPM: 120, trackDownbeat: 0 });
      useAudioStore.setState({
        isPlaying: false,
        currentTime: 10.3, // Slightly off beat
        cuePoint: 0,
      });
      const setCuePointSpy = vi.spyOn(useAudioStore.getState(), "setCuePoint");
      const seekSpy = vi.spyOn(useAudioStore.getState(), "seek");

      renderHook(() => useKeyboardShortcuts());

      fireKeyDown("c");

      // At 120 BPM, beat duration is 0.5s
      // 10.3s is closest to 10.5s
      expect(setCuePointSpy).toHaveBeenCalledWith(10.5);
      expect(seekSpy).toHaveBeenCalledWith(10.5);
    });
  });

  describe("R - Return to Cue", () => {
    it("always returns to cue point", () => {
      const returnToCueSpy = vi.spyOn(useAudioStore.getState(), "returnToCue");

      renderHook(() => useKeyboardShortcuts());

      useAudioStore.setState({
        currentTime: 10.0,
        cuePoint: 5.0,
      });

      fireKeyDown("r");

      expect(returnToCueSpy).toHaveBeenCalledTimes(1);
    });
  });

  describe("Arrow Keys - Navigation", () => {
    it("seeks backward 4 bars on left arrow", () => {
      useTempoStore.setState({ trackBPM: 120 }); // 4 beats per bar = 2s per bar
      useAudioStore.setState({ currentTime: 20.0 });
      const seekSpy = vi.spyOn(useAudioStore.getState(), "seek");

      renderHook(() => useKeyboardShortcuts());

      fireKeyDown("ArrowLeft");

      // 4 bars at 120 BPM = 8 seconds
      expect(seekSpy).toHaveBeenCalledWith(12.0);
    });

    it("seeks backward 1 bar on Ctrl+left arrow", () => {
      useTempoStore.setState({ trackBPM: 120 });
      useAudioStore.setState({ currentTime: 20.0 });
      const seekSpy = vi.spyOn(useAudioStore.getState(), "seek");

      renderHook(() => useKeyboardShortcuts());

      fireKeyDown("ArrowLeft", { ctrlKey: true });

      // 1 bar at 120 BPM = 2 seconds
      expect(seekSpy).toHaveBeenCalledWith(18.0);
    });

    it("seeks backward 8 bars on Shift+left arrow", () => {
      useTempoStore.setState({ trackBPM: 120 });
      useAudioStore.setState({ currentTime: 20.0 });
      const seekSpy = vi.spyOn(useAudioStore.getState(), "seek");

      renderHook(() => useKeyboardShortcuts());

      fireKeyDown("ArrowLeft", { shiftKey: true });

      // 8 bars at 120 BPM = 16 seconds
      expect(seekSpy).toHaveBeenCalledWith(4.0);
    });

    it("seeks forward 4 bars on right arrow", () => {
      useTempoStore.setState({ trackBPM: 120 });
      useAudioStore.setState({ currentTime: 20.0 });
      const seekSpy = vi.spyOn(useAudioStore.getState(), "seek");

      renderHook(() => useKeyboardShortcuts());

      fireKeyDown("ArrowRight");

      expect(seekSpy).toHaveBeenCalledWith(28.0);
    });

    it("clamps to 0 when seeking backward past start", () => {
      useTempoStore.setState({ trackBPM: 120 });
      useAudioStore.setState({ currentTime: 2.0 });
      const seekSpy = vi.spyOn(useAudioStore.getState(), "seek");

      renderHook(() => useKeyboardShortcuts());

      fireKeyDown("ArrowLeft"); // Would go to -6

      expect(seekSpy).toHaveBeenCalledWith(0);
    });
  });

  describe("Track Navigation", () => {
    it("goes to previous track on up arrow", () => {
      const previousTrackSpy = vi.spyOn(useTrackStore.getState(), "previousTrack");

      renderHook(() => useKeyboardShortcuts());

      fireKeyDown("ArrowUp");

      expect(previousTrackSpy).toHaveBeenCalledTimes(1);
    });

    it("goes to next track on down arrow", () => {
      const nextTrackSpy = vi.spyOn(useTrackStore.getState(), "nextTrack");

      renderHook(() => useKeyboardShortcuts());

      fireKeyDown("ArrowDown");

      expect(nextTrackSpy).toHaveBeenCalledTimes(1);
    });
  });

  describe("Zoom Controls", () => {
    it("zooms in on + key", () => {
      const zoomSpy = vi.spyOn(useWaveformStore.getState(), "zoom");

      renderHook(() => useKeyboardShortcuts());

      fireKeyDown("+");

      expect(zoomSpy).toHaveBeenCalledWith(1);
    });

    it("zooms in on = key", () => {
      const zoomSpy = vi.spyOn(useWaveformStore.getState(), "zoom");

      renderHook(() => useKeyboardShortcuts());

      fireKeyDown("=");

      expect(zoomSpy).toHaveBeenCalledWith(1);
    });

    it("zooms out on - key", () => {
      const zoomSpy = vi.spyOn(useWaveformStore.getState(), "zoom");

      renderHook(() => useKeyboardShortcuts());

      fireKeyDown("-");

      expect(zoomSpy).toHaveBeenCalledWith(-1);
    });

    it("resets zoom on 0 key", () => {
      const zoomToFitSpy = vi.spyOn(useWaveformStore.getState(), "zoomToFit");

      renderHook(() => useKeyboardShortcuts());

      fireKeyDown("0");

      expect(zoomToFitSpy).toHaveBeenCalledTimes(1);
    });
  });

  describe("Input Field Handling", () => {
    it("ignores shortcuts when typing in select", () => {
      const playSpy = vi.spyOn(useAudioStore.getState(), "play");

      renderHook(() => useKeyboardShortcuts());

      const select = document.createElement("select");
      document.body.appendChild(select);

      const event = new KeyboardEvent("keydown", {
        key: " ",
        bubbles: true,
        cancelable: true,
      });
      Object.defineProperty(event, "target", { value: select, enumerable: true });
      window.dispatchEvent(event);

      expect(playSpy).not.toHaveBeenCalled();

      document.body.removeChild(select);
    });

    it("ignores shortcuts when typing in textarea", () => {
      const playSpy = vi.spyOn(useAudioStore.getState(), "play");

      renderHook(() => useKeyboardShortcuts());

      const textarea = document.createElement("textarea");
      document.body.appendChild(textarea);

      const event = new KeyboardEvent("keydown", {
        key: " ",
        bubbles: true,
        cancelable: true,
      });
      Object.defineProperty(event, "target", { value: textarea, enumerable: true });
      window.dispatchEvent(event);

      expect(playSpy).not.toHaveBeenCalled();

      document.body.removeChild(textarea);
    });
  });
});
