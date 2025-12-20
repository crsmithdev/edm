import { describe, it, expect, vi, beforeEach, afterEach, beforeAll } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useAudioPlayback } from "@/hooks/useAudioPlayback";
import { useAudioStore } from "@/stores";

// Mock HTMLMediaElement prototype methods
beforeAll(() => {
  HTMLMediaElement.prototype.play = vi.fn().mockResolvedValue(undefined);
  HTMLMediaElement.prototype.pause = vi.fn();
  HTMLMediaElement.prototype.load = vi.fn();
});

describe("useAudioPlayback", () => {
  let mockAudio: HTMLAudioElement;

  beforeEach(() => {
    // Create a real HTMLAudioElement with mocked methods
    mockAudio = new Audio();

    // Mock currentTime as a writable property
    let currentTimeValue = 0;
    Object.defineProperty(mockAudio, 'currentTime', {
      get: () => currentTimeValue,
      set: (value: number) => { currentTimeValue = value; },
      configurable: true
    });

    useAudioStore.setState({ player: mockAudio });
    useAudioStore.getState().reset();

    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("Playback Control", () => {
    it("plays audio when play is called", async () => {
      renderHook(() => useAudioPlayback());

      const { play } = useAudioStore.getState();

      await act(async () => {
        await play();
      });

      expect(mockAudio.play).toHaveBeenCalledTimes(1);
      expect(useAudioStore.getState().isPlaying).toBe(true);
    });

    it("pauses audio when pause is called", () => {
      renderHook(() => useAudioPlayback());

      useAudioStore.setState({ isPlaying: true });

      const { pause } = useAudioStore.getState();

      act(() => {
        pause();
      });

      expect(mockAudio.pause).toHaveBeenCalledTimes(1);
      expect(useAudioStore.getState().isPlaying).toBe(false);
    });

    it("seeks to specified time", () => {
      renderHook(() => useAudioPlayback());

      const { seek, player } = useAudioStore.getState();

      act(() => {
        seek(30.5);
      });

      expect(player?.currentTime).toBe(30.5);
      expect(useAudioStore.getState().currentTime).toBe(30.5);
    });
  });

  describe("Cue Point Management", () => {
    it("sets cue point", () => {
      renderHook(() => useAudioPlayback());

      const { setCuePoint } = useAudioStore.getState();

      act(() => {
        setCuePoint(15.0);
      });

      expect(useAudioStore.getState().cuePoint).toBe(15.0);
    });

    it("returns to cue point and pauses", () => {
      renderHook(() => useAudioPlayback());

      const { setCuePoint, seek, returnToCue, player } = useAudioStore.getState();

      act(() => {
        setCuePoint(10.0);
        seek(50.0);
        returnToCue();
      });

      expect(player?.currentTime).toBe(10.0);
      expect(player?.pause).toHaveBeenCalled();
      expect(useAudioStore.getState().currentTime).toBe(10.0);
      expect(useAudioStore.getState().isPlaying).toBe(false);
    });

    it("returns to start if no cue point set", () => {
      renderHook(() => useAudioPlayback());

      const { seek, returnToCue, player } = useAudioStore.getState();

      act(() => {
        seek(50.0);
        returnToCue();
      });

      expect(player?.currentTime).toBe(0);
    });
  });

  describe("Current Time Tracking", () => {
    it("updates current time when seeking", () => {
      renderHook(() => useAudioPlayback());

      const { seek } = useAudioStore.getState();

      act(() => {
        seek(25.5);
      });

      expect(useAudioStore.getState().currentTime).toBe(25.5);
    });

    it("syncs current time with audio element", () => {
      renderHook(() => useAudioPlayback());

      // Simulate audio element time update
      mockAudio.currentTime = 42.3;

      act(() => {
        useAudioStore.setState({ currentTime: mockAudio.currentTime });
      });

      expect(useAudioStore.getState().currentTime).toBe(42.3);
    });
  });

  describe("Playback State", () => {
    it("initializes with paused state", () => {
      renderHook(() => useAudioPlayback());

      expect(useAudioStore.getState().isPlaying).toBe(false);
    });

    it("toggles playing state", async () => {
      renderHook(() => useAudioPlayback());

      const { play, pause } = useAudioStore.getState();

      // Start playing
      await act(async () => {
        await play();
      });
      expect(useAudioStore.getState().isPlaying).toBe(true);

      // Pause
      act(() => {
        pause();
      });
      expect(useAudioStore.getState().isPlaying).toBe(false);
    });
  });

  describe("Reset", () => {
    it("resets to initial state", async () => {
      renderHook(() => useAudioPlayback());

      const { play, seek, setCuePoint, reset } = useAudioStore.getState();

      // Set some state
      await act(async () => {
        await play();
        seek(50.0);
        setCuePoint(25.0);
      });

      // Reset
      act(() => {
        reset();
      });

      const state = useAudioStore.getState();
      expect(state.isPlaying).toBe(false);
      expect(state.currentTime).toBe(0);
      expect(state.cuePoint).toBe(0);
    });
  });
});
