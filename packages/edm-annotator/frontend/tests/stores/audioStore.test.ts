import { describe, it, expect, beforeEach, vi } from "vitest";
import { useAudioStore } from "@/stores/audioStore";

describe("audioStore", () => {
  let mockPlayer: HTMLAudioElement;

  beforeEach(() => {
    // Reset store before each test
    useAudioStore.getState().reset();
    useAudioStore.setState({ player: null });

    // Create a more complete mock audio element
    mockPlayer = {
      play: vi.fn().mockResolvedValue(undefined),
      pause: vi.fn(),
      currentTime: 0,
    } as unknown as HTMLAudioElement;
  });

  describe("setPlayer", () => {
    it("should set the audio player", () => {
      const { setPlayer } = useAudioStore.getState();

      setPlayer(mockPlayer);

      const { player } = useAudioStore.getState();
      expect(player).toBe(mockPlayer);
    });
  });

  describe("play", () => {
    it("should call play on player and update state", () => {
      const { setPlayer, play } = useAudioStore.getState();

      setPlayer(mockPlayer);
      play();

      expect(mockPlayer.play).toHaveBeenCalled();
      const { isPlaying } = useAudioStore.getState();
      expect(isPlaying).toBe(true);
    });

    it("should do nothing if player is not set", () => {
      const { play } = useAudioStore.getState();

      play();

      const { isPlaying } = useAudioStore.getState();
      expect(isPlaying).toBe(false);
    });

    it("should handle multiple play calls", () => {
      const { setPlayer, play } = useAudioStore.getState();

      setPlayer(mockPlayer);
      play();
      play();

      expect(mockPlayer.play).toHaveBeenCalledTimes(2);
    });
  });

  describe("pause", () => {
    it("should call pause on player and update state", () => {
      const { setPlayer, play, pause } = useAudioStore.getState();

      setPlayer(mockPlayer);
      play();
      pause();

      expect(mockPlayer.pause).toHaveBeenCalled();
      const { isPlaying } = useAudioStore.getState();
      expect(isPlaying).toBe(false);
    });

    it("should do nothing if player is not set", () => {
      const { pause } = useAudioStore.getState();

      pause();

      const { isPlaying } = useAudioStore.getState();
      expect(isPlaying).toBe(false);
    });

    it("should pause even if not playing", () => {
      const { setPlayer, pause } = useAudioStore.getState();

      setPlayer(mockPlayer);
      pause();

      expect(mockPlayer.pause).toHaveBeenCalled();
      const { isPlaying } = useAudioStore.getState();
      expect(isPlaying).toBe(false);
    });
  });

  describe("seek", () => {
    it("should update player currentTime and state", () => {
      const { setPlayer, seek } = useAudioStore.getState();

      setPlayer(mockPlayer);
      seek(30);

      expect(mockPlayer.currentTime).toBe(30);
      const { currentTime } = useAudioStore.getState();
      expect(currentTime).toBe(30);
    });

    it("should do nothing if player is not set", () => {
      const { seek } = useAudioStore.getState();

      seek(30);

      const { currentTime } = useAudioStore.getState();
      expect(currentTime).toBe(0);
    });

    it("should handle seeking to zero", () => {
      const { setPlayer, seek } = useAudioStore.getState();

      setPlayer(mockPlayer);
      seek(0);

      expect(mockPlayer.currentTime).toBe(0);
      const { currentTime } = useAudioStore.getState();
      expect(currentTime).toBe(0);
    });

    it("should handle seeking to fractional times", () => {
      const { setPlayer, seek } = useAudioStore.getState();

      setPlayer(mockPlayer);
      seek(45.678);

      expect(mockPlayer.currentTime).toBe(45.678);
      const { currentTime } = useAudioStore.getState();
      expect(currentTime).toBe(45.678);
    });

    it("should allow seeking while playing", () => {
      const { setPlayer, play, seek } = useAudioStore.getState();

      setPlayer(mockPlayer);
      play();
      seek(60);

      const { isPlaying, currentTime } = useAudioStore.getState();
      expect(isPlaying).toBe(true);
      expect(currentTime).toBe(60);
    });
  });

  describe("setCuePoint", () => {
    it("should update cue point", () => {
      const { setCuePoint } = useAudioStore.getState();

      setCuePoint(45);

      const { cuePoint } = useAudioStore.getState();
      expect(cuePoint).toBe(45);
    });

    it("should handle zero as cue point", () => {
      const { setCuePoint } = useAudioStore.getState();

      setCuePoint(0);

      const { cuePoint } = useAudioStore.getState();
      expect(cuePoint).toBe(0);
    });

    it("should update cue point multiple times", () => {
      const { setCuePoint } = useAudioStore.getState();

      setCuePoint(30);
      setCuePoint(60);
      setCuePoint(90);

      const { cuePoint } = useAudioStore.getState();
      expect(cuePoint).toBe(90);
    });
  });

  describe("returnToCue", () => {
    it("should seek to cue point and pause", () => {
      const { setPlayer, setCuePoint, returnToCue } =
        useAudioStore.getState();

      setPlayer(mockPlayer);
      setCuePoint(45);
      returnToCue();

      expect(mockPlayer.currentTime).toBe(45);
      expect(mockPlayer.pause).toHaveBeenCalled();
      const { currentTime, isPlaying } = useAudioStore.getState();
      expect(currentTime).toBe(45);
      expect(isPlaying).toBe(false);
    });

    it("should return to cue point while playing", () => {
      const { setPlayer, play, setCuePoint, returnToCue } =
        useAudioStore.getState();

      setPlayer(mockPlayer);
      setCuePoint(30);
      play();
      returnToCue();

      expect(mockPlayer.currentTime).toBe(30);
      expect(mockPlayer.pause).toHaveBeenCalled();
      const { isPlaying } = useAudioStore.getState();
      expect(isPlaying).toBe(false);
    });

    it("should do nothing if player is not set", () => {
      const { setCuePoint, returnToCue } = useAudioStore.getState();

      setCuePoint(30);
      returnToCue();

      const { currentTime } = useAudioStore.getState();
      expect(currentTime).toBe(0);
    });

    it("should return to default cue point (0)", () => {
      const { setPlayer, returnToCue } = useAudioStore.getState();

      setPlayer(mockPlayer);
      returnToCue();

      expect(mockPlayer.currentTime).toBe(0);
    });
  });

  describe("updateCurrentTime", () => {
    it("should update current time without seeking", () => {
      const { updateCurrentTime } = useAudioStore.getState();

      updateCurrentTime(75);

      const { currentTime } = useAudioStore.getState();
      expect(currentTime).toBe(75);
    });

    it("should not affect player currentTime", () => {
      const { setPlayer, updateCurrentTime } = useAudioStore.getState();

      setPlayer(mockPlayer);
      updateCurrentTime(75);

      expect(mockPlayer.currentTime).toBe(0); // Player not updated
    });

    it("should be useful for tracking playback progress", () => {
      const { updateCurrentTime } = useAudioStore.getState();

      // Simulate progress updates
      updateCurrentTime(1.0);
      updateCurrentTime(1.5);
      updateCurrentTime(2.0);

      const { currentTime } = useAudioStore.getState();
      expect(currentTime).toBe(2.0);
    });
  });

  describe("reset", () => {
    it("should reset playback state", () => {
      const { setPlayer, play, seek, setCuePoint, reset } =
        useAudioStore.getState();

      setPlayer(mockPlayer);
      play();
      seek(60);
      setCuePoint(30);

      reset();

      const { isPlaying, currentTime, cuePoint } = useAudioStore.getState();
      expect(isPlaying).toBe(false);
      expect(currentTime).toBe(0);
      expect(cuePoint).toBe(0);
    });

    it("should not clear player reference", () => {
      const { setPlayer, reset } = useAudioStore.getState();

      setPlayer(mockPlayer);
      reset();

      const { player } = useAudioStore.getState();
      expect(player).toBe(mockPlayer);
    });
  });

  describe("integration scenarios", () => {
    it("should handle full playback workflow", () => {
      const { setPlayer, play, seek, pause } = useAudioStore.getState();

      setPlayer(mockPlayer);
      play();

      let { isPlaying } = useAudioStore.getState();
      expect(isPlaying).toBe(true);

      seek(30);
      let { currentTime } = useAudioStore.getState();
      expect(currentTime).toBe(30);

      pause();
      isPlaying = useAudioStore.getState().isPlaying;
      expect(isPlaying).toBe(false);
    });

    it("should handle cue point workflow", () => {
      const { setPlayer, play, setCuePoint, seek, returnToCue } =
        useAudioStore.getState();

      setPlayer(mockPlayer);
      setCuePoint(45);
      play();
      seek(90);

      returnToCue();

      const { currentTime, isPlaying } = useAudioStore.getState();
      expect(currentTime).toBe(45);
      expect(isPlaying).toBe(false);
    });

    it("should handle repeated play/pause cycles", () => {
      const { setPlayer, play, pause } = useAudioStore.getState();

      setPlayer(mockPlayer);

      for (let i = 0; i < 5; i++) {
        play();
        pause();
      }

      expect(mockPlayer.play).toHaveBeenCalledTimes(5);
      expect(mockPlayer.pause).toHaveBeenCalledTimes(5);
      const { isPlaying } = useAudioStore.getState();
      expect(isPlaying).toBe(false);
    });

    it("should handle seeking during playback", () => {
      const { setPlayer, play, seek } = useAudioStore.getState();

      setPlayer(mockPlayer);
      play();

      seek(10);
      seek(20);
      seek(30);

      const { currentTime, isPlaying } = useAudioStore.getState();
      expect(currentTime).toBe(30);
      expect(isPlaying).toBe(true);
    });

    it("should maintain state consistency across operations", () => {
      const { setPlayer, play, seek, setCuePoint, updateCurrentTime } =
        useAudioStore.getState();

      setPlayer(mockPlayer);
      setCuePoint(15);
      play();
      updateCurrentTime(20);
      seek(30);

      const { isPlaying, currentTime, cuePoint } = useAudioStore.getState();
      expect(isPlaying).toBe(true);
      expect(currentTime).toBe(30);
      expect(cuePoint).toBe(15);
    });
  });

  describe("edge cases", () => {
    it("should handle seeking to same position", () => {
      const { setPlayer, seek } = useAudioStore.getState();

      setPlayer(mockPlayer);
      seek(30);
      seek(30);

      expect(mockPlayer.currentTime).toBe(30);
    });

    it("should handle play without pause", () => {
      const { setPlayer, play } = useAudioStore.getState();

      setPlayer(mockPlayer);
      play();
      play();
      play();

      expect(mockPlayer.play).toHaveBeenCalledTimes(3);
    });

    it("should handle setting cue point to current position", () => {
      const { setPlayer, seek, setCuePoint, returnToCue } =
        useAudioStore.getState();

      setPlayer(mockPlayer);
      seek(60);
      setCuePoint(60);
      returnToCue();

      const { currentTime } = useAudioStore.getState();
      expect(currentTime).toBe(60);
    });
  });
});
