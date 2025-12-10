import { useEffect, useRef } from "react";
import { useAudioStore } from "@/stores";

/**
 * Manages audio playback element and syncs with audio store
 */
export function useAudioPlayback() {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const { setPlayer, updateCurrentTime, pause } = useAudioStore();
  const animationFrameRef = useRef<number>();

  useEffect(() => {
    // Create audio element
    const audio = new Audio();
    audio.preload = "auto";
    audioRef.current = audio;
    setPlayer(audio);

    // Event handlers
    const handlePlay = () => {
      // Start animation loop for smooth currentTime updates
      const updateTime = () => {
        if (audio && !audio.paused) {
          updateCurrentTime(audio.currentTime);
          animationFrameRef.current = requestAnimationFrame(updateTime);
        }
      };
      updateTime();
    };

    const handlePause = () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      updateCurrentTime(audio.currentTime);
    };

    const handleEnded = () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      pause();
      updateCurrentTime(audio.currentTime);
    };

    const handleTimeUpdate = () => {
      // Fallback for when not using animation frame
      if (audio.paused) {
        updateCurrentTime(audio.currentTime);
      }
    };

    // Attach listeners
    audio.addEventListener("play", handlePlay);
    audio.addEventListener("pause", handlePause);
    audio.addEventListener("ended", handleEnded);
    audio.addEventListener("timeupdate", handleTimeUpdate);

    // Cleanup
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      audio.removeEventListener("play", handlePlay);
      audio.removeEventListener("pause", handlePause);
      audio.removeEventListener("ended", handleEnded);
      audio.removeEventListener("timeupdate", handleTimeUpdate);
      audio.pause();
      audio.src = "";
    };
  }, [setPlayer, updateCurrentTime, pause]);

  return audioRef;
}
