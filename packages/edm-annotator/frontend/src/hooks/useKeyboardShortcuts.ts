import { useEffect } from "react";
import {
  useAudioStore,
  useStructureStore,
  useTempoStore,
  useWaveformStore,
  useUIStore,
  useTrackStore,
} from "@/stores";
import { getBarDuration } from "@/utils/barCalculations";

/**
 * Handles keyboard shortcuts for the application
 */
export function useKeyboardShortcuts() {
  const { isPlaying, play, pause, seek, currentTime, returnToCue } =
    useAudioStore();
  const { addBoundary } = useStructureStore();
  const { setDownbeat, trackBPM } = useTempoStore();
  const { zoom, zoomToFit } = useWaveformStore();
  const { toggleQuantize, showStatus } = useUIStore();
  const { previousTrack, nextTrack } = useTrackStore();

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if typing in input or select
      const target = e.target as HTMLElement;
      if (
        target.tagName === "INPUT" ||
        target.tagName === "SELECT" ||
        target.tagName === "TEXTAREA"
      ) {
        return;
      }

      switch (e.key) {
        case " ": // Space - play/pause
          e.preventDefault();
          if (isPlaying) {
            pause();
          } else {
            play();
          }
          break;

        case "b": // B - add boundary at playhead
          e.preventDefault();
          addBoundary(currentTime);
          showStatus(`Added boundary at ${currentTime.toFixed(2)}s`);
          break;

        case "d": // D - set downbeat at playhead
          e.preventDefault();
          setDownbeat(currentTime);
          showStatus(`Downbeat set to ${currentTime.toFixed(2)}s`);
          break;

        case "q": // Q - toggle quantize
          e.preventDefault();
          toggleQuantize();
          break;

        case "c": // C - return to cue
        case "r": // R - return to cue
          e.preventDefault();
          returnToCue();
          break;

        case "ArrowLeft": // Left arrow - jump backward
          e.preventDefault();
          if (e.shiftKey) {
            // Shift+Left: -8 bars
            const barDuration = getBarDuration(trackBPM);
            seek(Math.max(0, currentTime - barDuration * 8));
          } else if (e.ctrlKey || e.metaKey) {
            // Ctrl+Left: -1 bar
            const barDuration = getBarDuration(trackBPM);
            seek(Math.max(0, currentTime - barDuration));
          } else {
            // Left: -4 bars
            const barDuration = getBarDuration(trackBPM);
            seek(Math.max(0, currentTime - barDuration * 4));
          }
          break;

        case "ArrowRight": // Right arrow - jump forward
          e.preventDefault();
          if (e.shiftKey) {
            // Shift+Right: +8 bars
            const barDuration = getBarDuration(trackBPM);
            seek(currentTime + barDuration * 8);
          } else if (e.ctrlKey || e.metaKey) {
            // Ctrl+Right: +1 bar
            const barDuration = getBarDuration(trackBPM);
            seek(currentTime + barDuration);
          } else {
            // Right: +4 bars
            const barDuration = getBarDuration(trackBPM);
            seek(currentTime + barDuration * 4);
          }
          break;

        case "ArrowUp": // Up - previous track
          e.preventDefault();
          previousTrack();
          break;

        case "ArrowDown": // Down - next track
          e.preventDefault();
          nextTrack();
          break;

        case "+": // + - zoom in
        case "=": // = - zoom in (same key without shift)
          e.preventDefault();
          zoom(1);
          break;

        case "-": // - - zoom out
        case "_": // _ - zoom out (shift + -)
          e.preventDefault();
          zoom(-1);
          break;

        case "0": // 0 - zoom to fit
          e.preventDefault();
          zoomToFit();
          showStatus("Zoom reset");
          break;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [
    isPlaying,
    play,
    pause,
    seek,
    currentTime,
    returnToCue,
    addBoundary,
    setDownbeat,
    toggleQuantize,
    showStatus,
    previousTrack,
    nextTrack,
    zoom,
    zoomToFit,
    trackBPM,
  ]);
}
