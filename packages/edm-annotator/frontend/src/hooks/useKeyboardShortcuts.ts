import { useEffect } from "react";
import {
  useAudioStore,
  useStructureStore,
  useTempoStore,
  useWaveformStore,
  useUIStore,
  useTrackStore,
} from "@/stores";
import { getBarDuration, getBeatDuration } from "@/utils/barCalculations";

/**
 * Handles keyboard shortcuts for the application
 */
export function useKeyboardShortcuts() {
  const { isPlaying, play, pause, seek, currentTime, returnToCue, setCuePoint } =
    useAudioStore();
  const { addBoundary } = useStructureStore();
  const { setDownbeat, trackBPM, trackDownbeat } = useTempoStore();
  const { zoom, zoomToFit } = useWaveformStore();
  const { toggleQuantize, showStatus, quantizeEnabled } = useUIStore();
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

        case "c": // C - set cue when stopped, return to cue when playing
          if (e.ctrlKey || e.metaKey) {
            // Allow Ctrl+C / Cmd+C for copy
            return;
          }
          e.preventDefault();
          if (isPlaying) {
            returnToCue();
            showStatus("Returned to cue");
          } else {
            // Snap to nearest beat if quantize enabled
            let cueTime = currentTime;
            if (quantizeEnabled && trackBPM > 0) {
              const beatDuration = getBeatDuration(trackBPM);
              const beatsFromDownbeat = (currentTime - trackDownbeat) / beatDuration;
              const nearestBeat = Math.round(beatsFromDownbeat);
              cueTime = trackDownbeat + nearestBeat * beatDuration;
            }
            setCuePoint(cueTime);
            showStatus(`Cue point set at ${cueTime.toFixed(2)}s`);
          }
          break;

        case "r": // R - return to cue (always)
          e.preventDefault();
          returnToCue();
          showStatus("Returned to cue");
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
    setCuePoint,
    addBoundary,
    setDownbeat,
    toggleQuantize,
    showStatus,
    previousTrack,
    nextTrack,
    zoom,
    zoomToFit,
    trackBPM,
    trackDownbeat,
    quantizeEnabled,
  ]);
}
