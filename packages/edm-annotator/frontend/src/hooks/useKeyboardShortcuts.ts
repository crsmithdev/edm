import { useEffect, useRef } from "react";
import {
  useAudioStore,
  useStructureStore,
  useTempoStore,
  useWaveformStore,
  useUIStore,
  useTrackStore,
} from "@/stores";
import { getBarDuration, getBeatDuration } from "@/utils/tempo";

/**
 * Handles keyboard shortcuts for the application
 */
export function useKeyboardShortcuts() {
  const { isPlaying, play, pause, seek, currentTime, returnToCue, setCuePoint, cuePoint } =
    useAudioStore();
  const { addBoundary } = useStructureStore();
  const { setDownbeat, trackBPM, trackDownbeat } = useTempoStore();
  const { zoom, zoomToFit } = useWaveformStore();
  const { toggleQuantize, showStatus, quantizeEnabled } = useUIStore();
  const { previousTrack, nextTrack } = useTrackStore();
  const isPreviewingRef = useRef(false);

  useEffect(() => {
    const getQuantizedPosition = (time: number): number => {
      if (!quantizeEnabled || trackBPM <= 0) return time;
      const beatDuration = getBeatDuration(trackBPM);
      const beatsFromDownbeat = (time - trackDownbeat) / beatDuration;
      const nearestBeat = Math.round(beatsFromDownbeat);
      return trackDownbeat + nearestBeat * beatDuration;
    };

    const isAtCuePoint = (): boolean => {
      const quantizedPosition = getQuantizedPosition(currentTime);
      const quantizedCue = getQuantizedPosition(cuePoint);
      return Math.abs(quantizedPosition - quantizedCue) < 0.01;
    };

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

        case "c": // C - smart cue behavior
          if (e.ctrlKey || e.metaKey) {
            // Allow Ctrl+C / Cmd+C for copy
            return;
          }
          e.preventDefault();

          // Prevent repeated keydown events while key is held
          if (e.repeat) {
            return;
          }

          if (isPlaying) {
            returnToCue();
            showStatus("Returned to cue");
          } else if (isAtCuePoint()) {
            // At cue point - start preview
            play();
            isPreviewingRef.current = true;
          } else {
            // Not at cue point - set cue
            const cueTime = getQuantizedPosition(currentTime);
            setCuePoint(cueTime);
            seek(cueTime);
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

    const handleKeyUp = (e: KeyboardEvent) => {
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
        case "c": // C - stop preview when released
          if (e.ctrlKey || e.metaKey) {
            return;
          }
          e.preventDefault();

          if (isPreviewingRef.current) {
            // Stop preview and return to cue
            returnToCue();
            isPreviewingRef.current = false;
          }
          break;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, [
    isPlaying,
    play,
    pause,
    seek,
    currentTime,
    returnToCue,
    setCuePoint,
    cuePoint,
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
