import { useState, useRef } from "react";
import { Play, Pause, RotateCcw } from "lucide-react";
import { useAudioStore, useUIStore, useTempoStore } from "@/stores";
import { Button, Tooltip } from "@/components/UI";
import { getBeatDuration } from "@/utils/tempo";

/**
 * Transport row with playback controls
 */
export function PlaybackControls() {
  const { isPlaying, play, pause, currentTime, returnToCue, setCuePoint, seek, cuePoint } =
    useAudioStore();
  const { quantizeEnabled } = useUIStore();
  const { trackBPM, trackDownbeat } = useTempoStore();
  const [isPreviewing, setIsPreviewing] = useState(false);
  const ignoreNextClick = useRef(false);

  const togglePlayback = () => {
    if (isPlaying) {
      pause();
    } else {
      play();
    }
  };

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

  const handleCueMouseDown = () => {
    if (isPlaying) {
      // Will return to cue on click
      return;
    }

    // Check if we're at the cue point
    if (isAtCuePoint()) {
      // Start preview playback
      play();
      setIsPreviewing(true);
    }
  };

  const handleCueMouseUp = () => {
    if (isPreviewing) {
      // Stop preview and return to cue
      returnToCue();
      setIsPreviewing(false);
      ignoreNextClick.current = true;
    }
  };

  const handleCueClick = () => {
    // Ignore click if we just finished a preview
    if (ignoreNextClick.current) {
      ignoreNextClick.current = false;
      return;
    }

    if (isPlaying) {
      // Return to cue and stop
      returnToCue();
    } else if (!isAtCuePoint()) {
      // Set cue point at current position
      const cueTime = getQuantizedPosition(currentTime);
      setCuePoint(cueTime);
      seek(cueTime);
    }
    // If at cue point and not playing, preview was handled by mouse down/up
  };

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(2, 1fr)",
        gap: "var(--space-3)",
        alignItems: "stretch",
      }}
    >
      <Tooltip content="Toggle playback" shortcut="Space">
        <Button
          onClick={togglePlayback}
          variant={isPlaying ? "danger" : "accent"}
          icon={isPlaying ? <Pause size={16} /> : <Play size={16} />}
        >
          {isPlaying ? "Pause" : "Play"}
        </Button>
      </Tooltip>
      <Tooltip
        content={
          isPlaying
            ? "Return to cue point and stop"
            : isAtCuePoint()
              ? "Hold to preview"
              : "Set cue point at current position"
        }
        shortcut="C / R"
      >
        <Button
          onClick={handleCueClick}
          onMouseDown={handleCueMouseDown}
          onMouseUp={handleCueMouseUp}
          variant="warning"
          icon={<RotateCcw size={16} />}
        >
          Cue
        </Button>
      </Tooltip>
    </div>
  );
}
