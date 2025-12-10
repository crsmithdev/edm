import { Play, Pause, RotateCcw, ChevronLeft, ChevronRight } from "lucide-react";
import { useAudioStore, useTrackStore } from "@/stores";
import { Button, Tooltip } from "@/components/UI";

/**
 * Transport row with playback controls and info displays
 */
export function PlaybackControls() {
  const { isPlaying, play, pause, returnToCue } = useAudioStore();
  const { nextTrack, previousTrack } = useTrackStore();

  const togglePlayback = () => {
    if (isPlaying) {
      pause();
    } else {
      play();
    }
  };

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(4, 1fr)",
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
      <Tooltip content="Return to cue point" shortcut="C / R">
        <Button
          onClick={returnToCue}
          variant="warning"
          icon={<RotateCcw size={16} />}
        >
          Cue
        </Button>
      </Tooltip>
      <Tooltip content="Previous track" shortcut="↑">
        <Button
          onClick={previousTrack}
          variant="secondary"
          icon={<ChevronLeft size={16} />}
        >
          Previous
        </Button>
      </Tooltip>
      <Tooltip content="Next track" shortcut="↓">
        <Button
          onClick={nextTrack}
          variant="secondary"
          icon={<ChevronRight size={16} />}
        >
          Next
        </Button>
      </Tooltip>
    </div>
  );
}
