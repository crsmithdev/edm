import { Play, Pause, RotateCcw, ChevronLeft, ChevronRight, Music, Clock, Zap } from "lucide-react";
import { useAudioStore, useTempoStore, useTrackStore } from "@/stores";
import { Button, InfoCard, Tooltip } from "@/components/UI";

/**
 * Transport row with playback controls and info displays
 */
export function PlaybackControls() {
  const { isPlaying, play, pause, currentTime, returnToCue } = useAudioStore();
  const { trackBPM, timeToBar } = useTempoStore();
  const { nextTrack, previousTrack } = useTrackStore();

  const currentBar = timeToBar(currentTime);

  const togglePlayback = () => {
    if (isPlaying) {
      pause();
    } else {
      play();
    }
  };

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = (seconds % 60).toFixed(2);
    return `${mins}:${secs.padStart(5, "0")}`;
  };

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "1fr auto",
        gap: "var(--space-5)",
        alignItems: "center",
        paddingBottom: "var(--space-5)",
      }}
    >
      {/* Left: Transport buttons */}
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

      {/* Right: Info displays */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap: "var(--space-3)",
        }}
      >
        <InfoCard label="BPM" value={trackBPM || "--"} icon={<Zap size={14} />} />
        <InfoCard label="Bar" value={currentBar} icon={<Music size={14} />} />
        <InfoCard label="Time" value={formatTime(currentTime)} icon={<Clock size={14} />} />
      </div>
    </div>
  );
}
