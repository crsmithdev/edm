import { useWaveformInteraction } from "@/hooks/useWaveformInteraction";
import { useWaveformStore, useUIStore, useTrackStore, useTempoStore, useAudioStore } from "@/stores";
import { WaveformCanvas } from "./WaveformCanvas";
import { BeatGrid } from "./BeatGrid";
import { Playhead } from "./Playhead";
import { BoundaryMarkers } from "./BoundaryMarkers";
import { RegionOverlays } from "./RegionOverlays";
import { InfoCard } from "@/components/UI";

/**
 * Container for waveform visualization and overlays
 */
export function WaveformContainer() {
  const { zoom, duration } = useWaveformStore();
  const { isDragging } = useUIStore();
  const { currentTrack } = useTrackStore();
  const { trackBPM, timeToBar } = useTempoStore();
  const { currentTime } = useAudioStore();
  const { handleMouseDown, handleMouseMove, handleMouseUp, handleWheel } =
    useWaveformInteraction();

  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = (seconds % 60).toFixed(2);
    return `${mins}:${secs.padStart(5, "0")}`;
  };

  const currentBar = timeToBar(currentTime);

  return (
    <div
      style={{
        background: "var(--bg-secondary)",
        padding: "var(--space-6)",
        borderRadius: "var(--radius-xl)",
        marginBottom: "var(--space-5)",
        border: "1px solid var(--border-primary)",
        boxShadow: "var(--shadow-md)",
        overflow: "hidden",
      }}
    >
      {/* Waveform Display */}
      <div
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onWheel={handleWheel}
        style={{
          position: "relative",
          width: "100%",
          height: "var(--waveform-height)",
          background: "var(--bg-tertiary)",
          border: "1px solid var(--border-subtle)",
          borderRadius: "var(--radius-lg)",
          cursor: isDragging ? "grabbing" : "crosshair",
          overflow: "hidden",
        }}
      >
        {/* Region overlays (behind everything) */}
        <RegionOverlays />

        {/* Waveform canvas */}
        <WaveformCanvas />

        {/* Beat grid overlay */}
        <BeatGrid />

        {/* Boundary markers */}
        <BoundaryMarkers />

        {/* Playhead (on top) */}
        <Playhead />
      </div>

      {/* Track Info and Zoom Controls */}
      <div
        style={{
          display: "flex",
          gap: "var(--space-5)",
          alignItems: "center",
          marginTop: "var(--space-4)",
          justifyContent: "space-between",
        }}
      >
        {/* Track metadata on left */}
        <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-1)" }}>
          <div
            style={{
              fontSize: "var(--font-size-lg)",
              fontWeight: "var(--font-weight-bold)",
              color: "var(--text-primary)",
            }}
          >
            {currentTrack || "No track loaded"}
          </div>
          {currentTrack && (
            <div
              style={{
                fontSize: "var(--font-size-xs)",
                color: "var(--text-tertiary)",
                textTransform: "uppercase",
                letterSpacing: "0.5px",
              }}
            >
              {formatDuration(duration)} duration
            </div>
          )}
        </div>

        {/* Status displays in middle */}
        <div style={{ display: "flex", gap: "var(--space-6)", alignItems: "flex-end" }}>
          <InfoCard label="BPM" value={trackBPM || "--"} />
          <InfoCard label="Bar" value={currentBar} />
          <InfoCard label="Time" value={formatTime(currentTime)} />
        </div>

        {/* Zoom controls on right */}
        <div style={{ display: "flex", gap: "var(--space-2)", alignItems: "center" }}>
        <button
          onClick={() => zoom(-1)}
          style={{
            padding: "var(--space-2) var(--space-4)",
            background: "var(--border-subtle)",
            color: "var(--text-secondary)",
            border: "none",
            borderRadius: "var(--radius-sm)",
            fontSize: "var(--font-size-lg)",
            cursor: "pointer",
            fontWeight: "var(--font-weight-semibold)",
          }}
        >
          −
        </button>
        <button
          onClick={() => zoom(1)}
          style={{
            padding: "var(--space-2) var(--space-4)",
            background: "var(--border-subtle)",
            color: "var(--text-secondary)",
            border: "none",
            borderRadius: "var(--radius-sm)",
            fontSize: "var(--font-size-lg)",
            cursor: "pointer",
            fontWeight: "var(--font-weight-semibold)",
          }}
        >
          +
        </button>
        <button
          onClick={() => {
            const { zoomToFit } = useWaveformStore.getState();
            zoomToFit();
          }}
          style={{
            padding: "var(--space-2) var(--space-4)",
            background: "var(--border-subtle)",
            color: "var(--text-secondary)",
            border: "none",
            borderRadius: "var(--radius-sm)",
            fontSize: "var(--font-size-sm)",
            cursor: "pointer",
          }}
        >
          Reset
        </button>
        </div>
      </div>
    </div>
  );
}
