import { useWaveformInteraction } from "@/hooks/useWaveformInteraction";
import { useWaveformStore, useUIStore, useTrackStore, useTempoStore } from "@/stores";
import { WaveformCanvas } from "./WaveformCanvas";
import { BeatGrid } from "./BeatGrid";
import { Playhead } from "./Playhead";
import { BoundaryMarkers } from "./BoundaryMarkers";
import { RegionOverlays } from "./RegionOverlays";

/**
 * Container for waveform visualization and overlays
 */
export function WaveformContainer() {
  const { zoom, duration } = useWaveformStore();
  const { isDragging } = useUIStore();
  const { currentTrack } = useTrackStore();
  const { trackBPM } = useTempoStore();
  const { handleMouseDown, handleMouseMove, handleMouseUp, handleWheel } =
    useWaveformInteraction();

  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div
      style={{
        background: "var(--bg-secondary)",
        padding: "var(--space-6)",
        borderRadius: "var(--radius-xl)",
        marginBottom: "var(--space-5)",
        border: "1px solid var(--border-primary)",
        boxShadow: "var(--shadow-md)",
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
          gap: "var(--space-2)",
          alignItems: "center",
          marginTop: "var(--space-4)",
          justifyContent: "space-between",
        }}
      >
        {/* Track metadata on left */}
        <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-1)" }}>
          <div
            style={{
              fontSize: "var(--font-size-base)",
              fontWeight: "var(--font-weight-semibold)",
              color: "var(--text-secondary)",
            }}
          >
            {currentTrack || "No track loaded"}
          </div>
          {currentTrack && (
            <div
              style={{
                fontSize: "var(--font-size-sm)",
                color: "var(--text-tertiary)",
                display: "flex",
                gap: "var(--space-3)",
              }}
            >
              <span>{trackBPM ? `${trackBPM} BPM` : "BPM not set"}</span>
              <span>•</span>
              <span>{formatDuration(duration)}</span>
            </div>
          )}
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
