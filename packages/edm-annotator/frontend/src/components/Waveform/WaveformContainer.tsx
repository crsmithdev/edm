import { useWaveformInteraction } from "@/hooks/useWaveformInteraction";
import { useWaveformStore, useUIStore } from "@/stores";
import { WaveformCanvas } from "./WaveformCanvas";
import { BeatGrid } from "./BeatGrid";
import { Playhead } from "./Playhead";
import { BoundaryMarkers } from "./BoundaryMarkers";
import { RegionOverlays } from "./RegionOverlays";

/**
 * Container for waveform visualization and overlays
 */
export function WaveformContainer() {
  const { zoom } = useWaveformStore();
  const { isDragging } = useUIStore();
  const { handleMouseDown, handleMouseMove, handleMouseUp, handleWheel } =
    useWaveformInteraction();

  return (
    <div
      style={{
        background: "#1E2139",
        padding: "24px",
        borderRadius: "14px",
        marginBottom: "20px",
        border: "1px solid rgba(91, 124, 255, 0.1)",
        boxShadow: "0 4px 6px rgba(0, 0, 0, 0.3)",
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
          height: "300px",
          background: "#151828",
          border: "1px solid #2A2F4C",
          borderRadius: "10px",
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

      {/* Zoom Controls */}
      <div
        style={{
          display: "flex",
          gap: "10px",
          alignItems: "center",
          marginTop: "16px",
          justifyContent: "flex-end",
        }}
      >
        <button
          onClick={() => zoom(-1)}
          style={{
            padding: "8px 16px",
            background: "#2A2F4C",
            color: "#E5E7EB",
            border: "none",
            borderRadius: "6px",
            fontSize: "16px",
            cursor: "pointer",
            fontWeight: 600,
          }}
        >
          −
        </button>
        <button
          onClick={() => zoom(1)}
          style={{
            padding: "8px 16px",
            background: "#2A2F4C",
            color: "#E5E7EB",
            border: "none",
            borderRadius: "6px",
            fontSize: "16px",
            cursor: "pointer",
            fontWeight: 600,
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
            padding: "8px 16px",
            background: "#2A2F4C",
            color: "#E5E7EB",
            border: "none",
            borderRadius: "6px",
            fontSize: "13px",
            cursor: "pointer",
          }}
        >
          Reset
        </button>
      </div>
    </div>
  );
}
