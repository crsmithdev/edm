import { useState } from "react";
import { useAudioStore, useStructureStore, useTempoStore, useUIStore, useTrackStore } from "@/stores";
import { trackService } from "@/services/api";

/**
 * Editing controls for annotations
 */
export function EditingControls() {
  const { currentTime } = useAudioStore();
  const { addBoundary, boundaries, regions } = useStructureStore();
  const { trackBPM, trackDownbeat, setBPM, setDownbeat, tapTempo, resetTapTempo } = useTempoStore();
  const { quantizeEnabled, toggleQuantize, showStatus } = useUIStore();
  const { currentTrack } = useTrackStore();

  const [bpmInput, setBpmInput] = useState(trackBPM.toString());
  const [isSaving, setIsSaving] = useState(false);

  const handleBPMChange = (value: string) => {
    setBpmInput(value);
    const bpm = parseFloat(value);
    if (!isNaN(bpm) && bpm > 0) {
      setBPM(bpm);
    }
  };

  const handleSetDownbeat = () => {
    setDownbeat(currentTime);
    showStatus(`Downbeat set to ${currentTime.toFixed(2)}s`);
  };

  const handleAddBoundary = () => {
    addBoundary(currentTime);
    showStatus(`Added boundary at ${currentTime.toFixed(2)}s`);
  };

  const handleTapTempo = () => {
    tapTempo();
    // Update input to reflect new BPM
    setTimeout(() => setBpmInput(trackBPM.toFixed(1)), 100);
  };

  const handleSave = async () => {
    if (!currentTrack) {
      showStatus("No track loaded");
      return;
    }

    setIsSaving(true);
    try {
      await trackService.saveAnnotation({
        filename: currentTrack,
        bpm: trackBPM,
        downbeat: trackDownbeat,
        boundaries: regions.map(r => ({ time: r.start, label: r.label })),
      });
      showStatus("Annotation saved successfully");
    } catch (error) {
      showStatus(`Error saving: ${error instanceof Error ? error.message : "Unknown error"}`);
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "12px",
        padding: "16px",
        background: "#1E2139",
        borderRadius: "10px",
        border: "1px solid rgba(91, 124, 255, 0.1)",
      }}
    >
      {/* BPM Controls */}
      <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
        <label htmlFor="bpm-input" style={{ color: "#9CA3AF", minWidth: "60px" }}>BPM:</label>
        <input
          id="bpm-input"
          type="number"
          value={bpmInput}
          onChange={(e) => handleBPMChange(e.target.value)}
          style={{
            padding: "8px 12px",
            background: "#151828",
            border: "1px solid #2A2F4C",
            borderRadius: "6px",
            color: "#E5E7EB",
            fontSize: "14px",
            width: "100px",
          }}
        />
        <button
          onClick={handleTapTempo}
          onDoubleClick={resetTapTempo}
          style={{
            padding: "8px 16px",
            background: "#5B7CFF",
            color: "#FFFFFF",
            border: "none",
            borderRadius: "6px",
            fontSize: "13px",
            cursor: "pointer",
          }}
          title="Click to tap tempo, double-click to reset"
        >
          Tap
        </button>
      </div>

      {/* Action Buttons */}
      <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
        <button
          onClick={handleAddBoundary}
          disabled={!currentTrack}
          style={{
            padding: "8px 16px",
            background: "#5B7CFF",
            color: "#FFFFFF",
            border: "none",
            borderRadius: "6px",
            fontSize: "13px",
            cursor: currentTrack ? "pointer" : "not-allowed",
            opacity: currentTrack ? 1 : 0.5,
          }}
        >
          Add Boundary (B)
        </button>

        <button
          onClick={handleSetDownbeat}
          disabled={!currentTrack}
          style={{
            padding: "8px 16px",
            background: "#FFB800",
            color: "#0F1419",
            border: "none",
            borderRadius: "6px",
            fontSize: "13px",
            fontWeight: 600,
            cursor: currentTrack ? "pointer" : "not-allowed",
            opacity: currentTrack ? 1 : 0.5,
          }}
        >
          Set Downbeat (D)
        </button>

        <button
          onClick={toggleQuantize}
          style={{
            padding: "8px 16px",
            background: quantizeEnabled ? "#00E6B8" : "#2A2F4C",
            color: quantizeEnabled ? "#0F1419" : "#9CA3AF",
            border: "none",
            borderRadius: "6px",
            fontSize: "13px",
            fontWeight: 600,
            cursor: "pointer",
          }}
        >
          Quantize (Q): {quantizeEnabled ? "ON" : "OFF"}
        </button>

        <button
          onClick={handleSave}
          disabled={!currentTrack || boundaries.length === 0 || isSaving}
          style={{
            padding: "8px 16px",
            background: "#00E6B8",
            color: "#0F1419",
            border: "none",
            borderRadius: "6px",
            fontSize: "13px",
            fontWeight: 600,
            cursor: currentTrack && boundaries.length > 0 && !isSaving ? "pointer" : "not-allowed",
            opacity: currentTrack && boundaries.length > 0 && !isSaving ? 1 : 0.5,
            marginLeft: "auto",
          }}
        >
          {isSaving ? "Saving..." : "Save Annotation"}
        </button>
      </div>

      {/* Stats */}
      <div style={{ fontSize: "12px", color: "#6B7280" }}>
        Boundaries: {boundaries.length} | Regions: {regions.length}
      </div>
    </div>
  );
}
