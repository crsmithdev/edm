import { useAudioStore, useStructureStore, useTempoStore, useUIStore, useTrackStore } from "@/stores";

/**
 * Editing row - three action buttons
 */
export function EditingControls() {
  const { currentTime } = useAudioStore();
  const { addBoundary } = useStructureStore();
  const { setDownbeat } = useTempoStore();
  const { quantizeEnabled, toggleQuantize, showStatus } = useUIStore();
  const { currentTrack } = useTrackStore();

  const handleSetDownbeat = () => {
    setDownbeat(currentTime);
    showStatus(`Downbeat set to ${currentTime.toFixed(2)}s`);
  };

  const handleAddBoundary = () => {
    addBoundary(currentTime);
    showStatus(`Added boundary at ${currentTime.toFixed(2)}s`);
  };

  const actionBtnStyle = {
    padding: "12px 20px",
    fontSize: "14px",
    fontWeight: 600,
    background: "#5B7CFF",
    color: "#FFFFFF",
    border: "none",
    borderRadius: "8px",
    cursor: "pointer",
    transition: "all 0.2s",
    whiteSpace: "nowrap" as const,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    height: "44px",
  };

  const actionBtnInactiveStyle = {
    ...actionBtnStyle,
    background: "#2A2F4C",
    color: "#9CA3AF",
  };

  const actionBtnActiveStyle = {
    ...actionBtnStyle,
    background: "#5B7CFF",
    color: "#FFFFFF",
  };

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(3, 1fr)",
        gap: "12px",
      }}
    >
      <button
        onClick={handleAddBoundary}
        disabled={!currentTrack}
        style={{
          ...actionBtnStyle,
          opacity: currentTrack ? 1 : 0.6,
          cursor: currentTrack ? "pointer" : "not-allowed",
        }}
      >
        + Boundary
      </button>
      <button
        onClick={handleSetDownbeat}
        disabled={!currentTrack}
        style={{
          ...actionBtnStyle,
          opacity: currentTrack ? 1 : 0.6,
          cursor: currentTrack ? "pointer" : "not-allowed",
        }}
      >
        Downbeat
      </button>
      <button
        onClick={toggleQuantize}
        style={quantizeEnabled ? actionBtnActiveStyle : actionBtnInactiveStyle}
      >
        Quantize
      </button>
    </div>
  );
}
