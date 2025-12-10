import { useAudioStore, useTempoStore, useUIStore } from "@/stores";
import { getBarDuration } from "@/utils/barCalculations";

/**
 * Navigation controls for jumping by bars/beats
 */
export function NavigationControls() {
  const { currentTime, seek } = useAudioStore();
  const { trackBPM, timeToBar } = useTempoStore();
  const { jumpMode, toggleJumpMode } = useUIStore();

  const currentBar = timeToBar(currentTime);
  const barDuration = getBarDuration(trackBPM);

  const jumpBars = (count: number) => {
    const newTime = currentTime + barDuration * count;
    seek(Math.max(0, newTime));
  };

  return (
    <div
      style={{
        display: "flex",
        gap: "8px",
        alignItems: "center",
        padding: "16px",
        background: "#1E2139",
        borderRadius: "10px",
        border: "1px solid rgba(91, 124, 255, 0.1)",
      }}
    >
      <div style={{ marginRight: "8px", color: "#9CA3AF" }}>
        <strong style={{ color: "#5B7CFF" }}>Bar:</strong> {currentBar}
      </div>

      <button
        onClick={() => jumpBars(-8)}
        style={{
          padding: "8px 12px",
          background: "#2A2F4C",
          color: "#E5E7EB",
          border: "none",
          borderRadius: "6px",
          fontSize: "13px",
          cursor: "pointer",
        }}
      >
        -8
      </button>

      <button
        onClick={() => jumpBars(-4)}
        style={{
          padding: "8px 12px",
          background: "#2A2F4C",
          color: "#E5E7EB",
          border: "none",
          borderRadius: "6px",
          fontSize: "13px",
          cursor: "pointer",
        }}
      >
        -4
      </button>

      <button
        onClick={() => jumpBars(-1)}
        style={{
          padding: "8px 12px",
          background: "#2A2F4C",
          color: "#E5E7EB",
          border: "none",
          borderRadius: "6px",
          fontSize: "13px",
          cursor: "pointer",
        }}
      >
        -1
      </button>

      <button
        onClick={() => jumpBars(1)}
        style={{
          padding: "8px 12px",
          background: "#2A2F4C",
          color: "#E5E7EB",
          border: "none",
          borderRadius: "6px",
          fontSize: "13px",
          cursor: "pointer",
        }}
      >
        +1
      </button>

      <button
        onClick={() => jumpBars(4)}
        style={{
          padding: "8px 12px",
          background: "#2A2F4C",
          color: "#E5E7EB",
          border: "none",
          borderRadius: "6px",
          fontSize: "13px",
          cursor: "pointer",
        }}
      >
        +4
      </button>

      <button
        onClick={() => jumpBars(8)}
        style={{
          padding: "8px 12px",
          background: "#2A2F4C",
          color: "#E5E7EB",
          border: "none",
          borderRadius: "6px",
          fontSize: "13px",
          cursor: "pointer",
        }}
      >
        +8
      </button>

      <button
        onClick={toggleJumpMode}
        style={{
          marginLeft: "auto",
          padding: "8px 16px",
          background: "#5B7CFF",
          color: "#FFFFFF",
          border: "none",
          borderRadius: "6px",
          fontSize: "13px",
          cursor: "pointer",
          minWidth: "70px",
        }}
      >
        {jumpMode === "bars" ? "Bars" : "Beats"}
      </button>
    </div>
  );
}
