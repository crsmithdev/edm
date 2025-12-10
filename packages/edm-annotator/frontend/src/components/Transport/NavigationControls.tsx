import { useAudioStore, useTempoStore, useUIStore } from "@/stores";
import { getBarDuration } from "@/utils/barCalculations";

/**
 * Navigation controls for jumping by bars/beats
 */
export function NavigationControls() {
  const { currentTime, seek } = useAudioStore();
  const { trackBPM } = useTempoStore();
  const { jumpMode, toggleJumpMode } = useUIStore();

  const barDuration = getBarDuration(trackBPM);

  const jumpBars = (count: number) => {
    const newTime = currentTime + barDuration * count;
    seek(Math.max(0, newTime));
  };

  const navBtnStyle = {
    padding: "12px 16px",
    fontSize: "13px",
    background: "#151828",
    color: "#E5E7EB",
    border: "1px solid #2A2F4C",
    borderRadius: "8px",
    fontWeight: 600,
    cursor: "pointer",
    transition: "all 0.2s",
    whiteSpace: "nowrap" as const,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    height: "44px",
    minWidth: "70px",
  };

  const actionBtnStyle = {
    padding: "12px 20px",
    fontSize: "14px",
    fontWeight: 600,
    background: "#2A2F4C",
    color: "#9CA3AF",
    border: "none",
    borderRadius: "8px",
    cursor: "pointer",
    transition: "all 0.2s",
    whiteSpace: "nowrap" as const,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    height: "44px",
    minWidth: "80px",
  };

  return (
    <div
      style={{
        display: "flex",
        gap: "10px",
        alignItems: "stretch",
        justifyContent: "center",
        paddingBottom: "20px",
        marginBottom: "20px",
        borderBottom: "1px solid #2A2F4C",
      }}
    >
      <button onClick={() => jumpBars(-16)} style={navBtnStyle}>
        ◀ 16
      </button>
      <button onClick={() => jumpBars(-8)} style={navBtnStyle}>
        ◀ 8
      </button>
      <button onClick={() => jumpBars(-4)} style={navBtnStyle}>
        ◀ 4
      </button>
      <button onClick={() => jumpBars(-2)} style={navBtnStyle}>
        ◀ 2
      </button>
      <button onClick={() => jumpBars(-1)} style={navBtnStyle}>
        ◀ 1
      </button>
      <button
        onClick={toggleJumpMode}
        style={actionBtnStyle}
      >
        {jumpMode === "bars" ? "Beats" : "Bars"}
      </button>
      <button onClick={() => jumpBars(1)} style={navBtnStyle}>
        1 ▶
      </button>
      <button onClick={() => jumpBars(2)} style={navBtnStyle}>
        2 ▶
      </button>
      <button onClick={() => jumpBars(4)} style={navBtnStyle}>
        4 ▶
      </button>
      <button onClick={() => jumpBars(8)} style={navBtnStyle}>
        8 ▶
      </button>
      <button onClick={() => jumpBars(16)} style={navBtnStyle}>
        16 ▶
      </button>
    </div>
  );
}
