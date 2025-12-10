import { ChevronLeft, ChevronRight } from "lucide-react";
import { useAudioStore, useTempoStore, useUIStore } from "@/stores";
import { getBarDuration } from "@/utils/barCalculations";
import { Button } from "@/components/UI";

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

  return (
    <div
      style={{
        display: "flex",
        gap: "var(--space-2)",
        alignItems: "stretch",
        justifyContent: "center",
        paddingBottom: "var(--space-5)",
        marginBottom: "var(--space-5)",
        borderBottom: "1px solid var(--border-subtle)",
      }}
    >
      <Button onClick={() => jumpBars(-16)} variant="ghost" size="sm" style={{ minWidth: "70px" }}>
        <ChevronLeft size={14} /> 16
      </Button>
      <Button onClick={() => jumpBars(-8)} variant="ghost" size="sm" style={{ minWidth: "70px" }}>
        <ChevronLeft size={14} /> 8
      </Button>
      <Button onClick={() => jumpBars(-4)} variant="ghost" size="sm" style={{ minWidth: "70px" }}>
        <ChevronLeft size={14} /> 4
      </Button>
      <Button onClick={() => jumpBars(-2)} variant="ghost" size="sm" style={{ minWidth: "70px" }}>
        <ChevronLeft size={14} /> 2
      </Button>
      <Button onClick={() => jumpBars(-1)} variant="ghost" size="sm" style={{ minWidth: "70px" }}>
        <ChevronLeft size={14} /> 1
      </Button>
      <Button
        onClick={toggleJumpMode}
        variant="secondary"
        size="sm"
        style={{ minWidth: "80px" }}
      >
        {jumpMode === "bars" ? "Beats" : "Bars"}
      </Button>
      <Button onClick={() => jumpBars(1)} variant="ghost" size="sm" style={{ minWidth: "70px" }}>
        1 <ChevronRight size={14} />
      </Button>
      <Button onClick={() => jumpBars(2)} variant="ghost" size="sm" style={{ minWidth: "70px" }}>
        2 <ChevronRight size={14} />
      </Button>
      <Button onClick={() => jumpBars(4)} variant="ghost" size="sm" style={{ minWidth: "70px" }}>
        4 <ChevronRight size={14} />
      </Button>
      <Button onClick={() => jumpBars(8)} variant="ghost" size="sm" style={{ minWidth: "70px" }}>
        8 <ChevronRight size={14} />
      </Button>
      <Button onClick={() => jumpBars(16)} variant="ghost" size="sm" style={{ minWidth: "70px" }}>
        16 <ChevronRight size={14} />
      </Button>
    </div>
  );
}
