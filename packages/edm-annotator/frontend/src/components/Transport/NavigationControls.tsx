import { ChevronLeft, ChevronRight } from "lucide-react";
import { useAudioStore, useTempoStore, useUIStore } from "@/stores";
import { getBarDuration, getBeatDuration } from "@/utils/tempo";
import { Button } from "@/components/UI";

/**
 * Navigation controls for jumping by bars/beats
 */
export function NavigationControls() {
  const { seek } = useAudioStore();
  const { trackBPM } = useTempoStore();
  const { jumpMode, toggleJumpMode } = useUIStore();

  const barDuration = getBarDuration(trackBPM);
  const beatDuration = getBeatDuration(trackBPM);

  const jump = (count: number) => {
    // Get current time from store to avoid stale closure
    const currentTime = useAudioStore.getState().currentTime;
    const duration = jumpMode === "beats" ? beatDuration : barDuration;
    const newTime = currentTime + duration * count;
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
      <Button onClick={() => jump(-16)} variant="ghost" size="sm" style={{ minWidth: "70px" }}>
        <ChevronLeft size={14} /> 16
      </Button>
      <Button onClick={() => jump(-8)} variant="ghost" size="sm" style={{ minWidth: "70px" }}>
        <ChevronLeft size={14} /> 8
      </Button>
      <Button onClick={() => jump(-4)} variant="ghost" size="sm" style={{ minWidth: "70px" }}>
        <ChevronLeft size={14} /> 4
      </Button>
      <Button onClick={() => jump(-2)} variant="ghost" size="sm" style={{ minWidth: "70px" }}>
        <ChevronLeft size={14} /> 2
      </Button>
      <Button onClick={() => jump(-1)} variant="ghost" size="sm" style={{ minWidth: "70px" }}>
        <ChevronLeft size={14} /> 1
      </Button>
      <Button
        onClick={toggleJumpMode}
        variant="primary"
        size="sm"
        style={{ minWidth: "80px" }}
      >
        {jumpMode === "beats" ? "Beats" : "Bars"}
      </Button>
      <Button onClick={() => jump(1)} variant="ghost" size="sm" style={{ minWidth: "70px" }}>
        1 <ChevronRight size={14} />
      </Button>
      <Button onClick={() => jump(2)} variant="ghost" size="sm" style={{ minWidth: "70px" }}>
        2 <ChevronRight size={14} />
      </Button>
      <Button onClick={() => jump(4)} variant="ghost" size="sm" style={{ minWidth: "70px" }}>
        4 <ChevronRight size={14} />
      </Button>
      <Button onClick={() => jump(8)} variant="ghost" size="sm" style={{ minWidth: "70px" }}>
        8 <ChevronRight size={14} />
      </Button>
      <Button onClick={() => jump(16)} variant="ghost" size="sm" style={{ minWidth: "70px" }}>
        16 <ChevronRight size={14} />
      </Button>
    </div>
  );
}
