import { ChevronLeft, ChevronRight, SkipBack, SkipForward } from "lucide-react";
import { useAudioStore, useTempoStore, useUIStore, useStructureStore, useWaveformStore } from "@/stores";
import { getBarDuration, getBeatDuration } from "@/utils/tempo";
import { Button, Tooltip } from "@/components/UI";

/**
 * Navigation controls for boundary/track navigation and jumping by bars/beats
 */
export function NavigationControls() {
  const { seek, currentTime } = useAudioStore();
  const { trackBPM } = useTempoStore();
  const { jumpMode, toggleJumpMode } = useUIStore();
  const { getNextBoundary, getPreviousBoundary } = useStructureStore();
  const { duration } = useWaveformStore();

  const barDuration = getBarDuration(trackBPM);
  const beatDuration = getBeatDuration(trackBPM);

  const jump = (count: number) => {
    // Get current time from store to avoid stale closure
    const currentTime = useAudioStore.getState().currentTime;
    const duration = jumpMode === "beats" ? beatDuration : barDuration;
    const newTime = currentTime + duration * count;
    seek(Math.max(0, newTime));
  };

  const handlePreviousBoundary = () => {
    const previous = getPreviousBoundary(currentTime);
    if (previous !== null) {
      seek(previous);
    }
  };

  const handleNextBoundary = () => {
    const next = getNextBoundary(currentTime);
    if (next !== null) {
      seek(next);
    }
  };

  const handleStart = () => {
    seek(0);
  };

  const handleEnd = () => {
    seek(duration);
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "var(--space-3)",
      }}
    >
      {/* Boundary navigation row */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(4, 1fr)",
          gap: "var(--space-3)",
        }}
      >
        <Tooltip content="Seek to start" shortcut="Home">
          <Button
            onClick={handleStart}
            variant="secondary"
            icon={<SkipBack size={16} />}
          >
            Start
          </Button>
        </Tooltip>
        <Tooltip content="Previous boundary" shortcut="↑">
          <Button
            onClick={handlePreviousBoundary}
            variant="secondary"
            icon={<ChevronLeft size={16} />}
          >
            Boundary
          </Button>
        </Tooltip>
        <Tooltip content="Next boundary" shortcut="↓">
          <Button
            onClick={handleNextBoundary}
            variant="secondary"
            icon={<ChevronRight size={16} />}
          >
            Boundary
          </Button>
        </Tooltip>
        <Tooltip content="Seek to end" shortcut="End">
          <Button
            onClick={handleEnd}
            variant="secondary"
            icon={<SkipForward size={16} />}
          >
            End
          </Button>
        </Tooltip>
      </div>

      {/* Bar/beat navigation */}
      <div
        style={{
          display: "flex",
          gap: "var(--space-2)",
          alignItems: "stretch",
          justifyContent: "center",
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
    </div>
  );
}
