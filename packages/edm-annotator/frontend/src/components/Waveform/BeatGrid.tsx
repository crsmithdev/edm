import { useMemo } from "react";
import { useWaveformStore, useTempoStore } from "@/stores";
import { getBarDuration, getBeatDuration } from "@/utils/barCalculations";

/**
 * Beat grid overlay showing bars and beats
 */
export function BeatGrid() {
  const { viewportStart, viewportEnd, duration } = useWaveformStore();
  const { trackBPM, trackDownbeat, timeToBar } = useTempoStore();

  const gridLines = useMemo(() => {
    if (trackBPM === 0 || duration === 0) return [];

    const lines: Array<{
      time: number;
      type: "downbeat" | "bar" | "beat";
      bar?: number;
    }> = [];

    const barDuration = getBarDuration(trackBPM);
    const beatDuration = getBeatDuration(trackBPM);

    // Calculate first bar in viewport
    const firstBar = Math.floor((viewportStart - trackDownbeat) / barDuration);
    const lastBar = Math.ceil((viewportEnd - trackDownbeat) / barDuration);

    // Add downbeat if visible
    if (
      trackDownbeat >= viewportStart &&
      trackDownbeat <= viewportEnd
    ) {
      lines.push({ time: trackDownbeat, type: "downbeat", bar: 1 });
    }

    // Add bar lines
    for (let bar = firstBar; bar <= lastBar; bar++) {
      const barTime = trackDownbeat + bar * barDuration;
      if (barTime >= viewportStart && barTime <= viewportEnd && barTime !== trackDownbeat) {
        const barNumber = bar + 1;
        lines.push({ time: barTime, type: "bar", bar: barNumber });
      }

      // Add beat lines within each bar
      for (let beat = 1; beat < 4; beat++) {
        const beatTime = barTime + beat * beatDuration;
        if (beatTime >= viewportStart && beatTime <= viewportEnd) {
          lines.push({ time: beatTime, type: "beat" });
        }
      }
    }

    return lines;
  }, [viewportStart, viewportEnd, duration, trackBPM, trackDownbeat]);

  const viewportDuration = viewportEnd - viewportStart;

  return (
    <>
      {gridLines.map((line, idx) => {
        const xPercent = ((line.time - viewportStart) / viewportDuration) * 100;

        return (
          <div
            key={idx}
            style={{
              position: "absolute",
              left: `${xPercent}%`,
              top: 0,
              height: "100%",
              width:
                line.type === "downbeat"
                  ? "3px"
                  : line.type === "bar"
                    ? "2px"
                    : "1px",
              background:
                line.type === "downbeat"
                  ? "#f44336"
                  : line.type === "bar"
                    ? "#ff9800"
                    : "#666",
              opacity: line.type === "downbeat" ? 0.8 : line.type === "bar" ? 0.6 : 0.4,
              pointerEvents: "none",
            }}
          >
            {line.type !== "beat" && line.bar && (
              <div
                style={{
                  position: "absolute",
                  top: "2px",
                  left: "50%",
                  transform: "translateX(-50%)",
                  background: "rgba(0,0,0,0.7)",
                  padding: "2px 4px",
                  borderRadius: "2px",
                  fontSize: "10px",
                  color: line.type === "downbeat" ? "#f44336" : "#ff9800",
                  whiteSpace: "nowrap",
                }}
              >
                {line.bar}
              </div>
            )}
          </div>
        );
      })}
    </>
  );
}
