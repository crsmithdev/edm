import { useMemo } from "react";
import { useWaveformStore, useTempoStore } from "@/stores";
import { getBarDuration, getBeatDuration } from "@/utils/barCalculations";

/**
 * Beat grid overlay showing bars and beats
 */
export function BeatGrid() {
  const { viewportStart, viewportEnd, duration } = useWaveformStore();
  const { trackBPM, trackDownbeat } = useTempoStore();

  const gridLines = useMemo(() => {
    if (trackBPM === 0 || duration === 0) return [];

    const lines: Array<{
      time: number;
      type: "downbeat" | "bar" | "beat";
      bar?: number;
    }> = [];

    const barDuration = getBarDuration(trackBPM);
    const beatDuration = getBeatDuration(trackBPM);
    const viewportDuration = viewportEnd - viewportStart;

    // Calculate zoom level and adjust granularity
    const zoomRatio = viewportDuration / duration;

    // Determine bar and beat granularity based on zoom level
    let barInterval = 1; // Show every bar
    let showBeats = false;

    if (zoomRatio > 0.5) {
      // Very zoomed out (>50% of track visible) - show every 16 bars, no beats
      barInterval = 16;
      showBeats = false;
    } else if (zoomRatio > 0.3) {
      // Zoomed out (30-50% visible) - show every 8 bars, no beats
      barInterval = 8;
      showBeats = false;
    } else if (zoomRatio > 0.2) {
      // Medium zoom (20-30% visible) - show every 4 bars, no beats
      barInterval = 4;
      showBeats = false;
    } else if (zoomRatio > 0.15) {
      // Medium-close zoom (15-20% visible) - show every 2 bars, no beats
      barInterval = 2;
      showBeats = false;
    } else if (zoomRatio > 0.1) {
      // Zoomed in (10-15% visible) - show all bars, no beats
      barInterval = 1;
      showBeats = false;
    } else {
      // Very zoomed in (<10% visible) - show all bars and beats
      barInterval = 1;
      showBeats = true;
    }

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

      // Skip if outside viewport or is the downbeat
      if (barTime < viewportStart || barTime > viewportEnd || barTime === trackDownbeat) {
        continue;
      }

      const barNumber = bar + 1;

      // Only add bar if it matches the interval
      if (bar % barInterval === 0) {
        lines.push({ time: barTime, type: "bar", bar: barNumber });

        // Add beat lines within this bar (only when zoomed in enough)
        if (showBeats) {
          for (let beat = 1; beat < 4; beat++) {
            const beatTime = barTime + beat * beatDuration;
            if (beatTime >= viewportStart && beatTime <= viewportEnd) {
              lines.push({ time: beatTime, type: "beat" });
            }
          }
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
                    : "#444",
              opacity: line.type === "downbeat" ? 0.8 : line.type === "bar" ? 0.6 : 0.25,
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
