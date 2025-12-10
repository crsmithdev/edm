import { useMemo } from "react";
import { useWaveformStore } from "@/stores";

/**
 * SVG-based 3-band waveform visualization
 */
export function WaveformCanvas() {
  const {
    waveformBass,
    waveformMids,
    waveformHighs,
    waveformTimes,
    viewportStart,
    viewportEnd,
  } = useWaveformStore();

  // Calculate visible sample range
  const visibleSamples = useMemo(() => {
    if (waveformTimes.length === 0) return { start: 0, end: 0, indices: [] };

    const startIdx = waveformTimes.findIndex((t) => t >= viewportStart);
    const endIdx = waveformTimes.findIndex((t) => t >= viewportEnd);

    const start = startIdx === -1 ? 0 : startIdx;
    const end = endIdx === -1 ? waveformTimes.length : endIdx;

    return {
      start,
      end,
      indices: Array.from({ length: end - start }, (_, i) => start + i),
    };
  }, [waveformTimes, viewportStart, viewportEnd]);

  // Generate stacked area path (filled polygon from baseline to top edge)
  const generateStackedPath = (
    bottomSamples: number[],
    topSamples: number[]
  ): string => {
    if (visibleSamples.indices.length === 0) return "";
    if (bottomSamples.length === 0 || topSamples.length === 0) return "";

    const viewportDuration = viewportEnd - viewportStart;
    if (viewportDuration <= 0) return "";

    const width = 100;
    const height = 100;
    const centerY = height / 2;

    // Build top edge (left to right)
    const topPoints = visibleSamples.indices
      .map((idx) => {
        const time = waveformTimes[idx];
        if (time === undefined) return null;
        const x = ((time - viewportStart) / viewportDuration) * width;
        const topValue = topSamples[idx] || 0;
        const y = centerY - (topValue * height) / 2;
        return `${x},${y}`;
      })
      .filter((p): p is string => p !== null);

    // Build bottom edge (right to left)
    const bottomPoints = visibleSamples.indices
      .map((idx) => {
        const time = waveformTimes[idx];
        if (time === undefined) return null;
        const x = ((time - viewportStart) / viewportDuration) * width;
        const bottomValue = bottomSamples[idx] || 0;
        const y = centerY - (bottomValue * height) / 2;
        return `${x},${y}`;
      })
      .filter((p): p is string => p !== null)
      .reverse();

    if (topPoints.length === 0 || bottomPoints.length === 0) return "";

    return `M${topPoints.join(" L")} L${bottomPoints.join(" L")} Z`;
  };

  // Calculate cumulative stacked values
  const stackedMidsPlusBass = useMemo(() => {
    return visibleSamples.indices.map((idx) => {
      const bass = waveformBass[idx] || 0;
      const mids = waveformMids[idx] || 0;
      return bass + mids;
    });
  }, [visibleSamples.indices, waveformBass, waveformMids]);

  const stackedAll = useMemo(() => {
    return visibleSamples.indices.map((idx) => {
      const bass = waveformBass[idx] || 0;
      const mids = waveformMids[idx] || 0;
      const highs = waveformHighs[idx] || 0;
      return bass + mids + highs;
    });
  }, [visibleSamples.indices, waveformBass, waveformMids, waveformHighs]);

  const zeroes = useMemo(
    () => new Array(visibleSamples.indices.length).fill(0),
    [visibleSamples.indices.length]
  );

  const bassOnly = useMemo(
    () => visibleSamples.indices.map((idx) => waveformBass[idx] || 0),
    [visibleSamples.indices, waveformBass]
  );

  return (
    <svg
      viewBox="0 0 100 100"
      preserveAspectRatio="none"
      style={{
        width: "100%",
        height: "100%",
        background: "#151828",
      }}
    >
      {/* Highs layer (top, stacked on bass + mids) */}
      <path
        d={generateStackedPath(stackedMidsPlusBass, stackedAll)}
        fill="rgba(91, 124, 255, 0.6)"
        stroke="none"
      />

      {/* Mids layer (middle, stacked on bass) */}
      <path
        d={generateStackedPath(bassOnly, stackedMidsPlusBass)}
        fill="rgba(0, 230, 184, 0.6)"
        stroke="none"
      />

      {/* Bass layer (bottom) */}
      <path
        d={generateStackedPath(zeroes, bassOnly)}
        fill="rgba(255, 107, 107, 0.6)"
        stroke="none"
      />
    </svg>
  );
}
