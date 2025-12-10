import { useMemo } from "react";
import { useWaveformStore } from "@/stores";

/**
 * SVG-based stacked area chart waveform visualization
 * Displays bass/mids/highs as cumulative stacked layers
 * Based on best practices from audio visualization research
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

  // Generate stacked area paths
  const { bassPath, midsPath, highsPath } = useMemo(() => {
    if (visibleSamples.indices.length === 0) {
      return { bassPath: "", midsPath: "", highsPath: "" };
    }

    const viewportDuration = viewportEnd - viewportStart;
    if (viewportDuration <= 0) {
      return { bassPath: "", midsPath: "", highsPath: "" };
    }

    const width = 100;
    const height = 100;

    // Calculate cumulative heights for each sample
    const cumulativeData = visibleSamples.indices.map((idx) => {
      const bass = Math.abs(waveformBass[idx] || 0);
      const mids = Math.abs(waveformMids[idx] || 0);
      const highs = Math.abs(waveformHighs[idx] || 0);
      const time = waveformTimes[idx];
      const x = ((time - viewportStart) / viewportDuration) * width;

      return {
        x,
        bass,
        bassTop: bass,
        midsTop: bass + mids,
        highsTop: bass + mids + highs,
      };
    });

    // Find max cumulative height for scaling
    const maxCumulative = Math.max(
      ...cumulativeData.map((d) => d.highsTop),
      0.001
    );
    const scale = (height * 0.9) / maxCumulative;

    // Generate bass area (bottom layer)
    const bassTopPoints = cumulativeData.map((d) => {
      const y = height - d.bassTop * scale;
      return `${d.x},${y}`;
    });

    const bassPath =
      bassTopPoints.length > 0
        ? `M0,${height} L${bassTopPoints.join(" L")} L${width},${height} Z`
        : "";

    // Generate mids area (middle layer)
    const midsTopPoints = cumulativeData.map((d) => {
      const y = height - d.midsTop * scale;
      return `${d.x},${y}`;
    });

    const midsBottomPoints = cumulativeData
      .map((d) => {
        const y = height - d.bassTop * scale;
        return `${d.x},${y}`;
      })
      .reverse();

    const midsPath =
      midsTopPoints.length > 0 && midsBottomPoints.length > 0
        ? `M${midsBottomPoints[midsBottomPoints.length - 1]} L${midsTopPoints.join(" L")} L${midsBottomPoints.join(" L")} Z`
        : "";

    // Generate highs area (top layer)
    const highsTopPoints = cumulativeData.map((d) => {
      const y = height - d.highsTop * scale;
      return `${d.x},${y}`;
    });

    const highsBottomPoints = cumulativeData
      .map((d) => {
        const y = height - d.midsTop * scale;
        return `${d.x},${y}`;
      })
      .reverse();

    const highsPath =
      highsTopPoints.length > 0 && highsBottomPoints.length > 0
        ? `M${highsBottomPoints[highsBottomPoints.length - 1]} L${highsTopPoints.join(" L")} L${highsBottomPoints.join(" L")} Z`
        : "";

    return {
      bassPath,
      midsPath,
      highsPath,
    };
  }, [
    visibleSamples,
    waveformBass,
    waveformMids,
    waveformHighs,
    waveformTimes,
    viewportStart,
    viewportEnd,
  ]);

  return (
    <svg
      viewBox="0 0 100 100"
      preserveAspectRatio="none"
      style={{
        width: "100%",
        height: "100%",
        background: "#0a0a12",
      }}
    >
      {/* Bass layer (bottom) - cyan */}
      <path
        d={bassPath}
        fill="rgba(0, 229, 204, 0.8)"
        stroke="none"
      />

      {/* Mids layer (middle) - purple */}
      <path
        d={midsPath}
        fill="rgba(123, 106, 255, 0.8)"
        stroke="none"
      />

      {/* Highs layer (top) - pink */}
      <path
        d={highsPath}
        fill="rgba(255, 107, 181, 0.8)"
        stroke="none"
      />

      {/* Center baseline for reference */}
      <line
        x1="0"
        y1="50"
        x2="100"
        y2="50"
        stroke="rgba(255, 255, 255, 0.1)"
        strokeWidth="0.3"
        strokeDasharray="2,2"
      />
    </svg>
  );
}
