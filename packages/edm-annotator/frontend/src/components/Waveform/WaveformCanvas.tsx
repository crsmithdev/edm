import { useMemo } from "react";
import { useWaveformStore } from "@/stores";

/**
 * SVG-based stacked area chart waveform visualization
 * Displays bass/mids/highs as cumulative stacked layers, mirrored along x-axis
 * Bass (inner), mids (middle), highs (outer) extend symmetrically from center
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

    const center = height / 2;
    const halfScale = scale / 2;

    // Generate bass area (innermost layer, mirrored)
    const bassTopPoints = cumulativeData.map((d) => {
      const y = center - d.bassTop * halfScale;
      return `${d.x},${y}`;
    });

    const bassBottomPoints = cumulativeData
      .map((d) => {
        const y = center + d.bassTop * halfScale;
        return `${d.x},${y}`;
      })
      .reverse();

    const bassPath =
      bassTopPoints.length > 0 && bassBottomPoints.length > 0
        ? `M${bassTopPoints.join(" L")} L${bassBottomPoints.join(" L")} Z`
        : "";

    // Generate mids area (middle layer, mirrored)
    const midsTopPoints = cumulativeData.map((d) => {
      const y = center - d.midsTop * halfScale;
      return `${d.x},${y}`;
    });

    const midsTopBasePoints = cumulativeData
      .map((d) => {
        const y = center - d.bassTop * halfScale;
        return `${d.x},${y}`;
      })
      .reverse();

    const midsBottomPoints = cumulativeData
      .map((d) => {
        const y = center + d.midsTop * halfScale;
        return `${d.x},${y}`;
      })
      .reverse();

    const midsBottomBasePoints = cumulativeData.map((d) => {
      const y = center + d.bassTop * halfScale;
      return `${d.x},${y}`;
    });

    const midsPath =
      midsTopPoints.length > 0
        ? `M${midsTopPoints.join(" L")} L${midsTopBasePoints.join(" L")} M${midsBottomBasePoints.join(" L")} L${midsBottomPoints.join(" L")} Z`
        : "";

    // Generate highs area (outermost layer, mirrored)
    const highsTopPoints = cumulativeData.map((d) => {
      const y = center - d.highsTop * halfScale;
      return `${d.x},${y}`;
    });

    const highsTopBasePoints = cumulativeData
      .map((d) => {
        const y = center - d.midsTop * halfScale;
        return `${d.x},${y}`;
      })
      .reverse();

    const highsBottomPoints = cumulativeData
      .map((d) => {
        const y = center + d.highsTop * halfScale;
        return `${d.x},${y}`;
      })
      .reverse();

    const highsBottomBasePoints = cumulativeData.map((d) => {
      const y = center + d.midsTop * halfScale;
      return `${d.x},${y}`;
    });

    const highsPath =
      highsTopPoints.length > 0
        ? `M${highsTopPoints.join(" L")} L${highsTopBasePoints.join(" L")} M${highsBottomBasePoints.join(" L")} L${highsBottomPoints.join(" L")} Z`
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
      {/* Bass layer (innermost, mirrored) - cyan */}
      <path
        d={bassPath}
        fill="rgba(0, 229, 204, 0.8)"
        stroke="none"
      />

      {/* Mids layer (middle, mirrored) - purple */}
      <path
        d={midsPath}
        fill="rgba(123, 106, 255, 0.8)"
        stroke="none"
      />

      {/* Highs layer (outermost, mirrored) - pink */}
      <path
        d={highsPath}
        fill="rgba(255, 107, 181, 0.8)"
        stroke="none"
      />

      {/* Center baseline */}
      <line
        x1="0"
        y1="50"
        x2="100"
        y2="50"
        stroke="rgba(255, 255, 255, 0.15)"
        strokeWidth="0.2"
      />
    </svg>
  );
}
