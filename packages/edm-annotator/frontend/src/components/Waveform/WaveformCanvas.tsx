import { useMemo } from "react";
import { useWaveformStore } from "@/stores";

/**
 * SVG-based 3-band waveform visualization
 * Displays bass/mids/highs in 3 separate horizontal bands, each mirrored vertically
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

  // Generate mirrored waveform path for a single band
  const generateBandPath = (
    samples: number[],
    offsetY: number,
    bandHeight: number
  ): string => {
    if (visibleSamples.indices.length === 0) return "";

    const viewportDuration = viewportEnd - viewportStart;
    if (viewportDuration <= 0) return "";

    const width = 100;
    const center = offsetY + bandHeight / 2;

    // Find max amplitude for scaling
    const maxAmp = Math.max(...visibleSamples.indices.map((idx) => Math.abs(samples[idx] || 0)));
    const scale = maxAmp > 0 ? (bandHeight / 2 * 0.9) / maxAmp : 1;

    // Build top half (left to right)
    const topPoints = visibleSamples.indices
      .map((idx) => {
        const time = waveformTimes[idx];
        if (time === undefined) return null;
        const x = ((time - viewportStart) / viewportDuration) * width;
        const amp = Math.abs(samples[idx] || 0);
        const y = center - amp * scale;
        return `${x},${y}`;
      })
      .filter((p): p is string => p !== null);

    // Build bottom half (right to left, mirrored)
    const bottomPoints = visibleSamples.indices
      .map((idx) => {
        const time = waveformTimes[idx];
        if (time === undefined) return null;
        const x = ((time - viewportStart) / viewportDuration) * width;
        const amp = Math.abs(samples[idx] || 0);
        const y = center + amp * scale;
        return `${x},${y}`;
      })
      .filter((p): p is string => p !== null)
      .reverse();

    if (topPoints.length === 0 || bottomPoints.length === 0) return "";

    return `M${topPoints.join(" L")} L${bottomPoints.join(" L")} Z`;
  };

  const bandHeight = 100 / 3;

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
      {/* Bass band (top third) - cyan */}
      <path
        d={generateBandPath(waveformBass, 0, bandHeight)}
        fill="rgba(0, 229, 204, 0.7)"
        stroke="none"
      />

      {/* Mids band (middle third) - purple */}
      <path
        d={generateBandPath(waveformMids, bandHeight, bandHeight)}
        fill="rgba(123, 106, 255, 0.7)"
        stroke="none"
      />

      {/* Highs band (bottom third) - pink */}
      <path
        d={generateBandPath(waveformHighs, bandHeight * 2, bandHeight)}
        fill="rgba(255, 107, 181, 0.7)"
        stroke="none"
      />

      {/* Separator lines */}
      <line
        x1="0"
        y1={bandHeight}
        x2="100"
        y2={bandHeight}
        stroke="rgba(123, 106, 255, 0.2)"
        strokeWidth="0.5"
      />
      <line
        x1="0"
        y1={bandHeight * 2}
        x2="100"
        y2={bandHeight * 2}
        stroke="rgba(123, 106, 255, 0.2)"
        strokeWidth="0.5"
      />
    </svg>
  );
}
