import { useMemo } from "react";
import { useWaveformStore, useAudioStore } from "@/stores";

interface OverviewWaveformProps {
  /** Time span shown in detail view (seconds) */
  detailSpan: number;
}

/**
 * Compact full-track waveform overview with moving playhead
 * Shows entire track duration with viewport indicator for detail view
 */
export function OverviewWaveform({ detailSpan }: OverviewWaveformProps) {
  const {
    waveformBass,
    waveformMids,
    waveformHighs,
    waveformTimes,
    duration,
  } = useWaveformStore();
  const { currentTime, seek } = useAudioStore();

  // Generate simplified waveform path for full track
  const waveformPath = useMemo(() => {
    if (waveformTimes.length === 0 || duration <= 0) return "";

    const width = 100;
    const height = 100;

    // Downsample for performance - take every Nth sample
    const targetSamples = 500;
    const step = Math.max(1, Math.floor(waveformTimes.length / targetSamples));

    const samples: { x: number; amplitude: number }[] = [];
    for (let i = 0; i < waveformTimes.length; i += step) {
      const bass = Math.abs(waveformBass[i] || 0);
      const mids = Math.abs(waveformMids[i] || 0);
      const highs = Math.abs(waveformHighs[i] || 0);
      const amplitude = bass + mids + highs;
      const x = (waveformTimes[i] / duration) * width;
      samples.push({ x, amplitude });
    }

    if (samples.length === 0) return "";

    // Find max for scaling
    const maxAmplitude = Math.max(...samples.map((s) => s.amplitude), 0.001);
    const scale = (height * 0.8) / maxAmplitude;
    const center = height / 2;
    const halfScale = scale / 2;

    // Generate mirrored path
    const topPoints = samples.map((s) => {
      const y = center - s.amplitude * halfScale;
      return `${s.x},${y}`;
    });

    const bottomPoints = samples
      .map((s) => {
        const y = center + s.amplitude * halfScale;
        return `${s.x},${y}`;
      })
      .reverse();

    return `M${topPoints.join(" L")} L${bottomPoints.join(" L")} Z`;
  }, [waveformBass, waveformMids, waveformHighs, waveformTimes, duration]);

  // Calculate playhead position
  const playheadPercent = duration > 0 ? (currentTime / duration) * 100 : 0;

  // Calculate viewport indicator position and width
  const viewportIndicator = useMemo(() => {
    if (duration <= 0) return { left: 0, width: 100 };

    // Detail view shows detailSpan seconds centered on currentTime
    const halfSpan = detailSpan / 2;
    let viewStart = currentTime - halfSpan;
    let viewEnd = currentTime + halfSpan;

    // Clamp to track boundaries
    if (viewStart < 0) {
      viewStart = 0;
      viewEnd = Math.min(detailSpan, duration);
    }
    if (viewEnd > duration) {
      viewEnd = duration;
      viewStart = Math.max(0, duration - detailSpan);
    }

    const left = (viewStart / duration) * 100;
    const width = ((viewEnd - viewStart) / duration) * 100;

    return { left, width };
  }, [currentTime, duration, detailSpan]);

  // Handle click to seek
  const handleClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percent = x / rect.width;
    const time = percent * duration;
    seek(Math.max(0, Math.min(duration, time)));
  };

  return (
    <div
      onClick={handleClick}
      style={{
        position: "relative",
        width: "100%",
        height: "60px",
        background: "var(--bg-tertiary)",
        border: "1px solid var(--border-subtle)",
        borderRadius: "var(--radius-md)",
        cursor: "pointer",
        overflow: "hidden",
      }}
    >
      {/* Waveform SVG */}
      <svg
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
        style={{
          width: "100%",
          height: "100%",
          background: "#0a0a12",
        }}
      >
        {/* Combined waveform - single color for overview */}
        <path d={waveformPath} fill="rgba(100, 140, 180, 0.6)" stroke="none" />

        {/* Center baseline */}
        <line
          x1="0"
          y1="50"
          x2="100"
          y2="50"
          stroke="rgba(255, 255, 255, 0.1)"
          strokeWidth="0.3"
        />
      </svg>

      {/* Viewport indicator - shows detail view extent */}
      <div
        style={{
          position: "absolute",
          left: `${viewportIndicator.left}%`,
          top: 0,
          width: `${viewportIndicator.width}%`,
          height: "100%",
          background: "rgba(26, 255, 239, 0.1)",
          borderLeft: "1px solid rgba(26, 255, 239, 0.4)",
          borderRight: "1px solid rgba(26, 255, 239, 0.4)",
          pointerEvents: "none",
        }}
      />

      {/* Playhead */}
      <div
        style={{
          position: "absolute",
          left: `${playheadPercent}%`,
          top: 0,
          width: "2px",
          height: "100%",
          background: "#1affef",
          boxShadow: "0 0 8px rgba(26, 255, 239, 0.6)",
          pointerEvents: "none",
          transform: "translateX(-1px)",
        }}
      />
    </div>
  );
}
