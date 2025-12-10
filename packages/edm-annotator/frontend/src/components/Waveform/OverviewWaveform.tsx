import { useMemo } from "react";
import { useWaveformStore, useAudioStore } from "@/stores";

/**
 * Compact full-track waveform overview with moving playhead
 * Shows entire track duration, waveform extends upward from baseline
 */
export function OverviewWaveform() {
  const {
    waveformBass,
    waveformMids,
    waveformHighs,
    waveformTimes,
    duration,
  } = useWaveformStore();
  const { currentTime, seek } = useAudioStore();

  // Generate simplified waveform path for full track (non-mirrored, extends upward)
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
    const scale = (height * 0.85) / maxAmplitude;
    const baseline = height; // Bottom of SVG

    // Generate path extending upward from baseline
    const points = samples.map((s) => {
      const y = baseline - s.amplitude * scale;
      return `${s.x},${y}`;
    });

    // Close path along baseline
    return `M0,${baseline} L${points.join(" L")} L${width},${baseline} Z`;
  }, [waveformBass, waveformMids, waveformHighs, waveformTimes, duration]);

  // Calculate playhead position
  const playheadPercent = duration > 0 ? (currentTime / duration) * 100 : 0;

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
      </svg>

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
