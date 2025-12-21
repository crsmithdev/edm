import { useMemo } from "react";
import { useWaveformStore, useAudioStore, useTempoStore, useUIStore, useStructureStore } from "@/stores";
import { timeToBar, barToTime } from "@/utils/tempo";
import { labelColors } from "@/utils/colors";

/**
 * Compact full-track waveform overview with moving playhead
 * Shows entire track duration, waveform extends upward from baseline
 * Click snaps to nearest bar
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
  const { trackBPM, trackDownbeat } = useTempoStore();
  const { quantizeEnabled } = useUIStore();
  const { regions, boundaries } = useStructureStore();

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

    // Apply smoothing for cleaner visual appearance
    const smoothingWindowSize = 2; // Fixed window for overview (reduced for sharper detail)
    const smoothedAmplitudes: number[] = [];
    for (let i = 0; i < samples.length; i++) {
      const halfWindow = Math.floor(smoothingWindowSize / 2);
      const windowStart = Math.max(0, i - halfWindow);
      const windowEnd = Math.min(samples.length, i + halfWindow + 1);

      let sum = 0;
      for (let j = windowStart; j < windowEnd; j++) {
        sum += samples[j].amplitude;
      }
      smoothedAmplitudes[i] = sum / (windowEnd - windowStart);
    }

    // Find max for scaling
    const maxAmplitude = Math.max(...smoothedAmplitudes, 0.001);
    const scale = (height * 0.85) / maxAmplitude;
    const baseline = height; // Bottom of SVG

    // Generate path extending upward from baseline
    const points = samples.map((s, i) => {
      const y = baseline - smoothedAmplitudes[i] * scale;
      return `${s.x},${y}`;
    });

    // Close path along baseline
    return `M0,${baseline} L${points.join(" L")} L${width},${baseline} Z`;
  }, [waveformBass, waveformMids, waveformHighs, waveformTimes, duration]);

  // Calculate playhead position
  const playheadPercent = duration > 0 ? (currentTime / duration) * 100 : 0;

  // Handle click to seek - snap to nearest bar if quantize enabled
  const handleClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percent = x / rect.width;
    const rawTime = percent * duration;

    // Snap to nearest bar if quantize enabled and BPM available
    let seekTime = rawTime;
    if (quantizeEnabled && trackBPM > 0) {
      const bar = timeToBar(rawTime, trackBPM, trackDownbeat);
      const nearestBar = Math.round(bar);
      seekTime = barToTime(nearestBar, trackBPM, trackDownbeat);
    }

    seek(Math.max(0, Math.min(duration, seekTime)));
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
        cursor: "default",
        overflow: "hidden",
      }}
    >
      {/* Region overlays (subtle, behind waveform) - skip default regions */}
      {duration > 0 &&
        regions
          .filter((region) => region.label !== "default")
          .map((region, idx) => {
            const leftPercent = (region.start / duration) * 100;
            const widthPercent = ((region.end - region.start) / duration) * 100;
            return (
              <div
                key={idx}
                style={{
                  position: "absolute",
                  left: `${leftPercent}%`,
                  top: 0,
                  width: `${widthPercent}%`,
                  height: "100%",
                  background: labelColors[region.label],
                  opacity: 0.35,
                  pointerEvents: "none",
                }}
              />
            );
          })}

      {/* Waveform SVG */}
      <svg
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
        style={{
          position: "relative",
          width: "100%",
          height: "100%",
          background: "#0a0a12",
        }}
      >
        {/* Combined waveform - single color for overview */}
        <path d={waveformPath} fill="rgba(100, 140, 180, 0.6)" stroke="none" />
      </svg>

      {/* Boundary markers (subtle) */}
      {duration > 0 &&
        boundaries.map((time, idx) => {
          const xPercent = (time / duration) * 100;
          return (
            <div
              key={idx}
              style={{
                position: "absolute",
                left: `${xPercent}%`,
                top: 0,
                width: "2px",
                height: "100%",
                background: "rgba(123, 106, 255, 0.7)",
                pointerEvents: "none",
              }}
            />
          );
        })}

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
