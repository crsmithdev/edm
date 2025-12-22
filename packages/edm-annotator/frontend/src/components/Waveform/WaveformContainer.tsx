import { useState } from "react";
import { useWaveformStore, useTrackStore, useTempoStore, useAudioStore } from "@/stores";
import { OverviewWaveform } from "./OverviewWaveform";
import { DetailWaveform } from "./DetailWaveform";
import { InfoCard } from "@/components/UI";

/** Default time span for detail view in seconds */
const DEFAULT_DETAIL_SPAN = 16;
const MIN_DETAIL_SPAN = 4;
const MAX_DETAIL_SPAN = 60;

// Helper to parse filename into artist and title
function parseFilename(filename: string): { artist: string; title: string } {
  const nameWithoutExt = filename.replace(/\.[^.]+$/, "");
  const parts = nameWithoutExt.split(" - ");

  if (parts.length >= 2) {
    return {
      artist: parts[0].trim(),
      title: parts.slice(1).join(" - ").trim(),
    };
  }

  return {
    artist: "",
    title: nameWithoutExt,
  };
}

/**
 * Container for dual waveform display:
 * - Overview: full track, moving playhead
 * - Detail: centered playhead, scrolling waveform
 */
export function WaveformContainer() {
  const { duration } = useWaveformStore();
  const { currentTrack } = useTrackStore();
  const { trackBPM, timeToBar } = useTempoStore();
  const { currentTime } = useAudioStore();

  // Detail view span (how many seconds to show around playhead)
  const [detailSpan, setDetailSpan] = useState(DEFAULT_DETAIL_SPAN);

  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = (seconds % 60).toFixed(2);
    return `${mins}:${secs.padStart(5, "0")}`;
  };

  const currentBar = timeToBar(currentTime);
  const { artist, title } = currentTrack ? parseFilename(currentTrack) : { artist: "", title: "" };

  const handleZoomIn = () => {
    setDetailSpan((prev) => Math.max(MIN_DETAIL_SPAN, prev / 1.5));
  };

  const handleZoomOut = () => {
    setDetailSpan((prev) => Math.min(MAX_DETAIL_SPAN, prev * 1.5));
  };

  const handleReset = () => {
    setDetailSpan(DEFAULT_DETAIL_SPAN);
  };

  return (
    <div
      style={{
        background: "var(--bg-secondary)",
        padding: "var(--space-6)",
        borderRadius: "var(--radius-xl)",
        marginBottom: "var(--space-5)",
        border: "1px solid var(--border-primary)",
        boxShadow: "var(--shadow-md)",
        overflow: "hidden",
      }}
    >
      {/* Overview Waveform (full track) */}
      <OverviewWaveform />

      {/* Detail Waveform (centered playhead) */}
      <div style={{ marginTop: "var(--space-3)" }}>
        <DetailWaveform span={detailSpan} onZoomIn={handleZoomIn} onZoomOut={handleZoomOut} />
      </div>

      {/* Track Info and Zoom Controls */}
      <div
        style={{
          display: "flex",
          gap: "var(--space-5)",
          alignItems: "center",
          marginTop: "var(--space-4)",
          justifyContent: "space-between",
        }}
      >
        {/* Track metadata on left */}
        <div
          style={{ display: "flex", flexDirection: "column", gap: "var(--space-1)" }}
        >
          {currentTrack && (
            <>
              <div
                style={{
                  fontSize: "20px",
                  fontWeight: "var(--font-weight-normal)",
                  letterSpacing: "var(--letter-spacing-tight)",
                  color: "var(--text-primary)",
                }}
              >
                {title}
              </div>
              <div
                style={{
                  fontSize: "var(--font-size-sm)",
                  color: "var(--text-tertiary)",
                  letterSpacing: "0.5px",
                }}
              >
                {artist}
              </div>
            </>
          )}
        </div>

        {/* Status displays in middle - centered */}
        <div style={{ display: "flex", gap: "var(--space-6)", alignItems: "center" }}>
          <InfoCard label="BPM" value={trackBPM || "--"} />
          <InfoCard label="Bar" value={currentTrack ? currentBar : "--"} />
          <InfoCard label="Time" value={currentTrack ? formatTime(currentTime) : "--"} />
        </div>

        {/* Zoom controls on right */}
        <div style={{ display: "flex", gap: "var(--space-2)", alignItems: "stretch" }}>
          <button
            onClick={handleZoomOut}
            style={{
              padding: "var(--space-2) var(--space-4)",
              background: "var(--border-subtle)",
              color: "var(--text-secondary)",
              border: "none",
              borderRadius: "var(--radius-sm)",
              fontSize: "var(--font-size-sm)",
              cursor: "pointer",
              fontWeight: "var(--font-weight-normal)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            −
          </button>
          <button
            onClick={handleZoomIn}
            style={{
              padding: "var(--space-2) var(--space-4)",
              background: "var(--border-subtle)",
              color: "var(--text-secondary)",
              border: "none",
              borderRadius: "var(--radius-sm)",
              fontSize: "var(--font-size-sm)",
              cursor: "pointer",
              fontWeight: "var(--font-weight-normal)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            +
          </button>
          <button
            onClick={handleReset}
            style={{
              padding: "var(--space-2) var(--space-4)",
              background: "var(--border-subtle)",
              color: "var(--text-secondary)",
              border: "none",
              borderRadius: "var(--radius-sm)",
              fontSize: "var(--font-size-sm)",
              cursor: "pointer",
              fontWeight: "var(--font-weight-normal)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            Reset
          </button>
        </div>
      </div>
    </div>
  );
}
