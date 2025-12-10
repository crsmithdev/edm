import { useAudioStore, useTempoStore, useTrackStore } from "@/stores";

/**
 * Transport row with playback controls and info displays
 */
export function PlaybackControls() {
  const { isPlaying, play, pause, currentTime, returnToCue } = useAudioStore();
  const { trackBPM, timeToBar } = useTempoStore();
  const { nextTrack, previousTrack } = useTrackStore();

  const currentBar = timeToBar(currentTime);

  const togglePlayback = () => {
    if (isPlaying) {
      pause();
    } else {
      play();
    }
  };

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = (seconds % 60).toFixed(2);
    return `${mins}:${secs.padStart(5, "0")}`;
  };

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "1fr auto",
        gap: "20px",
        alignItems: "center",
        paddingBottom: "20px",
      }}
    >
      {/* Left: Transport buttons */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(4, 1fr)",
          gap: "12px",
          alignItems: "stretch",
        }}
      >
        <button
          onClick={togglePlayback}
          style={{
            padding: "12px 20px",
            fontSize: "14px",
            fontWeight: 600,
            background: isPlaying ? "#FF6B6B" : "#00E6B8",
            color: isPlaying ? "#FFFFFF" : "#0F1419",
            border: "none",
            borderRadius: "8px",
            cursor: "pointer",
            transition: "all 0.2s",
            whiteSpace: "nowrap",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            height: "44px",
          }}
        >
          {isPlaying ? "⏸ Pause" : "▶ Play"}
        </button>
        <button
          onClick={returnToCue}
          style={{
            padding: "12px 20px",
            fontSize: "14px",
            fontWeight: 600,
            background: "#FFB800",
            color: "#0F1419",
            border: "none",
            borderRadius: "8px",
            cursor: "pointer",
            transition: "all 0.2s",
            whiteSpace: "nowrap",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            height: "44px",
          }}
        >
          ↺ Cue
        </button>
        <button
          onClick={previousTrack}
          style={{
            padding: "12px 20px",
            fontSize: "14px",
            fontWeight: 600,
            background: "#2A2F4C",
            color: "#E5E7EB",
            border: "none",
            borderRadius: "8px",
            cursor: "pointer",
            transition: "all 0.2s",
            whiteSpace: "nowrap",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            height: "44px",
          }}
        >
          ◀ Previous
        </button>
        <button
          onClick={nextTrack}
          style={{
            padding: "12px 20px",
            fontSize: "14px",
            fontWeight: 600,
            background: "#2A2F4C",
            color: "#E5E7EB",
            border: "none",
            borderRadius: "8px",
            cursor: "pointer",
            transition: "all 0.2s",
            whiteSpace: "nowrap",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            height: "44px",
          }}
        >
          Next ▶
        </button>
      </div>

      {/* Right: Info displays */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap: "12px",
        }}
      >
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            gap: "4px",
            padding: "8px 16px",
            background: "#151828",
            border: "1px solid #2A2F4C",
            borderRadius: "8px",
            height: "44px",
          }}
        >
          <span
            style={{
              fontSize: "11px",
              fontWeight: 600,
              color: "#6B7280",
              textTransform: "uppercase",
              letterSpacing: "0.5px",
            }}
          >
            BPM
          </span>
          <span
            style={{
              fontSize: "16px",
              fontWeight: 700,
              color: "#00E6B8",
              fontVariantNumeric: "tabular-nums",
            }}
          >
            {trackBPM || "--"}
          </span>
        </div>
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            gap: "4px",
            padding: "8px 16px",
            background: "#151828",
            border: "1px solid #2A2F4C",
            borderRadius: "8px",
            height: "44px",
          }}
        >
          <span
            style={{
              fontSize: "11px",
              fontWeight: 600,
              color: "#6B7280",
              textTransform: "uppercase",
              letterSpacing: "0.5px",
            }}
          >
            Bar
          </span>
          <span
            style={{
              fontSize: "16px",
              fontWeight: 700,
              color: "#00E6B8",
              fontVariantNumeric: "tabular-nums",
            }}
          >
            {currentBar}
          </span>
        </div>
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            gap: "4px",
            padding: "8px 16px",
            background: "#151828",
            border: "1px solid #2A2F4C",
            borderRadius: "8px",
            height: "44px",
          }}
        >
          <span
            style={{
              fontSize: "11px",
              fontWeight: 600,
              color: "#6B7280",
              textTransform: "uppercase",
              letterSpacing: "0.5px",
            }}
          >
            Time
          </span>
          <span
            style={{
              fontSize: "16px",
              fontWeight: 700,
              color: "#00E6B8",
              fontVariantNumeric: "tabular-nums",
            }}
          >
            {formatTime(currentTime)}
          </span>
        </div>
      </div>
    </div>
  );
}
