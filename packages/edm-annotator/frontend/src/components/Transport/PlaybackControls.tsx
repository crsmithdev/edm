import { useAudioStore } from "@/stores";

/**
 * Playback controls for play/pause
 */
export function PlaybackControls() {
  const { isPlaying, play, pause } = useAudioStore();

  const togglePlayback = () => {
    if (isPlaying) {
      pause();
    } else {
      play();
    }
  };

  return (
    <div
      style={{
        display: "flex",
        gap: "8px",
        alignItems: "center",
        padding: "16px",
        background: "#1E2139",
        borderRadius: "10px",
        border: "1px solid rgba(91, 124, 255, 0.1)",
      }}
    >
      <button
        onClick={togglePlayback}
        style={{
          padding: "12px 24px",
          background: isPlaying ? "#FFB800" : "#5B7CFF",
          color: isPlaying ? "#0F1419" : "#FFFFFF",
          border: "none",
          borderRadius: "6px",
          fontSize: "14px",
          fontWeight: 600,
          cursor: "pointer",
          minWidth: "100px",
        }}
      >
        {isPlaying ? "Pause" : "Play"} (Space)
      </button>
    </div>
  );
}
