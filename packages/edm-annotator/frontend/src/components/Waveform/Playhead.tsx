import { useAudioStore, useWaveformStore } from "@/stores";

/**
 * Playhead showing current playback position
 */
export function Playhead() {
  const { currentTime } = useAudioStore();
  const { viewportStart, viewportEnd } = useWaveformStore();

  // Only render if playhead is within viewport
  if (currentTime < viewportStart || currentTime > viewportEnd) {
    return null;
  }

  const viewportDuration = viewportEnd - viewportStart;
  const xPercent = ((currentTime - viewportStart) / viewportDuration) * 100;

  return (
    <div
      style={{
        position: "absolute",
        left: `${xPercent}%`,
        top: 0,
        width: "2px",
        height: "100%",
        background: "linear-gradient(180deg, #1affef 0%, #00e5cc 100%)",
        boxShadow: "0 0 15px rgba(26, 255, 239, 0.6), 0 0 30px rgba(26, 255, 239, 0.3)",
        pointerEvents: "none",
        zIndex: 10,
      }}
    />
  );
}
