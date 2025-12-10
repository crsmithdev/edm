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
        width: "3px",
        height: "100%",
        background: "#00E6B8",
        boxShadow: "0 0 10px rgba(0, 230, 184, 0.5)",
        pointerEvents: "none",
        zIndex: 10,
      }}
    />
  );
}
