import { useStructureStore, useWaveformStore } from "@/stores";
import { formatTime } from "@/utils/timeFormat";

interface BoundaryMarkersProps {
  /** Override viewport start (for centered playhead mode) */
  viewportStart?: number;
  /** Override viewport end (for centered playhead mode) */
  viewportEnd?: number;
}

/**
 * Boundary markers showing structure boundaries
 */
export function BoundaryMarkers(props: BoundaryMarkersProps) {
  const { boundaries, removeBoundary } = useStructureStore();
  const store = useWaveformStore();
  const viewportStart = props.viewportStart ?? store.viewportStart;
  const viewportEnd = props.viewportEnd ?? store.viewportEnd;

  const viewportDuration = viewportEnd - viewportStart;

  return (
    <>
      {boundaries.map((time, idx) => {
        // Only render if within viewport
        if (time < viewportStart || time > viewportEnd) {
          return null;
        }

        const xPercent = ((time - viewportStart) / viewportDuration) * 100;

        return (
          <div
            key={idx}
            style={{
              position: "absolute",
              left: `${xPercent}%`,
              top: 0,
              height: "100%",
              width: "3px",
              background: "#7b6aff",
              boxShadow: "0 0 10px rgba(123, 106, 255, 0.5)",
              cursor: "pointer",
              zIndex: 5,
            }}
            onClick={(e) => {
              e.stopPropagation();
              if (e.ctrlKey || e.metaKey) {
                removeBoundary(time);
              }
            }}
            title={`Boundary at ${formatTime(time)} (Ctrl+Click to remove)`}
          />
        );
      })}
    </>
  );
}
