import { useStructureStore, useWaveformStore } from "@/stores";
import { formatTime } from "@/utils/timeFormat";

/**
 * Boundary markers showing structure boundaries
 */
export function BoundaryMarkers() {
  const { boundaries, removeBoundary } = useStructureStore();
  const { viewportStart, viewportEnd } = useWaveformStore();

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
              background: "#00e5cc",
              boxShadow: "0 0 10px rgba(0, 229, 204, 0.5)",
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
          >
            <div
              style={{
                position: "absolute",
                top: "-25px",
                left: "50%",
                transform: "translateX(-50%)",
                background: "rgba(10, 10, 18, 0.95)",
                padding: "2px 8px",
                borderRadius: "4px",
                fontSize: "11px",
                whiteSpace: "nowrap",
                color: "#00e5cc",
                fontWeight: "600",
                border: "1px solid rgba(0, 229, 204, 0.3)",
              }}
            >
              {formatTime(time)}
            </div>
          </div>
        );
      })}
    </>
  );
}
