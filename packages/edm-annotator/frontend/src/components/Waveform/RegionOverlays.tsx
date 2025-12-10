import { useStructureStore, useWaveformStore } from "@/stores";
import { labelColors, labelBorderColors } from "@/utils/colors";

interface RegionOverlaysProps {
  /** Override viewport start (for centered playhead mode) */
  viewportStart?: number;
  /** Override viewport end (for centered playhead mode) */
  viewportEnd?: number;
}

/**
 * Colored region overlays showing structure sections
 */
export function RegionOverlays(props: RegionOverlaysProps) {
  const { regions } = useStructureStore();
  const store = useWaveformStore();
  const viewportStart = props.viewportStart ?? store.viewportStart;
  const viewportEnd = props.viewportEnd ?? store.viewportEnd;

  const viewportDuration = viewportEnd - viewportStart;

  return (
    <>
      {regions.map((region, idx) => {
        // Skip if region is completely outside viewport
        if (region.end < viewportStart || region.start > viewportEnd) {
          return null;
        }

        // Clamp region to viewport
        const visibleStart = Math.max(region.start, viewportStart);
        const visibleEnd = Math.min(region.end, viewportEnd);

        const leftPercent = ((visibleStart - viewportStart) / viewportDuration) * 100;
        const widthPercent =
          ((visibleEnd - visibleStart) / viewportDuration) * 100;

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
              border: `2px solid ${labelBorderColors[region.label]}`,
              borderTop: "none",
              borderBottom: "none",
              pointerEvents: "none",
              opacity: 0.4,
            }}
          />
        );
      })}
    </>
  );
}
