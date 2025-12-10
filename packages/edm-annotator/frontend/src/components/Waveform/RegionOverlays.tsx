import { useStructureStore, useWaveformStore } from "@/stores";
import { labelColors } from "@/utils/colors";

/**
 * Colored region overlays showing structure sections
 */
export function RegionOverlays() {
  const { regions } = useStructureStore();
  const { viewportStart, viewportEnd } = useWaveformStore();

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
              pointerEvents: "none",
              opacity: 0.3,
            }}
          />
        );
      })}
    </>
  );
}
