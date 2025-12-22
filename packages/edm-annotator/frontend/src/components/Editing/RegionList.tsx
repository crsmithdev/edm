import { Trash2 } from "lucide-react";
import { useStructureStore, useAudioStore, useTempoStore } from "@/stores";
import { formatTime } from "@/utils/timeFormat";
import type { SectionLabel } from "@/types/structure";

const VALID_LABELS: SectionLabel[] = [
  "intro",
  "buildup",
  "breakdown",
  "breakdown-buildup",
  "outro",
  "default",
];

/**
 * List of regions with label editing
 */
export function RegionList() {
  const { regions, setRegionLabel, removeBoundary } = useStructureStore();
  const { seek, play, currentTime } = useAudioStore();
  const { timeToBar } = useTempoStore();

  const handleDeleteRegion = (idx: number, region: { start: number; end: number }) => {
    // Can't delete if there's only one region
    if (regions.length <= 1) return;

    // For all regions except the first, remove the boundary at the start (merge with previous)
    // For the first region, remove the boundary at the end (merge with next)
    const boundaryToRemove = idx === 0 ? region.end : region.start;
    removeBoundary(boundaryToRemove);
  };

  return (
    <div
      style={{
        flex: 1,
        overflowY: "auto",
        padding: "var(--space-1) 0 var(--space-4) 0",
      }}
    >
      {/* Header */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "140px 100px 1fr 40px",
          gap: "var(--space-3)",
          padding: "var(--space-2) var(--space-4)",
          borderBottom: "2px solid var(--border-primary)",
          fontSize: "var(--font-size-xs)",
          fontWeight: "var(--font-weight-semibold)",
          color: "var(--text-tertiary)",
          textTransform: "uppercase",
          letterSpacing: "0.05em",
        }}
      >
        <div>Time</div>
        <div>Bars</div>
        <div>Label</div>
        <div></div>
      </div>

      {/* Regions */}
      {regions.map((region, idx) => {
        const startBar = timeToBar(region.start);
        const endBar = timeToBar(region.end);
        const isCurrentRegion = currentTime >= region.start && currentTime < region.end;

        return (
          <div
            key={idx}
            style={{
              display: "grid",
              gridTemplateColumns: "140px 100px 1fr 40px",
              gap: "var(--space-3)",
              padding: "var(--space-3) var(--space-4)",
              borderBottom: "1px solid var(--border-primary)",
              alignItems: "center",
              cursor: "pointer",
              background: isCurrentRegion ? "rgba(91, 124, 255, 0.15)" : "transparent",
              transition: "all var(--transition-base)",
            }}
            onClick={() => {
              seek(region.start);
              play();
            }}
            onMouseEnter={(e) => {
              if (!isCurrentRegion) {
                e.currentTarget.style.background = "var(--bg-elevated)";
              }
            }}
            onMouseLeave={(e) => {
              if (!isCurrentRegion) {
                e.currentTarget.style.background = "transparent";
              }
            }}
          >
            {/* Time */}
            <div
              style={{
                fontWeight: "var(--font-weight-semibold)",
                color: "var(--color-primary)",
                fontSize: "var(--font-size-sm)",
              }}
            >
              {formatTime(region.start)} - {formatTime(region.end)}
            </div>

            {/* Bars */}
            <div
              style={{
                color: "var(--text-secondary)",
                fontFamily: "var(--font-mono)",
                fontSize: "var(--font-size-sm)",
              }}
            >
              {startBar}-{endBar}
            </div>

            {/* Label Selector */}
            <select
              value={region.label}
              onChange={(e) => {
                e.stopPropagation();
                setRegionLabel(idx, e.target.value as SectionLabel);
              }}
              onClick={(e) => e.stopPropagation()}
              style={{
                padding: "var(--space-1) var(--space-3)",
                border: "1px solid var(--border-subtle)",
                background: "var(--bg-primary)",
                color: "var(--text-secondary)",
                borderRadius: "var(--radius-sm)",
                fontSize: "var(--font-size-sm)",
                fontWeight: "var(--font-weight-semibold)",
                textTransform: "capitalize",
                cursor: "pointer",
                transition: "all var(--transition-base)",
                maxWidth: "180px",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = "var(--bg-hover)";
                e.currentTarget.style.borderColor = "var(--border-focus)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = "var(--bg-primary)";
                e.currentTarget.style.borderColor = "var(--border-subtle)";
              }}
            >
              {VALID_LABELS.map((label) => (
                <option key={label} value={label}>
                  {label}
                </option>
              ))}
            </select>

            {/* Delete Button */}
            <button
              onClick={(e) => {
                e.stopPropagation();
                handleDeleteRegion(idx, region);
              }}
              disabled={regions.length <= 1}
              style={{
                padding: "var(--space-1)",
                border: "none",
                background: "transparent",
                color: regions.length <= 1 ? "var(--text-disabled)" : "var(--text-tertiary)",
                cursor: regions.length <= 1 ? "not-allowed" : "pointer",
                borderRadius: "var(--radius-sm)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                transition: "all var(--transition-base)",
                opacity: regions.length <= 1 ? 0.3 : 1,
              }}
              onMouseEnter={(e) => {
                if (regions.length > 1) {
                  e.currentTarget.style.background = "var(--bg-hover)";
                  e.currentTarget.style.color = "var(--color-error)";
                }
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = "transparent";
                e.currentTarget.style.color = regions.length <= 1 ? "var(--text-disabled)" : "var(--text-tertiary)";
              }}
            >
              <Trash2 size={16} />
            </button>
          </div>
        );
      })}
    </div>
  );
}
