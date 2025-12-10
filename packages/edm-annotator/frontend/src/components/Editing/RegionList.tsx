import { useStructureStore, useAudioStore, useTempoStore } from "@/stores";
import { formatTime } from "@/utils/timeFormat";
import type { SectionLabel } from "@/types/structure";

const VALID_LABELS: SectionLabel[] = [
  "intro",
  "buildup",
  "breakdown",
  "breakbuild",
  "outro",
  "unlabeled",
];

/**
 * List of regions with label editing
 */
export function RegionList() {
  const { regions, setRegionLabel } = useStructureStore();
  const { seek } = useAudioStore();
  const { timeToBar } = useTempoStore();

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
          gridTemplateColumns: "140px 100px 1fr",
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
      </div>

      {/* Regions */}
      {regions.map((region, idx) => {
        const startBar = timeToBar(region.start);
        const endBar = timeToBar(region.end);

        return (
          <div
            key={idx}
            style={{
              display: "grid",
              gridTemplateColumns: "140px 100px 1fr",
              gap: "var(--space-3)",
              padding: "var(--space-3) var(--space-4)",
              borderBottom: "1px solid var(--border-primary)",
              alignItems: "center",
              cursor: "pointer",
              background: "transparent",
              transition: "all var(--transition-base)",
            }}
            onClick={() => seek(region.start)}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = "var(--bg-elevated)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = "transparent";
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
                textTransform: "uppercase",
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
          </div>
        );
      })}
    </div>
  );
}
