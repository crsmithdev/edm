import { useStructureStore, useAudioStore } from "@/stores";
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

  return (
    <div
      style={{
        flex: 1,
        overflowY: "auto",
        padding: "var(--space-1) 0 var(--space-4) 0",
      }}
    >
      {regions.map((region, idx) => {
        return (
          <div
            key={idx}
            style={{
              padding: "var(--space-3) var(--space-4)",
              borderBottom: "1px solid var(--border-primary)",
              display: "flex",
              justifyContent: "space-between",
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
                minWidth: "120px",
                fontSize: "var(--font-size-sm)",
              }}
            >
              {formatTime(region.start)} - {formatTime(region.end)}
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
