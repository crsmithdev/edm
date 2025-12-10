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

  if (regions.length === 0) {
    return (
      <div
        style={{
          padding: "24px",
          background: "#1E2139",
          borderRadius: "10px",
          border: "1px solid rgba(91, 124, 255, 0.1)",
          textAlign: "center",
          color: "#6B7280",
        }}
      >
        No regions yet. Add boundaries to create regions.
      </div>
    );
  }

  return (
    <div
      style={{
        background: "#1E2139",
        borderRadius: "10px",
        border: "1px solid rgba(91, 124, 255, 0.1)",
        overflow: "hidden",
      }}
    >
      <div
        style={{
          padding: "16px",
          borderBottom: "1px solid rgba(91, 124, 255, 0.1)",
        }}
      >
        <h3 style={{ color: "#FFFFFF", fontSize: "16px", margin: 0 }}>
          Structure Regions
        </h3>
      </div>

      <div
        style={{
          maxHeight: "400px",
          overflowY: "auto",
        }}
      >
        {regions.map((region, idx) => {
          const duration = region.end - region.start;

          return (
            <div
              key={idx}
              style={{
                padding: "12px 16px",
                borderBottom:
                  idx < regions.length - 1
                    ? "1px solid rgba(91, 124, 255, 0.05)"
                    : "none",
                display: "flex",
                gap: "12px",
                alignItems: "center",
                cursor: "pointer",
                transition: "background 0.2s",
              }}
              onClick={() => seek(region.start)}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = "#252A45";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = "transparent";
              }}
            >
              {/* Region Info */}
              <div style={{ flex: 1, minWidth: 0 }}>
                <div
                  style={{
                    fontSize: "13px",
                    color: "#9CA3AF",
                    marginBottom: "4px",
                  }}
                >
                  {formatTime(region.start)} - {formatTime(region.end)}
                  <span style={{ marginLeft: "8px", color: "#6B7280" }}>
                    ({duration.toFixed(1)}s)
                  </span>
                </div>
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
                  padding: "6px 10px",
                  background: "#151828",
                  border: "1px solid #2A2F4C",
                  borderRadius: "6px",
                  color: "#E5E7EB",
                  fontSize: "13px",
                  cursor: "pointer",
                  minWidth: "120px",
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
    </div>
  );
}
