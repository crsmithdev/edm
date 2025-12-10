import { useState } from "react";
import { Save } from "lucide-react";
import { useStructureStore, useAudioStore, useTempoStore, useUIStore, useTrackStore } from "@/stores";
import { trackService } from "@/services/api";
import { formatTime } from "@/utils/timeFormat";
import type { SectionLabel } from "@/types/structure";
import { Button } from "@/components/UI";

const VALID_LABELS: SectionLabel[] = [
  "intro",
  "buildup",
  "breakdown",
  "breakbuild",
  "outro",
  "unlabeled",
];

/**
 * List of regions with label editing and save button
 */
export function RegionList() {
  const { regions, setRegionLabel, boundaries } = useStructureStore();
  const { seek } = useAudioStore();
  const { trackBPM, trackDownbeat } = useTempoStore();
  const { showStatus } = useUIStore();
  const { currentTrack } = useTrackStore();
  const [isSaving, setIsSaving] = useState(false);

  const handleSave = async () => {
    if (!currentTrack) {
      showStatus("No track loaded");
      return;
    }

    setIsSaving(true);
    try {
      await trackService.saveAnnotation({
        filename: currentTrack,
        bpm: trackBPM,
        downbeat: trackDownbeat,
        boundaries: regions.map(r => ({ time: r.start, label: r.label })),
      });
      showStatus("Annotation saved successfully");
    } catch (error) {
      showStatus(`Error saving: ${error instanceof Error ? error.message : "Unknown error"}`);
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div>
      {/* Header with save button */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: "15px",
        }}
      >
        <h2
          style={{
            marginBottom: "0",
            fontSize: "18px",
            fontWeight: 700,
            color: "#FFFFFF",
            letterSpacing: "-0.01em",
          }}
        >
          Regions (<span>{regions.length}</span>)
        </h2>
        <Button
          onClick={handleSave}
          disabled={!currentTrack || boundaries.length === 0 || isSaving}
          variant="primary"
          icon={<Save size={16} />}
        >
          {isSaving ? "Saving..." : "Save"}
        </Button>
      </div>

      {/* Region items */}
      <div>
        {regions.map((region, idx) => {
          return (
            <div
              key={idx}
              style={{
                background: "var(--bg-tertiary)",
                padding: "var(--space-3) var(--space-4)",
                borderRadius: "var(--radius-lg)",
                marginBottom: "var(--space-2)",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                transition: "all var(--transition-base)",
                border: "1px solid var(--border-subtle)",
                cursor: "pointer",
              }}
              onClick={() => seek(region.start)}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = "var(--bg-hover)";
                e.currentTarget.style.borderColor = "var(--border-focus)";
                e.currentTarget.style.transform = "translateY(-1px)";
                e.currentTarget.style.boxShadow = "var(--shadow-md)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = "var(--bg-tertiary)";
                e.currentTarget.style.borderColor = "var(--border-subtle)";
                e.currentTarget.style.transform = "translateY(0)";
                e.currentTarget.style.boxShadow = "none";
              }}
            >
              {/* Time */}
              <div
                style={{
                  fontWeight: "var(--font-weight-semibold)",
                  color: "var(--color-primary)",
                  minWidth: "120px",
                  fontSize: "var(--font-size-base)",
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
    </div>
  );
}
