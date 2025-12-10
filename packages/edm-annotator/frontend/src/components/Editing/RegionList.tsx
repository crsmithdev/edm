import { useState } from "react";
import { useStructureStore, useAudioStore, useTempoStore, useUIStore, useTrackStore } from "@/stores";
import { trackService } from "@/services/api";
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
        <button
          onClick={handleSave}
          disabled={!currentTrack || boundaries.length === 0 || isSaving}
          style={{
            background: "#5B7CFF",
            color: "#FFFFFF",
            border: "none",
            cursor: currentTrack && boundaries.length > 0 && !isSaving ? "pointer" : "not-allowed",
            fontWeight: 600,
            transition: "all 0.2s",
            borderRadius: "8px",
            padding: "10px 18px",
            fontSize: "14px",
            opacity: currentTrack && boundaries.length > 0 && !isSaving ? 1 : 0.5,
          }}
        >
          {isSaving ? "Saving..." : "💾 Save Annotation"}
        </button>
      </div>

      {/* Region items */}
      <div>
        {regions.map((region, idx) => {
          return (
            <div
              key={idx}
              style={{
                background: "#151828",
                padding: "14px 16px",
                borderRadius: "10px",
                marginBottom: "10px",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                transition: "all 0.2s",
                border: "1px solid #2A2F4C",
                cursor: "pointer",
              }}
              onClick={() => seek(region.start)}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = "#1A1F38";
                e.currentTarget.style.borderColor = "#5B7CFF";
                e.currentTarget.style.transform = "translateY(-1px)";
                e.currentTarget.style.boxShadow = "0 4px 8px rgba(0, 0, 0, 0.2)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = "#151828";
                e.currentTarget.style.borderColor = "#2A2F4C";
                e.currentTarget.style.transform = "translateY(0)";
                e.currentTarget.style.boxShadow = "none";
              }}
            >
              {/* Time */}
              <div
                style={{
                  fontWeight: 600,
                  color: "#5B7CFF",
                  minWidth: "120px",
                  fontSize: "14px",
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
                  padding: "6px 12px",
                  border: "1px solid #2A2F4C",
                  background: "#0F1419",
                  color: "#E5E7EB",
                  borderRadius: "6px",
                  fontSize: "12px",
                  fontWeight: 600,
                  textTransform: "uppercase",
                  cursor: "pointer",
                  transition: "all 0.2s",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = "#1A1F38";
                  e.currentTarget.style.borderColor = "#5B7CFF";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = "#0F1419";
                  e.currentTarget.style.borderColor = "#2A2F4C";
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
