import { useState, useEffect, useCallback } from "react";
import { Save } from "lucide-react";
import { useStructureStore, useTempoStore, useUIStore, useTrackStore } from "@/stores";
import { trackService } from "@/services/api";
import { Button, Tooltip } from "@/components/UI";

/**
 * Save button for saving annotations
 */
export function SaveButton() {
  const { regions, markAsSaved, setAnnotationTier, isDirty } = useStructureStore();
  const { trackBPM, trackDownbeat } = useTempoStore();
  const { showStatus } = useUIStore();
  const { currentTrack, updateTrackStatus } = useTrackStore();
  const [isSaving, setIsSaving] = useState(false);

  const hasChanges = isDirty();

  const handleSave = useCallback(async () => {
    if (!currentTrack) {
      showStatus("No track loaded");
      return;
    }

    setIsSaving(true);
    try {
      const response = await trackService.saveAnnotation({
        filename: currentTrack,
        bpm: trackBPM,
        downbeat: trackDownbeat,
        boundaries: regions.map((r) => ({ time: r.start, label: r.label })),
      });
      // Set tier to 1 (reference) since we just saved a hand-tagged annotation
      setAnnotationTier(1);
      markAsSaved();
      // Update track status in the track list
      updateTrackStatus(currentTrack, true, false);
      showStatus(`Saved ${response.boundaries_count} boundaries to ${response.output}`);
    } catch (error) {
      showStatus(`Error saving annotation: ${error instanceof Error ? error.message : "Unknown error"}`);
    } finally {
      setIsSaving(false);
    }
  }, [currentTrack, trackBPM, trackDownbeat, regions, showStatus, markAsSaved, setAnnotationTier, updateTrackStatus]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ctrl+S or Cmd+S for save
      if ((e.ctrlKey || e.metaKey) && e.key === "s") {
        e.preventDefault();
        if (currentTrack && regions.length > 0 && !isSaving && hasChanges) {
          handleSave();
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [currentTrack, regions.length, isSaving, hasChanges, handleSave]);

  return (
    <Tooltip content="Save annotation" shortcut="Ctrl+S">
      <Button
        onClick={handleSave}
        disabled={!currentTrack || regions.length === 0 || isSaving || !hasChanges}
        variant="primary"
        icon={<Save size={16} />}
        style={{ width: "auto" }}
      >
        {isSaving ? "Saving..." : "Save"}
      </Button>
    </Tooltip>
  );
}
