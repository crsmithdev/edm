import { useState } from "react";
import { Save } from "lucide-react";
import { useStructureStore, useTempoStore, useUIStore, useTrackStore } from "@/stores";
import { trackService } from "@/services/api";
import { Button, Tooltip } from "@/components/UI";

/**
 * Save button for saving annotations
 */
export function SaveButton() {
  const { regions, boundaries } = useStructureStore();
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
        boundaries: regions.map((r) => ({ time: r.start, label: r.label })),
      });
      showStatus("Annotation saved successfully");
    } catch (error) {
      showStatus(`Error saving: ${error instanceof Error ? error.message : "Unknown error"}`);
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <Tooltip content="Save annotation" shortcut="Ctrl+S">
      <Button
        onClick={handleSave}
        disabled={!currentTrack || boundaries.length === 0 || isSaving}
        variant="primary"
        icon={<Save size={16} />}
        style={{ width: "auto" }}
      >
        {isSaving ? "Saving..." : "Save"}
      </Button>
    </Tooltip>
  );
}
