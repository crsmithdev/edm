import { Plus, MapPin, Grid3x3 } from "lucide-react";
import { useAudioStore, useStructureStore, useTempoStore, useUIStore, useTrackStore } from "@/stores";
import { Button, Tooltip } from "@/components/UI";

/**
 * Editing row - three action buttons
 */
export function EditingControls() {
  const { currentTime } = useAudioStore();
  const { addBoundary } = useStructureStore();
  const { setDownbeat } = useTempoStore();
  const { quantizeEnabled, toggleQuantize, showStatus } = useUIStore();
  const { currentTrack } = useTrackStore();

  const handleSetDownbeat = () => {
    setDownbeat(currentTime);
    showStatus(`Downbeat set to ${currentTime.toFixed(2)}s`);
  };

  const handleAddBoundary = () => {
    addBoundary(currentTime);
    showStatus(`Added boundary at ${currentTime.toFixed(2)}s`);
  };

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(3, 1fr)",
        gap: "var(--space-3)",
        alignItems: "stretch",
      }}
    >
      <Tooltip content="Add boundary at current time" shortcut="B">
        <Button
          onClick={handleAddBoundary}
          disabled={!currentTrack}
          variant="primary"
          icon={<Plus size={16} />}
        >
          Boundary
        </Button>
      </Tooltip>
      <Tooltip content="Set downbeat at current time" shortcut="D">
        <Button
          onClick={handleSetDownbeat}
          disabled={!currentTrack}
          variant="primary"
          icon={<MapPin size={16} />}
        >
          Downbeat
        </Button>
      </Tooltip>
      <Tooltip content="Toggle boundary quantization to bars" shortcut="Q">
        <Button
          onClick={toggleQuantize}
          variant={quantizeEnabled ? "primary" : "secondary"}
          icon={<Grid3x3 size={16} />}
        >
          Quantize
        </Button>
      </Tooltip>
    </div>
  );
}
