import { useState, useCallback } from "react";
import { Download, Trash2 } from "lucide-react";
import { useStructureStore, useTrackStore, useUIStore, useTempoStore, useWaveformStore } from "@/stores";
import { Card, Button, Tooltip, ConfirmDialog } from "@/components/UI";
import { trackService } from "@/services/api";
import { RegionList } from "./RegionList";
import { SaveButton } from "./SaveButton";

/**
 * Container for regions list with header
 */
export function RegionListContainer() {
  const { regions, isDirty, setBoundaries, clearBoundaries, markAsSaved, annotationTier, setAnnotationTier } =
    useStructureStore();
  const { currentTrack } = useTrackStore();
  const { setBPM, setDownbeat } = useTempoStore();
  const { duration } = useWaveformStore();
  const { showStatus } = useUIStore();
  const [isLoading, setIsLoading] = useState(false);
  const [showConfirm, setShowConfirm] = useState<"load" | "clear" | null>(null);

  const loadGenerated = useCallback(async () => {
    if (!currentTrack) return;

    setIsLoading(true);
    try {
      const data = await trackService.loadGeneratedAnnotation(currentTrack);

      // Set boundaries from generated annotation
      // Boundaries from the file represent region starts
      // We need to add the end of the track as the final boundary
      const boundaryTimes = data.boundaries.map((b) => b.time);
      if (boundaryTimes[boundaryTimes.length - 1] !== duration) {
        boundaryTimes.push(duration);
      }
      setBoundaries(boundaryTimes);

      // Set region labels
      const { regions: newRegions } = useStructureStore.getState();
      data.boundaries.forEach((boundary, idx) => {
        if (idx < newRegions.length && boundary.label) {
          useStructureStore.getState().setRegionLabel(idx, boundary.label);
        }
      });

      // Update BPM and downbeat if available
      if (data.bpm) setBPM(data.bpm);
      if (data.downbeat !== undefined) setDownbeat(data.downbeat);

      // Set annotation tier to 2 (generated)
      setAnnotationTier(2);

      markAsSaved();
      showStatus(`Loaded ${data.boundaries.length} boundaries from generated annotation`);
    } catch (error) {
      showStatus(
        `Error loading generated annotation: ${error instanceof Error ? error.message : "Unknown error"}`
      );
    } finally {
      setIsLoading(false);
      setShowConfirm(null);
    }
  }, [currentTrack, duration, setBoundaries, setBPM, setDownbeat, markAsSaved, showStatus, setAnnotationTier]);

  const handleLoad = useCallback(async () => {
    if (!currentTrack) {
      showStatus("No track loaded");
      return;
    }

    // Check for unsaved changes
    if (isDirty()) {
      setShowConfirm("load");
      return;
    }

    await loadGenerated();
  }, [currentTrack, isDirty, showStatus, loadGenerated]);

  const doClear = useCallback(() => {
    clearBoundaries();
    markAsSaved();
    showStatus("Cleared all boundaries");
    setShowConfirm(null);
  }, [clearBoundaries, markAsSaved, showStatus]);

  const handleClear = useCallback(() => {
    // Check for unsaved changes
    if (isDirty()) {
      setShowConfirm("clear");
      return;
    }

    doClear();
  }, [isDirty, doClear]);

  return (
    <>
      <Card
        padding="sm"
        style={{
          overflow: "visible",
          display: "flex",
          flexDirection: "column",
          maxHeight: "600px",
          padding: 0,
        }}
      >
        <div
          style={{
            padding: "var(--space-4)",
            borderBottom: "1px solid var(--border-primary)",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            overflow: "visible",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "var(--space-3)" }}>
            <h3
              style={{
                color: "var(--text-primary)",
                fontSize: "20px",
                fontWeight: "var(--font-weight-normal)",
                letterSpacing: "var(--letter-spacing-tight)",
                margin: 0,
              }}
            >
              Annotations
            </h3>
            {annotationTier === 2 && (
              <span
                style={{
                  fontSize: "var(--font-size-xs)",
                  color: "var(--text-tertiary)",
                  background: "var(--bg-elevated)",
                  padding: "var(--space-1) var(--space-2)",
                  borderRadius: "var(--radius-sm)",
                  border: "1px solid var(--border-subtle)",
                  textTransform: "uppercase",
                  letterSpacing: "0.05em",
                  fontWeight: "var(--font-weight-semibold)",
                }}
              >
                Generated
              </span>
            )}
          </div>
          <div style={{ display: "flex", gap: "var(--space-2)" }}>
            <Tooltip content="Load generated boundaries">
              <Button
                onClick={handleLoad}
                disabled={!currentTrack || isLoading}
                variant="accent"
                icon={<Download size={16} />}
                style={{ width: "auto" }}
              >
                Load Generated
              </Button>
            </Tooltip>
            <Tooltip content="Clear all boundaries">
              <Button
                onClick={handleClear}
                disabled={!currentTrack || regions.length === 0}
                variant="secondary"
                icon={<Trash2 size={16} />}
                style={{ width: "auto" }}
              >
                Clear
              </Button>
            </Tooltip>
            <SaveButton />
          </div>
        </div>

        {regions.length === 0 ? (
          <div
            style={{
              padding: "var(--space-6)",
              textAlign: "center",
              color: "var(--text-muted)",
              fontSize: "var(--font-size-sm)",
            }}
          >
            No annotations defined
          </div>
        ) : (
          <RegionList />
        )}
      </Card>

      {showConfirm === "load" && (
        <ConfirmDialog
          title="Unsaved Changes"
          message="You have unsaved changes. Loading generated boundaries will discard these changes. Continue?"
          confirmText="Load Anyway"
          cancelText="Cancel"
          variant="warning"
          onConfirm={loadGenerated}
          onCancel={() => setShowConfirm(null)}
        />
      )}

      {showConfirm === "clear" && (
        <ConfirmDialog
          title="Unsaved Changes"
          message="You have unsaved changes. Clearing boundaries will discard these changes. Continue?"
          confirmText="Clear Anyway"
          cancelText="Cancel"
          variant="danger"
          onConfirm={doClear}
          onCancel={() => setShowConfirm(null)}
        />
      )}
    </>
  );
}
