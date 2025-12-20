import { useEffect } from "react";
import { Play, CheckCircle, Sparkles, Circle, ArrowUpDown, ArrowUp, ArrowDown } from "lucide-react";
import { useTrackStore, useWaveformStore, useTempoStore, useAudioStore, useStructureStore, useUIStore } from "@/stores";
import { trackService } from "@/services/api";
import { Card, TrackItemSkeleton, Button, Tooltip } from "@/components/UI";

// Helper to parse filename into artist and title
function parseFilename(filename: string): { artist: string; title: string } {
  const nameWithoutExt = filename.replace(/\.[^.]+$/, "");
  const parts = nameWithoutExt.split(" - ");

  if (parts.length >= 2) {
    return {
      artist: parts[0].trim(),
      title: parts.slice(1).join(" - ").trim(),
    };
  }

  return {
    artist: "",
    title: nameWithoutExt,
  };
}

/**
 * Track list sidebar with selection and loading
 */
export function TrackSelector() {
  const { tracks, selectedTrack, fetchTracks, selectTrack, setCurrentTrack, currentTrack, isLoading, sortBy, sortDirection, setSorting } = useTrackStore();
  const { setWaveformData, reset: resetWaveform } = useWaveformStore();
  const { setBPM, setDownbeat, reset: resetTempo } = useTempoStore();
  const { player, reset: resetAudio } = useAudioStore();
  const { reset: resetStructure, setBoundaries, markAsSaved, setAnnotationTier } = useStructureStore();
  const { showStatus } = useUIStore();

  // Auto-load track list on mount
  useEffect(() => {
    fetchTracks();
  }, [fetchTracks]);

  const handleLoadTrack = async () => {
    if (!selectedTrack) return;

    showStatus(`Loading ${selectedTrack}...`);

    try {
      // Load track data
      const data = await trackService.loadTrack(selectedTrack);

      // Reset all stores
      resetWaveform();
      resetTempo();
      resetStructure();
      resetAudio();

      // Update waveform store
      setWaveformData({
        waveform_bass: data.waveform_bass,
        waveform_mids: data.waveform_mids,
        waveform_highs: data.waveform_highs,
        waveform_times: data.waveform_times,
        duration: data.duration,
      });

      // Load boundaries from annotation if available, otherwise use default
      if (data.boundaries && data.boundaries.length > 0) {
        // Boundaries from the file represent region starts
        // We need to add the end of the track as the final boundary
        const boundaryTimes = data.boundaries.map((b) => b.time);
        if (boundaryTimes[boundaryTimes.length - 1] !== data.duration) {
          boundaryTimes.push(data.duration);
        }
        setBoundaries(boundaryTimes);

        // Set region labels
        const { regions } = useStructureStore.getState();
        data.boundaries.forEach((boundary, idx) => {
          if (idx < regions.length && boundary.label) {
            useStructureStore.getState().setRegionLabel(idx, boundary.label);
          }
        });

        // Set annotation tier (1 = reference, 2 = generated)
        setAnnotationTier(data.annotation_tier || null);
      } else {
        // Initialize with a single region spanning the entire track
        setBoundaries([0, data.duration]);
        setAnnotationTier(null);
      }

      // Update tempo store
      if (data.bpm) {
        setBPM(data.bpm);
      } else {
        // Set to 0 when no annotation exists (will display as "--")
        setBPM(0);
      }
      setDownbeat(data.downbeat);

      // Set audio source
      if (player) {
        player.src = trackService.getAudioUrl(selectedTrack);
        player.load();
      }

      // Mark initial state as saved
      markAsSaved();

      setCurrentTrack(selectedTrack);
      showStatus(`Loaded ${selectedTrack}`);
    } catch (error) {
      showStatus(`Error loading track: ${error instanceof Error ? error.message : "Unknown error"}`);
    }
  };

  return (
    <Card
      padding="sm"
      style={{
        display: "flex",
        flexDirection: "column",
        maxHeight: "600px",
        padding: 0,
      }}
    >
      <div
        style={{
          padding: "var(--space-3) var(--space-4)",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          position: "relative",
          overflow: "visible",
        }}
      >
        <h3
          style={{
            color: "var(--text-primary)",
            fontSize: "20px",
            fontWeight: "var(--font-weight-normal)",
            letterSpacing: "var(--letter-spacing-tight)",
            margin: 0,
          }}
        >
          Tracks
        </h3>
        <div style={{ width: "auto" }}>
          <Tooltip content="Load selected track">
            <Button
              onClick={handleLoadTrack}
              disabled={!selectedTrack || selectedTrack === currentTrack}
              variant="primary"
              icon={<Play size={16} />}
              style={{ width: "auto" }}
            >
              Load
            </Button>
          </Tooltip>
        </div>
      </div>

      {isLoading ? (
        <div>
          {[...Array(5)].map((_, i) => (
            <TrackItemSkeleton key={i} />
          ))}
        </div>
      ) : tracks.length === 0 ? (
        <div
          style={{
            padding: "var(--space-6)",
            textAlign: "center",
            color: "var(--text-muted)",
            fontSize: "var(--font-size-sm)",
          }}
        >
          No tracks loaded
        </div>
      ) : (
        <>
          {/* Sorting controls */}
          <div
            style={{
              padding: "var(--space-2) var(--space-4) var(--space-3)",
              borderBottom: "1px solid var(--border-primary)",
              display: "flex",
              gap: "var(--space-2)",
              alignItems: "center",
            }}
          >
            {(["artist", "title", "status"] as const).map((field) => {
              const isActive = sortBy === field;
              const Icon = isActive ? (sortDirection === "asc" ? ArrowUp : ArrowDown) : ArrowUpDown;

              return (
                <button
                  key={field}
                  onClick={() => setSorting(field)}
                  style={{
                    background: isActive ? "rgba(123, 106, 255, 0.1)" : "transparent",
                    color: isActive ? "var(--color-primary)" : "var(--text-secondary)",
                    border: `1px solid ${isActive ? "var(--color-primary)" : "var(--border-primary)"}`,
                    borderRadius: "var(--radius-sm)",
                    padding: "var(--space-1) var(--space-2)",
                    fontSize: "var(--font-size-xs)",
                    cursor: "pointer",
                    display: "flex",
                    alignItems: "center",
                    gap: "var(--space-1)",
                    transition: "all var(--transition-base)",
                    fontWeight: "var(--font-weight-normal)",
                  }}
                  onMouseEnter={(e) => {
                    if (!isActive) {
                      e.currentTarget.style.background = "var(--bg-elevated)";
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (!isActive) {
                      e.currentTarget.style.background = "transparent";
                    }
                  }}
                >
                  <Icon size={12} />
                  {field.charAt(0).toUpperCase() + field.slice(1)}
                </button>
              );
            })}
          </div>

          <div
            style={{
              flex: 1,
              overflowY: "auto",
              overflow: "hidden auto",
              padding: "var(--space-1) 0 var(--space-4) 0",
            }}
          >
            {tracks.map((track) => {
            const isSelected = track.filename === selectedTrack;
            const { artist, title } = parseFilename(track.filename);

            // Determine status icon
            let StatusIcon;
            let statusColor;
            if (track.has_reference) {
              StatusIcon = CheckCircle;
              statusColor = "var(--color-success)";
            } else if (track.has_generated) {
              StatusIcon = Sparkles;
              statusColor = "rgba(123, 106, 255, 0.8)";
            } else {
              StatusIcon = Circle;
              statusColor = "var(--text-muted)";
            }

            return (
              <div
                key={track.filename}
                onClick={() => selectTrack(track.filename)}
                style={{
                  padding: "var(--space-2) var(--space-4)",
                  borderBottom: "1px solid var(--border-primary)",
                  cursor: "pointer",
                  background: isSelected ? "rgba(91, 124, 255, 0.1)" : "transparent",
                  transition: "all var(--transition-base)",
                }}
                onMouseEnter={(e) => {
                  if (!isSelected) {
                    e.currentTarget.style.background = "var(--bg-elevated)";
                  }
                }}
                onMouseLeave={(e) => {
                  if (!isSelected) {
                    e.currentTarget.style.background = "transparent";
                  }
                }}
              >
                <div
                  style={{
                    fontSize: "13px",
                    color: isSelected ? "var(--color-primary)" : "var(--text-secondary)",
                    fontWeight: "var(--font-weight-normal)",
                    marginBottom: "var(--space-1)",
                    wordBreak: "break-word",
                    display: "flex",
                    alignItems: "center",
                    gap: "var(--space-2)",
                  }}
                >
                  <StatusIcon
                    size={14}
                    style={{
                      color: statusColor,
                      flexShrink: 0,
                    }}
                  />
                  <span>{title}</span>
                </div>
                {artist && (
                  <div
                    style={{
                      fontSize: "var(--font-size-sm)",
                      color: "var(--text-muted)",
                      paddingLeft: "calc(14px + var(--space-2))", // Align with title text
                    }}
                  >
                    {artist}
                  </div>
                )}
              </div>
            );
          })}
          </div>
        </>
      )}
    </Card>
  );
}
