import { useEffect } from "react";
import { Play, CheckCircle } from "lucide-react";
import { useTrackStore, useWaveformStore, useTempoStore, useAudioStore, useStructureStore, useUIStore } from "@/stores";
import { trackService } from "@/services/api";
import { Card, TrackItemSkeleton, Button, Tooltip } from "@/components/UI";

/**
 * Track list sidebar with selection and loading
 */
export function TrackSelector() {
  const { tracks, selectedTrack, fetchTracks, selectTrack, setCurrentTrack, currentTrack, isLoading } = useTrackStore();
  const { setWaveformData, reset: resetWaveform } = useWaveformStore();
  const { setBPM, setDownbeat, reset: resetTempo } = useTempoStore();
  const { player, reset: resetAudio } = useAudioStore();
  const { reset: resetStructure } = useStructureStore();
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
          padding: "var(--space-4)",
          borderBottom: "1px solid var(--border-primary)",
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
            fontSize: "var(--font-size-lg)",
            fontWeight: "var(--font-weight-semibold)",
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

            return (
              <div
                key={track.filename}
                onClick={() => selectTrack(track.filename)}
                style={{
                  padding: "var(--space-3) var(--space-4)",
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
                    fontSize: "var(--font-size-sm)",
                    color: isSelected ? "var(--color-primary)" : "var(--text-secondary)",
                    fontWeight: isSelected ? "var(--font-weight-semibold)" : "var(--font-weight-normal)",
                    marginBottom: "var(--space-1)",
                    wordBreak: "break-word",
                    display: "flex",
                    alignItems: "center",
                    gap: "var(--space-2)",
                  }}
                >
                  {track.has_reference && (
                    <CheckCircle
                      size={14}
                      style={{
                        color: "var(--color-success)",
                        flexShrink: 0,
                      }}
                    />
                  )}
                  <span>{track.filename}</span>
                </div>
                <div
                  style={{
                    fontSize: "var(--font-size-xs)",
                    color: "var(--text-muted)",
                  }}
                >
                  {track.has_reference && "Reference annotation"}
                  {track.has_generated && !track.has_reference && "Generated annotation"}
                  {!track.has_reference && !track.has_generated && "No annotation"}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </Card>
  );
}
