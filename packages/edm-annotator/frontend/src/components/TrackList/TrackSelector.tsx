import { useEffect } from "react";
import { Music, CheckCircle, FileQuestion } from "lucide-react";
import { useTrackStore, useWaveformStore, useTempoStore, useAudioStore, useStructureStore, useUIStore } from "@/stores";
import { trackService } from "@/services/api";
import { Card, EmptyState, TrackItemSkeleton } from "@/components/UI";

/**
 * Track list sidebar with selection and loading
 */
export function TrackSelector() {
  const { tracks, selectedTrack, fetchTracks, selectTrack, setCurrentTrack, isLoading } = useTrackStore();
  const { setWaveformData, reset: resetWaveform } = useWaveformStore();
  const { setBPM, setDownbeat, reset: resetTempo } = useTempoStore();
  const { player, reset: resetAudio } = useAudioStore();
  const { reset: resetStructure } = useStructureStore();
  const { showStatus } = useUIStore();

  useEffect(() => {
    fetchTracks();
  }, [fetchTracks]);

  const handleTrackClick = async (filename: string) => {
    selectTrack(filename);
    showStatus(`Loading ${filename}...`);

    try {
      // Load track data
      const data = await trackService.loadTrack(filename);

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
      }
      setDownbeat(data.downbeat);

      // Set audio source
      if (player) {
        player.src = trackService.getAudioUrl(filename);
        player.load();
      }

      setCurrentTrack(filename);
      showStatus(`Loaded ${filename}`);
    } catch (error) {
      showStatus(`Error loading track: ${error instanceof Error ? error.message : "Unknown error"}`);
    }
  };

  if (isLoading) {
    return (
      <Card style={{ display: "flex", flexDirection: "column", maxHeight: "600px" }}>
        <div
          style={{
            padding: "var(--space-4)",
            borderBottom: "1px solid var(--border-primary)",
          }}
        >
          <h3 style={{ color: "var(--text-primary)", fontSize: "var(--font-size-lg)", margin: 0 }}>
            Tracks
          </h3>
        </div>
        <div>
          {[...Array(5)].map((_, i) => (
            <TrackItemSkeleton key={i} />
          ))}
        </div>
      </Card>
    );
  }

  if (tracks.length === 0) {
    return (
      <Card>
        <EmptyState
          icon={<FileQuestion size={48} />}
          title="No Tracks Found"
          description="No audio files found. Check EDM_AUDIO_DIR environment variable or add audio files to your tracks directory."
        />
      </Card>
    );
  }

  return (
    <Card
      padding="sm"
      style={{
        overflow: "hidden",
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
          alignItems: "center",
          gap: "var(--space-2)",
        }}
      >
        <Music size={18} style={{ color: "var(--color-primary)" }} />
        <h3
          style={{
            color: "var(--text-primary)",
            fontSize: "var(--font-size-lg)",
            fontWeight: "var(--font-weight-semibold)",
            margin: 0,
          }}
        >
          Tracks ({tracks.length})
        </h3>
      </div>

      <div
        style={{
          flex: 1,
          overflowY: "auto",
        }}
      >
        {tracks.map((track) => {
          const isSelected = track.filename === selectedTrack;

          return (
            <div
              key={track.filename}
              onClick={() => handleTrackClick(track.filename)}
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
    </Card>
  );
}
