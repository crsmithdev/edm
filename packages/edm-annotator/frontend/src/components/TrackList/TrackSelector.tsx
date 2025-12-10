import { useEffect } from "react";
import { useTrackStore, useWaveformStore, useTempoStore, useAudioStore, useStructureStore, useUIStore } from "@/stores";
import { trackService } from "@/services/api";

/**
 * Track list sidebar with selection and loading
 */
export function TrackSelector() {
  const { tracks, selectedTrack, fetchTracks, selectTrack, setCurrentTrack, isLoading } = useTrackStore();
  const { setWaveformData, reset: resetWaveform } = useWaveformStore();
  const { setBPM, setDownbeat, reset: resetTempo } = useTempoStore();
  const { player, reset: resetAudio } = useAudioStore();
  const { setBoundaries, reset: resetStructure } = useStructureStore();
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
      <div
        style={{
          padding: "24px",
          background: "#1E2139",
          borderRadius: "10px",
          border: "1px solid rgba(91, 124, 255, 0.1)",
          textAlign: "center",
          color: "#9CA3AF",
        }}
      >
        Loading tracks...
      </div>
    );
  }

  if (tracks.length === 0) {
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
        No audio files found. Check EDM_AUDIO_DIR environment variable.
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
        display: "flex",
        flexDirection: "column",
        maxHeight: "600px",
      }}
    >
      <div
        style={{
          padding: "16px",
          borderBottom: "1px solid rgba(91, 124, 255, 0.1)",
        }}
      >
        <h3 style={{ color: "#FFFFFF", fontSize: "16px", margin: 0 }}>
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
                padding: "12px 16px",
                borderBottom: "1px solid rgba(91, 124, 255, 0.05)",
                cursor: "pointer",
                background: isSelected ? "rgba(91, 124, 255, 0.1)" : "transparent",
                transition: "all 0.2s",
              }}
              onMouseEnter={(e) => {
                if (!isSelected) {
                  e.currentTarget.style.background = "#252A45";
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
                  color: isSelected ? "#5B7CFF" : "#E5E7EB",
                  fontWeight: isSelected ? 600 : 400,
                  marginBottom: "4px",
                  wordBreak: "break-word",
                }}
              >
                {track.has_reference && (
                  <span
                    style={{
                      color: "#00E6B8",
                      marginRight: "6px",
                      fontSize: "12px",
                    }}
                  >
                    ✓
                  </span>
                )}
                {track.filename}
              </div>
              <div style={{ fontSize: "11px", color: "#6B7280" }}>
                {track.has_reference && "Reference"}
                {track.has_generated && !track.has_reference && "Generated"}
                {!track.has_reference && !track.has_generated && "No annotation"}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
