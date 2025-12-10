import { useEffect } from "react";
import { useAudioPlayback } from "./hooks/useAudioPlayback";
import { useKeyboardShortcuts } from "./hooks/useKeyboardShortcuts";
import { useTrackStore } from "./stores";

// Components
import { WaveformContainer } from "./components/Waveform/WaveformContainer";
import { PlaybackControls } from "./components/Transport/PlaybackControls";
import { NavigationControls } from "./components/Transport/NavigationControls";
import { EditingControls } from "./components/Editing/EditingControls";
import { RegionList } from "./components/Editing/RegionList";
import { TrackSelector } from "./components/TrackList/TrackSelector";
import { StatusToast } from "./components/Layout/StatusToast";

/**
 * Main application component for EDM Structure Annotator
 */
function App() {
  const fetchTracks = useTrackStore((state) => state.fetchTracks);

  // Initialize audio playback
  useAudioPlayback();

  // Set up keyboard shortcuts
  useKeyboardShortcuts();

  // Fetch tracks on mount
  useEffect(() => {
    fetchTracks();
  }, [fetchTracks]);

  return (
    <div
      style={{
        maxWidth: "1600px",
        margin: "0 auto",
        padding: "24px",
      }}
    >
      {/* Header */}
      <header style={{ marginBottom: "24px" }}>
        <h1
          style={{
            color: "#FFFFFF",
            fontSize: "28px",
            fontWeight: 700,
            letterSpacing: "-0.02em",
            margin: 0,
          }}
        >
          EDM Structure Annotator
        </h1>
      </header>

      {/* Main Content */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 300px",
          gap: "20px",
        }}
      >
        {/* Left Column: Waveform and Controls */}
        <div>
          {/* Waveform */}
          <WaveformContainer />

          {/* Transport Controls */}
          <PlaybackControls />

          <div style={{ height: "16px" }} />

          {/* Navigation Controls */}
          <NavigationControls />

          <div style={{ height: "16px" }} />

          {/* Editing Controls */}
          <EditingControls />

          <div style={{ height: "16px" }} />

          {/* Region List */}
          <RegionList />
        </div>

        {/* Right Column: Track Selector */}
        <div>
          <TrackSelector />
        </div>
      </div>

      {/* Status Toast */}
      <StatusToast />

      {/* Keyboard Shortcuts Help */}
      <div
        style={{
          marginTop: "24px",
          padding: "16px",
          background: "#1E2139",
          borderRadius: "10px",
          border: "1px solid rgba(91, 124, 255, 0.1)",
          fontSize: "12px",
          color: "#6B7280",
        }}
      >
        <strong style={{ color: "#9CA3AF" }}>Keyboard Shortcuts:</strong>{" "}
        Space=Play/Pause | B=Add Boundary | D=Set Downbeat | Q=Toggle Quantize |
        C/R=Return to Cue | ←/→=Jump | ↑/↓=Previous/Next Track | +/-=Zoom | 0=Zoom
        Reset
      </div>
    </div>
  );
}

export default App;
