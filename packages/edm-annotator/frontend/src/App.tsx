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

      {/* Waveform */}
      <WaveformContainer />

      {/* Two Column Layout */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "3fr 2fr",
          gap: "20px",
          marginBottom: "20px",
        }}
      >
        {/* Left Column: Controls */}
        <div
          style={{
            background: "#1E2139",
            padding: "20px",
            borderRadius: "14px",
            border: "1px solid rgba(91, 124, 255, 0.1)",
            boxShadow: "0 4px 6px rgba(0, 0, 0, 0.3)",
            display: "flex",
            flexDirection: "column",
            gap: "20px",
          }}
        >
          {/* Transport Row */}
          <PlaybackControls />

          {/* Editing Row */}
          <EditingControls />

          {/* Navigation Row */}
          <NavigationControls />

          {/* Regions List */}
          <RegionList />
        </div>

        {/* Right Column: Track List */}
        <div>
          <TrackSelector />
        </div>
      </div>

      {/* Instructions */}
      <div
        style={{
          marginTop: "30px",
          fontSize: "12px",
          lineHeight: "1.8",
          color: "#6B7280",
        }}
      >
        <strong style={{ color: "#9CA3AF", fontWeight: 600 }}>Workflow:</strong>{" "}
        Shift+Click waveform to add boundaries → Label regions using dropdowns in list below
        <br />
        <strong style={{ color: "#9CA3AF", fontWeight: 600 }}>Navigate:</strong>{" "}
        Left/Right ±4 bars | Ctrl+Left/Right ±1 bar | Shift+Left/Right ±8 bars |
        Up/Down previous/next track
        <br />
        <strong style={{ color: "#9CA3AF", fontWeight: 600 }}>Playback:</strong>{" "}
        Click waveform to set cue point | Spacebar play/pause | C/R return to cue |
        Drag to pan | Scroll to zoom
        <br />
        <strong style={{ color: "#9CA3AF", fontWeight: 600 }}>Editing:</strong>{" "}
        B add boundary | D set downbeat | Q toggle quantize
      </div>

      {/* Status Toast */}
      <StatusToast />
    </div>
  );
}

export default App;
