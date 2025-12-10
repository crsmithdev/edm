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
import { KeyboardHints } from "./components/UI";

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
        padding: "var(--space-6)",
      }}
    >
      {/* Header */}
      <header style={{ marginBottom: "var(--space-6)" }}>
        <h1
          style={{
            color: "var(--text-primary)",
            fontSize: "var(--font-size-3xl)",
            fontWeight: "var(--font-weight-bold)",
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
          gap: "var(--space-5)",
          marginBottom: "var(--space-5)",
        }}
      >
        {/* Left Column: Controls */}
        <div
          style={{
            background: "var(--bg-secondary)",
            padding: "var(--space-5)",
            borderRadius: "var(--radius-xl)",
            border: "1px solid var(--border-primary)",
            boxShadow: "var(--shadow-md)",
            display: "flex",
            flexDirection: "column",
            gap: "var(--space-5)",
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
          marginTop: "var(--space-8)",
          fontSize: "var(--font-size-sm)",
          lineHeight: "var(--line-height-relaxed)",
          color: "var(--text-muted)",
        }}
      >
        <strong style={{ color: "var(--text-tertiary)", fontWeight: "var(--font-weight-semibold)" }}>Workflow:</strong>{" "}
        Shift+Click waveform to add boundaries → Label regions using dropdowns in list below
        <br />
        <strong style={{ color: "var(--text-tertiary)", fontWeight: "var(--font-weight-semibold)" }}>Navigate:</strong>{" "}
        Left/Right ±4 bars | Ctrl+Left/Right ±1 bar | Shift+Left/Right ±8 bars |
        Up/Down previous/next track
        <br />
        <strong style={{ color: "var(--text-tertiary)", fontWeight: "var(--font-weight-semibold)" }}>Playback:</strong>{" "}
        Click waveform to set cue point | Spacebar play/pause | C/R return to cue |
        Drag to pan | Scroll to zoom
        <br />
        <strong style={{ color: "var(--text-tertiary)", fontWeight: "var(--font-weight-semibold)" }}>Editing:</strong>{" "}
        B add boundary | D set downbeat | Q toggle quantize
      </div>

      {/* Status Toast */}
      <StatusToast />

      {/* Keyboard Shortcuts Helper */}
      <KeyboardHints />
    </div>
  );
}

export default App;
