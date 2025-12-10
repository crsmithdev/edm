import { useState } from "react";
import { useAudioPlayback } from "./hooks/useAudioPlayback";
import { useKeyboardShortcuts } from "./hooks/useKeyboardShortcuts";

// Components
import { WaveformContainer } from "./components/Waveform/WaveformContainer";
import { PlaybackControls } from "./components/Transport/PlaybackControls";
import { NavigationControls } from "./components/Transport/NavigationControls";
import { EditingControls } from "./components/Editing/EditingControls";
import { RegionListContainer } from "./components/Editing/RegionListContainer";
import { TrackSelector } from "./components/TrackList/TrackSelector";
import { StatusToast } from "./components/Layout/StatusToast";
import { KeyboardHints, KeyboardShortcutsLink } from "./components/UI";

/**
 * Main application component for EDM Structure Annotator
 */
function App() {
  const [showShortcuts, setShowShortcuts] = useState(false);

  // Initialize audio playback
  useAudioPlayback();

  // Set up keyboard shortcuts
  useKeyboardShortcuts();

  return (
    <div
      style={{
        maxWidth: "1600px",
        margin: "0 auto",
        padding: "var(--space-6)",
        overflowX: "hidden",
      }}
    >
      {/* Header */}
      <header
        style={{
          marginBottom: "var(--space-6)",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
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
        <KeyboardShortcutsLink onClick={() => setShowShortcuts(true)} />
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
        {/* Left Column: Controls and Regions */}
        <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-5)" }}>
          {/* Controls Card */}
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
          </div>

          {/* Regions List */}
          <RegionListContainer />
        </div>

        {/* Right Column: Track List */}
        <div>
          <TrackSelector />
        </div>
      </div>

      {/* Status Toast */}
      <StatusToast />

      {/* Keyboard Shortcuts Modal */}
      {showShortcuts && <KeyboardHints onClose={() => setShowShortcuts(false)} />}
    </div>
  );
}

export default App;
