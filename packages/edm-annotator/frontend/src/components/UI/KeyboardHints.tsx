import { useEffect } from "react";
import { X } from "lucide-react";

interface Shortcut {
  key: string;
  description: string;
  category: string;
}

const shortcuts: Shortcut[] = [
  { key: "Space", description: "Play/Pause", category: "Playback" },
  { key: "C", description: "Set cue (stopped) / Return to cue (playing)", category: "Playback" },
  { key: "R", description: "Return to cue point", category: "Playback" },
  { key: "B", description: "Add boundary at playhead", category: "Editing" },
  { key: "D", description: "Set downbeat", category: "Editing" },
  { key: "Q", description: "Toggle quantize", category: "Editing" },
  { key: "←", description: "Jump back 4 bars", category: "Navigation" },
  { key: "→", description: "Jump forward 4 bars", category: "Navigation" },
  { key: "Ctrl + ←", description: "Jump back 1 bar", category: "Navigation" },
  { key: "Ctrl + →", description: "Jump forward 1 bar", category: "Navigation" },
  { key: "Shift + ←", description: "Jump back 8 bars", category: "Navigation" },
  { key: "Shift + →", description: "Jump forward 8 bars", category: "Navigation" },
  { key: "↑", description: "Previous track", category: "Navigation" },
  { key: "↓", description: "Next track", category: "Navigation" },
  { key: "Drag", description: "Scrub playback position", category: "Waveform" },
  { key: "Shift + Drag", description: "Scrub (bypass quantize)", category: "Waveform" },
  { key: "Ctrl + Click", description: "Add boundary at click position", category: "Waveform" },
  { key: "Scroll", description: "Zoom in/out", category: "Waveform" },
];

/**
 * Keyboard shortcuts trigger link
 */
export function KeyboardShortcutsLink({ onClick }: { onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      style={{
        background: "transparent",
        border: "none",
        color: "var(--text-tertiary)",
        fontSize: "var(--font-size-lg)",
        cursor: "pointer",
        padding: 0,
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.color = "var(--text-secondary)";
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.color = "var(--text-tertiary)";
      }}
      title="Keyboard shortcuts"
    >
      ⌨️
    </button>
  );
}

/**
 * Keyboard shortcuts overlay
 */
export function KeyboardHints({ onClose }: { onClose: () => void }) {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  const categories = Array.from(new Set(shortcuts.map((s) => s.category)));

  return (
    <>
      {/* Modal Overlay */}
      <div
        style={{
          position: "fixed",
          inset: 0,
          background: "rgba(0, 0, 0, 0.7)",
          zIndex: "var(--z-modal)",
          animation: "fadeIn 0.2s ease-out",
        }}
        onClick={onClose}
      />
          <div
            style={{
              position: "fixed",
              top: "50%",
              left: "50%",
              transform: "translate(-50%, -50%)",
              background: "var(--bg-secondary)",
              border: "1px solid var(--border-subtle)",
              borderRadius: "var(--radius-xl)",
              boxShadow: "var(--shadow-lg)",
              maxWidth: "800px",
              maxHeight: "80vh",
              width: "90%",
              overflow: "hidden",
              zIndex: "calc(var(--z-modal) + 1)",
              animation: "slideUp 0.2s ease-out",
            }}
          >
            {/* Header */}
            <div
              style={{
                padding: "var(--space-5)",
                borderBottom: "1px solid var(--border-subtle)",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              <h2
                style={{
                  fontSize: "var(--font-size-xl)",
                  fontWeight: "var(--font-weight-bold)",
                  color: "var(--text-primary)",
                  margin: 0,
                }}
              >
                Keyboard Shortcuts
              </h2>
              <button
                onClick={onClose}
                style={{
                  background: "transparent",
                  border: "none",
                  color: "var(--text-tertiary)",
                  cursor: "pointer",
                  padding: "var(--space-2)",
                  display: "flex",
                  alignItems: "center",
                }}
              >
                <X size={20} />
              </button>
            </div>

            {/* Content */}
            <div
              style={{
                padding: "var(--space-5)",
                overflowY: "auto",
                maxHeight: "calc(80vh - 80px)",
              }}
            >
              {categories.map((category) => (
                <div key={category} style={{ marginBottom: "var(--space-6)" }}>
                  <h3
                    style={{
                      fontSize: "var(--font-size-sm)",
                      fontWeight: "var(--font-weight-semibold)",
                      color: "var(--text-tertiary)",
                      textTransform: "uppercase",
                      letterSpacing: "0.5px",
                      marginBottom: "var(--space-3)",
                    }}
                  >
                    {category}
                  </h3>
                  <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-2)" }}>
                    {shortcuts
                      .filter((s) => s.category === category)
                      .map((shortcut, idx) => (
                        <div
                          key={idx}
                          style={{
                            display: "flex",
                            justifyContent: "space-between",
                            alignItems: "center",
                            padding: "var(--space-2)",
                            background: "var(--bg-tertiary)",
                            borderRadius: "var(--radius-sm)",
                          }}
                        >
                          <span
                            style={{
                              fontSize: "var(--font-size-sm)",
                              color: "var(--text-secondary)",
                            }}
                          >
                            {shortcut.description}
                          </span>
                          <kbd
                            style={{
                              padding: "var(--space-1) var(--space-2)",
                              background: "var(--bg-primary)",
                              border: "1px solid var(--border-subtle)",
                              borderRadius: "var(--radius-sm)",
                              fontSize: "var(--font-size-xs)",
                              fontWeight: "var(--font-weight-semibold)",
                              color: "var(--color-primary)",
                              fontFamily: "monospace",
                            }}
                          >
                            {shortcut.key}
                          </kbd>
                        </div>
                      ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
    </>
  );
}
